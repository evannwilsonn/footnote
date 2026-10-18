// Command builddemo writes docs/index.html from web/template.html and the results files.
//
//	go run ./cmd/builddemo
//
// It reads results/choice.json (the settings picked on the tuning articles), results/ingest.json,
// results/retrieval.json and results/answers-test-k<k>.json, and picks the example questions for
// the page from the test run: some answered correctly, some correctly declined, some wrong.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
)

type source struct {
	Title string `json:"title"`
	Text  string `json:"text"`
	Gold  bool   `json:"gold"`
}

type record struct {
	ID         string   `json:"id"`
	Question   string   `json:"question"`
	Doc        string   `json:"doc"`
	Impossible bool     `json:"impossible"`
	Gold       []string `json:"gold"`
	Condition  string   `json:"condition"`
	Answer     string   `json:"answer"`
	Abstained  bool     `json:"abstained"`
	EM         float64  `json:"em"`
	F1         float64  `json:"f1"`
	Retrieved  bool     `json:"retrieved"`
	Cited      []int    `json:"cited"`
	Sources    []source `json:"sources,omitempty"`
}

type side struct {
	Answer    string   `json:"answer"`
	Abstained bool     `json:"abstained"`
	EM        float64  `json:"em"`
	F1        float64  `json:"f1"`
	Retrieved bool     `json:"retrieved"`
	Cited     []int    `json:"cited"`
	Sources   []source `json:"sources,omitempty"`
}

type example struct {
	Group      string   `json:"group"`
	Question   string   `json:"question"`
	Doc        string   `json:"doc"`
	Impossible bool     `json:"impossible"`
	Gold       []string `json:"gold"`
	Closed     side     `json:"closed"`
	RAG        side     `json:"rag"`
}

func read(path string, v any) {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("%s: %v", path, err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		log.Fatalf("%s: %v", path, err)
	}
}

// compact encodes v for an inline <script>, so "</" can't close the tag early.
func compact(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		log.Fatal(err)
	}
	return strings.ReplaceAll(string(b), "</", `<\/`)
}

func main() {
	var choice struct {
		Chunker string `json:"chunker"`
		Method  string `json:"method"`
		K       int    `json:"k"`
	}
	read("results/choice.json", &choice)
	var ingest, retrieval map[string]any
	read("results/ingest.json", &ingest)
	read("results/retrieval.json", &retrieval)
	var gate map[string]any
	read("results/gate.json", &gate)
	gate["shipped"] = 0.6392 // the threshold the server ships with (FOOTNOTE_MIN_SIMILARITY)
	var answers struct {
		K       int            `json:"k"`
		Summary map[string]any `json:"summary"`
		Records []record       `json:"records"`
	}
	read(fmt.Sprintf("results/answers-test-k%d.json", choice.K), &answers)

	// Pair each question's closed-book and retrieval answers.
	byID := map[string]*example{}
	var order []string
	for _, r := range answers.Records {
		ex := byID[r.ID]
		if ex == nil {
			ex = &example{Question: r.Question, Doc: r.Doc, Impossible: r.Impossible, Gold: r.Gold}
			byID[r.ID] = ex
			order = append(order, r.ID)
		}
		s := side{Answer: r.Answer, Abstained: r.Abstained, EM: r.EM, F1: r.F1, Retrieved: r.Retrieved, Cited: r.Cited, Sources: r.Sources}
		if r.Condition == "rag" {
			ex.RAG = s
		} else {
			ex.Closed = s
		}
	}
	sort.Strings(order) // stable and arbitrary: ids are hashes, so this doesn't favour any article

	// One example per article per group keeps the docket varied.
	pick := func(group string, n int, keep func(e *example) bool) []example {
		var out []example
		seen := map[string]bool{}
		for _, id := range order {
			e := byID[id]
			if len(out) == n || seen[e.Doc] || len(e.Question) > 90 || !keep(e) {
				continue
			}
			seen[e.Doc] = true
			c := *e
			c.Group = group
			out = append(out, c)
		}
		return out
	}
	var examples []example
	// Right with a citation, where the model alone got it wrong: what retrieval adds.
	examples = append(examples, pick("right", 5, func(e *example) bool {
		return !e.Impossible && e.RAG.EM == 1 && e.Closed.EM == 0 && len(e.RAG.Cited) > 0
	})...)
	examples = append(examples, pick("declined", 3, func(e *example) bool { return e.Impossible && e.RAG.Abstained })...)
	examples = append(examples, pick("wrong", 2, func(e *example) bool { return e.Impossible && !e.RAG.Abstained })...)
	examples = append(examples, pick("wrong", 2, func(e *example) bool {
		return !e.Impossible && !e.RAG.Abstained && e.RAG.F1 == 0
	})...)

	page, err := os.ReadFile("web/template.html")
	if err != nil {
		log.Fatal(err)
	}
	html := string(page)
	fills := map[string]string{
		"{{RETRIEVAL}}": compact(retrieval),
		"{{ANSWERS}}":   compact(map[string]any{"k": answers.K, "summary": answers.Summary}),
		"{{EXAMPLES}}":  compact(examples),
		"{{INGEST}}":    compact(ingest),
		"{{CHOSEN}}":    compact(choice),
		"{{GATE}}":      compact(gate),
	}
	for k, v := range fills {
		if strings.Count(html, k) != 1 {
			log.Fatalf("template must contain %s exactly once", k)
		}
		html = strings.Replace(html, k, v, 1)
	}
	if err := os.MkdirAll("docs", 0o755); err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("docs/index.html", []byte(html), 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote docs/index.html (%d bytes, %d example questions)\n", len(html), len(examples))
}
