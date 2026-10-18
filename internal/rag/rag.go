// Package rag ties retrieval and generation together: find chunks for a question, show them to a
// small local model as numbered sources, and parse an answer with citations. It also holds the
// SQuAD scoring used to evaluate it.
package rag

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/evannwilsonn/footnote/internal/ollama"
	"github.com/evannwilsonn/footnote/internal/store"
)

// ---------- Retrieval ----------

type Method string

const (
	Lexical Method = "lexical"
	Dense   Method = "dense"
	Hybrid  Method = "hybrid"
)

// QueryPrefix is what nomic-embed-text expects in front of a search query (documents get
// "search_document: "). Whether it helps is measured, not assumed.
const QueryPrefix = "search_query: "
const DocPrefix = "search_document: "

type Retriever struct {
	Store   *store.Store
	LLM     *ollama.Client
	Chunker string
	Method  Method
	Prefix  bool
}

// Fuse merges ranked lists with reciprocal-rank fusion: each list adds 1/(60 + rank) to a
// chunk's score. It needs no score calibration between the two kinds of search.
func Fuse(lists [][]store.Hit, k int) []store.Hit {
	const c = 60.0
	score := map[int64]float64{}
	first := map[int64]store.Hit{}
	for _, list := range lists {
		for rank, h := range list {
			score[h.ChunkID] += 1 / (c + float64(rank+1))
			if _, ok := first[h.ChunkID]; !ok {
				first[h.ChunkID] = h
			}
		}
	}
	out := make([]store.Hit, 0, len(score))
	for id, s := range score {
		h := first[id]
		h.Score = s
		out = append(out, h)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].ChunkID < out[j].ChunkID
	})
	if len(out) > k {
		out = out[:k]
	}
	return out
}

func (r *Retriever) EmbedQuestion(ctx context.Context, q string) ([]float32, error) {
	if r.Prefix {
		q = QueryPrefix + q
	}
	v, err := r.LLM.Embed(ctx, []string{q})
	if err != nil {
		return nil, err
	}
	return v[0], nil
}

// Search returns the top k chunks. vec may be nil, in which case it is computed when needed.
func (r *Retriever) Search(ctx context.Context, question string, vec []float32, k int) ([]store.Hit, error) {
	if r.Method != Lexical && vec == nil {
		var err error
		if vec, err = r.EmbedQuestion(ctx, question); err != nil {
			return nil, err
		}
	}
	switch r.Method {
	case Lexical:
		return r.Store.Lexical(ctx, r.Chunker, question, k)
	case Dense:
		return r.Store.Dense(ctx, r.Chunker, vec, k)
	case Hybrid:
		const pool = 20
		lex, err := r.Store.Lexical(ctx, r.Chunker, question, pool)
		if err != nil {
			return nil, err
		}
		den, err := r.Store.Dense(ctx, r.Chunker, vec, pool)
		if err != nil {
			return nil, err
		}
		return Fuse([][]store.Hit{lex, den}, k), nil
	}
	return nil, fmt.Errorf("unknown method %q", r.Method)
}

// ---------- Answering ----------

const ragSystem = `You answer questions using only the numbered sources you are given.
Reply with the shortest exact phrase from the sources that answers the question, usually a few words, then the number of the source you used in brackets. For example: Richard Fleischer [2]
If the sources do not contain the answer, reply exactly: I don't know.`

const closedSystem = `Answer the question with the shortest phrase that answers it, usually a few words.
If you are not sure of the answer, reply exactly: I don't know.`

type Answer struct {
	Text      string        `json:"answer"` // the answer, citations removed
	Raw       string        `json:"raw"`    // what the model wrote
	Abstained bool          `json:"abstained"`
	Cited     []int         `json:"cited"` // 1-based source numbers the model cited
	Sources   []store.Hit   `json:"-"`
	Duration  time.Duration `json:"-"`
	Prompt    int           `json:"promptTokens"`
}

// Prompt builds the user message: numbered sources, then the question.
func Prompt(question string, hits []store.Hit) string {
	var b strings.Builder
	for i, h := range hits {
		fmt.Fprintf(&b, "[%d] (%s)\n%s\n\n", i+1, h.Title, strings.TrimSpace(h.Text))
	}
	b.WriteString("Question: " + question)
	return b.String()
}

// Ask answers from the given sources; with no sources it answers closed-book.
func Ask(ctx context.Context, llm *ollama.Client, question string, hits []store.Hit) (Answer, error) {
	msgs := []ollama.Message{{Role: "system", Content: closedSystem}, {Role: "user", Content: question}}
	if len(hits) > 0 {
		msgs = []ollama.Message{{Role: "system", Content: ragSystem}, {Role: "user", Content: Prompt(question, hits)}}
	}
	res, err := llm.Chat(ctx, msgs, 40)
	if err != nil {
		return Answer{}, err
	}
	a := Parse(res.Text, len(hits))
	a.Sources, a.Duration, a.Prompt = hits, res.TotalDuration, res.PromptTokens
	return a, nil
}

var citeRe = regexp.MustCompile(`\[(\d+)\]`)

// Parse separates the answer from its [n] citations and spots "I don't know".
func Parse(raw string, sources int) Answer {
	a := Answer{Raw: raw}
	for _, m := range citeRe.FindAllStringSubmatch(raw, -1) {
		if n, err := strconv.Atoi(m[1]); err == nil && n >= 1 && n <= sources {
			a.Cited = append(a.Cited, n)
		}
	}
	text := citeRe.ReplaceAllString(raw, "")
	if i := strings.IndexByte(text, '\n'); i >= 0 {
		text = text[:i]
	}
	text = strings.TrimSpace(strings.TrimRight(strings.TrimSpace(text), "."))
	a.Text = text
	n := Normalize(text)
	a.Abstained = n == "" || strings.HasPrefix(n, "i dont know") || strings.HasPrefix(n, "i do not know") ||
		strings.Contains(n, "sources do not contain") || strings.Contains(n, "not mentioned")
	if a.Abstained {
		a.Text = ""
	}
	return a
}

// ---------- SQuAD scoring ----------

var articles = regexp.MustCompile(`\b(a|an|the)\b`)

// Normalize follows the official SQuAD script: lower case, drop punctuation and articles,
// collapse whitespace.
func Normalize(s string) string {
	s = strings.ToLower(s)
	s = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			return -1
		}
		return r
	}, s)
	s = articles.ReplaceAllString(s, " ")
	return strings.Join(strings.Fields(s), " ")
}

func ExactMatch(pred string, golds []string) float64 {
	for _, g := range golds {
		if Normalize(pred) == Normalize(g) {
			return 1
		}
	}
	return 0
}

func F1(pred string, golds []string) float64 {
	best := 0.0
	p := strings.Fields(Normalize(pred))
	for _, g := range golds {
		gt := strings.Fields(Normalize(g))
		counts := map[string]int{}
		for _, t := range gt {
			counts[t]++
		}
		same := 0
		for _, t := range p {
			if counts[t] > 0 {
				same++
				counts[t]--
			}
		}
		if same == 0 {
			continue
		}
		prec, rec := float64(same)/float64(len(p)), float64(same)/float64(len(gt))
		if f := 2 * prec * rec / (prec + rec); f > best {
			best = f
		}
	}
	return best
}
