// Command footnote answers questions about a set of documents, citing the passages it used.
//
//	footnote ingest                      chunk, embed and store the documents (all chunkers)
//	footnote eval-retrieval              recall@k and MRR for every chunker and search method
//	footnote eval-answers -split S -n N  answer quality with and without retrieval
//	footnote ask "question"              answer one question
//	footnote serve -addr :8080           HTTP API: POST /ask {"question": "..."}
//
// Settings come from the environment: DATABASE_URL, OLLAMA_URL, EMBED_MODEL, CHAT_MODEL, and for
// ask/serve FOOTNOTE_CHUNKER, FOOTNOTE_METHOD, FOOTNOTE_K.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/evannwilsonn/footnote/internal/corpus"
	"github.com/evannwilsonn/footnote/internal/ollama"
	"github.com/evannwilsonn/footnote/internal/rag"
	"github.com/evannwilsonn/footnote/internal/store"
)

const (
	dataPath  = "data/dev-v2.0.json"
	splitSeed = 2026
	tuneDocs  = 10 // of 35 articles; the other 25 are the test set
)

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

type app struct {
	ctx   context.Context
	store *store.Store
	llm   *ollama.Client
	docs  []corpus.Document
	qs    []corpus.Question
}

func setup(needData bool) (*app, error) {
	ctx := context.Background()
	st, err := store.Open(ctx, env("DATABASE_URL", "postgres://rag:rag@localhost:5432/rag"))
	if err != nil {
		return nil, err
	}
	a := &app{ctx: ctx, store: st, llm: ollama.New(env("OLLAMA_URL", "http://localhost:11434"), env("EMBED_MODEL", "nomic-embed-text"), env("CHAT_MODEL", "qwen2.5:1.5b"))}
	if needData {
		if a.docs, a.qs, err = corpus.Load(dataPath); err != nil {
			return nil, err
		}
	}
	return a, nil
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println("wrote", path)
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func round(v float64, places int) float64 {
	p := math.Pow(10, float64(places))
	return math.Round(v*p) / p
}

// ---------- ingest ----------

func ingest(args []string) error {
	fs := flag.NewFlagSet("ingest", flag.ExitOnError)
	only := fs.String("chunkers", strings.Join(corpus.SortedNames(), ","), "comma-separated chunkers")
	fs.Parse(args)
	a, err := setup(true)
	if err != nil {
		return err
	}
	if err := a.store.Migrate(a.ctx); err != nil {
		return err
	}
	if err := a.store.ReplaceDocuments(a.ctx, a.docs); err != nil {
		return err
	}
	report := map[string]any{"documents": len(a.docs), "bytes": totalBytes(a.docs), "embedModel": a.llm.EmbedModel}
	for _, name := range strings.Split(*only, ",") {
		c, ok := corpus.Chunkers[name]
		if !ok {
			return fmt.Errorf("unknown chunker %q", name)
		}
		var chunks []corpus.Chunk
		for _, d := range a.docs {
			chunks = append(chunks, c.Split(d)...)
		}
		start := time.Now()
		// Embeddings are cached so a failed database write never means embedding everything again.
		path := a.cachePath("chunks-" + name)
		vecs, cached := readVectors(path, len(chunks))
		for i := 0; !cached && i < len(chunks); i += 16 {
			batch := chunks[i:min(i+16, len(chunks))]
			texts := make([]string, len(batch))
			for j, ch := range batch {
				texts[j] = rag.DocPrefix + a.docs[ch.DocID].Title + "\n" + ch.Text
			}
			v, err := a.llm.Embed(a.ctx, texts)
			if err != nil {
				return err
			}
			vecs = append(vecs, v...)
			if (i/16)%20 == 0 {
				log.Printf("%s: embedded %d of %d chunks", name, len(vecs), len(chunks))
			}
		}
		if !cached {
			if err := writeVectors(path, vecs); err != nil {
				return err
			}
		}
		embedSecs := time.Since(start).Seconds()
		if err := a.store.ReplaceChunks(a.ctx, name, chunks, vecs); err != nil {
			return err
		}
		words := 0
		for _, ch := range chunks {
			words += len(strings.Fields(ch.Text))
		}
		log.Printf("%s: %d chunks stored in %.0fs", name, len(chunks), time.Since(start).Seconds())
		report[name] = map[string]any{"chunks": len(chunks), "meanWords": round(float64(words)/float64(len(chunks)), 1), "embedSeconds": round(embedSecs, 1)}
	}
	return writeJSON("results/ingest.json", report)
}

func totalBytes(docs []corpus.Document) int {
	n := 0
	for _, d := range docs {
		n += len(d.Text)
	}
	return n
}

// ---------- question embeddings (cached) ----------

// readVectors loads n cached vectors from path; ok is false when the file is missing or the wrong size.
func readVectors(path string, n int) (out [][]float32, ok bool) {
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) != n*store.Dim*4 {
		return nil, false
	}
	out = make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, store.Dim)
		for j := range out[i] {
			out[i][j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[(i*store.Dim+j)*4:]))
		}
	}
	return out, true
}

func writeVectors(path string, vecs [][]float32) error {
	raw := make([]byte, 0, len(vecs)*store.Dim*4)
	for _, v := range vecs {
		for _, x := range v {
			raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(x))
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

func (a *app) cachePath(kind string) string {
	return fmt.Sprintf("results/cache/%s-%s.bin", kind, strings.ReplaceAll(a.llm.EmbedModel, ":", "_"))
}

// questionVectors embeds every question once per prefix setting and caches the result.
func (a *app) questionVectors(prefix bool) ([][]float32, error) {
	path := a.cachePath(fmt.Sprintf("questions-prefix%v", prefix))
	if out, ok := readVectors(path, len(a.qs)); ok {
		return out, nil
	}
	log.Printf("embedding %d questions (prefix %v)", len(a.qs), prefix)
	out := make([][]float32, 0, len(a.qs))
	for i := 0; i < len(a.qs); i += 64 {
		batch := a.qs[i:min(i+64, len(a.qs))]
		texts := make([]string, len(batch))
		for j, q := range batch {
			texts[j] = q.Text
			if prefix {
				texts[j] = rag.QueryPrefix + q.Text
			}
		}
		v, err := a.llm.Embed(a.ctx, texts)
		if err != nil {
			return nil, err
		}
		out = append(out, v...)
	}
	return out, writeVectors(path, out)
}

// ---------- eval-retrieval ----------

type retrievalScores struct {
	Questions int                `json:"questions"`
	Recall    map[string]float64 `json:"recall"` // "1", "3", "5", "10", "20"
	MRR       float64            `json:"mrr"`
	MeanMs    float64            `json:"meanMs"`
}

var cutoffs = []int{1, 3, 5, 10, 20}

func evalRetrieval(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval", flag.ExitOnError)
	fs.Parse(args)
	a, err := setup(true)
	if err != nil {
		return err
	}
	tune, test, tuneSet := corpus.SplitByDocument(a.docs, a.qs, tuneDocs, splitSeed)
	index := map[string]int{}
	for i, q := range a.qs {
		index[q.ID] = i
	}
	vecs := map[bool][][]float32{}
	for _, p := range []bool{false, true} {
		if vecs[p], err = a.questionVectors(p); err != nil {
			return err
		}
	}
	type setting struct {
		name   string
		method rag.Method
		prefix bool
	}
	settings := []setting{
		{"lexical", rag.Lexical, false},
		{"dense-noprefix", rag.Dense, false},
		{"dense", rag.Dense, true},
		{"hybrid", rag.Hybrid, true},
	}
	out := map[string]any{"split": map[string]any{"seed": splitSeed, "tuneDocuments": tuneDocs, "tuneDocIds": keys(tuneSet)}}
	for _, split := range []struct {
		name string
		qs   []corpus.Question
	}{{"tune", tune}, {"test", test}} {
		res := map[string]map[string]retrievalScores{}
		for _, ch := range corpus.SortedNames() {
			res[ch] = map[string]retrievalScores{}
			for _, s := range settings {
				r := &rag.Retriever{Store: a.store, LLM: a.llm, Chunker: ch, Method: s.method, Prefix: s.prefix}
				hits := make([]int, len(cutoffs))
				var mrr, ms float64
				n := 0
				for _, q := range split.qs {
					if q.Impossible {
						continue
					}
					n++
					start := time.Now()
					found, err := r.Search(a.ctx, q.Text, vecs[s.prefix][index[q.ID]], 20)
					if err != nil {
						return err
					}
					ms += float64(time.Since(start).Microseconds()) / 1000
					rank := 0
					for i, h := range found {
						if h.DocID == q.DocID && h.Chunk().Contains(q.Answers) {
							rank = i + 1
							break
						}
					}
					if rank > 0 {
						mrr += 1 / float64(rank)
						for i, c := range cutoffs {
							if rank <= c {
								hits[i]++
							}
						}
					}
				}
				sc := retrievalScores{Questions: n, Recall: map[string]float64{}, MRR: round(mrr/float64(n), 4), MeanMs: round(ms/float64(n), 2)}
				for i, c := range cutoffs {
					sc.Recall[strconv.Itoa(c)] = round(float64(hits[i])/float64(n), 4)
				}
				res[ch][s.name] = sc
				log.Printf("%-4s %-5s %-14s recall@1 %.3f @5 %.3f @20 %.3f  mrr %.3f  %.1f ms", split.name, ch, s.name, sc.Recall["1"], sc.Recall["5"], sc.Recall["20"], sc.MRR, sc.MeanMs)
			}
		}
		out[split.name] = res
	}
	return writeJSON("results/retrieval.json", out)
}

func keys(m map[int]bool) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

// ---------- eval-answers ----------

type answerRecord struct {
	ID         string   `json:"id"`
	Question   string   `json:"question"`
	Doc        string   `json:"doc"`
	Impossible bool     `json:"impossible"`
	Gold       []string `json:"gold"`
	Condition  string   `json:"condition"` // "closed" or "rag"
	Answer     string   `json:"answer"`
	Raw        string   `json:"raw"`
	Abstained  bool     `json:"abstained"`
	EM         float64  `json:"em"`
	F1         float64  `json:"f1"`
	Retrieved  bool     `json:"retrieved"` // rag only: an answer-bearing chunk was among the sources
	Cited      []int    `json:"cited"`
	Sources    []source `json:"sources,omitempty"`
	Ms         float64  `json:"ms"`
}

type source struct {
	Title string `json:"title"`
	Text  string `json:"text"`
	Gold  bool   `json:"gold"`
}

func evalAnswers(args []string) error {
	fs := flag.NewFlagSet("eval-answers", flag.ExitOnError)
	splitName := fs.String("split", "tune", "tune or test")
	n := fs.Int("n", 100, "answerable and unanswerable questions each")
	k := fs.Int("k", 3, "sources shown to the model")
	chunker := fs.String("chunker", "para", "chunker")
	method := fs.String("method", "dense", "lexical, dense or hybrid")
	tag := fs.String("tag", "", "name for this run (default split-k)")
	fs.Parse(args)
	a, err := setup(true)
	if err != nil {
		return err
	}
	tune, test, _ := corpus.SplitByDocument(a.docs, a.qs, tuneDocs, splitSeed)
	pool := tune
	if *splitName == "test" {
		pool = test
	}
	var ans, imp []corpus.Question
	for _, q := range pool {
		if q.Impossible {
			imp = append(imp, q)
		} else {
			ans = append(ans, q)
		}
	}
	rng := rand.New(rand.NewSource(splitSeed + 1))
	rng.Shuffle(len(ans), func(i, j int) { ans[i], ans[j] = ans[j], ans[i] })
	rng.Shuffle(len(imp), func(i, j int) { imp[i], imp[j] = imp[j], imp[i] })
	sample := append(append([]corpus.Question{}, ans[:min(*n, len(ans))]...), imp[:min(*n, len(imp))]...)

	name := *tag
	if name == "" {
		name = fmt.Sprintf("%s-k%d", *splitName, *k)
	}
	cache := "results/cache/answers-" + name + ".jsonl"
	done := map[string]answerRecord{}
	if f, err := os.Open(cache); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for sc.Scan() {
			var r answerRecord
			if json.Unmarshal(sc.Bytes(), &r) == nil {
				done[r.ID+"/"+r.Condition] = r
			}
		}
		f.Close()
	}
	os.MkdirAll(filepath.Dir(cache), 0o755)
	out, err := os.OpenFile(cache, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer out.Close()

	r := &rag.Retriever{Store: a.store, LLM: a.llm, Chunker: *chunker, Method: rag.Method(*method), Prefix: true}
	var records []answerRecord
	for i, q := range sample {
		golds := make([]string, len(q.Answers))
		for j, s := range q.Answers {
			golds[j] = s.Text
		}
		for _, cond := range []string{"closed", "rag"} {
			if rec, ok := done[q.ID+"/"+cond]; ok {
				records = append(records, rec)
				continue
			}
			var hits []store.Hit
			if cond == "rag" {
				if hits, err = r.Search(a.ctx, q.Text, nil, *k); err != nil {
					return err
				}
			}
			start := time.Now()
			res, err := rag.Ask(a.ctx, a.llm, q.Text, hits)
			if err != nil {
				return err
			}
			rec := answerRecord{ID: q.ID, Question: q.Text, Doc: a.docs[q.DocID].Title, Impossible: q.Impossible, Gold: golds, Condition: cond,
				Answer: res.Text, Raw: res.Raw, Abstained: res.Abstained, Cited: res.Cited, Ms: round(float64(time.Since(start).Milliseconds()), 0)}
			if !q.Impossible && !res.Abstained {
				rec.EM, rec.F1 = rag.ExactMatch(res.Text, golds), round(rag.F1(res.Text, golds), 4)
			}
			for _, h := range hits {
				g := h.DocID == q.DocID && h.Chunk().Contains(q.Answers)
				rec.Retrieved = rec.Retrieved || g
				rec.Sources = append(rec.Sources, source{Title: h.Title, Text: h.Text, Gold: g})
			}
			line, _ := json.Marshal(rec)
			out.Write(append(line, '\n'))
			records = append(records, rec)
		}
		if (i+1)%20 == 0 {
			log.Printf("%s: %d of %d questions", name, i+1, len(sample))
		}
	}
	return writeJSON("results/answers-"+name+".json", summarize(name, *chunker, *method, *k, records))
}

func summarize(name, chunker, method string, k int, recs []answerRecord) map[string]any {
	type agg struct {
		n, abst, found      int
		em, f1, ms, emFound float64
	}
	g := map[string]*agg{}
	for _, r := range recs {
		key := r.Condition + map[bool]string{true: "/unanswerable", false: "/answerable"}[r.Impossible]
		if g[key] == nil {
			g[key] = &agg{}
		}
		x := g[key]
		x.n++
		x.ms += r.Ms
		if r.Abstained {
			x.abst++
		}
		x.em += r.EM
		x.f1 += r.F1
		if r.Retrieved {
			x.found++
			x.emFound += r.EM
		}
	}
	summary := map[string]any{}
	for key, x := range g {
		s := map[string]any{"questions": x.n, "abstained": round(float64(x.abst)/float64(x.n), 4), "meanMs": round(x.ms/float64(x.n), 0)}
		if strings.HasSuffix(key, "/answerable") {
			s["exactMatch"] = round(x.em/float64(x.n), 4)
			s["f1"] = round(x.f1/float64(x.n), 4)
			if strings.HasPrefix(key, "rag") {
				s["sourceHadAnswer"] = round(float64(x.found)/float64(x.n), 4)
				if x.found > 0 {
					s["exactMatchWhenSourceHadAnswer"] = round(x.emFound/float64(x.found), 4)
				}
			}
		}
		summary[key] = s
	}
	return map[string]any{"name": name, "chunker": chunker, "method": method, "k": k, "summary": summary, "records": recs}
}

// ---------- ask and serve ----------

func (a *app) retriever() (*rag.Retriever, int) {
	k, _ := strconv.Atoi(env("FOOTNOTE_K", "3"))
	return &rag.Retriever{Store: a.store, LLM: a.llm, Chunker: env("FOOTNOTE_CHUNKER", "para"), Method: rag.Method(env("FOOTNOTE_METHOD", "dense")), Prefix: true}, k
}

type askResponse struct {
	Question  string   `json:"question"`
	Answer    string   `json:"answer"`
	Abstained bool     `json:"abstained"`
	Cited     []int    `json:"cited"`
	Sources   []hitOut `json:"sources"`
	Ms        int64    `json:"ms"`
}

type hitOut struct {
	N     int     `json:"n"`
	Title string  `json:"title"`
	Text  string  `json:"text"`
	Score float64 `json:"score"`
	Kept  bool    `json:"kept"` // passed the relevance grader and was shown to the model
}

func (a *app) answer(q string) (askResponse, error) {
	start := time.Now()
	r, k := a.retriever()
	hits, err := r.Search(a.ctx, q, nil, k)
	if err != nil {
		return askResponse{}, err
	}
	// With dense search, decline without asking the model when even the best passage is a weak
	// match. The default threshold was picked on the tuning articles (see eval-gate).
	minSim, _ := strconv.ParseFloat(env("FOOTNOTE_MIN_SIMILARITY", "0.6392"), 64)
	declined := rag.Answer{Text: "I don't know.", Abstained: true}
	// Then the course's relevance grader (CRAG) drops passages the model judges off topic, and
	// the model answers from the rest. On the test questions this took unanswerable declines from
	// 24% to 59% (see eval-graders). FOOTNOTE_GRADER=off skips it.
	kept := make([]bool, len(hits))
	var res rag.Answer
	if r.Method == rag.Dense && (len(hits) == 0 || hits[0].Score < minSim) {
		res = declined
	} else {
		var use []store.Hit
		var idx []int // position in hits of each passage in use
		for i, h := range hits {
			ok := true
			if env("FOOTNOTE_GRADER", "relevance") != "off" {
				if ok, err = rag.GradeDoc(a.ctx, a.llm, rag.Relevance, q, h); err != nil {
					return askResponse{}, err
				}
			}
			if ok {
				kept[i] = true
				use = append(use, h)
				idx = append(idx, i)
			}
		}
		if len(use) == 0 {
			res = declined
		} else if res, err = rag.Ask(a.ctx, a.llm, q, use); err != nil {
			return askResponse{}, err
		}
		// Citations number the passages the model saw; renumber them as the full list.
		for j, c := range res.Cited {
			if c >= 1 && c <= len(idx) {
				res.Cited[j] = idx[c-1] + 1
			}
		}
	}
	out := askResponse{Question: q, Answer: res.Text, Abstained: res.Abstained, Cited: res.Cited, Ms: time.Since(start).Milliseconds()}
	for i, h := range hits {
		out.Sources = append(out.Sources, hitOut{N: i + 1, Title: h.Title, Text: h.Text, Score: round(h.Score, 4), Kept: kept[i]})
	}
	return out, nil
}

func ask(args []string) error {
	if len(args) == 0 {
		return errors.New(`usage: footnote ask "question"`)
	}
	a, err := setup(false)
	if err != nil {
		return err
	}
	res, err := a.answer(strings.Join(args, " "))
	if err != nil {
		return err
	}
	if res.Abstained {
		fmt.Println("I don't know.")
	} else {
		fmt.Println(res.Answer)
	}
	for _, s := range res.Sources {
		mark := " "
		for _, c := range res.Cited {
			if c == s.N {
				mark = "*"
			}
		}
		text := s.Text
		if len(text) > 160 {
			text = text[:160] + "…"
		}
		fmt.Printf("%s[%d] %s: %s\n", mark, s.N, s.Title, strings.ReplaceAll(text, "\n", " "))
	}
	fmt.Printf("(%d ms)\n", res.Ms)
	return nil
}

func serve(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", ":8080", "listen address")
	fs.Parse(args)
	a, err := setup(false)
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"ok": true}`))
	})
	mux.HandleFunc("POST /ask", func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			Question string `json:"question"`
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&in); err != nil || strings.TrimSpace(in.Question) == "" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error": "send {\"question\": \"...\"}"}`))
			return
		}
		res, err := a.answer(strings.TrimSpace(in.Question))
		if err != nil {
			log.Printf("ask: %v", err)
			w.WriteHeader(http.StatusBadGateway)
			json.NewEncoder(w).Encode(map[string]string{"error": "the model or database is unavailable"})
			return
		}
		json.NewEncoder(w).Encode(res)
	})
	// The page's self-hosted typeface (Crimson Pro, SIL OFL), from docs/fonts like on GitHub Pages.
	mux.Handle("GET /fonts/", http.FileServer(http.Dir("docs")))
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, "docs/index.html")
	})
	log.Printf("footnote: listening on %s (POST /ask)", *addr)
	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	return srv.ListenAndServe()
}

func main() {
	log.SetFlags(log.Ltime)
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: footnote ingest | eval-retrieval | eval-answers | eval-gate | eval-translate | eval-graders | ask \"question\" | serve")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "ingest":
		err = ingest(os.Args[2:])
	case "eval-retrieval":
		err = evalRetrieval(os.Args[2:])
	case "eval-answers":
		err = evalAnswers(os.Args[2:])
	case "eval-translate":
		err = evalTranslate(os.Args[2:])
	case "eval-graders":
		err = evalGraders(os.Args[2:])
	case "eval-gate":
		err = evalGate(os.Args[2:])
	case "ask":
		err = ask(os.Args[2:])
	case "serve":
		err = serve(os.Args[2:])
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		log.Fatal(err)
	}
}
