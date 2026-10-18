package main

// eval-translate and eval-graders test the techniques from "Learn RAG From Scratch" on the same
// questions as everything else.
//
//   eval-translate: does rewriting the question (multi-query, RAG-Fusion, step-back, HyDE) find
//                   the answer passage more often than the question alone?
//   eval-graders:   do the course's yes/no graders (CRAG document grading, adaptive RAG's
//                   hallucination check) stop the model answering questions the documents can't?
//
// Every model call is cached in results/cache/, so both commands can be stopped and resumed.

import (
	"bufio"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/evannwilsonn/footnote/internal/corpus"
	"github.com/evannwilsonn/footnote/internal/rag"
	"github.com/evannwilsonn/footnote/internal/store"
)

// jsonCache is an append-only file of {"key": ..., "value": ...} lines.
type jsonCache struct {
	f    *os.File
	seen map[string]json.RawMessage
}

func openCache(path string) (*jsonCache, error) {
	c := &jsonCache{seen: map[string]json.RawMessage{}}
	if f, err := os.Open(path); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for sc.Scan() {
			var row struct {
				Key   string          `json:"key"`
				Value json.RawMessage `json:"value"`
			}
			if json.Unmarshal(sc.Bytes(), &row) == nil {
				c.seen[row.Key] = row.Value
			}
		}
		f.Close()
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	c.f = f
	return c, err
}

// cached returns the stored value for key, or computes, stores and returns it.
func cached[T any](c *jsonCache, key string, compute func() (T, error)) (T, error) {
	var v T
	if raw, ok := c.seen[key]; ok {
		return v, json.Unmarshal(raw, &v)
	}
	v, err := compute()
	if err != nil {
		return v, err
	}
	raw, _ := json.Marshal(v)
	line, _ := json.Marshal(map[string]any{"key": key, "value": json.RawMessage(raw)})
	c.f.Write(append(line, '\n'))
	c.seen[key] = raw
	return v, nil
}

func hashKey(parts ...string) string {
	h := sha1.Sum([]byte(strings.Join(parts, "\x00")))
	return hex.EncodeToString(h[:8])
}

// ---------- eval-translate ----------

type translations struct {
	Multi    []string `json:"multi"`
	Fusion   []string `json:"fusion"`
	StepBack string   `json:"stepBack"`
	HyDE     string   `json:"hyde"`
	Ms       struct {
		Multi, Fusion, StepBack, HyDE float64
	} `json:"ms"`
}

func evalTranslate(args []string) error {
	fs := flag.NewFlagSet("eval-translate", flag.ExitOnError)
	splitName := fs.String("split", "tune", "tune or test")
	n := fs.Int("n", 150, "answerable questions to sample")
	chunker := fs.String("chunker", "para", "chunker")
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
	var qs []corpus.Question
	for _, q := range pool {
		if !q.Impossible {
			qs = append(qs, q)
		}
	}
	rng := rand.New(rand.NewSource(splitSeed + 7))
	rng.Shuffle(len(qs), func(i, j int) { qs[i], qs[j] = qs[j], qs[i] })
	qs = qs[:min(*n, len(qs))]

	cache, err := openCache("results/cache/translate.jsonl")
	if err != nil {
		return err
	}
	defer cache.f.Close()

	const pool20 = 20
	dense := func(text, prefix string) ([]store.Hit, error) {
		v, err := a.llm.Embed(a.ctx, []string{prefix + text})
		if err != nil {
			return nil, err
		}
		return a.store.Dense(a.ctx, *chunker, v[0], pool20)
	}
	denseAll := func(texts []string) ([][]store.Hit, error) {
		var out [][]store.Hit
		for _, t := range texts {
			h, err := dense(t, rag.QueryPrefix)
			if err != nil {
				return nil, err
			}
			out = append(out, h)
		}
		return out, nil
	}

	methods := []string{"question", "multi-query", "rag-fusion", "rag-fusion+question", "step-back", "hyde", "hyde-as-document"}
	ranks := map[string][]int{}
	llmMs := map[string]float64{}
	empty := map[string]int{}
	for i, q := range qs {
		tr, err := cached(cache, "tr:"+q.ID, func() (translations, error) {
			var t translations
			var err error
			s := time.Now()
			if t.Multi, err = rag.MultiQuery(a.ctx, a.llm, q.Text); err != nil {
				return t, err
			}
			t.Ms.Multi = float64(time.Since(s).Milliseconds())
			s = time.Now()
			if t.Fusion, err = rag.FusionQueries(a.ctx, a.llm, q.Text); err != nil {
				return t, err
			}
			t.Ms.Fusion = float64(time.Since(s).Milliseconds())
			s = time.Now()
			if t.StepBack, err = rag.StepBack(a.ctx, a.llm, q.Text); err != nil {
				return t, err
			}
			t.Ms.StepBack = float64(time.Since(s).Milliseconds())
			s = time.Now()
			if t.HyDE, err = rag.HyDE(a.ctx, a.llm, q.Text); err != nil {
				return t, err
			}
			t.Ms.HyDE = float64(time.Since(s).Milliseconds())
			return t, nil
		})
		if err != nil {
			return err
		}
		llmMs["multi-query"] += tr.Ms.Multi
		llmMs["rag-fusion"] += tr.Ms.Fusion
		llmMs["rag-fusion+question"] += tr.Ms.Fusion
		llmMs["step-back"] += tr.Ms.StepBack
		llmMs["hyde"] += tr.Ms.HyDE
		llmMs["hyde-as-document"] += tr.Ms.HyDE

		base, err := dense(q.Text, rag.QueryPrefix)
		if err != nil {
			return err
		}
		results := map[string][]store.Hit{"question": base}
		// The course retrieves for the generated queries only; "+question" also keeps the original.
		withFallback := func(gen []string, method string) []string {
			if len(gen) == 0 {
				empty[method]++
				return []string{q.Text}
			}
			return gen
		}
		lists, err := denseAll(withFallback(tr.Multi, "multi-query"))
		if err != nil {
			return err
		}
		results["multi-query"] = rag.MaxSim(lists, pool20)
		if lists, err = denseAll(withFallback(tr.Fusion, "rag-fusion")); err != nil {
			return err
		}
		results["rag-fusion"] = rag.Fuse(lists, pool20)
		results["rag-fusion+question"] = rag.Fuse(append(lists, base), pool20)
		sb, err := dense(tr.StepBack, rag.QueryPrefix)
		if err != nil {
			return err
		}
		results["step-back"] = rag.Fuse([][]store.Hit{base, sb}, pool20)
		if results["hyde"], err = dense(tr.HyDE, rag.QueryPrefix); err != nil {
			return err
		}
		if results["hyde-as-document"], err = dense(tr.HyDE, rag.DocPrefix); err != nil {
			return err
		}
		for _, m := range methods {
			rank := 0
			for j, h := range results[m] {
				if h.DocID == q.DocID && h.Chunk().Contains(q.Answers) {
					rank = j + 1
					break
				}
			}
			ranks[m] = append(ranks[m], rank)
		}
		if (i+1)%25 == 0 {
			log.Printf("translate %s: %d of %d", *splitName, i+1, len(qs))
		}
	}

	out := map[string]any{"split": *splitName, "questions": len(qs), "chunker": *chunker}
	scores := map[string]any{}
	for _, m := range methods {
		hits := make([]int, len(cutoffs))
		var mrr float64
		for _, r := range ranks[m] {
			if r == 0 {
				continue
			}
			mrr += 1 / float64(r)
			for i, c := range cutoffs {
				if r <= c {
					hits[i]++
				}
			}
		}
		rec := map[string]float64{}
		for i, c := range cutoffs {
			rec[strconv.Itoa(c)] = round(float64(hits[i])/float64(len(qs)), 4)
		}
		scores[m] = map[string]any{"recall": rec, "mrr": round(mrr/float64(len(qs)), 4), "llmSeconds": round(llmMs[m]/float64(len(qs))/1000, 2), "noQueriesParsed": empty[m]}
		log.Printf("%-20s recall@1 %.3f @3 %.3f @5 %.3f @10 %.3f  mrr %.3f  +%.1f s", m, rec["1"], rec["3"], rec["5"], rec["10"], mrr/float64(len(qs)), llmMs[m]/float64(len(qs))/1000)
	}
	out["methods"] = scores
	// A few examples of what the model wrote, for the README and the page.
	var samples []map[string]any
	for _, q := range qs[:min(4, len(qs))] {
		var tr translations
		json.Unmarshal(cache.seen["tr:"+q.ID], &tr)
		samples = append(samples, map[string]any{"question": q.Text, "multi": tr.Multi, "fusion": tr.Fusion, "stepBack": tr.StepBack, "hyde": tr.HyDE})
	}
	out["samples"] = samples
	return writeJSON(fmt.Sprintf("results/translate-%s.json", *splitName), out)
}

// ---------- eval-graders ----------

func evalGraders(args []string) error {
	fs := flag.NewFlagSet("eval-graders", flag.ExitOnError)
	run := fs.String("answers", "results/answers-test-k3.json", "the answer run to grade")
	n := fs.Int("n", 80, "answerable and unanswerable questions each, taken in the run's (random) order")
	minSim := fs.Float64("gate", 0.6392, "the shipped similarity gate, to compare and combine with the graders")
	fs.Parse(args)
	a, err := setup(true)
	if err != nil {
		return err
	}
	// Top-1 similarity per question, for the gate rows (question vectors are cached).
	vecs, err := a.questionVectors(true)
	if err != nil {
		return err
	}
	topScore := map[string]float64{}
	for i, q := range a.qs {
		hits, err := a.store.Dense(a.ctx, "para", vecs[i], 1)
		if err != nil {
			return err
		}
		if len(hits) > 0 {
			topScore[q.ID] = hits[0].Score
		}
	}
	raw, err := os.ReadFile(*run)
	if err != nil {
		return err
	}
	var file struct {
		Records []answerRecord `json:"records"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return err
	}
	var recs []answerRecord
	nAns, nImp := 0, 0
	for _, r := range file.Records {
		if r.Condition != "rag" {
			continue
		}
		if r.Impossible && nImp < *n {
			nImp++
			recs = append(recs, r)
		} else if !r.Impossible && nAns < *n {
			nAns++
			recs = append(recs, r)
		}
	}
	cache, err := openCache("results/cache/graders.jsonl")
	if err != nil {
		return err
	}
	defer cache.f.Close()

	hits := func(r answerRecord) []store.Hit {
		out := make([]store.Hit, len(r.Sources))
		for i, s := range r.Sources {
			out[i] = store.Hit{Title: s.Title, Text: s.Text}
		}
		return out
	}
	gradeAll := func(r answerRecord, g rag.Grader) ([]bool, error) {
		out := make([]bool, len(r.Sources))
		for i, h := range hits(r) {
			ok, err := cached(cache, "doc:"+string(g)+":"+r.ID+":"+strconv.Itoa(i), func() (bool, error) { return rag.GradeDoc(a.ctx, a.llm, g, r.Question, h) })
			if err != nil {
				return nil, err
			}
			out[i] = ok
		}
		return out, nil
	}
	// answerWith answers from the kept sources; keeping all of them reuses the original answer,
	// which the same deterministic prompt would reproduce.
	answerWith := func(r answerRecord, keep []bool) (rag.Answer, []store.Hit, error) {
		all := hits(r)
		var kept []store.Hit
		for i, k := range keep {
			if k {
				kept = append(kept, all[i])
			}
		}
		if len(kept) == 0 {
			return rag.Answer{Abstained: true}, nil, nil
		}
		if len(kept) == len(all) {
			return rag.Answer{Text: r.Answer, Abstained: r.Abstained}, kept, nil
		}
		keyParts := []string{r.ID}
		for _, h := range kept {
			keyParts = append(keyParts, h.Text)
		}
		ans, err := cached(cache, "ans:"+hashKey(keyParts...), func() (rag.Answer, error) { return rag.Ask(a.ctx, a.llm, r.Question, kept) })
		return ans, kept, err
	}
	grounded := func(ans rag.Answer, docs []store.Hit) (bool, error) {
		if ans.Abstained {
			return true, nil
		}
		keyParts := []string{ans.Text}
		for _, h := range docs {
			keyParts = append(keyParts, h.Text)
		}
		return cached(cache, "ground:"+hashKey(keyParts...), func() (bool, error) { return rag.Grounded(a.ctx, a.llm, ans.Text, docs) })
	}

	type tally struct {
		ansN, impN                      int
		em, f1, ansAbstain, impDeclined float64
	}
	pipelines := []string{"no graders", "+ hallucination check", "relevance grader (course)", "answer grader", "answer grader + hallucination check", "similarity gate (shipped)", "gate + relevance grader"}
	t := map[string]*tally{}
	for _, p := range pipelines {
		t[p] = &tally{}
	}
	// How often each grader says yes to a passage that does, or doesn't, hold the answer.
	type agree struct{ tp, fp, fn, tn int }
	agreement := map[rag.Grader]*agree{rag.Relevance: {}, rag.Answers: {}}

	add := func(p string, r answerRecord, ans rag.Answer) {
		s := t[p]
		if r.Impossible {
			s.impN++
			if ans.Abstained {
				s.impDeclined++
			}
			return
		}
		s.ansN++
		if ans.Abstained {
			s.ansAbstain++
			return
		}
		s.em += rag.ExactMatch(ans.Text, r.Gold)
		s.f1 += rag.F1(ans.Text, r.Gold)
	}
	declined := rag.Answer{Abstained: true}

	for i, r := range recs {
		all := hits(r)
		base := rag.Answer{Text: r.Answer, Abstained: r.Abstained}
		add("no graders", r, base)
		score, ok := topScore[r.ID]
		if !ok {
			return fmt.Errorf("no similarity score for %s", r.ID)
		}
		gatedOut := score < *minSim
		if gatedOut {
			add("similarity gate (shipped)", r, declined)
		} else {
			add("similarity gate (shipped)", r, base)
		}
		ok, err := grounded(base, all)
		if err != nil {
			return err
		}
		if ok {
			add("+ hallucination check", r, base)
		} else {
			add("+ hallucination check", r, declined)
		}
		for _, g := range []rag.Grader{rag.Relevance, rag.Answers} {
			keep, err := gradeAll(r, g)
			if err != nil {
				return err
			}
			if !r.Impossible {
				ag := agreement[g]
				for j, s := range r.Sources {
					switch {
					case keep[j] && s.Gold:
						ag.tp++
					case keep[j] && !s.Gold:
						ag.fp++
					case !keep[j] && s.Gold:
						ag.fn++
					default:
						ag.tn++
					}
				}
			}
			ans, docs, err := answerWith(r, keep)
			if err != nil {
				return err
			}
			if g == rag.Relevance {
				add("relevance grader (course)", r, ans)
				if gatedOut {
					add("gate + relevance grader", r, declined)
				} else {
					add("gate + relevance grader", r, ans)
				}
				continue
			}
			add("answer grader", r, ans)
			ok, err := grounded(ans, docs)
			if err != nil {
				return err
			}
			if !ok {
				ans = declined
			}
			add("answer grader + hallucination check", r, ans)
		}
		if (i+1)%20 == 0 {
			log.Printf("graders: %d of %d", i+1, len(recs))
		}
	}

	out := map[string]any{"answers": *run, "perGroup": *n}
	rows := []map[string]any{}
	for _, p := range pipelines {
		s := t[p]
		row := map[string]any{
			"pipeline":             p,
			"exactMatch":           round(s.em/float64(s.ansN), 4),
			"f1":                   round(s.f1/float64(s.ansN), 4),
			"answerableDeclined":   round(s.ansAbstain/float64(s.ansN), 4),
			"unanswerableDeclined": round(s.impDeclined/float64(s.impN), 4),
			"answerable":           s.ansN,
			"unanswerable":         s.impN,
		}
		rows = append(rows, row)
		log.Printf("%-38s EM %.3f  F1 %.3f  declines: answerable %.3f, unanswerable %.3f", p, row["exactMatch"], row["f1"], row["answerableDeclined"], row["unanswerableDeclined"])
	}
	out["pipelines"] = rows
	ag := map[string]any{}
	for g, x := range agreement {
		ag[string(g)] = map[string]any{
			"yesWhenPassageHoldsAnswer": round(float64(x.tp)/float64(max(1, x.tp+x.fn)), 4),
			"yesWhenPassageDoesNot":     round(float64(x.fp)/float64(max(1, x.fp+x.tn)), 4),
			"passages":                  x.tp + x.fp + x.fn + x.tn,
		}
		log.Printf("grader %-9s yes on answer passages %.3f, yes on others %.3f", g, float64(x.tp)/float64(max(1, x.tp+x.fn)), float64(x.fp)/float64(max(1, x.fp+x.tn)))
	}
	out["graderAgreement"] = ag
	return writeJSON("results/graders.json", out)
}
