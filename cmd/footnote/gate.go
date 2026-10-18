package main

// eval-gate tests a cheap fix for confabulation: skip the model and say "I don't know" when the
// best passage is not similar enough to the question. The threshold is picked on the tuning
// articles from retrieval scores alone. Its effect on the test run is then exact, because a gate
// that fires before generation turns that question's answer into an abstention and changes nothing else.

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/evannwilsonn/footnote/internal/corpus"
)

type scored struct {
	score      float64
	impossible bool
}

// auc is the chance a random unanswerable question scores lower than a random answerable one.
func auc(xs []scored) float64 {
	var lower, pairs float64
	for _, a := range xs {
		if a.impossible {
			continue
		}
		for _, b := range xs {
			if !b.impossible {
				continue
			}
			pairs++
			if b.score < a.score {
				lower++
			} else if b.score == a.score {
				lower += 0.5
			}
		}
	}
	return lower / pairs
}

// gated returns the share of answerable and of unanswerable questions scoring below t.
func gated(xs []scored, t float64) (ans, imp float64) {
	var na, ni float64
	for _, x := range xs {
		if x.impossible {
			ni++
			if x.score < t {
				imp++
			}
		} else {
			na++
			if x.score < t {
				ans++
			}
		}
	}
	return ans / na, imp / ni
}

func evalGate(args []string) error {
	fs := flag.NewFlagSet("eval-gate", flag.ExitOnError)
	chunker := fs.String("chunker", "para", "chunker")
	run := fs.String("answers", "results/answers-test-k3.json", "answer run to apply the gate to")
	fs.Parse(args)
	a, err := setup(true)
	if err != nil {
		return err
	}
	vecs, err := a.questionVectors(true)
	if err != nil {
		return err
	}
	tune, test, _ := corpus.SplitByDocument(a.docs, a.qs, tuneDocs, splitSeed)
	index := map[string]int{}
	for i, q := range a.qs {
		index[q.ID] = i
	}
	top := map[string]float64{}
	score := func(qs []corpus.Question) ([]scored, error) {
		out := make([]scored, 0, len(qs))
		for _, q := range qs {
			hits, err := a.store.Dense(a.ctx, *chunker, vecs[index[q.ID]], 1)
			if err != nil {
				return nil, err
			}
			s := 0.0
			if len(hits) > 0 {
				s = hits[0].Score
			}
			top[q.ID] = s
			out = append(out, scored{s, q.Impossible})
		}
		return out, nil
	}
	tuneScores, err := score(tune)
	if err != nil {
		return err
	}
	testScores, err := score(test)
	if err != nil {
		return err
	}

	// Pick the threshold on the tuning articles: the most unanswerable questions caught for each
	// answerable one lost (Youden's J), scanning every observed score.
	cands := make([]float64, 0, len(tuneScores))
	for _, x := range tuneScores {
		cands = append(cands, x.score)
	}
	sort.Float64s(cands)
	best, bestJ := 0.0, -1.0
	for _, t := range cands {
		ga, gi := gated(tuneScores, t)
		if j := gi - ga; j > bestJ {
			best, bestJ = t, j
		}
	}
	tuneA, tuneI := gated(tuneScores, best)
	testA, testI := gated(testScores, best)
	log.Printf("top-1 similarity AUC: tune %.3f, test %.3f", auc(tuneScores), auc(testScores))
	log.Printf("threshold %.4f (tune): gates %.1f%% of unanswerable and %.1f%% of answerable on tune; %.1f%% and %.1f%% on test",
		best, 100*tuneI, 100*tuneA, 100*testI, 100*testA)

	// Apply thresholds to the test answer run: Youden's pick plus milder ones that, on the tuning
	// articles, would block at most 5%, 10% or 20% of answerable questions.
	raw, err := os.ReadFile(*run)
	if err != nil {
		return err
	}
	var answers struct {
		Records []answerRecord `json:"records"`
	}
	if err := json.Unmarshal(raw, &answers); err != nil {
		return err
	}
	apply := func(t float64) (map[string]float64, error) {
		var nAns, nImp, em, f1, declined, lostRight float64
		for _, r := range answers.Records {
			if r.Condition != "rag" {
				continue
			}
			s, ok := top[r.ID]
			if !ok {
				return nil, fmt.Errorf("no score for %s", r.ID)
			}
			gate := s < t
			if r.Impossible {
				nImp++
				if gate || r.Abstained {
					declined++
				}
				continue
			}
			nAns++
			if gate {
				if r.EM == 1 {
					lostRight++
				}
				continue
			}
			em += r.EM
			f1 += r.F1
		}
		return map[string]float64{
			"threshold":             round(t, 4),
			"exactMatch":            round(em/nAns, 4),
			"f1":                    round(f1/nAns, 4),
			"unanswerableDeclined":  round(declined/nImp, 4),
			"correctAnswersBlocked": round(lostRight/nAns, 4),
		}, nil
	}
	var answerable []float64
	for _, x := range tuneScores {
		if !x.impossible {
			answerable = append(answerable, x.score)
		}
	}
	sort.Float64s(answerable)
	points := []struct {
		Name string
		T    float64
	}{{"none", -1}}
	for _, p := range []float64{0.05, 0.10, 0.20} {
		points = append(points, struct {
			Name string
			T    float64
		}{fmt.Sprintf("block %.0f%% on tune", 100*p), answerable[int(p*float64(len(answerable)))]})
	}
	points = append(points, struct {
		Name string
		T    float64
	}{"youden", best})
	var curve []map[string]any
	for _, p := range points {
		m, err := apply(p.T)
		if err != nil {
			return err
		}
		log.Printf("%-18s t=%.4f  exact match %.3f  F1 %.3f  unanswerable declined %.3f  correct answers blocked %.3f",
			p.Name, p.T, m["exactMatch"], m["f1"], m["unanswerableDeclined"], m["correctAnswersBlocked"])
		row := map[string]any{"name": p.Name}
		for k, v := range m {
			row[k] = v
		}
		curve = append(curve, row)
	}
	return writeJSON("results/gate.json", map[string]any{
		"chunker":   *chunker,
		"auc":       map[string]float64{"tune": round(auc(tuneScores), 3), "test": round(auc(testScores), 3)},
		"gatedTune": map[string]float64{"answerable": round(tuneA, 4), "unanswerable": round(tuneI, 4)},
		"gatedTest": map[string]float64{"answerable": round(testA, 4), "unanswerable": round(testI, 4)},
		"onTestRun": curve,
	})
}
