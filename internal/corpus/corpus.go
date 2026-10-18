// Package corpus turns SQuAD 2.0 into documents and questions, and splits documents into chunks.
//
// Each Wikipedia article becomes one document: its paragraphs joined by blank lines, the way
// text comes out of a PDF. Every answer is stored as a character span in that document, so a
// retrieved chunk can be checked by whether it contains the span.
package corpus

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strings"
)

type Document struct {
	ID    int
	Title string
	Text  string
}

type Span struct {
	Start, End int // byte offsets into Document.Text
	Text       string
}

type Question struct {
	ID         string
	DocID      int
	Text       string
	Answers    []Span // empty when Impossible
	Impossible bool
}

type Chunk struct {
	DocID      int
	Ord        int
	Start, End int // byte offsets into the document
	Text       string
}

type squadFile struct {
	Data []struct {
		Title      string `json:"title"`
		Paragraphs []struct {
			Context string `json:"context"`
			QAs     []struct {
				ID           string `json:"id"`
				Question     string `json:"question"`
				IsImpossible bool   `json:"is_impossible"`
				Answers      []struct {
					Text        string `json:"text"`
					AnswerStart int    `json:"answer_start"`
				} `json:"answers"`
			} `json:"qas"`
		} `json:"paragraphs"`
	} `json:"data"`
}

// Load reads the SQuAD 2.0 dev file. SQuAD's answer_start counts characters, not bytes, so
// offsets are converted through each paragraph's rune index.
func Load(path string) ([]Document, []Question, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w (run scripts/get-data.sh)", path, err)
	}
	var f squadFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, nil, err
	}
	var docs []Document
	var qs []Question
	for i, a := range f.Data {
		var b strings.Builder
		for j, p := range a.Paragraphs {
			if j > 0 {
				b.WriteString("\n\n")
			}
			base := b.Len()
			b.WriteString(p.Context)
			byteAt := runeOffsets(p.Context)
			for _, q := range p.QAs {
				qq := Question{ID: q.ID, DocID: i, Text: strings.TrimSpace(q.Question), Impossible: q.IsImpossible}
				for _, ans := range q.Answers {
					if ans.AnswerStart >= len(byteAt) {
						continue
					}
					start := base + byteAt[ans.AnswerStart]
					qq.Answers = append(qq.Answers, Span{Start: start, End: start + len(ans.Text), Text: ans.Text})
				}
				qs = append(qs, qq)
			}
		}
		docs = append(docs, Document{ID: i, Title: strings.ReplaceAll(a.Title, "_", " "), Text: b.String()})
	}
	// Check every answer span points at its text.
	for _, q := range qs {
		for _, s := range q.Answers {
			if docs[q.DocID].Text[s.Start:s.End] != s.Text {
				return nil, nil, fmt.Errorf("question %s: answer offset does not match its text", q.ID)
			}
		}
	}
	return docs, qs, nil
}

// runeOffsets maps a character index to a byte index (with one extra entry for the end).
func runeOffsets(s string) []int {
	out := make([]int, 0, len(s)+1)
	for i := range s {
		out = append(out, i)
	}
	return append(out, len(s))
}

// SplitByDocument puts the questions of `n` randomly chosen documents in the first set and the
// rest in the second. Splitting by document keeps tuning and testing on different articles.
func SplitByDocument(docs []Document, qs []Question, n int, seed int64) (tune, test []Question, tuneDocs map[int]bool) {
	ids := make([]int, len(docs))
	for i := range ids {
		ids[i] = i
	}
	rand.New(rand.NewSource(seed)).Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
	tuneDocs = map[int]bool{}
	for _, id := range ids[:n] {
		tuneDocs[id] = true
	}
	for _, q := range qs {
		if tuneDocs[q.DocID] {
			tune = append(tune, q)
		} else {
			test = append(test, q)
		}
	}
	return tune, test, tuneDocs
}

// ---------- Chunking ----------

// A Chunker splits one document into chunks with byte offsets.
type Chunker struct {
	Name    string
	Words   int // window size in words; 0 means split on blank lines
	Overlap int // words shared by neighbouring windows
}

var Chunkers = map[string]Chunker{
	"para": {Name: "para"},
	"w100": {Name: "w100", Words: 100, Overlap: 25},
	"w300": {Name: "w300", Words: 300, Overlap: 50},
}

var wordRe = regexp.MustCompile(`\S+`)

func (c Chunker) Split(d Document) []Chunk {
	var out []Chunk
	if c.Words == 0 {
		start := 0
		for start < len(d.Text) {
			end := strings.Index(d.Text[start:], "\n\n")
			if end < 0 {
				end = len(d.Text)
			} else {
				end += start
			}
			if strings.TrimSpace(d.Text[start:end]) != "" {
				out = append(out, Chunk{DocID: d.ID, Ord: len(out), Start: start, End: end, Text: d.Text[start:end]})
			}
			start = end + 2
		}
		return out
	}
	words := wordRe.FindAllStringIndex(d.Text, -1)
	step := c.Words - c.Overlap
	for i := 0; i < len(words); i += step {
		j := min(i+c.Words, len(words))
		s, e := words[i][0], words[j-1][1]
		out = append(out, Chunk{DocID: d.ID, Ord: len(out), Start: s, End: e, Text: d.Text[s:e]})
		if j == len(words) {
			break
		}
	}
	return out
}

// Contains reports whether the chunk holds any of the answer spans in full.
func (c Chunk) Contains(spans []Span) bool {
	for _, s := range spans {
		if s.Start >= c.Start && s.End <= c.End {
			return true
		}
	}
	return false
}

// SortedNames lists the chunker names in a fixed order.
func SortedNames() []string {
	names := make([]string, 0, len(Chunkers))
	for n := range Chunkers {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
