package corpus

import (
	"strings"
	"testing"
)

func doc() Document {
	text := "First paragraph about Normans.\n\nSecond paragraph, café and Normandy.\n\nThird one."
	return Document{ID: 0, Title: "T", Text: text}
}

func TestParagraphChunksKeepOffsets(t *testing.T) {
	d := doc()
	chunks := Chunkers["para"].Split(d)
	if len(chunks) != 3 {
		t.Fatalf("want 3 paragraphs, got %d", len(chunks))
	}
	for _, c := range chunks {
		if d.Text[c.Start:c.End] != c.Text || strings.Contains(c.Text, "\n\n") {
			t.Fatalf("bad chunk %+v", c)
		}
	}
}

func TestWindowChunksOverlapAndCoverEverything(t *testing.T) {
	words := make([]string, 250)
	for i := range words {
		words[i] = "w"
	}
	d := Document{Text: strings.Join(words, " ")}
	chunks := Chunker{Words: 100, Overlap: 25}.Split(d)
	// windows start at words 0, 75, 150; the last reaches the end.
	if len(chunks) != 3 || chunks[len(chunks)-1].End != len(d.Text) {
		t.Fatalf("got %d chunks, last ends at %d of %d", len(chunks), chunks[len(chunks)-1].End, len(d.Text))
	}
	if chunks[1].Start >= chunks[0].End {
		t.Fatal("neighbouring windows should overlap")
	}
}

func TestContains(t *testing.T) {
	c := Chunk{Start: 10, End: 50}
	if !c.Contains([]Span{{Start: 20, End: 30}}) {
		t.Fatal("span inside the chunk")
	}
	if c.Contains([]Span{{Start: 45, End: 55}}) {
		t.Fatal("a span that runs past the end is not contained")
	}
}

func TestRuneOffsets(t *testing.T) {
	// "é" is two bytes: character 5 ("x") is byte 6.
	if got := runeOffsets("café x")[5]; got != 6 {
		t.Fatalf("got %d", got)
	}
}

func TestLoadRealData(t *testing.T) {
	docs, qs, err := Load("../../data/dev-v2.0.json")
	if err != nil {
		t.Skip("data not downloaded:", err)
	}
	if len(docs) != 35 || len(qs) != 11873 {
		t.Fatalf("got %d documents, %d questions", len(docs), len(qs))
	}
	tune, test, _ := SplitByDocument(docs, qs, 10, 2026)
	if len(tune)+len(test) != len(qs) {
		t.Fatal("split lost questions")
	}
}
