package store

import (
	"context"
	"os"
	"testing"

	"github.com/evannwilsonn/footnote/internal/corpus"
)

// These tests need a real Postgres with pgvector. CI starts one as a service container and sets
// TEST_DATABASE_URL; locally, `docker compose up -d db` and
// TEST_DATABASE_URL=postgres://rag:rag@localhost:5432/rag go test ./internal/store
// Without it they're skipped, so `go test ./...` still runs anywhere.
func open(t *testing.T) *Store {
	t.Helper()
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		t.Skip("TEST_DATABASE_URL not set")
	}
	ctx := context.Background()
	s, err := Open(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.DB.Close)
	if err := s.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	return s
}

// unit returns a Dim-length vector pointing mostly along axis i, so nearest-neighbour order is known.
func unit(i int, lean float32) []float32 {
	v := make([]float32, Dim)
	v[i] = 1
	v[(i+1)%Dim] = lean
	return v
}

func TestStoreRoundTrip(t *testing.T) {
	s := open(t)
	ctx := context.Background()
	docs := []corpus.Document{
		{ID: 1, Title: "Steam engine", Text: "High-pressure steam drove the first locomotives."},
		{ID: 2, Title: "Photosynthesis", Text: "Plants turn sunlight, water and carbon dioxide into sugar."},
		{ID: 3, Title: "Tides", Text: "The moon's gravity pulls the oceans into two bulges."},
	}
	if err := s.ReplaceDocuments(ctx, docs); err != nil {
		t.Fatal(err)
	}
	chunks := make([]corpus.Chunk, len(docs))
	vecs := make([][]float32, len(docs))
	for i, d := range docs {
		chunks[i] = corpus.Chunk{DocID: d.ID, Ord: 0, Start: 0, End: len(d.Text), Text: d.Text}
		vecs[i] = unit(i*10, 0.1)
	}
	if err := s.ReplaceChunks(ctx, "para", chunks, vecs); err != nil {
		t.Fatal(err)
	}

	if n, err := s.Count(ctx, "para"); err != nil || n != 3 {
		t.Fatalf("Count = %d, %v; want 3", n, err)
	}

	// A query vector next to document 2's points at it first.
	hits, err := s.Dense(ctx, "para", unit(10, 0.2), 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].DocID != 2 || hits[0].Title != "Photosynthesis" {
		t.Fatalf("Dense top hit = %+v; want document 2", hits)
	}
	if hits[0].Score <= hits[1].Score {
		t.Fatalf("Dense scores not descending: %v then %v", hits[0].Score, hits[1].Score)
	}

	// Full-text search stems "pulling" to match "pulls".
	lex, err := s.Lexical(ctx, "para", "what is pulling the oceans?", 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(lex) == 0 || lex[0].DocID != 3 {
		t.Fatalf("Lexical top hit = %+v; want document 3", lex)
	}

	// A question with only stop words matches nothing rather than erroring.
	if none, err := s.Lexical(ctx, "para", "what is the", 3); err != nil || len(none) != 0 {
		t.Fatalf("stop-word query = %v, %v; want no hits", none, err)
	}

	// Replacing one chunker's chunks leaves the count right, and unknown chunkers are refused.
	if err := s.ReplaceChunks(ctx, "para", chunks[:1], vecs[:1]); err != nil {
		t.Fatal(err)
	}
	if n, _ := s.Count(ctx, "para"); n != 1 {
		t.Fatalf("after replace Count = %d; want 1", n)
	}
	if _, err := s.Dense(ctx, "nope", vecs[0], 1); err == nil {
		t.Fatal("unknown chunker accepted")
	}
}
