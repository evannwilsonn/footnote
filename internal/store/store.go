// Package store keeps documents and chunks in Postgres: full-text search with a tsvector column
// and vector search with pgvector, both on the same table.
package store

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgxvec "github.com/pgvector/pgvector-go/pgx"

	"github.com/evannwilsonn/footnote/internal/corpus"
)

const Dim = 768 // nomic-embed-text

type Store struct{ DB *pgxpool.Pool }

type Hit struct {
	ChunkID    int64
	DocID      int
	Title      string
	Ord        int
	Start, End int
	Text       string
	Score      float64
}

func (h Hit) Chunk() corpus.Chunk {
	return corpus.Chunk{DocID: h.DocID, Ord: h.Ord, Start: h.Start, End: h.End, Text: h.Text}
}

func Open(ctx context.Context, url string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		return nil, err
	}
	// Teach every connection the vector type, so COPY can send embeddings in binary.
	// The type has to exist first, which on a fresh database means creating the extension here.
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		if _, err := conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
			return err
		}
		return pgxvec.RegisterTypes(ctx, conn)
	}
	db, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if err := db.Ping(ctx); err != nil {
		return nil, fmt.Errorf("postgres: %w (is it running, and is DATABASE_URL right?)", err)
	}
	return &Store{DB: db}, nil
}

func (s *Store) Migrate(ctx context.Context) error {
	_, err := s.DB.Exec(ctx, fmt.Sprintf(`
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
  id    int PRIMARY KEY,
  title text NOT NULL,
  body  text NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id         bigserial PRIMARY KEY,
  chunker    text NOT NULL,
  doc_id     int  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord        int  NOT NULL,
  start_byte int  NOT NULL,
  end_byte   int  NOT NULL,
  body       text NOT NULL,
  tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,
  embedding  vector(%d) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_chunker ON chunks (chunker);`, Dim))
	return err
}

func (s *Store) ReplaceDocuments(ctx context.Context, docs []corpus.Document) error {
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `TRUNCATE documents CASCADE`); err != nil {
		return err
	}
	rows := make([][]any, len(docs))
	for i, d := range docs {
		rows[i] = []any{d.ID, d.Title, d.Text}
	}
	if _, err := tx.CopyFrom(ctx, pgx.Identifier{"documents"}, []string{"id", "title", "body"}, pgx.CopyFromRows(rows)); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// ReplaceChunks stores one chunker's chunks with their embeddings and builds that chunker's HNSW
// index. Each chunker gets its own partial index so a filtered search stays within its own set.
func (s *Store) ReplaceChunks(ctx context.Context, chunker string, chunks []corpus.Chunk, vecs [][]float32) error {
	if _, ok := corpus.Chunkers[chunker]; !ok {
		return fmt.Errorf("unknown chunker %q", chunker)
	}
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `DELETE FROM chunks WHERE chunker = $1`, chunker); err != nil {
		return err
	}
	rows := make([][]any, len(chunks))
	for i, c := range chunks {
		rows[i] = []any{chunker, c.DocID, c.Ord, c.Start, c.End, c.Text, pgvector.NewVector(vecs[i])}
	}
	cols := []string{"chunker", "doc_id", "ord", "start_byte", "end_byte", "body", "embedding"}
	if _, err := tx.CopyFrom(ctx, pgx.Identifier{"chunks"}, cols, pgx.CopyFromRows(rows)); err != nil {
		return err
	}
	idx := "chunks_hnsw_" + chunker
	if _, err := tx.Exec(ctx, fmt.Sprintf(`DROP INDEX IF EXISTS %s; CREATE INDEX %s ON chunks USING hnsw (embedding vector_cosine_ops) WHERE chunker = '%s'`, idx, idx, chunker)); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func (s *Store) Count(ctx context.Context, chunker string) (int, error) {
	var n int
	err := s.DB.QueryRow(ctx, `SELECT count(*) FROM chunks WHERE chunker = $1`, chunker).Scan(&n)
	return n, err
}

func scan(rows pgx.Rows) ([]Hit, error) {
	defer rows.Close()
	var out []Hit
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.ChunkID, &h.DocID, &h.Title, &h.Ord, &h.Start, &h.End, &h.Text, &h.Score); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

func chunkerLiteral(chunker string) (string, error) {
	if _, ok := corpus.Chunkers[chunker]; !ok {
		return "", fmt.Errorf("unknown chunker %q", chunker)
	}
	return chunker, nil
}

// Lexical ranks chunks with Postgres full-text search. The question's words (English-stemmed,
// stop words removed) are OR-ed together, and matches are ordered by ts_rank_cd, which rewards
// chunks where the matched words appear close together.
func (s *Store) Lexical(ctx context.Context, chunker, question string, k int) ([]Hit, error) {
	ch, err := chunkerLiteral(chunker)
	if err != nil {
		return nil, err
	}
	rows, err := s.DB.Query(ctx, fmt.Sprintf(`
WITH q AS (
  SELECT to_tsquery('english', coalesce(nullif(array_to_string(ARRAY(
    SELECT quote_literal(l) FROM unnest(tsvector_to_array(to_tsvector('english', $1))) AS l), ' | '), ''), 'zzzz')) AS q)
SELECT c.id, c.doc_id, d.title, c.ord, c.start_byte, c.end_byte, c.body, ts_rank_cd(c.tsv, q.q, 1)::float8
FROM chunks c JOIN documents d ON d.id = c.doc_id, q
WHERE c.chunker = '%s' AND c.tsv @@ q.q
ORDER BY 8 DESC, c.id
LIMIT $2`, ch), question, k)
	if err != nil {
		return nil, err
	}
	return scan(rows)
}

// Dense ranks chunks by cosine similarity of embeddings, using the chunker's HNSW index.
func (s *Store) Dense(ctx context.Context, chunker string, vec []float32, k int) ([]Hit, error) {
	ch, err := chunkerLiteral(chunker)
	if err != nil {
		return nil, err
	}
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `SET LOCAL hnsw.ef_search = 100`); err != nil {
		return nil, err
	}
	rows, err := tx.Query(ctx, fmt.Sprintf(`
SELECT c.id, c.doc_id, d.title, c.ord, c.start_byte, c.end_byte, c.body, (1 - (c.embedding <=> $1))::float8
FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE c.chunker = '%s'
ORDER BY c.embedding <=> $1
LIMIT $2`, ch), pgvector.NewVector(vec), k)
	if err != nil {
		return nil, err
	}
	return scan(rows)
}
