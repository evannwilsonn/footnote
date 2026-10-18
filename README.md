# Footnote

Question answering over a set of documents, with citations. Footnote stores the documents in Postgres, finds the passages most likely to hold the answer with pgvector, and hands them to a small model running locally as numbered exhibits. It asks for the shortest answer the exhibits support, with the exhibit number, or "I don't know". It's written in Go, uses free local models through Ollama, and ships as three containers.

**Live page:** [evannwilsonn.github.io/footnote](https://evannwilsonn.github.io/footnote/) has the results and example transcripts. The live question box appears only when the page is served by `footnote serve`.

```
$ footnote ask "Who patented a high-pressure compound engine in 1805?"
Arthur Woolf
*[1] Steam engine: A method to lessen the magnitude of this heating and cooling was invented in 1804 by British engineer Arthur Woolf, who patented hi…
 [2] Steam engine: Around 1800 Richard Trevithick and, separately, Oliver Evans in 1801 introduced engines using high-pressure steam; Trevithick obtai…
 [3] Steam engine: In 1781 James Watt patented a steam engine that produced continuous rotary motion. Watt's ten-horsepower engines enabled a wide ran…
(4092 ms)
```

The `*` marks the exhibit the model cited.

## 1. Problem

Ask a small language model a factual question about a specific document and it will often answer confidently and wrongly, because it is answering from whatever it absorbed in training. The goal is a system that answers from a given set of documents instead, shows where each answer came from, and says "I don't know" when the documents don't say. Success is measured on questions with known answers: how often the right passage is found, how often the final answer matches, and how often the system declines questions that have no answer.

## 2. Data

The [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dev set (Rajpurkar, Jia and Liang, 2018; CC BY-SA 4.0) has 35 Wikipedia articles, 1,204 paragraphs, about 970 KB of text, and 11,873 questions written by crowd workers. 5,945 questions have answers marked as character spans in the text. The other 5,928 were written to look answerable from the article when they aren't, which makes them a direct test of "I don't know".

The articles are split 10 for tuning and 25 for testing (seed 2026). Questions about the same article never cross the split, so nothing chosen on the tuning articles has seen the test articles. Every choice below (chunk size, search method, threshold) was made on the tuning articles, and the test numbers were run once, at the end.

`scripts/get-data.sh` downloads the file. The loader converts SQuAD's character offsets to byte offsets and checks every answer span against the text.

## 3. Baseline

There are two baselines, one per stage.

- **Retrieval:** Postgres full-text search (`tsvector`, `ts_rank_cd`) over the same chunks. On the test articles it ranks the passage holding the answer first 41.7% of the time, and in the top 3 66.3%.
- **Answering:** the same model with no documents ("closed book"). On 200 answerable test questions it gets 6.5% exact match and 0.16 word-overlap F1. On 200 unanswerable ones it says "I don't know" 9.5% of the time.

## 4. Model

```
question ─► nomic-embed-text ─► pgvector HNSW (cosine) ─► top 3 paragraphs ─► Qwen2.5 1.5B ─► answer + [n]
                                     │                                       ▲
                       Postgres: documents, chunks, tsvector, vector(768)    └ prompt: answer only from the
                                                                               exhibits, cite, or "I don't know"
```

- **Chunks:** three ways of cutting the documents are stored side by side: paragraphs, 100-word windows with 25 words of overlap, and 300-word windows with 50. Each chunk keeps its byte range in the source document, so a retrieved chunk can be checked against the answer span exactly.
- **Embeddings:** `nomic-embed-text` (768 dimensions) through Ollama, with its `search_document:` / `search_query:` prefixes. They are loaded with `COPY` and indexed with one partial HNSW index per chunker.
- **Search:** lexical (full text), dense (pgvector cosine), or hybrid (reciprocal-rank fusion of the top 20 from each, k = 60).
- **Answering:** `qwen2.5:1.5b` at temperature 0 with a fixed seed, given the top passages as numbered exhibits. The reply is parsed for citations and abstentions, and scored with the official SQuAD normalisation (exact match and token F1).
- **Declining:** before answering, the server skips the model if even the best passage is a weak match, then asks the model whether each passage is on topic and answers only from those it keeps (see Improvement).
- **Serving:** `footnote serve` exposes `POST /ask` and `GET /health` and serves the results page.

## 5. Evaluation

**Retrieval** was scored on 4,119 answerable test questions. A chunk counts only if it contains the whole answer span.

| Cut into paragraphs | Top 1 | Top 3 | Top 5 | Top 20 | MRR |
|---|---|---|---|---|---|
| Word search (baseline) | 41.7% | 66.3% | 76.3% | 93.6% | 0.567 |
| Meaning search | **73.1%** | **89.4%** | 92.8% | 97.9% | **0.818** |
| Both, fused | 67.6% | 87.2% | 91.7% | 98.9% | 0.783 |

Fusion loses at the top because word search is weak enough here that it mostly pulls the right passage down from first place; it only wins at top 20. The `search_query:` prefix the embedding model was trained with made no measurable difference (73.1% vs 72.9% top 1).

Chunk size was compared on the tuning articles with meaning search:

| Cut into | Chunks | Words each | Top 1 | Top 3 |
|---|---|---|---|---|
| **Paragraphs** | 1,204 | 126.6 | **70.9%** | **85.0%** |
| 100-word windows | 2,038 | 99.3 | 63.9% | 81.5% |
| 300-word windows | 617 | 294.2 | 53.1% | 74.3% |

Paragraphs won on the tuning articles and held up on the test articles.

**Answers** were scored on 200 answerable and 200 unanswerable test questions, sampled at random, with the top 3 paragraphs as exhibits.

| | From memory | With exhibits |
|---|---|---|
| Exact match, answerable | 6.5% | **48.5%** |
| Word-overlap F1, answerable | 0.161 | **0.622** |
| Said "I don't know", unanswerable | 9.5% | 14.0% |
| Wrongly said "I don't know", answerable | 8.0% | 3.0% |
| Seconds per answer (2 CPU cores) | 0.7 | 5.8 |

An exhibit held the answer for 87.5% of answerable questions, and when one did, the model matched the gold answer exactly 53.7% of the time. So most of the remaining errors are the model's, not the search's. The weak spot is the unanswerable half: with exhibits in hand, the model still answers 86% of questions the documents can't answer.

## 6. Improvement

The obvious fix for confabulation is to decline before asking the model when even the best passage is a weak match. `footnote eval-gate` tests this. It picks thresholds on the tuning articles from retrieval scores alone, then applies them to the test answers above. The effect there is exact, because a gate that fires before generation only turns that question's answer into "I don't know".

| Threshold (set on tuning articles) | Min. similarity | Exact match | Declines unanswerable |
|---|---|---|---|
| No gate | – | 48.5% | 14.0% |
| **Blocks 5% of answerable** | **0.639** | **48.0%** | **20.5%** |
| Blocks 10% | 0.668 | 44.5% | 25.5% |
| Blocks 20% | 0.696 | 43.0% | 38.5% |
| Balanced (Youden's J) | 0.758 | 28.5% | 68.5% |

It only half works. The top passage's similarity barely separates answerable from unanswerable questions (AUC 0.62 on tuning, 0.67 on test, where 0.5 is chance), because SQuAD's unanswerable questions were written against the same paragraphs and match the text about as well as real ones. The server ships the mildest threshold, which buys 6.5 more points of correct declines for half a point of exact match. Closing more of the gap needed a second look at the passages themselves: the relevance grader from the course, below, more than doubles the declines. A larger model or a trained answerability classifier would be the next step. The model also sometimes answers from its own knowledge despite the prompt: asked for the capital of Australia, it says "Canberra" while being shown three passages about the state of Victoria that don't mention it.

### Techniques from "Learn RAG From Scratch"

Lance Martin's freeCodeCamp course ([video](https://youtu.be/sVcwVQRHIc8), [code](https://github.com/langchain-ai/rag-from-scratch)) builds its techniques with LangChain and GPT-3.5. They are rebuilt here in Go without LangChain (`internal/rag/course.go`), with the course's prompts, and run on the same 1.5B model and the same test articles. The grader prompts come from the LangGraph CRAG and adaptive-RAG examples the course runs.

**Graders** (`footnote eval-graders`, the first 80 answerable and 80 unanswerable questions of the answer run above):

| Pipeline | Exact match | Declines answerable | Declines unanswerable |
|---|---|---|---|
| Plain retrieval | 45.0% | 3.8% | 16.3% |
| Similarity gate (before) | 45.0% | 5.0% | 23.8% |
| Relevance grader (course) | 50.0% | 10.0% | 55.0% |
| **Gate + relevance grader (now shipped)** | **50.0%** | **11.3%** | **58.8%** |
| Stricter "does it answer?" grader | 47.5% | 12.5% | 50.0% |
| Hallucination check | 17.5% | 53.8% | 60.0% |
| Stricter grader + hallucination check | 30.0% | 40.0% | 65.0% |

- **Relevance grader:** asks the model, passage by passage, whether the passage is on topic; the model then answers from the passages it kept. It said yes to 94% of passages holding the answer and 25% of the rest.
- **Shipped:** on top of the similarity gate it more than doubles correct declines and adds five points of exact match, for six more points of answerable questions declined. The server now runs it by default (`FOOTNOTE_GRADER=off` turns it off). On live questions, the page marks the passages it set aside.
- **Cost:** it adds about 3.5 s per answer (13.3 s against 9.8 s for four test questions on the same, busier machine).
- **Hallucination check:** asks whether the answer is supported by the passages. It rejected so many correct answers that it cuts exact match to a third. At this size, judging an answer is much harder for the model than judging a passage.
- **Noise:** with 80 questions per group, differences under about 10 points are within noise. The relevance grader's gain on declines (+35 points) is well clear of that.

**Rewriting the question** (`footnote eval-translate -split test -n 60`, 60 answerable test questions, the passage holding the answer must come back):

| Search with | Top 1 | Top 3 | MRR | Model time |
|---|---|---|---|---|
| **The question as asked** | **65.0%** | **91.7%** | **0.785** | – |
| Multi-query (5 rewrites, best match each) | 56.7% | 83.3% | 0.702 | +10.2 s |
| RAG-Fusion (4 queries, RRF) | 63.3% | 83.3% | 0.747 | +9.5 s |
| RAG-Fusion + the question | 66.7% | 83.3% | 0.772 | +9.5 s |
| Step-back + the question | 68.3% | 88.3% | 0.786 | +2.0 s |
| HyDE (a written answer, embedded) | 48.3% | 75.0% | 0.609 | +14.0 s |

None of the rewrites beat plain search by more than two questions in 60, and all of them were worse at top 3, which is what the model reads. HyDE did worst. The small model's made-up answer passages state confident details of their own, which pull search toward the wrong paragraph: asked when seafloor spreading was discovered, it wrote a passage dating it to the 1950s and crediting a team led by Harry Hess and Maurice Ewing. SQuAD's questions are written from the paragraphs, so they already share their wording, and rewriting has little to fix. The server keeps plain search. On vaguer questions from real users, rewriting would have more to do.

Running the evaluation also produced two robustness fixes. Embeddings are cached to disk, so a failed database write never means embedding everything again. The Ollama client retries server errors, because the model runner crashed once during an hour-long run.

## 7. Deployment

```sh
scripts/get-data.sh               # SQuAD 2.0 dev set into data/ (not committed)
docker compose up -d db ollama
docker compose run --rm models    # pulls nomic-embed-text and qwen2.5:1.5b into the ollama volume
docker compose run --rm ingest    # loads SQuAD, embeds the paragraphs, builds the index (about 7 min on CPU)
docker compose up -d api
curl -s localhost:8080/ask -d '{"question": "Who patented a high-pressure compound engine in 1805?"}'
```

`db` is `pgvector/pgvector:0.8.0-pg16` and `ollama` is `ollama/ollama:0.40.6`. `api` is this repo's Dockerfile: a static Go binary on Alpine, running as a non-root user, about 27 MB. Open `localhost:8080` for the results page with a live question box.

`POST /ask` returns `{question, answer, abstained, cited, sources: [{n, title, text, score, kept}], ms}`; `kept` is false for passages the relevance grader set aside. Settings come from the environment: `DATABASE_URL`, `OLLAMA_URL`, `EMBED_MODEL`, `CHAT_MODEL`, `FOOTNOTE_CHUNKER` (para), `FOOTNOTE_METHOD` (dense), `FOOTNOTE_K` (3), `FOOTNOTE_MIN_SIMILARITY` (0.6392) and `FOOTNOTE_GRADER` (relevance, or off).

## Reproducing the numbers

With Postgres (pgvector) and Ollama running locally:

```sh
scripts/get-data.sh
go build -o bin/footnote ./cmd/footnote
./bin/footnote ingest -chunkers para,w100,w300                                   # about 23 min on 2 CPU cores
./bin/footnote eval-retrieval                                                    # results/retrieval.json
./bin/footnote eval-answers -split test -n 200 -k 3 -chunker para -method dense  # about 1.5 h; resumable
./bin/footnote eval-gate                                                         # results/gate.json
./bin/footnote eval-graders -n 80                                                # results/graders.json, about 1.5 h
./bin/footnote eval-translate -split test -n 60                                  # results/translate-test.json, about 1 h
go run ./cmd/builddemo                                                           # docs/index.html
go test ./...
```

`results/choice.json` records the settings picked on the tuning articles. The page design is described in [web/DESIGN.md](web/DESIGN.md).

## Layout

```
cmd/footnote/       CLI and HTTP server: ingest, eval-retrieval, eval-answers, eval-gate, eval-graders, eval-translate, ask, serve
cmd/builddemo/      fills web/template.html with the results to make docs/index.html
internal/corpus/    SQuAD loader, byte-exact answer spans, chunkers, tuning/test split
internal/store/     Postgres schema, COPY loading, full-text and pgvector search
internal/ollama/    embeddings and chat client
internal/rag/       retrieval methods, fusion, prompt, citation parsing, SQuAD scoring, the course's rewrites and graders
```

## Limits

- The documents are Wikipedia articles from SQuAD, chosen because the questions come with known answers. Pointing Footnote at other files means swapping the loader; the search, database and server stay the same.
- Exact match is strict: "in the 10th century" against "10th century" scores 0. F1 gives partial credit.
- With 200 questions per group, the answer-level numbers carry about ±7 points of sampling noise at 95% confidence. The retrieval numbers use every question.
- Everything runs on CPU. A GPU would make answers several times faster, but wouldn't change the results.
