package rag

// Techniques from "Learn RAG From Scratch" (Lance Martin, LangChain; freeCodeCamp, 2024), rebuilt
// without LangChain: four ways to rewrite a question before retrieval (multi-query, RAG-Fusion,
// step-back, HyDE), and the yes/no graders from its CRAG and adaptive-RAG sections.
//
// The query-rewriting prompts are copied from the course's notebooks (langchain-ai/rag-from-scratch,
// rag_from_scratch_5_to_9.ipynb). The grader prompts aren't in that repo: they come from the
// LangGraph CRAG and adaptive-RAG examples the course runs, whose relevance grader the transcript
// quotes. The course runs these on GPT-3.5 and Command R; here they run on the same 1.5B model
// that answers. One small difference: the notebook's RRF counts ranks from 0, Fuse from 1.

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/evannwilsonn/footnote/internal/ollama"
	"github.com/evannwilsonn/footnote/internal/store"
)

const multiQueryPrompt = `You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: %s`

const fusionPrompt = `You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries related to: %s
Output (4 queries):`

// The course asks for a "scientific paper passage"; the topic word is dropped because these
// documents are encyclopedia articles, as the course suggests tuning the prompt per domain.
const hydePrompt = `Please write a passage to answer the question
Question: %s
Passage:`

// Step-back uses the course's few-shot prompt, from the step-back prompting paper.
const stepBackSystem = `You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:`

var stepBackShots = [][2]string{
	{"Could the members of The Police perform lawful arrests?", "what can the members of The Police do?"},
	{"Jan Sindel’s was born in what country?", "what is Jan Sindel’s personal history?"},
}

// lines turns a list the model wrote into clean queries: numbering, bullets and blanks removed.
func lines(text string, max int) []string {
	var out []string
	for _, l := range strings.Split(text, "\n") {
		l = strings.TrimSpace(l)
		l = strings.TrimLeft(l, "-*•0123456789.) ")
		l = strings.Trim(l, `"`)
		if len(l) < 8 || strings.HasSuffix(l, ":") {
			continue
		}
		out = append(out, l)
		if len(out) == max {
			break
		}
	}
	return out
}

func complete(ctx context.Context, llm *ollama.Client, msgs []ollama.Message, max int) (string, error) {
	res, err := llm.Chat(ctx, msgs, max)
	return strings.TrimSpace(res.Text), err
}

// MultiQuery asks for five rewordings of the question.
func MultiQuery(ctx context.Context, llm *ollama.Client, q string) ([]string, error) {
	t, err := complete(ctx, llm, []ollama.Message{{Role: "user", Content: fmt.Sprintf(multiQueryPrompt, q)}}, 200)
	return lines(t, 5), err
}

// FusionQueries asks for four related search queries, for RAG-Fusion.
func FusionQueries(ctx context.Context, llm *ollama.Client, q string) ([]string, error) {
	t, err := complete(ctx, llm, []ollama.Message{{Role: "user", Content: fmt.Sprintf(fusionPrompt, q)}}, 160)
	return lines(t, 4), err
}

// StepBack asks for a more general version of the question.
func StepBack(ctx context.Context, llm *ollama.Client, q string) (string, error) {
	msgs := []ollama.Message{{Role: "system", Content: stepBackSystem}}
	for _, s := range stepBackShots {
		msgs = append(msgs, ollama.Message{Role: "user", Content: s[0]}, ollama.Message{Role: "assistant", Content: s[1]})
	}
	msgs = append(msgs, ollama.Message{Role: "user", Content: q})
	t, err := complete(ctx, llm, msgs, 60)
	if l := lines(t, 1); len(l) == 1 {
		t = l[0]
	}
	return t, err
}

// HyDE writes a hypothetical passage that answers the question; the passage is what gets embedded.
func HyDE(ctx context.Context, llm *ollama.Client, q string) (string, error) {
	return complete(ctx, llm, []ollama.Message{{Role: "user", Content: fmt.Sprintf(hydePrompt, q)}}, 160)
}

// MaxSim merges several result lists by each chunk's best similarity: the course's "unique union"
// of multi-query results, given an order so it can be scored at k.
func MaxSim(lists [][]store.Hit, k int) []store.Hit {
	best := map[int64]store.Hit{}
	for _, l := range lists {
		for _, h := range l {
			if b, ok := best[h.ChunkID]; !ok || h.Score > b.Score {
				best[h.ChunkID] = h
			}
		}
	}
	out := make([]store.Hit, 0, len(best))
	for _, h := range best {
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

// ---------- graders ----------

var yesNo = map[string]any{
	"type":       "object",
	"properties": map[string]any{"binary_score": map[string]any{"type": "string", "enum": []string{"yes", "no"}}},
	"required":   []string{"binary_score"},
}

func grade(ctx context.Context, llm *ollama.Client, system, user string) (bool, error) {
	res, err := llm.ChatFormat(ctx, []ollama.Message{{Role: "system", Content: system}, {Role: "user", Content: user}}, 20, yesNo)
	if err != nil {
		return false, err
	}
	var v struct {
		Score string `json:"binary_score"`
	}
	if err := json.Unmarshal([]byte(res.Text), &v); err != nil {
		return false, fmt.Errorf("grader reply %q: %w", res.Text, err)
	}
	return v.Score == "yes", nil
}

// The course's document grader (CRAG and adaptive RAG): deliberately lenient, to catch retrievals
// that are plainly off topic.
const relevanceGrader = `You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.`

// A stricter variant: SQuAD's unanswerable questions are on topic by design, so a topic check
// can't catch them. This asks the question the abstention actually depends on.
const answerGrader = `You are a grader checking whether a retrieved document answers a user question.
Grade it 'yes' only if the document states the answer to the question. If the document is about the same topic but does not state the answer, grade it 'no'.
Give a binary score 'yes' or 'no'.`

// The course's hallucination grader (adaptive RAG): is the answer supported by the documents?
const groundingGrader = `You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.`

type Grader string

const (
	Relevance Grader = "relevance" // the course's prompt
	Answers   Grader = "answers"   // the stricter variant
)

// GradeDoc grades one retrieved passage for a question.
func GradeDoc(ctx context.Context, llm *ollama.Client, g Grader, question string, doc store.Hit) (bool, error) {
	system := relevanceGrader
	if g == Answers {
		system = answerGrader
	}
	return grade(ctx, llm, system, fmt.Sprintf("Retrieved document:\n\n%s\n\nUser question: %s", strings.TrimSpace(doc.Text), question))
}

// Grounded asks whether an answer is supported by the passages it was given.
func Grounded(ctx context.Context, llm *ollama.Client, answer string, docs []store.Hit) (bool, error) {
	var b strings.Builder
	for _, d := range docs {
		b.WriteString(strings.TrimSpace(d.Text) + "\n\n")
	}
	return grade(ctx, llm, groundingGrader, fmt.Sprintf("Set of facts:\n\n%s\nLLM generation: %s", b.String(), answer))
}
