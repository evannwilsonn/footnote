package rag

import (
	"testing"

	"github.com/evannwilsonn/footnote/internal/store"
)

func TestFuseRewardsAgreement(t *testing.T) {
	a := []store.Hit{{ChunkID: 1}, {ChunkID: 2}, {ChunkID: 3}}
	b := []store.Hit{{ChunkID: 3}, {ChunkID: 4}, {ChunkID: 1}}
	got := Fuse([][]store.Hit{a, b}, 3)
	// 1: 1/61 + 1/63, 3: 1/63 + 1/61, tie broken by id; both beat chunks found by one list only.
	if got[0].ChunkID != 1 || got[1].ChunkID != 3 || len(got) != 3 {
		t.Fatalf("got %v", []int64{got[0].ChunkID, got[1].ChunkID, got[2].ChunkID})
	}
}

func TestParse(t *testing.T) {
	a := Parse("Richard Fleischer [2].", 3)
	if a.Text != "Richard Fleischer" || len(a.Cited) != 1 || a.Cited[0] != 2 || a.Abstained {
		t.Fatalf("got %+v", a)
	}
	if !Parse("I don't know.", 3).Abstained {
		t.Fatal("did not spot an abstention")
	}
	if got := Parse("1066 [7]", 3).Cited; len(got) != 0 {
		t.Fatalf("kept a citation to a source that wasn't shown: %v", got)
	}
}

func TestSquadScoring(t *testing.T) {
	if Normalize("The  Eiffel-Tower!") != "eiffeltower" {
		t.Fatalf("normalize: %q", Normalize("The  Eiffel-Tower!"))
	}
	if ExactMatch("the Normans", []string{"Normans"}) != 1 {
		t.Fatal("articles should not matter")
	}
	// pred "in northern france" vs gold "northern france": precision 2/3, recall 1, F1 0.8.
	if f := F1("in northern France", []string{"northern France"}); f < 0.799 || f > 0.801 {
		t.Fatalf("F1 = %v", f)
	}
	if F1("Paris", []string{"London"}) != 0 {
		t.Fatal("no overlap should score 0")
	}
}
