# Design notes

The page is a court deposition transcript. A question-answering system that has to show its sources is close to a witness who has to point at the record, so the page borrows those conventions: pleading paper with line numbers down the left margin, a double vermilion margin rule, a caption block, testimony set as Q. and A., and retrieved passages filed as numbered exhibits with stickers.

## Layout

A caption block opens the page with the title ("Answer from the record."), the corpus, the models, and the settings chosen on the tuning articles. Below it a docket lists example questions in three groups: answered from the record, correctly declined, and got it wrong. Picking one fills the transcript with the question, the answer from memory alone, the answer with exhibits and its citations, and a one-line ruling against the gold answer. The exhibits follow, with the gold answer highlighted when an exhibit holds it. "Findings of fact" holds the retrieval chart, the chunk-size table and the closed-book vs retrieval verdicts, and "Objections" lists the limits. When the page is served by `footnote serve`, a live question box appears. On GitHub Pages it stays hidden, because there's no server to answer.

Line numbers are recomputed from the rendered layout, so they stay aligned at any width. Below 900px the columns stack and the right-hand rule drops away. Below 520px the sheet runs edge to edge with a narrower margin.

## Color

Light mode is paper `#fbfaf6` on a desk of `#e7e4dc`, with near-black ink. The margin rule is vermilion `#c4361f` and the exhibit stickers are mint `#bfe6cf`. Dark mode is its own set of values rather than an inversion: paper `#15161a` on `#0f1013`, with the rule lifted to `#e0573d` and the stickers darkened to `#1f4a36` with light text.

Chart series were checked with the dataviz palette validator against both surfaces:

| Series | Light | Dark |
|---|---|---|
| word search (lexical) | `#c4361f` | `#e0573d` |
| meaning search (dense) | `#1b7f5f` | `#2ea57c` |
| both, fused (hybrid) | `#4251b8` | `#7683ea` |

Each series also has its own marker shape (square, circle, diamond), shown in the legend too, so color is never the only cue.

## Type

Libre Bodoni sets the caption and headings, Public Sans the running text, and Cutive Mono the line numbers and Q./A. marks, like a court reporter's typewriter.
