# Design notes

The page is a page of a scholarly book lying on a dark desk, the same composition as the project's thumbnail. A question-answering system that cites its sources works like a footnoted text: a question, an answer, and a small superscript number that sends you to the exhibit at the foot of the page. So each example is set as a Q. and an A., the answer is run over with a yellow highlighter and carries the number of the passage it cites, and the retrieved passages sit below a short footnote rule, numbered the way the model cites them: the cited one in roman, the others in italic and faded.

## Layout

- **The desk and the page:** a cream page column (max 960px) on a dark desk, with a faint binding shadow down its left edge.
- **Running head:** the wordmark and stack in italic, the source link, and a lamp switch for light and dark.
- **Title:** "Footnote" in bold, an italic subtitle, a short rule, the lede, then a colophon line between hairlines: source, article and paragraph counts, the search method and how many passages the model reads.
- **Margin figures:** the two headline numbers (right passage found first; exact answers with and without the documents) are computed from the results. On a desk 1260px or wider they sit in the dark margins beside the page, in big serif figures; narrower, they sit on the page under the colophon.
- **The reader:** the index of test questions is on the left, one ruled entry per question, each with its coordinate (A1 to C4), a status mark (✓ answered from the passages, – correctly declined, ✗ wrong) and the question. The chosen one is highlighted. The question and answers are on the right: the answer from memory (struck through when wrong), then the answer after reading, highlighted, with its superscript citation, a one-line ruling against the gold answer, and the passages as footnotes, with the gold answer highlighted in the one that holds it. When the page is served by `footnote serve`, a live question box appears above the index and answers set on the same page. On GitHub Pages it stays hidden, because there's no server to answer.
- **Findings:** numbered sections (§ 01 to § 07) with the number and a short label in a left column; tables use book rules (heavier top and bottom, hairlines between rows) and the shipped row is highlighted.

Below 980px the index moves above the answer, as two columns, and the section numbers go inline. Below 560px the index is a single column.

## Color

| Token | Light (page on the desk) | Dark (lamp on) |
|---|---|---|
| Desk | `#2c2a27` | `#131210` |
| Page | `#fbf8f0` | `#22201c` |
| Ink | `#26221c` | `#ece4d2` |
| Soft text | `#4a443a` | `#cec5b2` |
| Faint text | `#6b6458` | `#a89f8b` |
| Highlighter | `#f6e27a` | `#f6e27a` |
| Selected row | `#f6e27a` | `#3b3524` with a highlighter edge |

Dark mode turns the page dark and the type warm and light. The highlighter keeps its colour, with dark ink on it, but is used only on the answer and the gold span in a passage; selected rows get a dim ochre instead.

| Series | Light | Dark |
|---|---|---|
| word search (lexical) | `#a8590b` | `#e0a050` |
| meaning search (dense) | `#41398f` | `#aaa3f0` |
| both, fused (hybrid) | `#1d6e68` | `#5cc0b4` |

Each series also has its own marker shape (square, circle, diamond), shown in the legend and labelled at the line ends, so colour is never the only cue. The status marks use symbols and screen-reader text as well as colour.

## Type

Crimson Pro throughout, self-hosted from `docs/fonts` (SIL OFL 1.1, licence alongside): bold for the title, headings and the answer, italic for the subtitle, labels and uncited footnotes, regular for running text. `footnote serve` serves `/fonts/` from the same folder.
