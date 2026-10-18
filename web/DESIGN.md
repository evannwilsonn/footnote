# Design notes

The page is a microfiche reader. A question-answering system that cites its sources works like someone at a library reader: pick a frame off the fiche, project it, and read the passage it points to. So the example questions sit on a fiche card as a grid of frames with coordinates (A1 to C4), and choosing one projects it onto the reader screen beside it, with the retrieved passages underneath, numbered the way the model cites them.

## Layout

- **Nameplate:** the wordmark, the stack, the source link, and a lamp switch for light and dark.
- **Opening:** the headline ("Every answer, with its frame number.") and a short lede, then a dark header strip in the style of a fiche title band: source, article and paragraph counts, the search method and how many passages the model reads.
- **The reader:** the fiche is on the left. It is a dark film card with a 4 × 3 grid of frames, each showing its coordinate, a status mark (✓ answered from the passages, – correctly declined, ✗ wrong) and the question. The reader screen is on the right, a pale projection with registration corners. It shows the question, the answer from memory beside the answer after reading, a one-line ruling against the gold answer, and the passages, with the gold answer highlighted in the one that holds it. When the page is served by `footnote serve`, a live question box appears at the top of the fiche and answers project onto the same screen. On GitHub Pages it stays hidden, because there's no server to answer.
- **Findings:** one long panel indexed 01 to 06 in a left column:
  - 01: retrieval, as a chart
  - 02: chunk size, as a table
  - 03: answering, as paired bars, from memory vs after reading
  - 04: the similarity gate
  - 05: limits
  - 06: running it

Below 980px the fiche moves above the screen and the findings index goes inline. Below 560px the fiche becomes two columns and drops its axis labels, since each frame carries its own coordinate.

## Color

Diazo fiche prints violet on clear film, so violet is the ink and the projection lamp gives the amber accent.

| Token | Light | Dark |
|---|---|---|
| Housing (page) | `#d8d5e2` | `#0e0c17` |
| Screen (panels) | `#f7f5fc` | `#1b1830` |
| Ink | `#1d1838` | `#ece9ff` |
| Violet (headings, links) | `#3d2a9e` | `#b7a8ff` |
| Lamp (accent) | `#e08a1e` | `#f2a54a` |
| Film (fiche, code) | `#1d1934` | `#07060e` |
| Answer highlight | `#ffe2a8` | `#5b3f0b` |

Dark mode is its own set of values rather than an inversion. Chart series were checked with the dataviz palette validator against each mode's screen colour:

| Series | Light | Dark |
|---|---|---|
| word search (lexical) | `#b0600c` | `#cf7f2a` |
| meaning search (dense) | `#5a3ccf` | `#9077ee` |
| both, fused (hybrid) | `#007c99` | `#1fa6b4` |

Each series also has its own marker shape (square, circle, diamond), shown in the legend and labelled at the line ends, so colour is never the only cue. The status marks on the frames use symbols and screen-reader text as well as colour.

## Type

- **Syne (800):** the wordmark, headline and section numbers. It is wide and mechanical, like the stencilled lettering on reader hardware.
- **Figtree:** running text and the projected questions.
- **Martian Mono:** coordinates, labels and the header strip.
