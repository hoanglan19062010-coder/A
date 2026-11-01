# Cardigan girl: self-critique

Judged like any illustration, with the Aoi design sheet as the professional bar (about 9/10).
Scores are my own assessment. The third column is after the user's seven notes (see `LESSONS.md`).

| Area | First try | Redraw | After your notes | What still falls short |
|---|---|---|---|---|
| Proportion and balance | 3 | 7 | 8 | The knee now sits at 0.71 of the height, like the Aoi sheet and the book. The torso is still a little long for her height. |
| Gesture | 2 | 4 | 5 | The relaxed hands help, but it is still a stiff design-sheet stance with no weight shift through the shoulders. |
| Face | 4 | 7 | 8 | Fuller cheeks, a gradient iris with reflections and hatching, and lash flicks. The expression is still neutral and generic. |
| Hair | 2 | 6 | 7 | Cel shadows shaped like strands, long highlight strands, and fine pointed ends. The dark strands under the jaw read a little like spikes. |
| Hands and feet | 1 | 3 | 6 | Hands are built with the book's wedge-and-segments method, and the feet sit inside the loafers. Finger joints and knuckles are only implied. |
| Clothing and folds | 3 | 6 | 7 | The sleeves now show compression, half-lock and hanging folds. Cardigan body folds are still soft stripes. |
| Line | 3 | 5 | 5 | Tapered lines and a heavier outer contour. Lines are still mechanical, with no light-side thinning or broken lines. |
| Colour and value | 4 | 7 | 7 | One light source, computed cast shadows, a clear value read. Still one shadow tone per material and no bounce light. |
| **Overall** | **3** | **6** | **7** | Clearly anime and coherent, but it reads as clean vector-style art rather than a hand-painted design sheet. |

`fixes_closeups.png` shows each of the seven notes before and after, and `fixes_before_after.png` shows the whole figure.

## How it was made
- `engine.py` is a small raster paint engine: pressure-tapered brush strokes, polygon fills, layers with normal, lock-alpha, multiply and screen modes, and blur.
- Stage 1 draws the construction: head sphere and guide lines, 3-box torso, hip box, S-curve line of action, joints, and a head-unit ruler.
- Stage 2 draws lineart over the bones. Stage 3 adds flat colours. Stage 4 adds clipped shadows, highlights and the outer contour.
- Run `python3 render.py all` to rebuild every stage into `out/`.

## What I would fix next
1. Gesture: a real weight shift through the shoulders, and one hand doing something.
2. Line: thin on the lit side, heavy in shadow, and broken where the light hits.
3. Hair: lighter, more varied strands behind the neck instead of dark spikes.
4. Folds: pull the cardigan body folds from tension points (shoulder, buttons) like the sleeves now do.

`LESSONS.md` logs every part that was scrapped and the rule learned from it.
