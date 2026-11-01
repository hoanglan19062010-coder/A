# Cardigan girl: self-critique

Judged like any illustration, with the Aoi design sheet as the professional bar (about 9/10).
Scores are my own assessment.

| Area | First try (scrapped) | Redraw | What still falls short |
|---|---|---|---|
| Proportion and balance | 3 | 7 | 6.35 heads and the weight ankle sits under the pit of the neck. The legs still read long and tube-like, and the knees have almost no anatomy. |
| Gesture | 2 | 4 | A stiff design-sheet pose. Only the head tilt, the knees turning in and the angled forearms give it life. Aoi's slight lean and dropped shoulders feel human; this does not yet. |
| Face | 4 | 7 | The book's eye method reads well and the face is appealing. It is generic: neutral expression, flat eye whites, a tiny mouth. |
| Hair | 2 | 6 | The bell-shaped bob with swept bangs reads correctly. The shine band is too evenly spaced, and there are no thin see-through strands like the ones that make Aoi's hair airy. |
| Clothing and folds | 3 | 6 | Cardigan, rib cuffs, placket, button skirt and loafers all read. The cardigan folds are soft stripes rather than folds pulled from tension points, so the knit has no weight. The hands hide in sweater paws, which dodges the hardest part. |
| Line | 3 | 5 | Tapered, colour-matched lines and a heavier outer contour. Still mechanical: every curve is perfectly smooth, with no breaks where the light hits and none of the sketchy energy of the Aoi sheet. |
| Colour and value | 4 | 7 | Cohesive palette, one light source from the upper left, cast shadows computed from the real shapes, a clear value read. Only one shadow tone per material and no bounce light. |
| **Overall** | **3** | **6** | It is clearly anime and coherent now, but it reads as clean vector-style art, not a hand-painted design sheet. |

## How it was made
- `engine.py` is a small raster paint engine: pressure-tapered brush strokes, polygon fills, layers with normal, lock-alpha, multiply and screen modes, and blur.
- Stage 1 draws the construction: head sphere and guide lines, 3-box torso, hip box, S-curve line of action, joints, and a head-unit ruler.
- Stage 2 draws lineart over the bones. Stage 3 adds flat colours. Stage 4 adds clipped shadows, highlights and the outer contour.
- Run `python3 render.py all` to rebuild every stage into `out/`.

## What I would fix next
1. Gesture: a real weight shift through the shoulders, and one hand doing something.
2. Line: thin on the lit side, heavy in shadow, and broken where the light hits.
3. Hair: irregular shine shapes and a few stray strands.
4. Folds: start each fold from a tension point (shoulder, elbow, button) per the book's clothing chapter.

`LESSONS.md` logs every part that was scrapped and the rule learned from it.
