# Metrics

How each score of the benchmark is computed. Commands are in [../USAGE.md](../USAGE.md); file formats in
[data.md](data.md) and [../protocol/README.md](../protocol/README.md).

## 1. Common conventions

- 48 scene families × 9 cells (`P0/P1/P2` × `AN/AA/AC`) = 432 videos per model. `P` is draft polish, `A` the
  relation between the draft's source appearance and the target appearance in the prompt.
- Videos are aligned in **seconds**, normally over the first 5 s. No per-video registration (shift, scale, rotation or
  time warp) is fitted to a generated video.
- Sampling differs per metric: geometry, camera and CLIP use the 40 instants `t = k/8 s, k = 0…39`; QA uses 8 sparse
  instants (Count uses 3 of them); image and aesthetic quality use every native frame of the first 5 s.
- Missing results, parse failures, not-applicable items and real zeros are recorded separately. A GT object that is not
  visible is not counted as a generation miss; a missing prediction for a valid GT counts as a failure.
- How a model receives the 832×480 draft (whole-canvas resize, centre crop, letterbox, …) is declared in
  `eval/configs/models.yaml`; GT boxes and masks are mapped to the output frame with it. Nothing is fitted to the output.

## 2. Overview

| group | metric | meaning | direction |
|---|---|---|---|
| Entity & spatial | Global Count | the counted category is exactly right at all three sampled instants | ↑ |
| | Local Count | per-instant exact count inside a GT region | ↑ |
| | BBox IoU | projected GT box vs tracked box after one video-level assignment | ↑ |
| Motion & event | Matched Track Coverage | share of GT-valid instants at which a matched track has a box (dynamic entities) | ↑ |
| | ISR | events whose action and outcome are both judged correct | ↑ |
| | Timing Coverage | expected events whose required boundaries are localized | ↑ |
| Camera | ViPE Rot / Trans / CamMC | camera trajectory error against the reference draft (D series) | ↓ |
| Render fidelity | RF | target color, material, texture, environment, lighting questions | ↑ |
| | Image / Aesthetic / CLIP | MUSIQ, LAION aesthetic, frame–text CLIP similarity | ↑ |
| Appearance persistence | AP | target attribute kept over time, gated by a correct anchor | ↑ |
| Disentanglement | SL (color) | conflicting source color still present (AC cells) | ↓ |

Proportions are reported ×100, CLIP as cosine ×100; camera errors keep their units. RF, the three quality scores and AP
are reported separately; there is no composite score.

## 3. Ground truth: two pipelines

**Visible instance masks (QA and Count).** `worlddraft_eval/gt/visible.py` renders per-pixel instance IDs from the `P2_AA`
scene at the 8 QA instants (`data/gt/<case>/`). Masks give visibility, visible area, centroid and visible bbox.
The draft's canvas mapping (scale and padding from `preprocessing.json`) is applied to GT; padding is part of the
input canvas. `gt_not_visible` marks questions without visible evidence (AP needs at least two visible instants).

**Projected entity boxes (IoU and tracks).** `worlddraft_eval/gt/entities.py` evaluates each semantic entity's geometry
(including animation and deformation) at every Blender frame, takes the world-space AABB corners, projects them through
the scene camera, clips to the canvas and maps them to the draft canvas (`data/gt/entities/<case>.json`,
81 frames at 16 fps). These are **amodal** boxes: occlusion is not tested, so a fully occluded entity can still be
scored, and an AABB projection can be wider than the silhouette. They are a different annotation from the visible masks.

## 4. BBox IoU

1. SAM 3 detects and tracks the first 5 s with semantic text prompts (no target color or material), score ≥ 0.2.
2. Predicted boxes are read at the 40 instants (output frame `round(t × fps)`); GT uses the matching 16 fps index.
3. Dynamic or interacting entities are evaluated. For GT entity `e` and predicted track `j`, the mean IoU over the
   GT-valid instants of `e` is computed (missing box = 0); incompatible categories cannot match.
4. One Hungarian assignment per video maximizes the mean track IoU; a match needs a mean ≥ 0.05.
5. With the assignment fixed, per-instant IoU (unmatched or missing = 0) is averaged per entity, then over entities.

```text
IoU_video = mean_e [ mean_{t ∈ V_e} IoU(B[e,t], B̂[a(e),t]) ]
```

When SAM 3 finds no instance of a category in the whole video, SAM 2.1 (Hiera-Large) is initialized from that
category's first valid GT box and propagated (for the draft, also for entities left unmatched by the assignment). The
metric is therefore detection and tracking **with a GT-box-initialized fallback**, not detection independent of GT.
Extra unassigned predictions do not lower IoU (Count covers quantity).

## 5. Matched Track Coverage

Same assignment, dynamic entities only: the share of an entity's GT-valid instants at which its assigned track has a
box, averaged over entities; an entity without a match scores 0, a video without dynamic entities is not applicable.
The 0.05 threshold applies to the whole-track match; once matched, an instant counts as covered even if its IoU is low.
This measures temporal coverage of matched tracks, not path accuracy (it is not ADE, IDF1 or HOTA).

## 6. Camera (ViPE)

D01–D08 only, all cells and models. The reference of a scene is the **ViPE estimate of its draft `P2_AA`**.

- ViPE 1.2.0 (`nv-tlabs/vipe`), pose-only configuration (default pipeline without post depth alignment; UniDepth
  keyframe depth is still used), fixed intrinsics from the Blender camera scaled to the output resolution, and masks of
  the dynamic entities.
- Both trajectories are sampled at the 40 instants and expressed relative to their first pose; each translation is
  divided by its own maximum norm (+1e-9), so absolute scale is not compared.

```text
Rot   = mean_t angle(R_g(t)ᵀ R_v(t))                       (degrees)
Trans = mean_t || u_g(t) − u_v(t) ||₂                      (unitless)
CamMC = mean_t || [R_g(t) | u_g(t)] − [R_v(t) | u_v(t)] ||_F
```

The reference is an estimate, not the Blender extrinsics, so estimator error exists on both sides; per-trajectory
normalization hides differences in motion amplitude and near-static cameras amplify noise. The draft `P2_AA` compared
with itself is excluded from aggregates.

## 7. QA: Count, RF, AP, SL

Judge: `Qwen/Qwen3.6-27B` on vLLM, temperature 0, thinking disabled. Questions: `eval/protocol/qa/questions.csv`;
per-scene settings: `eval/protocol/questions/<case>.json`. QA instants: `[0, 0.6875, 1.4375, 2.125, 2.875, 3.5625, 4.3125, 5]` s, nearest output frame. Evidence is the full frame plus zooms of GT regions (strict visible GT bbox,
no margin); regions come from the reference GT, never from the generated video.

**Count** uses `[0, 2.125, 5]` s. Global GT is the number of visible entities of the category; local GT counts visible
same-category mask centroids inside a GT region (left/top inclusive, right/bottom exclusive). Answers are non-negative
integers matched exactly.

```text
GlobalCount = mean_q Π_{t ∈ 3 instants} 1(n̂[q,t] = n[q,t])
LocalCount  = mean_q mean_t 1(n̂[q,t] = n[q,t])
```

**RF** asks binary questions on color, material, texture, environment and lighting. Each question uses one frame: the
instant with the largest visible GT mask area (earliest on ties); scene-level questions use the first frame. Every
question has a literal negation of the same proposition (`Is this car green?` expected yes; `Is this car not green?`
expected no). Each answer is scored against its expected answer; RF_total averages questions (not the five groups).

**AP** gates persistence by correctness. Each applicable entity attribute has an anchor question Q1 (target attribute
correct at one visible instant) and a persistence question Q2 (kept across the visible instants, at least two), each
with its literal negation (`Does this car not remain green throughout the sequence?` negates persistence as a whole).

```text
AP_q = 1(Q1 correct) × 1(Q2 correct);   AP_video = mean over Q2 propositions
```

A wrong or unanswered anchor scores 0. Reported for color, material and texture. AP is sparse multi-frame consistency;
it does not prove every frame is correct.

**SL (color)** only in AC cells, only for source colors that conflict with the target and are actually encoded in that
polish (from the audit of the scene materials, `eval/protocol/qa/source-material-audit.json`). SL is the share of
`yes` answers to "is the source color present" questions (lower is better), averaged over sampled instants per question,
then over questions.

## 8. Events: ISR and Timing Coverage

Event definitions: `eval/protocol/events/`; instances and boundaries exported from Blender state rules:
`data/gt/events/registry.json`.

**ISR.** Each event is judged on 12 frames covering 0–5 s with zooms of the participants. The judge answers action,
outcome and participants; success requires action **and** outcome to match their expected answers. Negative
statements use `It is not the case that: <proposition>`. Events and polarities are weighted equally.

**Timing Coverage.** Only events with complete evaluable boundaries are used. The judge first localizes the event on
11 frames over 0–5 s, then refines each boundary within ±0.625 s at 0.125 s steps; a boundary time is the midpoint of
the adjacent displayed frames. An event counts when all required boundaries are localized, indices are valid and
onset ≤ offset. Invalid answers are re-asked twice automatically (strict chronological schema, then clarified joint
boundaries).

```text
TimingCoverage_video = localized expected events / expected events
```

No threshold on the error against the GT time is applied: this is boundary-localization coverage, not timing accuracy.

## 9. Image quality, aesthetic quality, CLIP alignment

Image and aesthetic quality follow VBench `imaging_quality` and `aesthetic_quality`, averaged over every native frame
of the first 5 s.

- **Image quality**: MUSIQ with the SPAQ checkpoint (`pyiqa`); frames whose long side exceeds 512 px are resized to 512
  (bilinear, no antialias); the frame mean is divided by 100 and reported ×100.
- **Aesthetic quality**: CLIP ViT-L/14 image features (224 resize/centre crop, L2-normalized) with the LAION linear
  head `sa_0_4_vit_l_14_linear.pth`; the mean is divided by 10 and reported ×100.
- **CLIP alignment**: OpenAI CLIP ViT-B/32; the target prompt is split into sentences (long sentences at punctuation,
  never truncated); every sampled frame (40 instants) against every sentence, equally weighted:

```text
CLIP_video = 100 × mean_{t, s} cos(image[t], text[s])
```

These are per-frame scores (no temporal model); high quality does not imply world preservation.

## 10. Aggregation

1. Score each video as above.
2. Average the selected cells within a scene family, equally.
3. Average families equally (macro average).
4. Every table and metric uses the families that **all models in the table complete** for it; the number of families is
   reported with each table. The canonical operating point is `(P1, AN)`; the full grid needs all nine cells
   (SL: the three AC cells).

**Draft support (Q2).** With A fixed, compare P0/P1/P2 per family (`Δ01 = s(P1) − s(P0)`, `Δ12 = s(P2) − s(P1)`), then
average; negative gains are kept (for camera errors a negative difference is an improvement).

**Source–target disentanglement (Q3).** With P fixed, compare AN/AA/AC: conflict RF, `CP = RF(AA) − RF(AC)`, SL and the
AA − AC difference of each world-preservation metric. CP across polish levels uses only color questions that conflict
explicitly in all three P levels and are answerable for every model in the table (`common_color_*.csv`).
