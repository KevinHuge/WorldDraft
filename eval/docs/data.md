# Data

`bash eval/download_data.sh` puts the benchmark data in `data/` (or `$WORLDDRAFT_DATA`). Paths stored inside these
files are written as `data/...` (inside the data directory) or relative to the repository; `worlddraft_eval/common/paths.py`
resolves both.

```
data/
├── drafts/   432 draft videos with preprocessing metadata (~110 MB)
├── gt/       ground truth exported from the Blender scenes (~110 MB)
└── scenes/   432 Blender scenes, optional (--with-scenes, ~560 MB)
```

## Scenes and cells

48 scene families `A01`…`F08`; the letter is the main challenge: **A** multi-entity choreography, **B** multi-stage
dynamics, **C** occlusion and reappearance, **D** camera–world coupling, **E** articulated / non-rigid dynamics,
**F** mixed compositional scenes. Each family has nine cells `<P>_<A>`:

| | AN (neutral) | AA (aligned) | AC (conflict) |
|---|---|---|---|
| **P0** proxy / abstract draft | `P0_AN` | `P0_AA` | `P0_AC` |
| **P1** simulation draft | `P1_AN` | `P1_AA` | `P1_AC` |
| **P2** high-quality previs | `P2_AN` | `P2_AA` | `P2_AC` |

The nine cells share the world (entities, motion, events, camera) and the target prompt; polish changes how much visual
information the draft shows, alignment changes whether the draft's appearance agrees with the target text.

## drafts/

```
drafts/<case>/target_prompt.txt                       target prompt shared by the nine cells
drafts/<case>/<case>_<cell>/<case>_<cell>.mp4         832×480, 81 frames, 16 fps (5.0625 s)
drafts/<case>/<case>_<cell>/preprocessing.json
```

`preprocessing.json` records how the Blender render (`source`, 50 frames) became the draft: `size`, `fps`, `frames`,
`frame_indices` (source frame of every draft frame; uniform over the timeline, with repeats), `source_metadata`,
the aspect-preserving resize with gray letterbox (`spatial_preprocessing`) and the sha256 of
the source, the draft and the prompt. Evaluation uses it to map Blender coordinates and times to the draft canvas.

## gt/

| path | content | used by |
|---|---|---|
| `<case>/gt.json`, `instances/*.png`, `rendered_ids/*.png`, `overlays/*.png` | visible-instance ground truth at the 8 QA instants: per-entity visibility, mask area, centroid, visible bbox; 16-bit instance-ID masks (pixel value = `instance_id`, 0 = background) | QA, Count |
| `full-entity/F03/` | finer re-export of F03 (every domino as its own entity) used for its QA | QA |
| `entities/<case>.json` | projected amodal entity boxes for all 81 frames (`frames[].boxes[<entity>] = {box, visible, source_box}`), entity labels, `dynamic` flags | geometry |
| `interaction-regions/<case>.json` | interaction regions per frame for event-question zooms | events |
| `events/registry.json` | event instances with start/end times and frames from Blender state rules (`instances[]`), sources and definitions | events, Timing |
| `events/source_states/<case>.json` | the evaluated Blender state signals behind the boundaries | events |
| `camera/<case>.json` | D series: camera-to-world trajectory per frame (position, quaternion xyzw, intrinsics; OpenGL convention, Blender units) | camera |

Pixel coordinates use a top-left origin and exclusive bbox maxima; GT is expressed on the 832×480 draft canvas
(including the letterbox). Each GT file records the scene file, preprocessing file and their sha256 it was made from.
Six instance masks are entirely 0 on purpose: the entity is fully occluded at that instant (C03, C05, C07).

## scenes/

`scenes/<case>/` holds the nine `<case>_<cell>.blend` files (Blender 5.2; all assets packed), `target_prompt.txt`,
`appearance.json` (target and source appearance of the scene's objects) and `render.py`:

```bash
cd data/scenes/A01
blender --background --factory-startup --disable-autoexec --python render.py -- --cell P1_AA --frame 1
blender --background --factory-startup --disable-autoexec --python render.py -- --cell P2_AA --animation
```

PNGs are written to `renders/<cell>/` (or `--output`). Render engine, resolution, samples and frame range are those
saved in the scene; Cycles uses the CPU by default, Eevee needs EGL on a headless Linux machine.

## Regenerating the ground truth

Needs the scenes and the `bpy` environment (`eval/requirements-gt.txt`), from the repository root with
`PYTHONPATH=eval`:

| GT | command |
|---|---|
| visible instances | `python -m worlddraft_eval.gt.visible --cases A01 --output-dir <dir>` (writes `<dir>/A01/`) |
| projected entity boxes | `python -m worlddraft_eval.gt.entities --out <dir> [--cases ...]` |
| interaction regions | `python -m worlddraft_eval.gt.regions --out <dir> [--cases ...]` |
| event registry | `python -m worlddraft_eval.gt.events [--out <dir>]` |
| camera trajectories (D series) | `python -m worlddraft_eval.gt.camera --out <dir> [--cases ...]` |

Each command lists its options with `--help`. Write to a new directory and compare before replacing `data/gt`;
changing the GT changes its hashes, so evaluate with a new result directory.
