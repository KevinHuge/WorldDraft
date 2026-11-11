# Protocol files

The inputs that define what is asked and scored, used as given by the evaluation. Metric definitions are in
[../docs/metrics.md](../docs/metrics.md).

| path | content | read by |
|---|---|---|
| `qa/questions.csv` | the question bank: 9,875 questions for the 48 scenes | QA stage |
| `qa/provenance.json`, `qa/factual-contrasts.json`, `qa/source-material-audit.json`, `qa/coverage.json` | where every question comes from (prompt sentence, appearance record or manual verification), the attribute replacements of negated questions, the audit of the source colors in the AC scenes, counts | QA stage (frozen with each run), report, tests |
| `questions/<case>.json` | per-scene QA settings: input files, judge, sampling instants, entity groups and labels, per-question bindings | QA stage |
| `events/` | event semantics (`events.json`), timeline descriptions (`descriptions.json`), boundary rules (`timing_rules.json`), compiled motion protocol, negated statements, stage settings | event stage |
| `motion-scope.json` | per-scene motion scope and interaction-region definitions | event stage, GT export |
| `entities.json` | which Blender objects form each evaluated entity, their labels and detector prompts | geometry stage, GT export |
| `contracts/<case>.json` | entities and attributes exported into the visible-instance GT | GT export |

## Question bank

`qa/questions.csv` keeps the column layout of [ELLA's DPG-Bench](https://github.com/TencentQQGYLab/ELLA/blob/main/dpg_bench/dpg_bench.csv)
(without `text`) and adds the benchmark's own fields; one row is one question:

| column | meaning |
|---|---|
| `item_id`, `proposition_id` | scene and question number; together the question key |
| `dependency` | `proposition_id` of the parent question (`0` = none); AP persistence questions depend on their anchor |
| `category_broad`, `category_detailed`, `tuple`, `keywords` | descriptive fields for reading the bank; not parsed for scoring |
| `question_natural_language` | the question sent to the judge |
| `attribute` | `color` / `material` / `texture` / `environment` / `lighting`; empty for counts |
| `entity_id` | GT entity or entity group the question is about; empty for scene-level questions |
| `metric` | `EntityCount`, `RF`, `AP` or `SL` |
| `expected_answer` | `yes` / `no` for RF and AP; empty for counts (the GT count is filled in per instant) |
| `applies_to` | `ALL` or the cells it is asked in (`AC` for SL) |

| metric | questions | notes |
|---|---:|---|
| EntityCount | 434 | global (full frame) and local (GT region) counts at three instants; exact integer match |
| RF | 3,378 | 1,689 propositions × {original, literal negation}; color 1,574, material 718, texture 694, environment 200, lighting 192 |
| AP | 5,564 | 1,391 attributes × {anchor, persistence} × {original, negation} |
| SL | 499 | AC cells only; source colors that conflict with the target in that polish |

**Literal negation.** Each RF and AP proposition is asked with its literal negation (`Is this car green?` → yes,
`Is this car not green?` → no), each scored against its own expected answer; a consistently wrong judge cannot gain
by always answering one way. The negation of persistence applies to the whole statement (`Does this car not remain green throughout the sequence?`). Event statements are negated as `It is not the case that: <proposition>`. AP scores a persistence question only when its anchor (same entity, part, attribute and target at one visible instant) is answered correctly. Count and SL have no negated pair; SL reports the share of `yes`.

**Settings.** `questions/<case>.json` names its inputs in `files` (`data/...` entries live in the data directory,
others are relative to the settings file), the 8 QA instants (`qa.times_s`) and 3 count instants, entity groups for
counting (`qa.groups`), per-question bindings (count scope `global`/`region`, cell restrictions for SL), and
`qa.entity_labels` naming every visible GT instance shown to the judge. Where a scene's source appearance differs
between polish levels, `bindings` restricts or rephrases the SL question per cell (recorded in
`qa/source-material-audit.json`).
