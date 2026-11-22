# Paper factory v3: redesign brief

Written 22 Nov 2026 from a read of the thesis repo `main` (commit 04a5dcc). For the coding agent: read this, then split the work
into the subagents in section 7.

## 1. Goal and the one rule

Goal: papers that vary as much as real papers do, in every field, without hand-writing more templates per field.

Rule: **form comes from real papers, facts come from the world, choices come from code. Qwen only writes prose
inside a small, fixed frame.** Every v2 failure came from asking Qwen to choose (one disease), count (numbers),
name (Mora / Mora App) or keep a long context consistent (copied claims). v3 removes those jobs from Qwen.

| Job | v2 | v3 |
|---|---|---|
| Section layout | 3 hand-written templates per field | one real paper's layout per synthetic paper (form card) |
| Style axes | 5 hand-written lists | measured from the same real paper |
| Tables | Qwen writes them, a check compares | code renders them; Qwen never types a table |
| Reference list, selection counts | forbidden / missing | code makes both |
| Paper types | synthesis or argument per field | mix measured from the corpus, incl. primary reports |
| Topics (llm) | OpenAlex (classic NLP only) | hand-curated list |
| Clusters | even split, no links | uneven sizes, bridge papers |

## 2. New step: profile the 500 papers (run once, results checked in)

`paperfactory profile-corpus --in <folder of .md> --out packs/forms/<field>.json`

One **form card** per real paper. It holds form only, never a sentence of the paper.

By code (no model):
- heading tree (levels and text), words per section, paragraph count and length per section
- tables: count, position (section), columns, rows, caption length
- citations: style (`[1]`, author-year, footnote), count, citations per 1,000 words, reference count
- sentence length (mean, spread), share of sentences with "we" / passive, hedging words per 1,000 words
- abstract: structured or not, length
- word and 3-word-phrase counts, summed per field (the baseline for section 6)

By Qwen (one short call per paper, closed lists, JSON schema, temperature 0):
- paper type: one of the field's list (trial report, systematic review, case note, method paper, ...)
- for each heading: `generic` (Introduction, Methods, Limitations) or `topical` (names the paper's subject)
- opening move of the introduction: one of a closed list

By Semantic Scholar (batch endpoint, by DOI or title match; cache the answers):
- publication types, fields of study, year, venue, reference count, open-access licence
- use these to cross-check Qwen's paper type and to drop papers without an open licence

Checks: a card with fewer than 3 level-2 headings, or broken tables, is dropped and counted (bad Markdown
conversion). Report per field: cards kept, paper-type mix, heading-set count.

## 3. Form cards replace templates and style lists

- Each synthetic paper gets one donor card, sampled without replacement, matched on field and paper type.
  With about 125 cards per field and 40 papers, no two papers share a layout.
- `template.md` for the paper is built from the donor: generic headings are kept as written; topical headings
  become slots that the paper plan fills with its own subject (one Qwen call, schema-bound).
- Section word budgets, table count and position, citation style, voice and sentence length come from the donor.
  `style.py` keeps only tone and opening, and takes the opening from the donor too.
- `genre.json` loses `templates` and most of `style`. Hand-written templates stay as the fallback when a field
  has no cards.

## 4. Paper kinds: three roles, mixed as in the corpus

A genre is the role a paper has towards its facts, the same in every field:

- **synthesis**: pools named earlier studies (built).
- **primary**: reports one named study of the world as its own. The world marks it (`"self": <subject id>`); the
  study's name must be in the title, the abstract and every table; the tagger maps "we" and "this study" to it.
  Its facts stay facts about a named study, so seed, check and QA do not change shape.
- **argument**: commentary, position, doctrinal (built; law's main kind).

The share of each per field comes from the paper types in section 2. A primary report and a synthesis that pools
it sit in the same cluster: that gives cross-paper questions for free.

## 5. Writing within Qwen's limits

- **Section by section.** One call per level-2 section: the section's brief, only the facts and numbers that
  belong in it, the style lines, and the last paragraph of the previous section. No call sees the whole paper or
  another paper. This keeps every prompt far below the 65k window, where Qwen follows rules best.
- **Tables by code.** Code renders every table from `number_rows` in the donor's shape and puts it in place.
  Qwen writes the sentence that introduces it. The numbers check then covers prose only, where repeats are few.
- **References by code.** Each named study gets one reference line (invented venue and year from the world,
  names from `names.py`); the donor's citation style decides `[n]` or author-year. Background claims get no
  invented reference: the writer makes none.
- **Selection counts by code** (records found, screened, excluded, included) for review-type papers, consistent
  with the number of studies pooled.
- **Outcome types per cluster.** `effect` becomes a list per field (medicine: hazard ratio, mean difference in a
  score, response-rate ratio); code picks one per cluster. Age range and follow-up come from ClinicalTrials.gov
  per condition (extend `topics.fetch_ranges`), not one range for all.
- **Repairs stay local**: a failed check rewrites that section only.
- Decision needed: PaperOrchestra writes a whole paper from one template. Either drive it section by section, or
  add a small own writer (`write_sections.py`) and keep PaperOrchestra for comparison. Measure both on 5 papers.

## 6. Worlds and measures

Worlds:
- cluster sizes uneven (for 40 papers and 5 clusters: 14 / 10 / 8 / 5 / 3), set in the config
- 3 to 4 bridge papers per field whose facts use subjects of two clusters
- llm topics: a hand-written list in `packs/topics/llm.json` (retrieval, prompting, fine-tuning, evaluation,
  code generation, long context, multilingual, compression, agents, safety), same file format
- every invented study, method, case and material name: Semantic Scholar title search; an exact match is an error

Gates for the trial run (5 papers per field), all reported in `report.json`:

| Measure | How | Bar |
|---|---|---|
| Layout variety | distinct level-2 heading sets / papers | 1.0 |
| Phrase sameness | share of papers containing each 3-word phrase, against the real-paper baseline | no phrase above 3x its real rate |
| Real or synthetic | Haiku sees one real and one synthetic paper of the same field and type, picks the synthetic | under 65% (chance 50%) |
| Planted-sentence leak | existing test, run with Haiku as well as Qwen | 10% or less, both |
| Numbers | existing check, prose only | 100% |
| CI follows size | `quant.errors` (patch 0001) | 0 errors |

The real-or-synthetic test is the number behind the thesis claim "reads like a real paper". Report it per field.

## 7. Subagents

Agree the form-card schema (section 2) first, in one short step, then run 1, 2 and 4 in parallel.

| # | Subagent | Owns | Depends on |
|---|---|---|---|
| 1 | Corpus profiler | new `profile.py`, `packs/forms/`, Semantic Scholar client, tests | schema |
| 2 | World | `world.py`, `topics.py`, `quant.py`, field packs: uneven clusters, bridges, outcome types, llm topics, `self` flag | none |
| 3 | Writer | `materials.py`, `style.py`, `write.py`, genre packs: donor templates, code tables, references, section calls | 1 (schema), 2 (`self`) |
| 4 | Measures | `evaluate.py` additions: variety, sameness, real-or-synthetic, name search | schema |
| 5 | Reviewer (read-only) | reads the trial papers and the real papers of the same type; top 5 gaps with quotes | 1 to 4 done |

Each subagent: new prompt versions beside the old ones (`docs/prompt-writing.md`), tests in `tests/`, no edits
outside its files. Subagent 5 changes nothing and calls no model on the cluster.

## 8. The academic-research-skills repository

Licence: CC BY-NC 4.0, and it is a Claude Code plugin with a person confirming each stage. So: do not copy its
files into this repository, and it cannot run on Qwen. Use it in two ways:

- **As the reviewer** (subagent 5): install the plugin and run its paper-reviewer panel on the trial papers. It
  replaces the ad-hoc reviewer prompt, and its `statistical_reporting_standards.md` is a ready checklist for the
  tables.
- **As a list of ideas to build in our own code**: the six dimensions of `shared/style_calibration_protocol.md`
  are the form-card measurements of section 2; the patterns in `academic-paper/references/writing_quality_check.md`
  (throat-clearing openers, groups of three, mirrored sentences, dash and semicolon rates) become code checks
  beside the banned list.

## 9. Order

1. Apply patch 0001 (CI width follows study size).
2. Form-card schema, then subagents 1, 2, 4.
3. Subagent 3.
4. Trial run, 5 papers per field; gates of section 6; subagent 5.
5. Full run of 40 per field (plan 55 to reach 40 that pass).

## 10. Risks

- Markdown from PDFs: broken tables give wrong form cards. Check 20 cards per field by hand before use.
- Law: few openly licensed papers. If under 60 cards, allow each donor twice with different tone and opening.
- Donor leakage: a card must hold no topical words. Test: no card field contains a word of the donor's title.
- Primary reports change the tagger. Build them last inside subagent 3 and keep them out of the first trial if
  they slip.
