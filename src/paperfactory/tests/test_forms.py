"""Form cards: a donor's layout without its words; donor layouts as templates (no model calls, no donor files)."""
from __future__ import annotations

import json
import random

from paperfactory import forms, materials, packs, plan, style
from paperfactory.tests.test_plan import _world

SENT = "We measured the response in every group and compared it with the control. The difference was small. "
TOPIC_WORDS = ("zanzibar", "pemphigoid", "rituximab", "tocilizumab", "appendix")


def _donor(i: int, rng: random.Random) -> str:
    mids = rng.sample(["Search strategy", "Eligibility criteria", "Rituximab in Zanzibar", "Risk of bias",
                       "Statistical analysis", "Pemphigoid outcomes", "Data extraction", "Tocilizumab dosing"],
                      rng.randint(2, 5))
    out = [f"# Donor paper {i} about pemphigoid", "", "## Abstract", SENT * 3, "## 1. Introduction", SENT * 12]
    out += ["## 2. Methods", SENT * 2]
    for m in mids[:2]:
        out += [f"### 2.{mids.index(m) + 1} {m}", SENT * 6]
    for m in mids[2:]:
        out += [f"## {m}", SENT * 8, "| a | b |", "|---|---|", "| 1 | 2 |", ""]
    out += ["## Discussion", SENT * 10, "### Limitations", SENT * 3, "## Conclusions", SENT * 3,
            "## References", "1. Someone. Rituximab in Zanzibar.", "## Appendix A", SENT]
    return "\n\n".join(out)


def _folder(tmp_path, n: int = 30):
    rng = random.Random(4)
    d = tmp_path / "donors" / "medicine"
    d.mkdir(parents=True)
    man = []
    for i in range(n):
        (d / f"m{i:03d}.md").write_text(_donor(i, rng), encoding="utf-8")
        man.append({"id": f"m{i:03d}", "title": f"Donor paper {i} about pemphigoid",
                    "paper_type": "narrative_review" if i % 3 else "randomised_trial",
                    "licence": "CC-BY"})
    (d / "bad.md").write_text("no headings here. " * 50, encoding="utf-8")
    (tmp_path / "donors" / "manifest.jsonl").write_text("\n".join(map(json.dumps, man)), encoding="utf-8")
    return d


def test_generic_headings() -> None:
    assert forms.is_generic("Materials and Methods") and forms.is_generic("Risk of bias")
    assert not forms.is_generic("Rituximab in Zanzibar") and not forms.is_generic("")
    assert forms._clean("2.1 **Search strategy**.") == "Search strategy"
    assert forms._clean("IV. Discussion") == "Discussion"


def test_heading_and_table_syntax_from_converters() -> None:
    headings = forms._headings(["  ## Introduction", "<h2>Methods</h2>", "Results", "---"])
    assert headings == [(0, 2, "Introduction"), (1, 2, "Methods"), (3, 2, "Results")]
    assert forms._table_count(["name | result", "--- | :---:", "A | 2"], 0, 3) == 1


def test_topical_and_two_section_layouts_are_usable() -> None:
    card = {"outline": [
        {"level": 2, "text": "Abstract"},
        {"level": 2, "text": "Introduction"},
        {"level": 2, "text": None},
    ]}
    assert forms.usable(card) is None


def test_a_card_keeps_the_layout_and_none_of_the_donors_words(tmp_path) -> None:
    prof = forms.profile(_folder(tmp_path), tmp_path / "donors" / "manifest.jsonl")
    assert len(prof["cards"]) == 30 and "bad" in prof["dropped"]
    assert prof["paper_types"] == {"narrative_review": 20, "randomised_trial": 10}
    assert prof["layouts"] >= 10
    text = json.dumps(prof["cards"]).lower()
    assert not any(w in text for w in TOPIC_WORDS)                 # no topical word, no reference, no appendix
    c = prof["cards"][0]
    heads = [x["text"] for x in c["outline"]]
    assert heads[0] == "Abstract" and "Introduction" in heads and "References" not in heads
    assert abs(sum(x.get("share", 0) for x in c["outline"]) - 1) < 0.02
    assert c["we_share"] > 0.3 and 5 < c["sentence_words"] < 15


def test_donor_layouts_become_the_papers_templates(tmp_path, monkeypatch) -> None:
    prof = forms.profile(_folder(tmp_path), tmp_path / "donors" / "manifest.jsonl")
    monkeypatch.setattr(forms, "FORMS", tmp_path / "forms")
    forms.write("medicine", prof)
    f, g = packs.field("medicine"), packs.genre("synthesis")
    donors = forms.load(f, g)
    assert len(donors) == 20                                        # the genre's donor_types: reviews only
    w = _world(f, g, 20)
    cards = style.cards(list(w["papers"]), f, g, 1, donors)
    assert len({c["donor"] for c in cards.values()}) == 20          # every donor once before any twice
    layouts = set()
    for pid, c in cards.items():
        c["template_text"] = forms.template_text({"outline": c.pop("outline")}, w, f, pid, 1)
        t = c["template_text"]
        assert t.startswith("# {TITLE}\n\n## Abstract") and t.count("## Abstract") == 1
        heads = [x for x in t.splitlines() if x.startswith("#")]
        assert len(heads) == len(set(heads))
        assert not any(wd in t.lower() for wd in TOPIC_WORDS)
        layouts.add(tuple(heads[1:]))
        assert 4 <= c["pages"] <= 8 and "words per sentence" in c["rhythm"]
        # the content plan and the writer's notes work on a donor layout as on a template file
        cp = plan.make(w, f, g, pid, packs.card_template(g, c), 1)
        assert cp["sections"] == plan.leaves(t)
        _, log = materials.materials_v3(w, f, g, pid, c, cp)
        assert all(x["section"] in log for x in cp["placement"])
        d = packs.render_paper_template(f, g, c, tmp_path / "t" / pid)
        assert (d / "template.md").read_text(encoding="utf-8") == t
        assert "words per sentence" in (d / "guidelines.md").read_text(encoding="utf-8")
    assert len(layouts) >= 15                                       # the hand-written templates: 3 per field


def test_without_form_cards_the_genre_templates_are_used(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(forms, "FORMS", tmp_path / "none")
    f, g = packs.field("law"), packs.genre("argument")
    assert forms.load(f, g) == []
    cards = style.cards(["A", "B"], f, g, 1, [])
    assert "donor" not in cards["A"] and packs.card_template(g, cards["A"]).startswith("# {TITLE}")
