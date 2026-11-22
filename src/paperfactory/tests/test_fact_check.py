"""PaperOrchestra fact_check: where a planted number sits (no model calls)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "external" / "paperorchestra"))
from utils import fact_check  # noqa: E402

TABLE = "| Trial | Countries (n) | Patients randomised (n) |\n|---|---|---|\n| TR-1 | 13 | 412 |\n"


def test_table_unit_ignores_case():
    """Seen in v3 baseline MED-01: header 'Countries (n)', fact unit 'countries'; was rejected."""
    assert fact_check.count_places(TABLE, "13", "countries")["table"] == 1


def test_table_unit_still_required():
    assert fact_check.count_places(TABLE, "13", "kg")["table"] == 0


def test_text_unit_unchanged():
    md = "Patients came from 13 countries.\n"
    assert fact_check.count_places(md, "13", "countries") == {"table": 0, "text": 1}
