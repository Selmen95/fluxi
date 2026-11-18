"""Ingestion of three unstable Excel layouts into one canonical table.

Canonical schema: date (datetime64), store (str), sku (str), units (int).
Each parser sniffs its layout instead of trusting fixed coordinates, and the
pipeline returns a quality report next to the data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

CANONICAL = ["date", "store", "sku", "units"]


@dataclass
class QualityReport:
    source: str
    layout: str
    rows_read: int = 0
    rows_kept: int = 0
    dropped_blank: int = 0
    dropped_duplicate: int = 0
    dropped_footer: int = 0
    unparsable_units: int = 0
    notes: list[str] = field(default_factory=list)


class UnknownLayout(ValueError):
    pass


def _rows(path: Path, sheet=None):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.active
    return [list(r) for r in ws.iter_rows(values_only=True)], wb.sheetnames


def _find_header(rows, required: set[str]) -> int:
    for i, r in enumerate(rows[:20]):
        cells = {str(c).strip().lower() for c in r if c is not None}
        if required <= cells:
            return i
    raise UnknownLayout("header row not found")


def _to_int(x):
    """Parse '12', '12 pcs', '12,0', -3 ... Returns None when unparsable."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(round(x))
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(x))
    return int(round(float(m.group().replace(",", ".")))) if m else None


def detect_layout(path: Path) -> str:
    rows, sheets = _rows(path)
    if len(sheets) > 1:
        return "per_store_sheets"
    for r in rows[:20]:
        low = [str(c).strip().lower() for c in r if c is not None]
        if "code article" in low and "magasin" in low:
            return "long_localized"
        if "store" in low and "sku" in low and "total" in low:
            return "wide_dates"
    raise UnknownLayout(f"cannot classify {path.name}")


def parse_long_localized(path: Path):
    rows, _ = _rows(path)
    h = _find_header(rows, {"date", "magasin", "code article"})
    head = [str(c).strip().lower() for c in rows[h]]
    ix = {k: head.index(k) for k in ("date", "magasin", "code article", "qte vendue")}
    rep = QualityReport(path.name, "long_localized")
    out = []
    for r in rows[h + 1:]:
        rep.rows_read += 1
        if r[ix["date"]] is None:
            rep.dropped_blank += 1
            continue
        d = pd.to_datetime(r[ix["date"]], format="%d/%m/%Y", errors="coerce")
        u = _to_int(r[ix["qte vendue"]])
        if pd.isna(d) or u is None:
            rep.unparsable_units += 1
            continue
        out.append((d, r[ix["magasin"]], r[ix["code article"]], u))
    return pd.DataFrame(out, columns=CANONICAL), rep


def parse_wide_dates(path: Path):
    rows, _ = _rows(path)
    h = _find_header(rows, {"store", "sku", "total"})
    head = rows[h]
    date_cols = [i for i, c in enumerate(head) if hasattr(c, "year")]
    rep = QualityReport(path.name, "wide_dates")
    out = []
    for r in rows[h + 1:]:
        rep.rows_read += 1
        if r[0] is None:
            rep.dropped_blank += 1
            continue
        if str(r[0]).strip().upper() == "TOTAL":
            rep.dropped_footer += 1
            continue
        for i in date_cols:
            u = _to_int(r[i])
            if u is None:
                rep.unparsable_units += 1
                continue
            out.append((pd.Timestamp(head[i]), r[0], r[1], u))
    return pd.DataFrame(out, columns=CANONICAL), rep


def parse_per_store_sheets(path: Path):
    wb = load_workbook(path, read_only=True, data_only=True)
    rep = QualityReport(path.name, "per_store_sheets")
    out = []
    for name in wb.sheetnames:
        rows = [list(r) for r in wb[name].iter_rows(values_only=True)]
        h = _find_header(rows, {"sku", "day", "units"})
        for r in rows[h + 1:]:
            rep.rows_read += 1
            if r[0] is None:
                rep.dropped_blank += 1
                continue
            u = _to_int(r[2])
            if u is None:
                rep.unparsable_units += 1
                continue
            out.append((pd.Timestamp(r[1]), name, r[0], u))
    return pd.DataFrame(out, columns=CANONICAL), rep


PARSERS = {
    "long_localized": parse_long_localized,
    "wide_dates": parse_wide_dates,
    "per_store_sheets": parse_per_store_sheets,
}


def ingest(paths: list[Path]):
    """Parse every file, de-duplicate on (date, store, sku), return data and reports."""
    frames, reports = [], []
    for p in paths:
        df, rep = PARSERS[detect_layout(Path(p))](Path(p))
        before = len(df)
        df = df.drop_duplicates(subset=["date", "store", "sku"], keep="first")
        rep.dropped_duplicate = before - len(df)
        rep.rows_kept = len(df)
        frames.append(df)
        reports.append(rep)
    data = pd.concat(frames, ignore_index=True).sort_values(["store", "sku", "date"]).reset_index(drop=True)
    return data, reports
