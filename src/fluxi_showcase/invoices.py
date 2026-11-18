"""Invoice control: match supplier invoice lines against purchase orders."""
from __future__ import annotations

import pandas as pd

PRICE_TOLERANCE = 0.02  # 2 percent
TOTAL_TOLERANCE = 0.01  # absolute currency units


def check_invoices(invoices: pd.DataFrame, purchase_orders: pd.DataFrame,
                   price_tolerance: float = PRICE_TOLERANCE) -> pd.DataFrame:
    """Return one row per finding: invoice, kind, detail."""
    findings = []
    po = purchase_orders.set_index(["po", "sku"])

    # duplicate: identical (po, sku, qty, price) lines billed under another invoice number
    key = ["po", "sku", "qty", "unit_price"]
    sig = invoices.groupby("invoice").apply(
        lambda g: tuple(sorted(map(tuple, g[key].values.tolist()))), include_groups=False)
    seen = {}
    for inv, s in sig.items():
        if s in seen:
            findings.append((inv, "duplicate_invoice", f"same lines as {seen[s]}"))
        else:
            seen[s] = inv

    for inv, g in invoices.groupby("invoice"):
        for _, l in g.iterrows():
            k = (l["po"], l["sku"])
            if k not in po.index:
                findings.append((inv, "unknown_sku", f"{l['sku']} not on {l['po']}"))
                continue
            ordered = po.loc[k]
            if l["qty"] > ordered["qty"]:
                findings.append((inv, "over_billed_qty", f"{l['sku']}: {l['qty']} billed vs {ordered['qty']} ordered"))
            drift = l["unit_price"] / ordered["unit_cost"] - 1
            if abs(drift) > price_tolerance:
                findings.append((inv, "price_drift", f"{l['sku']}: {drift:+.1%} vs agreed price"))
        computed = round(g["line_total"].sum(), 2)
        declared = g["invoice_total"].iloc[0]
        if abs(computed - declared) > TOTAL_TOLERANCE:
            findings.append((inv, "total_mismatch", f"lines sum to {computed}, header says {declared}"))
    return pd.DataFrame(findings, columns=["invoice", "kind", "detail"]).drop_duplicates(["invoice", "kind"])


def evaluate(findings: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Precision and recall against the injected anomalies (synthetic ground truth)."""
    f = set(map(tuple, findings[["invoice", "kind"]].values.tolist()))
    t = set(map(tuple, truth[["invoice", "kind"]].values.tolist()))
    tp = len(f & t)
    return {"injected": len(t), "flagged": len(f), "true_positive": tp,
            "precision": tp / len(f) if f else 1.0, "recall": tp / len(t) if t else 1.0}
