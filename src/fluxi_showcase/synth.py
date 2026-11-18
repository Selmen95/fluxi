"""Deterministic synthetic data generator (fixed seed).

Produces a small retail world: stores, SKUs, suppliers, stock, sales history
and purchase orders / invoices with injected anomalies. Sales are also written
as three deliberately unstable Excel layouts to exercise the ingestion code.
Nothing here is derived from any real dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

SEED = 42
N_SKUS = 40
N_DAYS = 90
START = pd.Timestamp("2025-01-06")  # a Monday


@dataclass
class World:
    stores: list[str]
    skus: pd.DataFrame        # sku, supplier, unit_cost, pack_size, lead_time_days
    suppliers: pd.DataFrame   # supplier, min_order_value
    sales: pd.DataFrame       # date, store, sku, units, revenue
    stock: pd.DataFrame       # store, sku, on_hand, on_order
    purchase_orders: pd.DataFrame
    invoices: pd.DataFrame
    injected_anomalies: pd.DataFrame  # ground truth for invoice control


def build_world(seed: int = SEED) -> World:
    rng = np.random.default_rng(seed)
    stores = [f"STORE_{i:02d}" for i in range(1, 7)]

    skus = pd.DataFrame({
        "sku": [f"SKU{i:03d}" for i in range(1, N_SKUS + 1)],
        "supplier": [f"SUP_{rng.integers(1, 6):02d}" for _ in range(N_SKUS)],
        "unit_cost": np.round(rng.uniform(0.5, 12.0, N_SKUS), 2),
        "pack_size": rng.choice([1, 6, 12, 24], N_SKUS),
        "lead_time_days": rng.choice([2, 3, 5, 7], N_SKUS),
    })
    skus["unit_price"] = np.round(skus["unit_cost"] * rng.uniform(1.25, 1.6, N_SKUS), 2)
    suppliers = pd.DataFrame({
        "supplier": sorted(skus["supplier"].unique()),
    })
    suppliers["min_order_value"] = rng.choice([100.0, 150.0, 250.0], len(suppliers))

    dates = pd.date_range(START, periods=N_DAYS, freq="D")
    weekday_factor = np.array([0.85, 0.85, 0.9, 0.95, 1.1, 1.4, 1.2])
    rows = []
    for store in stores:
        store_scale = rng.uniform(0.6, 1.5)
        for sku in skus["sku"]:
            base = rng.gamma(2.0, 2.0) * store_scale
            for d in dates:
                lam = base * weekday_factor[d.weekday()]
                rows.append((d, store, sku, int(rng.poisson(lam))))
    sales = pd.DataFrame(rows, columns=["date", "store", "sku", "units"])
    sales = sales.merge(skus[["sku", "unit_price"]], on="sku")
    sales["revenue"] = (sales["units"] * sales["unit_price"]).round(2)
    sales = sales.drop(columns="unit_price")

    recent = sales[sales["date"] > dates[-1] - pd.Timedelta(days=14)]
    avg = recent.groupby(["store", "sku"])["units"].mean().rename("avg").reset_index()
    stock = avg.copy()
    stock["on_hand"] = (stock["avg"] * rng.uniform(0.0, 9.0, len(stock))).round().astype(int)
    stock["on_order"] = (stock["avg"] * rng.choice([0, 0, 3, 6], len(stock))).round().astype(int)
    stock = stock.drop(columns="avg")

    purchase_orders, invoices, truth = _build_orders_and_invoices(rng, skus)
    return World(stores, skus, suppliers, sales, stock, purchase_orders, invoices, truth)


def _build_orders_and_invoices(rng, skus):
    po_rows, inv_rows, truth = [], [], []
    for n in range(1, 61):
        po = f"PO{n:04d}"
        supplier = f"SUP_{rng.integers(1, 6):02d}"
        pool = skus[skus["supplier"] == supplier]
        if pool.empty:
            pool = skus
        picks = pool.sample(min(3, len(pool)), random_state=int(rng.integers(1_000_000)))
        for _, s in picks.iterrows():
            qty = int(s["pack_size"] * rng.integers(1, 6))
            po_rows.append((po, supplier, s["sku"], qty, s["unit_cost"]))
    purchase_orders = pd.DataFrame(po_rows, columns=["po", "supplier", "sku", "qty", "unit_cost"])

    kinds = ["price_drift", "over_billed_qty", "unknown_sku", "total_mismatch", "duplicate_invoice"]
    for po, g in purchase_orders.groupby("po"):
        inv_no = f"INV{po[2:]}"
        rows = []
        for _, l in g.iterrows():
            rows.append({"invoice": inv_no, "po": po, "supplier": l["supplier"], "sku": l["sku"],
                         "qty": int(l["qty"]), "unit_price": float(l["unit_cost"])})
        for r in rows:
            r["line_total"] = round(r["qty"] * r["unit_price"], 2)
            r["invoice_total"] = None
        total = round(sum(r["line_total"] for r in rows), 2)
        for r in rows:
            r["invoice_total"] = total
        inv_rows.extend(rows)
    invoices = pd.DataFrame(inv_rows)

    # inject anomalies on a known subset
    chosen = rng.choice(invoices["invoice"].unique(), size=15, replace=False)
    dup_rows = []
    for i, inv in enumerate(chosen):
        kind = kinds[i % len(kinds)]
        idx = invoices.index[invoices["invoice"] == inv]
        first = idx[0]
        if kind == "price_drift":
            invoices.loc[first, "unit_price"] = round(invoices.loc[first, "unit_price"] * 1.12, 2)
            invoices.loc[first, "line_total"] = round(invoices.loc[first, "qty"] * invoices.loc[first, "unit_price"], 2)
            invoices.loc[idx, "invoice_total"] = round(invoices.loc[idx, "line_total"].sum(), 2)
        elif kind == "over_billed_qty":
            invoices.loc[first, "qty"] = int(invoices.loc[first, "qty"] * 2)
            invoices.loc[first, "line_total"] = round(invoices.loc[first, "qty"] * invoices.loc[first, "unit_price"], 2)
            invoices.loc[idx, "invoice_total"] = round(invoices.loc[idx, "line_total"].sum(), 2)
        elif kind == "unknown_sku":
            invoices.loc[first, "sku"] = "SKU999"
        elif kind == "total_mismatch":
            invoices.loc[idx, "invoice_total"] = round(invoices.loc[first, "invoice_total"] + 37.5, 2)
        elif kind == "duplicate_invoice":
            dup = invoices.loc[idx].copy()
            dup["invoice"] = dup["invoice"] + "_B"
            dup_rows.append(dup)
            kind_inv = inv + "_B"
            truth.append((kind_inv, kind))
            continue
        truth.append((inv, kind))
    if dup_rows:
        invoices = pd.concat([invoices] + dup_rows, ignore_index=True)
    return purchase_orders, invoices, pd.DataFrame(truth, columns=["invoice", "kind"])


# --- messy Excel writers ---------------------------------------------------

def write_excel_formats(world: World, out_dir: Path) -> list[Path]:
    """Write the sales history as three unstable layouts, split by store."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s = world.sales
    stores = world.stores
    a_stores, b_stores, c_stores = stores[:2], stores[2:4], stores[4:]
    paths = [
        _write_a(s[s["store"].isin(a_stores)], out_dir / "export_a.xlsx"),
        _write_b(s[s["store"].isin(b_stores)], out_dir / "export_b.xlsx"),
        _write_c(s[s["store"].isin(c_stores)], out_dir / "export_c.xlsx", c_stores),
    ]
    return paths


def _write_a(df: pd.DataFrame, path: Path) -> Path:
    """Long format, title block, localized headers, decimal commas as text."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Export"
    ws.append(["Sales extract"])
    ws.append(["Generated by synthetic exporter"])
    ws.append([])
    ws.append(["Date", "Magasin", "Code article", "Qte vendue", "CA"])
    for r in df.itertuples():
        ws.append([r.date.strftime("%d/%m/%Y"), r.store, r.sku, r.units,
                   f"{r.revenue:.2f}".replace(".", ",")])
    wb.save(path)
    return path


def _write_b(df: pd.DataFrame, path: Path) -> Path:
    """Wide format: one column per day, trailing total column and footer row."""
    wide = df.pivot_table(index=["store", "sku"], columns="date", values="units", aggfunc="sum").reset_index()
    wb = Workbook()
    ws = wb.active
    ws.append(["store", "sku"] + [d.to_pydatetime() for d in wide.columns[2:]] + ["Total"])
    for _, row in wide.iterrows():
        vals = list(row.values[2:])
        ws.append([row["store"], row["sku"]] + vals + [sum(vals)])
    ws.append(["TOTAL", None] + [None] * (len(wide.columns) - 2) + [int(df["units"].sum())])
    wb.save(path)
    return path


def _write_c(df: pd.DataFrame, path: Path, stores: list[str]) -> Path:
    """One sheet per store, units as text with a suffix, returns, blank and duplicate rows."""
    wb = Workbook()
    wb.remove(wb.active)
    for store in stores:
        ws = wb.create_sheet(store)
        ws.append([f"Daily sales - {store}"])
        ws.append(["SKU", "Day", "Units"])
        part = df[df["store"] == store]
        for i, r in enumerate(part.itertuples()):
            ws.append([r.sku, r.date.to_pydatetime(), f"{r.units} pcs"])
            if i % 500 == 0:
                ws.append([None, None, None])
            if i % 700 == 0:
                ws.append([r.sku, r.date.to_pydatetime(), f"{r.units} pcs"])  # duplicate
    wb.save(path)
    return path
