"""Replenishment rules.

target stock   = mean_daily_demand * (lead_time + review_period) + safety_stock
safety_stock   = z * std_daily_demand * sqrt(lead_time + review_period)
order quantity = max(0, target - on_hand - on_order), rounded up to a pack size.
Supplier-level minimum order value: lines are held back unless at least one is urgent
(projected stock-out before the next delivery).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

Z_BY_SERVICE = {0.90: 1.28, 0.95: 1.65, 0.98: 2.05}


def demand_stats(sales: pd.DataFrame, window_days: int = 28) -> pd.DataFrame:
    end = sales["date"].max()
    w = sales[sales["date"] > end - pd.Timedelta(days=window_days)]
    g = w.groupby(["store", "sku"])["units"]
    return pd.DataFrame({"mean_daily": g.mean(), "std_daily": g.std(ddof=0)}).reset_index()


def propose_orders(sales, stock, skus, suppliers, review_days: int = 7,
                   service_level: float = 0.95, window_days: int = 28) -> pd.DataFrame:
    z = Z_BY_SERVICE[service_level]
    d = demand_stats(sales, window_days)
    df = (stock.merge(d, on=["store", "sku"])
               .merge(skus[["sku", "supplier", "unit_cost", "pack_size", "lead_time_days"]], on="sku"))
    horizon = df["lead_time_days"] + review_days
    df["safety_stock"] = z * df["std_daily"] * np.sqrt(horizon)
    df["target"] = df["mean_daily"] * horizon + df["safety_stock"]
    df["raw_need"] = (df["target"] - df["on_hand"] - df["on_order"]).clip(lower=0)
    df["order_qty"] = [math.ceil(n / p) * p if n > 0 else 0 for n, p in zip(df["raw_need"], df["pack_size"])]
    df["urgent"] = (df["on_hand"] + df["on_order"]) < df["mean_daily"] * df["lead_time_days"]
    df["line_value"] = df["order_qty"] * df["unit_cost"]
    df = df[df["order_qty"] > 0].copy()

    sup = df.groupby("supplier").agg(value=("line_value", "sum"), any_urgent=("urgent", "any")).reset_index()
    sup = sup.merge(suppliers, on="supplier")
    sup["released"] = (sup["value"] >= sup["min_order_value"]) | sup["any_urgent"]
    sup["reason"] = np.where(sup["value"] >= sup["min_order_value"], "min_value_reached",
                             np.where(sup["any_urgent"], "released_urgent", "held_below_minimum"))
    df = df.merge(sup[["supplier", "released", "reason"]], on="supplier")
    cols = ["supplier", "store", "sku", "on_hand", "on_order", "mean_daily", "target",
            "order_qty", "urgent", "line_value", "released", "reason"]
    return df[cols].sort_values(["supplier", "store", "sku"]).reset_index(drop=True)
