import pandas as pd
import pytest

from fluxi_showcase import agents, ingest, invoices, replenishment, synth
from fluxi_showcase.cli import run_demo


@pytest.fixture(scope="module")
def world():
    return synth.build_world()


@pytest.fixture(scope="module")
def ingested(world, tmp_path_factory):
    paths = synth.write_excel_formats(world, tmp_path_factory.mktemp("x"))
    return paths, *ingest.ingest(paths)


def test_generator_is_deterministic():
    a, b = synth.build_world(), synth.build_world()
    pd.testing.assert_frame_equal(a.sales, b.sales)
    pd.testing.assert_frame_equal(a.invoices, b.invoices)


def test_layouts_detected(ingested):
    paths, _, reports = ingested
    assert [r.layout for r in reports] == ["long_localized", "wide_dates", "per_store_sheets"]


def test_ingestion_is_lossless(world, ingested):
    _, data, reports = ingested
    truth = world.sales[["date", "store", "sku", "units"]].sort_values(["store", "sku", "date"]).reset_index(drop=True)
    assert data.equals(truth)
    assert sum(r.dropped_duplicate for r in reports) > 0  # layout C duplicates were removed
    assert reports[1].dropped_footer == 1


def test_unknown_layout(tmp_path):
    from openpyxl import Workbook
    wb = Workbook(); wb.active.append(["nothing", "useful"]); wb.save(tmp_path / "bad.xlsx")
    with pytest.raises(ingest.UnknownLayout):
        ingest.detect_layout(tmp_path / "bad.xlsx")


def test_to_int_variants():
    assert ingest._to_int("12 pcs") == 12
    assert ingest._to_int("3,0") == 3
    assert ingest._to_int(None) is None
    assert ingest._to_int("n/a") is None


def test_replenishment_rules(world):
    p = replenishment.propose_orders(world.sales, world.stock, world.skus, world.suppliers)
    packs = world.skus.set_index("sku")["pack_size"]
    assert (p["order_qty"] > 0).all()
    assert all(q % packs[s] == 0 for q, s in zip(p["order_qty"], p["sku"]))
    # higher service level never lowers the order
    hi = replenishment.propose_orders(world.sales, world.stock, world.skus, world.suppliers, service_level=0.98)
    assert hi["order_qty"].sum() >= p["order_qty"].sum()


def _tiny(on_hand, min_value):
    dates = pd.date_range("2025-01-01", periods=28)
    sales = pd.DataFrame({"date": dates, "store": "S", "sku": "A", "units": [2, 4] * 14})
    stock = pd.DataFrame({"store": ["S"], "sku": ["A"], "on_hand": [on_hand], "on_order": [0]})
    skus = pd.DataFrame({"sku": ["A"], "supplier": ["X"], "unit_cost": [1.0], "pack_size": [6], "lead_time_days": [2]})
    sup = pd.DataFrame({"supplier": ["X"], "min_order_value": [min_value]})
    return replenishment.propose_orders(sales, stock, skus, sup)


def test_supplier_minimum_hold_and_urgent_release():
    held = _tiny(on_hand=20, min_value=1000.0)      # enough cover for lead time, below minimum
    assert not held["released"].iloc[0] and held["reason"].iloc[0] == "held_below_minimum"
    urgent = _tiny(on_hand=1, min_value=1000.0)     # stock-out before delivery
    assert urgent["released"].iloc[0] and urgent["reason"].iloc[0] == "released_urgent"
    ok = _tiny(on_hand=20, min_value=1.0)
    assert ok["reason"].iloc[0] == "min_value_reached"


def test_invoice_control_matches_ground_truth(world):
    f = invoices.check_invoices(world.invoices, world.purchase_orders)
    m = invoices.evaluate(f, world.injected_anomalies)
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_clean_invoices_not_flagged(world):
    clean = world.invoices[~world.invoices["invoice"].isin(world.injected_anomalies["invoice"])]
    clean = clean[~clean["invoice"].str.endswith("_B")]
    assert invoices.check_invoices(clean, world.purchase_orders).empty


def test_agent_gates_side_effects(world):
    o = agents.Orchestrator(agents.build_tools(world, world.sales), agents.ScriptedPlanner())
    o.run("reorder this week")
    assert [s.status for s in o.trace] == ["ok", "pending_approval"]
    o2 = agents.Orchestrator(agents.build_tools(world, world.sales), agents.ScriptedPlanner(), approver=lambda c: True)
    o2.run("reorder this week")
    assert [s.status for s in o2.trace] == ["ok", "ok"]
    o3 = agents.Orchestrator(agents.build_tools(world, world.sales), agents.ScriptedPlanner(), approver=lambda c: False)
    o3.run("reorder this week")
    assert o3.trace[-1].status == "rejected"


def test_agent_validation_and_budget(world):
    class Bad:
        def next(self, goal, history):
            return agents.ToolCall("propose_replenishment", {"service_level": "high"})
    o = agents.Orchestrator(agents.build_tools(world, world.sales), Bad(), max_steps=3)
    assert "budget" in o.run("x")
    assert all(s.status == "error" for s in o.trace)

    class Unknown:
        def next(self, goal, history):
            return agents.ToolCall("drop_database", {})
    o = agents.Orchestrator(agents.build_tools(world, world.sales), Unknown(), max_steps=1)
    o.run("x")
    assert o.trace[0].summary == "unknown tool"


def test_demo_end_to_end(tmp_path):
    r = run_demo(tmp_path)
    assert r["lossless"]
    assert r["invoice_metrics"]["recall"] == 1.0
