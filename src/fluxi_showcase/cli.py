"""Command line demo: generate synthetic data, ingest it, run every module."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from . import agents, ingest, invoices, replenishment, synth


def run_demo(out_dir: Path) -> dict:
    world = synth.build_world()
    paths = synth.write_excel_formats(world, out_dir)
    data, reports = ingest.ingest(paths)
    # Ingestion must reproduce the generator's ground truth exactly.
    truth = world.sales[["date", "store", "sku", "units"]].sort_values(["store", "sku", "date"]).reset_index(drop=True)
    lossless = bool(data.equals(truth))

    proposal = replenishment.propose_orders(data, world.stock, world.skus, world.suppliers)
    findings = invoices.check_invoices(world.invoices, world.purchase_orders)
    metrics = invoices.evaluate(findings, world.injected_anomalies)

    orch = agents.Orchestrator(agents.build_tools(world, data), agents.ScriptedPlanner())
    answer = orch.run("Reorder proposal for this week")
    return {"reports": reports, "lossless": lossless, "proposal": proposal,
            "invoice_metrics": metrics, "agent_answer": answer, "trace": orch.trace}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="where to write the synthetic Excel files")
    a = ap.parse_args()
    out = a.out or Path(tempfile.mkdtemp(prefix="fluxi_demo_"))
    r = run_demo(out)
    print(f"Excel files written to {out}")
    for rep in r["reports"]:
        print(f"[ingest] {rep.source:14s} layout={rep.layout:16s} read={rep.rows_read:6d} kept={rep.rows_kept:6d} "
              f"blank={rep.dropped_blank} footer={rep.dropped_footer} duplicates={rep.dropped_duplicate}")
    print(f"[ingest] matches generator ground truth exactly: {r['lossless']}")
    p = r["proposal"]
    print(f"[replenishment] {len(p)} proposed lines, {int(p['released'].sum())} released, "
          f"{int((~p['released']).sum())} held below supplier minimum")
    m = r["invoice_metrics"]
    print(f"[invoices] injected={m['injected']} flagged={m['flagged']} "
          f"precision={m['precision']:.2f} recall={m['recall']:.2f} (synthetic ground truth)")
    print(f"[agent] {r['agent_answer']}")
    for s in r["trace"]:
        print(f"        step {s.step}: {s.tool} -> {s.status} ({s.summary})")


if __name__ == "__main__":
    main()
