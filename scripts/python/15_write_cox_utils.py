from __future__ import annotations

from pathlib import Path


def write_cox(path: Path, res: dict) -> None:
    lines = [
        f"MODELO={res['title']}",
        f"STATUS={res['status']}",
        f"ROWS={res['rows']}",
        f"EVENTS={res['events']}",
        f"COVARIAVEIS={','.join(res['features'])}",
    ]
    if res["c_index"] is not None:
        lines.append(f"C_INDEX={res['c_index']:.6f}")
    if res["reason"]:
        lines.append(f"RAZAO={res['reason']}")

    lines.extend(["", "[HAZARD_RATIOS]"])
    if len(res["summary"]) == 0:
        lines.append("sem_resultado")
    else:
        for _, r in res["summary"].iterrows():
            lines.append(
                f"{r['COVARIAVEL']}: HR={float(r['exp(coef)']):.6f}; p={float(r['p']):.6g}; "
                f"IC95=[{float(r['exp(coef) lower 95%']):.6f},{float(r['exp(coef) upper 95%']):.6f}]"
            )

    lines.extend(["", "[PH_TEST]"])
    if len(res["ph"]) == 0:
        lines.append("sem_resultado")
    else:
        for _, r in res["ph"].iterrows():
            lines.append(f"{r['COVARIAVEL']}: p={float(r['p']):.6g}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
