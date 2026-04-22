from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "project_config.json").exists():
            return candidate
    raise FileNotFoundError("project_config.json nao encontrado")


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_num(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()
    s = s.str.replace("R$", "", regex=False).str.replace("%", "", regex=False).str.replace(" ", "", regex=False)
    both_mask = s.str.contains(",", regex=False) & s.str.contains(".", regex=False)
    s.loc[both_mask] = s.loc[both_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    comma_only = s.str.contains(",", regex=False) & (~s.str.contains(".", regex=False))
    s.loc[comma_only] = s.loc[comma_only].str.replace(",", ".", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def parse_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str).str.strip(), errors="coerce")


def parse_kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line and not line.startswith("["):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def build_episode_dataframe(daily_df: pd.DataFrame, equip: str) -> pd.DataFrame:
    df = daily_df.copy()
    df["EQUIPAMENTO"] = df["EQUIPAMENTO"].astype(str).str.strip().str.upper()
    df = df[df["EQUIPAMENTO"] == equip].copy()
    if df.empty:
        raise ValueError(f"Sem dados para equipamento {equip}")

    df["DATA_DT"] = parse_date(df["DATA_DIA"])
    for col in [
        "T_DIAS",
        "EVENT_OBSERVED",
        "CENSORED",
        "CARGA_TON_ACUM_EP_LAG1",
        "CARGA_TON_30D_MEAN_LAG1",
        "FLAG_RECUP_7D_LAG1",
        "DIAS_DESDE_ULT_RECUP",
        "DOS_VAZAO_MEDIA_LAG1",
        "DOS_VELOC_MEDIA_LAG1",
        "DOS_COBERTURA_24H_PCT_LAG1",
    ]:
        if col in df.columns:
            df[col] = to_num(df[col])

    rows: list[dict] = []
    for eid, g in df.groupby("EPISODIO_ID", sort=True):
        g = g.sort_values("DATA_DT")
        t_dias = to_num(g["T_DIAS"])
        duration = float(t_dias.max()) if t_dias.notna().any() else np.nan
        rows.append(
            {
                "EPISODIO_ID": str(eid),
                "DATA_INICIO": g["DATA_DT"].iloc[0],
                "DATA_FIM": g["DATA_DT"].iloc[-1],
                "ANO_INICIO": int(g["DATA_DT"].iloc[0].year) if pd.notna(g["DATA_DT"].iloc[0]) else np.nan,
                "DURATION_DIAS": duration,
                "EVENT_OBSERVED": int(to_num(g["EVENT_OBSERVED"]).max()),
                "CENSORED": int(to_num(g["CENSORED"]).max()),
                "CARGA_ACUM_EP": float(g["CARGA_TON_ACUM_EP_LAG1"].dropna().iloc[-1])
                if g["CARGA_TON_ACUM_EP_LAG1"].dropna().size
                else np.nan,
                "MEDIA_CARGA_30D": float(g["CARGA_TON_30D_MEAN_LAG1"].mean()),
                "FLAG_EMENDA_7D": float(g["FLAG_RECUP_7D_LAG1"].max()),
                "DIAS_DESDE_ULT_EMENDA": float(g["DIAS_DESDE_ULT_RECUP"].dropna().iloc[-1])
                if g["DIAS_DESDE_ULT_RECUP"].dropna().size
                else np.nan,
                "DOS_VAZAO_MEDIA_EP": float(g["DOS_VAZAO_MEDIA_LAG1"].mean()),
                "DOS_VELOC_MEDIA_EP": float(g["DOS_VELOC_MEDIA_LAG1"].mean()),
                "DOS_COBERTURA_MEDIA_EP": float(g["DOS_COBERTURA_24H_PCT_LAG1"].mean()),
            }
        )

    ep = pd.DataFrame(rows).sort_values("DATA_INICIO").reset_index(drop=True)
    ep = ep.dropna(subset=["DURATION_DIAS", "EVENT_OBSERVED"]).copy()
    ep["EVENT_OBSERVED"] = ep["EVENT_OBSERVED"].astype(int)
    return ep


def build_temporal_folds(n_rows: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if n_rows < 15:
        return []
    test_size = max(5, int(round(0.2 * n_rows)))
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for frac in [0.60, 0.70, 0.80]:
        n_train = int(np.floor(frac * n_rows))
        start = n_train
        end = min(n_rows, start + test_size)
        if n_train < 8 or (end - start) < 4:
            continue
        train_idx = np.arange(0, n_train)
        test_idx = np.arange(start, end)
        folds.append((f"FOLD_{int(frac*100)}", train_idx, test_idx))
    return folds


@dataclass
class ModelSpec:
    name: str
    fitter_factory: Callable[[], object]
    features: list[str]
    penalizer: float | None = None


def fit_model(spec: ModelSpec, train_df: pd.DataFrame) -> object:
    cols = ["DURATION_DIAS", "EVENT_OBSERVED", *spec.features]
    train = train_df[cols].copy()
    for c in spec.features:
        train[c] = to_num(train[c])
        med = train[c].median(skipna=True)
        if np.isnan(med):
            med = 0.0
        train[c] = train[c].fillna(med)
    train["DURATION_DIAS"] = to_num(train["DURATION_DIAS"])
    train["EVENT_OBSERVED"] = to_num(train["EVENT_OBSERVED"]).fillna(0).astype(int)
    train = train.dropna(subset=["DURATION_DIAS"]).copy()
    train = train[train["DURATION_DIAS"] >= 0].copy()

    model = spec.fitter_factory()
    if isinstance(model, CoxPHFitter):
        if spec.penalizer is not None:
            model = CoxPHFitter(penalizer=float(spec.penalizer))
        model.fit(train, duration_col="DURATION_DIAS", event_col="EVENT_OBSERVED")
    else:
        model.fit(train, duration_col="DURATION_DIAS", event_col="EVENT_OBSERVED")
    return model


def prepare_test_matrix(spec: ModelSpec, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df[spec.features].copy()
    for c in spec.features:
        tr = to_num(train_df[c]) if c in train_df.columns else pd.Series(dtype=float)
        med = tr.median(skipna=True)
        if np.isnan(med):
            med = 0.0
        out[c] = to_num(out[c]).fillna(med)
    return out


def predict_risk_score(spec: ModelSpec, model: object, x_test: pd.DataFrame) -> np.ndarray:
    if spec.name == "CoxPH":
        risk = np.asarray(model.predict_partial_hazard(x_test)).reshape(-1)
        return -risk
    med = np.asarray(model.predict_median(x_test)).reshape(-1)
    inf_mask = ~np.isfinite(med)
    if inf_mask.any():
        expv = np.asarray(model.predict_expectation(x_test)).reshape(-1)
        med[inf_mask] = expv[inf_mask]
    return med


def predict_event_prob_at_horizon(model: object, x_test: pd.DataFrame, horizon: int) -> np.ndarray:
    surv = model.predict_survival_function(x_test, times=[horizon])
    if isinstance(surv, pd.DataFrame):
        s = surv.T.iloc[:, 0].astype(float).values
    else:
        s = np.asarray(surv).reshape(-1)
    return 1.0 - np.clip(s, 0.0, 1.0)


def eval_calibration(
    p_event: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    horizon: int,
    model_name: str,
    fold_id: str,
) -> tuple[float, float, list[dict], list[dict]]:
    y = np.where((t <= horizon) & (e == 1), 1.0, 0.0)
    valid = ~((t <= horizon) & (e == 0))
    if valid.sum() < 3:
        return np.nan, np.nan, [], []

    yv = y[valid]
    pv = p_event[valid]
    brier = float(np.mean((yv - pv) ** 2))

    sample_rows = [
        {
            "MODEL": model_name,
            "FOLD": fold_id,
            "HORIZON_DIAS": int(horizon),
            "P_EVENT": float(p),
            "Y_EVENT": float(yobs),
        }
        for p, yobs in zip(pv, yv)
    ]

    bins_n = 5 if len(pv) >= 15 else (3 if len(pv) >= 8 else 2)
    try:
        bins = pd.qcut(pd.Series(pv), q=bins_n, duplicates="drop")
    except Exception:
        bins = pd.Series(["BIN_1"] * len(pv))
    df = pd.DataFrame({"p": pv, "y": yv, "bin": bins.astype(str)})
    points = (
        df.groupby("bin", as_index=False)
        .agg(P_EVENT_MEAN=("p", "mean"), Y_EVENT_RATE=("y", "mean"), N=("y", "count"))
        .sort_values("P_EVENT_MEAN")
        .reset_index(drop=True)
    )
    if len(points) == 0:
        return brier, np.nan, [], sample_rows
    calib_mae = float(np.average(np.abs(points["Y_EVENT_RATE"] - points["P_EVENT_MEAN"]), weights=points["N"]))
    point_rows = [
        {
            "MODEL": model_name,
            "FOLD": fold_id,
            "HORIZON_DIAS": int(horizon),
            "BIN": int(i + 1),
            "P_EVENT_MEAN": float(r["P_EVENT_MEAN"]),
            "Y_EVENT_RATE": float(r["Y_EVENT_RATE"]),
            "N": int(r["N"]),
        }
        for i, (_, r) in enumerate(points.iterrows())
    ]
    return brier, calib_mae, point_rows, sample_rows


def build_calibration_plot(sample_df: pd.DataFrame, horizon: int, out_path: Path) -> None:
    df = sample_df[sample_df["HORIZON_DIAS"] == horizon].copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if len(df) == 0:
        ax.text(0.5, 0.5, f"Sem dados de calibracao ({horizon} dias)", ha="center", va="center")
        ax.axis("off")
    else:
        for model, g in df.groupby("MODEL"):
            bins_n = 5 if len(g) >= 20 else (4 if len(g) >= 12 else 3)
            try:
                bins = pd.qcut(g["P_EVENT"], q=bins_n, duplicates="drop")
            except Exception:
                bins = pd.Series(["BIN_1"] * len(g), index=g.index)
            gg = (
                g.assign(BIN=bins.astype(str))
                .groupby("BIN", as_index=False)
                .agg(P_EVENT_MEAN=("P_EVENT", "mean"), Y_EVENT_RATE=("Y_EVENT", "mean"))
                .sort_values("P_EVENT_MEAN")
            )
            ax.plot(gg["P_EVENT_MEAN"], gg["Y_EVENT_RATE"], marker="o", label=model)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Probabilidade prevista de evento")
        ax.set_ylabel("Frequencia observada de evento")
        ax.set_title(f"Calibracao - Horizonte {horizon} dias")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def model_display_name(name: str) -> str:
    mapping = {
        "CoxPH": "Cox",
        "WeibullAFT": "Weibull AFT",
        "LogLogisticAFT": "Log-Logistic AFT",
        "LogNormalAFT": "Log-Normal AFT",
    }
    return mapping.get(name, name)


def format_decimal_pt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def main() -> None:
    root = find_project_root(Path(__file__).resolve().parent)
    cfg = load_cfg(root / "project_config.json")
    model_dir = root / "02_model_input" / "modelo01"
    out_plot_dir = root / "outputs" / "relatorio_graficos"
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    ds_path = root / '02_model_input' / 'modelo01' / 'modelo01_d02_survival_dataset.csv'
    if not ds_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {ds_path}")

    daily = pd.read_csv(ds_path, sep=";", encoding="utf-8-sig", low_memory=False)
    ep = build_episode_dataframe(daily, equip="D02")
    folds = build_temporal_folds(len(ep))
    if len(folds) == 0:
        raise ValueError("Amostra insuficiente para validacao temporal entre modelos")

    specs = [
        ModelSpec(
            name="CoxPH",
            fitter_factory=lambda: CoxPHFitter(penalizer=0.15),
            features=["CARGA_ACUM_EP", "MEDIA_CARGA_30D", "FLAG_EMENDA_7D", "DIAS_DESDE_ULT_EMENDA"],
            penalizer=0.15,
        ),
        ModelSpec(
            name="WeibullAFT",
            fitter_factory=lambda: WeibullAFTFitter(),
            features=["CARGA_ACUM_EP", "MEDIA_CARGA_30D", "FLAG_EMENDA_7D", "DIAS_DESDE_ULT_EMENDA"],
        ),
        ModelSpec(
            name="LogNormalAFT",
            fitter_factory=lambda: LogNormalAFTFitter(),
            features=["CARGA_ACUM_EP", "MEDIA_CARGA_30D", "FLAG_EMENDA_7D", "DIAS_DESDE_ULT_EMENDA"],
        ),
        ModelSpec(
            name="LogLogisticAFT",
            fitter_factory=lambda: LogLogisticAFTFitter(),
            features=["CARGA_ACUM_EP", "MEDIA_CARGA_30D", "FLAG_EMENDA_7D", "DIAS_DESDE_ULT_EMENDA"],
        ),
    ]

    fold_rows: list[dict] = []
    calib_points_rows: list[dict] = []
    calib_sample_rows: list[dict] = []

    horizons = [180, 365]
    for fold_id, train_idx, test_idx in folds:
        train = ep.iloc[train_idx].copy()
        test = ep.iloc[test_idx].copy()
        t_test = to_num(test["DURATION_DIAS"]).values
        e_test = to_num(test["EVENT_OBSERVED"]).fillna(0).astype(int).values

        for spec in specs:
            row = {
                "MODEL": spec.name,
                "FOLD": fold_id,
                "TRAIN_ROWS": int(len(train)),
                "TEST_ROWS": int(len(test)),
                "TRAIN_EVENTS": int(to_num(train["EVENT_OBSERVED"]).fillna(0).sum()),
                "TEST_EVENTS": int(e_test.sum()),
                "STATUS": "OK",
                "ERROR": "",
                "C_INDEX": np.nan,
                "BRIER_180": np.nan,
                "BRIER_365": np.nan,
                "CAL_MAE_180": np.nan,
                "CAL_MAE_365": np.nan,
            }

            try:
                model = fit_model(spec, train)
                x_test = prepare_test_matrix(spec, train, test)
                score = predict_risk_score(spec, model, x_test)
                if np.isfinite(score).sum() >= 3:
                    row["C_INDEX"] = float(concordance_index(t_test, score, e_test))

                for hz in horizons:
                    p_event = predict_event_prob_at_horizon(model, x_test, hz)
                    brier, mae, point_rows, sample_rows = eval_calibration(
                        p_event=p_event, t=t_test, e=e_test, horizon=hz, model_name=spec.name, fold_id=fold_id
                    )
                    row[f"BRIER_{hz}"] = brier
                    row[f"CAL_MAE_{hz}"] = mae
                    calib_points_rows.extend(point_rows)
                    calib_sample_rows.extend(sample_rows)
            except Exception as exc:
                row["STATUS"] = "FAIL"
                row["ERROR"] = f"{type(exc).__name__}: {exc}"

            fold_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)
    calib_points_df = pd.DataFrame(calib_points_rows)
    calib_sample_df = pd.DataFrame(calib_sample_rows)

    ok = folds_df[folds_df["STATUS"] == "OK"].copy()
    metrics_df = (
        ok.groupby("MODEL", as_index=False)
        .agg(
            FOLDS_OK=("MODEL", "count"),
            C_INDEX_MEAN=("C_INDEX", "mean"),
            C_INDEX_STD=("C_INDEX", "std"),
            BRIER_180_MEAN=("BRIER_180", "mean"),
            BRIER_365_MEAN=("BRIER_365", "mean"),
            CAL_MAE_180_MEAN=("CAL_MAE_180", "mean"),
            CAL_MAE_365_MEAN=("CAL_MAE_365", "mean"),
            TEST_EVENTS_TOTAL=("TEST_EVENTS", "sum"),
        )
        .sort_values("C_INDEX_MEAN", ascending=False)
        .reset_index(drop=True)
    )
    metrics_df["BRIER_MEAN"] = metrics_df[["BRIER_180_MEAN", "BRIER_365_MEAN"]].mean(axis=1)
    metrics_df["CAL_MAE_MEAN"] = metrics_df[["CAL_MAE_180_MEAN", "CAL_MAE_365_MEAN"]].mean(axis=1)

    # Decision rule (non-subjective)
    chosen_model = "CoxPH"
    decision_reason = "Sem alternativa com ganho consistente; manter CoxPH por interpretabilidade e robustez."
    cond_rows: list[dict] = []
    if (metrics_df["MODEL"] == "CoxPH").any():
        base = metrics_df.loc[metrics_df["MODEL"] == "CoxPH"].iloc[0]
        candidates = []
        for _, r in metrics_df.iterrows():
            if r["MODEL"] == "CoxPH":
                continue
            cond_cindex = bool(r["C_INDEX_MEAN"] >= (base["C_INDEX_MEAN"] + 0.01))
            cond_calib = bool(r["CAL_MAE_MEAN"] <= (base["CAL_MAE_MEAN"] - 0.005))
            cond_consist = bool(r["C_INDEX_STD"] <= (base["C_INDEX_STD"] + 0.01))
            all_ok = cond_cindex and cond_calib and cond_consist
            cond_rows.append(
                {
                    "MODEL": r["MODEL"],
                    "COND_CINDEX": cond_cindex,
                    "COND_CALIB": cond_calib,
                    "COND_CONSISTENCIA": cond_consist,
                    "DECISAO_CANDIDATO": all_ok,
                }
            )
            if all_ok:
                candidates.append(r)
        if len(candidates) > 0:
            best = sorted(candidates, key=lambda x: float(x["C_INDEX_MEAN"]), reverse=True)[0]
            chosen_model = str(best["MODEL"])
            decision_reason = "Modelo alternativo superou CoxPH em C-index de forma consistente e calibracao."

    cond_df = pd.DataFrame(cond_rows)

    # Cox B exploratory note (small sample)
    coxb_txt = root / "outputs" / "relatorio_graficos" / "cox_B_results.txt"
    coxb_kv = parse_kv_file(coxb_txt)
    coxb_rows = coxb_kv.get("ROWS", "6")
    coxb_events = coxb_kv.get("EVENTS", "5")

    # Save tables
    out_folds = model_dir / "modelo01_d02_model_compare_folds.csv"
    out_metrics = model_dir / "modelo01_d02_model_compare_metrics.csv"
    out_cond = model_dir / "modelo01_d02_model_compare_conditions.csv"
    out_calib_points = model_dir / "modelo01_d02_model_compare_calibration_points.csv"
    out_calib_samples = model_dir / "modelo01_d02_model_compare_calibration_samples.csv"
    out_decision = model_dir / "modelo01_d02_model_compare_decision.txt"

    folds_df.to_csv(out_folds, sep=";", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_metrics, sep=";", index=False, encoding="utf-8-sig")
    cond_df.to_csv(out_cond, sep=";", index=False, encoding="utf-8-sig")
    calib_points_df.to_csv(out_calib_points, sep=";", index=False, encoding="utf-8-sig")
    calib_sample_df.to_csv(out_calib_samples, sep=";", index=False, encoding="utf-8-sig")

    # Plots
    cidx = metrics_df.sort_values("C_INDEX_MEAN", ascending=False).copy()
    cidx["MODEL_LABEL"] = cidx["MODEL"].map(model_display_name)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(cidx["MODEL_LABEL"], cidx["C_INDEX_MEAN"], yerr=cidx["C_INDEX_STD"].fillna(0.0), capsize=4)
    for b, (_, row) in zip(bars, cidx.iterrows()):
        model = str(row["MODEL"])
        if model == "CoxPH":
            b.set_color("#1f77b4")
        else:
            b.set_color("#8a8a8a")
        cidx_std = float(row["C_INDEX_STD"]) if pd.notna(row["C_INDEX_STD"]) else 0.0
        ax.text(
            b.get_x() + b.get_width() / 2,
            float(row["C_INDEX_MEAN"]) + cidx_std + 0.012,
            format_decimal_pt(float(row["C_INDEX_MEAN"]), 3),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("C-index medio na validacao temporal (quanto maior, melhor)")
    ax.set_title("Comparacao entre modelos por C-index")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "model_compare_cindex.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    mae_plot = metrics_df.sort_values("CAL_MAE_MEAN", ascending=True)
    mae_plot["MODEL_LABEL"] = mae_plot["MODEL"].map(model_display_name)
    best_calib_model = str(mae_plot.iloc[0]["MODEL"]) if len(mae_plot) > 0 else ""
    bars = ax.bar(mae_plot["MODEL_LABEL"], mae_plot["CAL_MAE_MEAN"])
    for b, (_, row) in zip(bars, mae_plot.iterrows()):
        model = str(row["MODEL"])
        if model == best_calib_model:
            b.set_color("#2ca02c")
        elif model == "CoxPH":
            b.set_color("#1f77b4")
        else:
            b.set_color("#8a8a8a")
        ax.text(
            b.get_x() + b.get_width() / 2,
            float(row["CAL_MAE_MEAN"]) + 0.0035,
            format_decimal_pt(float(row["CAL_MAE_MEAN"]), 3),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Erro medio de calibracao (quanto menor, melhor)")
    ax.set_title("Comparacao entre modelos por calibracao")
    ax.set_ylim(0.0, float(mae_plot["CAL_MAE_MEAN"].max()) + 0.012)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "model_compare_calibration_mae.png", dpi=150)
    plt.close(fig)

    build_calibration_plot(calib_sample_df, 180, out_plot_dir / "model_compare_calibration_180d.png")
    build_calibration_plot(calib_sample_df, 365, out_plot_dir / "model_compare_calibration_365d.png")

    lines = [
        "COMPARACAO_OBJETIVA_MODELOS_SURVIVAL_D02",
        "",
        "REGRA_DE_DECISAO",
        "- Se modelo alternativo melhorar C-index de forma consistente e calibrar melhor, ele ganha.",
        "- Se ganho for pequeno/instavel, manter Cox por interpretabilidade e robustez.",
        f"- Cox B (com dosadoras) e exploratorio por amostra pequena ({coxb_rows} episodios/{coxb_events} eventos).",
        "",
        f"MODELO_ESCOLHIDO={chosen_model}",
        f"JUSTIFICATIVA={decision_reason}",
        "",
        "[METRICAS_RESUMO]",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"{r['MODEL']}: C_INDEX={float(r['C_INDEX_MEAN']):.4f} (std={float(r['C_INDEX_STD']) if pd.notna(r['C_INDEX_STD']) else 0:.4f}); "
            f"CAL_MAE={float(r['CAL_MAE_MEAN']):.4f}; BRIER={float(r['BRIER_MEAN']):.4f}; FOLDS={int(r['FOLDS_OK'])}"
        )

    if len(cond_df) > 0:
        lines.extend(["", "[CHECAGEM_CONDICOES_VS_COXPH]"])
        for _, r in cond_df.iterrows():
            lines.append(
                f"{r['MODEL']}: CINDEX_OK={bool(r['COND_CINDEX'])}; CALIB_OK={bool(r['COND_CALIB'])}; "
                f"CONSISTENCIA_OK={bool(r['COND_CONSISTENCIA'])}; CANDIDATO={bool(r['DECISAO_CANDIDATO'])}"
            )

    out_decision.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # log
    logs_dir = root / cfg["folders"]["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"modelo_d02_model_compare_{stamp}.log"
    log_path.write_text(
        "\n".join(
            [
                "pipeline=modelo01_d02_model_compare",
                f"input_dataset={ds_path}",
                f"episodes={len(ep)}",
                f"folds={len(folds)}",
                f"chosen_model={chosen_model}",
                f"output_metrics={out_metrics}",
                f"output_decision={out_decision}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"MODELOS_AVALIADOS: {len(metrics_df)}")
    print(f"FOLDS_TEMPORAIS: {len(folds)}")
    print(f"MODELO_ESCOLHIDO: {chosen_model}")
    print(f"METRICAS: {out_metrics}")
    print(f"DECISAO: {out_decision}")
    print(f"GRAFICO_CINDEX: {out_plot_dir / 'model_compare_cindex.png'}")
    print(f"GRAFICO_CAL_180: {out_plot_dir / 'model_compare_calibration_180d.png'}")
    print(f"GRAFICO_CAL_365: {out_plot_dir / 'model_compare_calibration_365d.png'}")
    print(f"GRAFICO_CAL_MAE: {out_plot_dir / 'model_compare_calibration_mae.png'}")
    print(f"LOG: {log_path}")


if __name__ == "__main__":
    main()
