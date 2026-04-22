from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter
from lifelines.statistics import proportional_hazard_test


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "project_config.json").exists():
            return candidate
    raise FileNotFoundError("project_config.json nao encontrado em nenhum diretorio pai")


def parse_date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str).str.strip(), errors="coerce")


def to_num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def safe_last(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.iloc[-1])


def fit_weibull(episode_df: pd.DataFrame, equip: str, out_txt: Path) -> dict:
    data = episode_df.copy()
    data = data.dropna(subset=["DURATION_DIAS", "EVENT_OBSERVED"])
    durations = data["DURATION_DIAS"].astype(float)
    events = data["EVENT_OBSERVED"].astype(int)

    # Weibull parametric fit is unstable/undefined with tiny samples or no observed events.
    if len(data) < 2 or int(events.sum()) < 1:
        reason = (
            f"dados_insuficientes (observacoes={len(data)}, eventos={int(events.sum())})"
        )
        out_txt.write_text(
            "\n".join(
                [
                    f"EQUIPAMENTO={equip}",
                    "MODELO=WEIBULL",
                    "STATUS=SKIPPED",
                    f"RAZAO={reason}",
                    f"OBSERVACOES={len(data)}",
                    f"EVENTOS={int(events.sum())}",
                    "BETA_SHAPE=",
                    "ETA_SCALE=",
                    "LOGLIK=",
                    "AIC=",
                    "INTERPRETACAO_BETA=",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "SKIPPED",
            "reason": reason,
            "events": int(events.sum()),
            "n": int(len(data)),
            "beta": np.nan,
            "eta": np.nan,
            "loglik": np.nan,
            "aic": np.nan,
            "beta_interp": "",
        }

    wf = WeibullFitter()
    wf.fit(durations=durations, event_observed=events, label=equip)

    beta = float(wf.rho_)
    eta = float(wf.lambda_)
    loglik = float(wf.log_likelihood_)
    aic = float((2 * 2) - (2 * loglik))

    if beta > 1:
        beta_interp = "DESGASTE_PROGRESSIVO"
    elif abs(beta - 1.0) < 0.05:
        beta_interp = "FALHA_ALEATORIA"
    else:
        beta_interp = "FALHA_PRECOCE"

    ci_rho_low = np.nan
    ci_rho_high = np.nan
    ci_lambda_low = np.nan
    ci_lambda_high = np.nan
    try:
        s = wf.summary
        if "rho_" in s.index:
            ci_rho_low = float(s.loc["rho_", "coef lower 95%"])
            ci_rho_high = float(s.loc["rho_", "coef upper 95%"])
        if "lambda_" in s.index:
            ci_lambda_low = float(s.loc["lambda_", "coef lower 95%"])
            ci_lambda_high = float(s.loc["lambda_", "coef upper 95%"])
    except Exception:
        pass

    lines = [
        f"EQUIPAMENTO={equip}",
        "MODELO=WEIBULL",
        "STATUS=OK",
        f"OBSERVACOES={len(data)}",
        f"EVENTOS={int(events.sum())}",
        f"BETA_SHAPE={beta:.6f}",
        f"BETA_CI95_LOW={ci_rho_low:.6f}" if np.isfinite(ci_rho_low) else "BETA_CI95_LOW=",
        f"BETA_CI95_HIGH={ci_rho_high:.6f}" if np.isfinite(ci_rho_high) else "BETA_CI95_HIGH=",
        f"ETA_SCALE={eta:.6f}",
        f"ETA_CI95_LOW={ci_lambda_low:.6f}" if np.isfinite(ci_lambda_low) else "ETA_CI95_LOW=",
        f"ETA_CI95_HIGH={ci_lambda_high:.6f}" if np.isfinite(ci_lambda_high) else "ETA_CI95_HIGH=",
        f"LOGLIK={loglik:.6f}",
        f"AIC={aic:.6f}",
        f"INTERPRETACAO_BETA={beta_interp}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "OK",
        "beta": beta,
        "eta": eta,
        "loglik": loglik,
        "aic": aic,
        "events": int(events.sum()),
        "n": int(len(data)),
        "beta_interp": beta_interp,
    }


def prepare_episode_features(daily_df: pd.DataFrame, equip: str) -> pd.DataFrame:
    df = daily_df.copy()
    df["EQUIPAMENTO"] = df["EQUIPAMENTO"].astype(str).str.strip().str.upper()
    df = df[df["EQUIPAMENTO"] == equip].copy()
    if df.empty:
        raise ValueError(f"Sem dados diarios para {equip}")

    df["DATA_DT"] = parse_date_series(df["DATA_DIA"])
    for col in [
        "DURATION_DIAS",
        "EVENT_OBSERVED",
        "CENSORED",
        "T_DIAS",
        "EVENT_TROCA",
        "EVENT_RECUP",
        "CARGA_TON_ACUM_EP_LAG1",
        "CARGA_TON_30D_MEAN_LAG1",
        "FLAG_RECUP_7D_LAG1",
        "FLAG_RECUP_14D_LAG1",
        "DIAS_DESDE_ULT_RECUP",
        "DOS_VAZAO_MEDIA_LAG1",
        "DOS_VELOC_MEDIA_LAG1",
        "DOS_COBERTURA_24H_PCT_LAG1",
        "HAS_DOSADORAS_DADO_LAG1",
        "CLIMA_TEMP_MEDIA_C_LAG1",
        "CLIMA_TEMP_MAX_C_LAG1",
        "CLIMA_TEMP_MIN_C_LAG1",
        "CLIMA_UMIDADE_REL_MEDIA_PCT_LAG1",
        "HAS_CLIMA_DADO_LAG1",
        "TEMP_PROCESSO_C_LAG1",
        "TEMP_PROCESSO_7D_MEAN_LAG1",
        "TEMP_PROCESSO_30D_MEAN_LAG1",
        "HAS_TEMP_PROCESSO_DADO_LAG1",
    ]:
        if col in df.columns:
            df[col] = to_num(df[col])

    if "EPISODIO_ID" not in df.columns:
        raise ValueError("Coluna EPISODIO_ID nao encontrada no dataset survival")

    rows = []
    for eid, g in df.groupby("EPISODIO_ID", sort=True):
        g = g.sort_values("DATA_DT")
        event_obs = int(g["EVENT_OBSERVED"].max())
        censored = int(g["CENSORED"].max())
        duration = safe_last(g["T_DIAS"])
        if np.isnan(duration):
            duration = float(g["T_DIAS"].max())

        row = {
            "EQUIPAMENTO": equip,
            "EPISODIO_ID": eid,
            "DATA_INICIO": g["DATA_DIA"].iloc[0],
            "DATA_FIM": g["DATA_DIA"].iloc[-1],
            "ANO_INICIO": int(g["DATA_DT"].iloc[0].year) if pd.notna(g["DATA_DT"].iloc[0]) else np.nan,
            "DURATION_DIAS": float(duration),
            "EVENT_OBSERVED": event_obs,
            "CENSORED": censored,
            "CARGA_ACUM_EP": safe_last(g["CARGA_TON_ACUM_EP_LAG1"]) if "CARGA_TON_ACUM_EP_LAG1" in g else np.nan,
            "MEDIA_CARGA_30D": float(g["CARGA_TON_30D_MEAN_LAG1"].mean()) if "CARGA_TON_30D_MEAN_LAG1" in g else np.nan,
            "FLAG_EMENDA_7D": float(g["FLAG_RECUP_7D_LAG1"].max()) if "FLAG_RECUP_7D_LAG1" in g else np.nan,
            "FLAG_EMENDA_14D": float(g["FLAG_RECUP_14D_LAG1"].max()) if "FLAG_RECUP_14D_LAG1" in g else np.nan,
            "DIAS_DESDE_ULT_EMENDA": safe_last(g["DIAS_DESDE_ULT_RECUP"]) if "DIAS_DESDE_ULT_RECUP" in g else np.nan,
            "DOS_VAZAO_MEDIA_EP": float(g["DOS_VAZAO_MEDIA_LAG1"].mean()) if "DOS_VAZAO_MEDIA_LAG1" in g else np.nan,
            "DOS_VELOC_MEDIA_EP": float(g["DOS_VELOC_MEDIA_LAG1"].mean()) if "DOS_VELOC_MEDIA_LAG1" in g else np.nan,
            "DOS_COBERTURA_MEDIA_EP": float(g["DOS_COBERTURA_24H_PCT_LAG1"].mean()) if "DOS_COBERTURA_24H_PCT_LAG1" in g else np.nan,
            "DOS_DISPONIBILIDADE_EP": float(g["HAS_DOSADORAS_DADO_LAG1"].mean()) if "HAS_DOSADORAS_DADO_LAG1" in g else np.nan,
            "CLIMA_TEMP_MEDIA_EP": float(g["CLIMA_TEMP_MEDIA_C_LAG1"].mean()) if "CLIMA_TEMP_MEDIA_C_LAG1" in g else np.nan,
            "CLIMA_TEMP_MAX_EP": float(g["CLIMA_TEMP_MAX_C_LAG1"].mean()) if "CLIMA_TEMP_MAX_C_LAG1" in g else np.nan,
            "CLIMA_TEMP_MIN_EP": float(g["CLIMA_TEMP_MIN_C_LAG1"].mean()) if "CLIMA_TEMP_MIN_C_LAG1" in g else np.nan,
            "CLIMA_UMIDADE_MEDIA_EP": float(g["CLIMA_UMIDADE_REL_MEDIA_PCT_LAG1"].mean()) if "CLIMA_UMIDADE_REL_MEDIA_PCT_LAG1" in g else np.nan,
            "CLIMA_DISPONIBILIDADE_EP": float(g["HAS_CLIMA_DADO_LAG1"].mean()) if "HAS_CLIMA_DADO_LAG1" in g else np.nan,
            "TEMP_PROCESSO_MEDIA_EP": float(g["TEMP_PROCESSO_C_LAG1"].mean()) if "TEMP_PROCESSO_C_LAG1" in g else np.nan,
            "TEMP_PROCESSO_7D_MEDIA_EP": float(g["TEMP_PROCESSO_7D_MEAN_LAG1"].mean()) if "TEMP_PROCESSO_7D_MEAN_LAG1" in g else np.nan,
            "TEMP_PROCESSO_30D_MEDIA_EP": float(g["TEMP_PROCESSO_30D_MEAN_LAG1"].mean()) if "TEMP_PROCESSO_30D_MEAN_LAG1" in g else np.nan,
            "TEMP_PROCESSO_DISPONIBILIDADE_EP": float(g["HAS_TEMP_PROCESSO_DADO_LAG1"].mean()) if "HAS_TEMP_PROCESSO_DADO_LAG1" in g else np.nan,
        }
        rows.append(row)

    ep_df = pd.DataFrame(rows).sort_values(["DATA_INICIO", "EPISODIO_ID"]).reset_index(drop=True)
    return ep_df


def run_cox(
    df: pd.DataFrame,
    features: Iterable[str],
    out_txt: Path,
    title: str,
    penalizer: float = 0.1,
    min_rows: int | None = None,
) -> dict:
    cols = ["DURATION_DIAS", "EVENT_OBSERVED", *features]
    model_df = df[cols].copy()
    for c in features:
        model_df[c] = to_num(model_df[c])

    for c in features:
        med = model_df[c].median(skipna=True)
        if np.isnan(med):
            med = 0.0
        model_df[c] = model_df[c].fillna(med)

    model_df = model_df.dropna(subset=["DURATION_DIAS", "EVENT_OBSERVED"]).copy()
    model_df = model_df[model_df["DURATION_DIAS"] >= 0].copy()
    model_df["EVENT_OBSERVED"] = model_df["EVENT_OBSERVED"].astype(int)

    required_rows = max(10, len(features) + 3) if min_rows is None else int(min_rows)
    if len(model_df) < required_rows:
        out_txt.write_text(
            "\n".join(
                [
                    f"MODELO={title}",
                    "STATUS=SKIPPED",
                    f"RAZAO=amostra_insuficiente ({len(model_df)} linhas para {len(features)} covariaveis; minimo={required_rows})",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "SKIPPED",
            "rows": int(len(model_df)),
            "events": int(model_df["EVENT_OBSERVED"].sum()),
            "c_index": None,
            "features_used": list(features),
            "features": list(features),
            "reason": f"amostra_insuficiente ({len(model_df)} linhas para {len(features)} covariaveis; minimo={required_rows})",
            "summary": pd.DataFrame(),
            "ph": pd.DataFrame(),
            "title": title,
        }

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(model_df, duration_col="DURATION_DIAS", event_col="EVENT_OBSERVED")

        ph_test = proportional_hazard_test(cph, model_df, time_transform="rank")
        ph_summary = ph_test.summary.copy()
        ph_summary = ph_summary.reset_index().rename(columns={"index": "COVARIAVEL"})

        cindex = float(cph.concordance_index_)
        events = int(model_df["EVENT_OBSERVED"].sum())
    except Exception as exc:
        out_txt.write_text(
            "\n".join(
                [
                    f"MODELO={title}",
                    "STATUS=FAIL",
                    f"RAZAO=erro_no_ajuste ({type(exc).__name__}: {exc})",
                    f"COVARIAVEIS={','.join(features)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "FAIL",
            "rows": int(len(model_df)),
            "events": int(model_df["EVENT_OBSERVED"].sum()),
            "c_index": None,
            "features_used": list(features),
            "features": list(features),
            "reason": f"erro_no_ajuste ({type(exc).__name__}: {exc})",
            "error": str(exc),
            "summary": pd.DataFrame(),
            "ph": pd.DataFrame(),
            "title": title,
        }

    lines = [
        f"MODELO={title}",
        "STATUS=OK",
        f"ROWS={len(model_df)}",
        f"EVENTS={events}",
        f"C_INDEX={cindex:.6f}",
        f"COVARIAVEIS={','.join(features)}",
        "",
        "[COEFICIENTES]",
    ]

    s = cph.summary.reset_index().rename(columns={"covariate": "COVARIAVEL"})
    keep_cols = [
        "COVARIAVEL",
        "coef",
        "exp(coef)",
        "se(coef)",
        "p",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
    ]
    s = s[keep_cols]

    for _, r in s.iterrows():
        lines.append(
            f"{r['COVARIAVEL']}: HR={float(r['exp(coef)']):.6f}; p={float(r['p']):.6g}; "
            f"IC95=[{float(r['exp(coef) lower 95%']):.6f},{float(r['exp(coef) upper 95%']):.6f}]"
        )

    lines.extend(["", "[PH_TEST_RANK]"])
    if "p" in ph_summary.columns:
        for _, r in ph_summary.iterrows():
            lines.append(f"{r['COVARIAVEL']}: p={float(r['p']):.6g}")
    else:
        lines.append("PH test nao disponivel")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "OK",
        "rows": int(len(model_df)),
        "events": events,
        "c_index": cindex,
        "features_used": list(features),
        "features": list(features),
        "reason": "",
        "summary": s.copy(),
        "ph": ph_summary.copy(),
        "title": title,
    }


def create_km_plot(episode_df: pd.DataFrame, equip: str, out_png: Path) -> None:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=episode_df["DURATION_DIAS"], event_observed=episode_df["EVENT_OBSERVED"], label=equip)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(f"Curva Kaplan-Meier - {equip}")
    ax.set_xlabel("Dias no episodio")
    ax.set_ylabel("Sobrevivencia")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_covariate_coverage_report(daily_df: pd.DataFrame, out_csv: Path) -> None:
    df = daily_df.copy()
    df["DATA_DT"] = parse_date_series(df["DATA_DIA"])
    df["ANO"] = df["DATA_DT"].dt.year

    cov_cols = [
        "PRODUCAO_TONELADAS_DIA_LAG1",
        "CARGA_TON_7D_MEAN_LAG1",
        "CARGA_TON_30D_MEAN_LAG1",
        "CARGA_TON_30D_P95_LAG1",
        "CARGA_TON_ACUM_EP_LAG1",
        "FLAG_RECUP_7D_LAG1",
        "FLAG_RECUP_14D_LAG1",
        "DIAS_DESDE_ULT_RECUP",
        "DOS_VAZAO_MEDIA_LAG1",
        "DOS_VELOC_MEDIA_LAG1",
        "DOS_COBERTURA_24H_PCT_LAG1",
        "HAS_DOSADORAS_DADO_LAG1",
        "CLIMA_TEMP_MEDIA_C_LAG1",
        "CLIMA_TEMP_MAX_C_LAG1",
        "CLIMA_TEMP_MIN_C_LAG1",
        "CLIMA_UMIDADE_REL_MEDIA_PCT_LAG1",
        "HAS_CLIMA_DADO_LAG1",
        "TEMP_PROCESSO_C_LAG1",
        "TEMP_PROCESSO_7D_MEAN_LAG1",
        "TEMP_PROCESSO_30D_MEAN_LAG1",
        "HAS_TEMP_PROCESSO_DADO_LAG1",
    ]
    cov_cols = [c for c in cov_cols if c in df.columns]

    rows = []
    total = len(df)
    for c in cov_cols:
        pct = float(df[c].notna().mean() * 100.0) if total > 0 else 0.0
        rows.append({"ESCOPO": "OVERALL", "ANO": "", "COVARIAVEL": c, "LINHAS": total, "PCT_PREENCHIMENTO": round(pct, 2)})

    for year, g in df.groupby("ANO", dropna=True):
        n = len(g)
        for c in cov_cols:
            pct = float(g[c].notna().mean() * 100.0) if n > 0 else 0.0
            rows.append({"ESCOPO": "BY_YEAR", "ANO": int(year), "COVARIAVEL": c, "LINHAS": n, "PCT_PREENCHIMENTO": round(pct, 2)})

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, sep=";", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train survival models for modelo01")
    parser.add_argument("--equipamento", default="D02", help="Equipamento piloto (default: D02)")
    args = parser.parse_args()
    equip = str(args.equipamento).strip().upper()

    script_dir = Path(__file__).resolve().parent
    base = find_project_root(script_dir)
    cfg = load_cfg(base / "project_config.json")
    logs_dir = base / cfg["folders"]["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_dir = base / "02_model_input" / "modelo01"
    ds_path = model_dir / f"modelo01_{equip.lower()}_survival_dataset.csv"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset survival nao encontrado: {ds_path}")

    daily_df = pd.read_csv(ds_path, sep=";", encoding="utf-8-sig", low_memory=False)
    ep_df = prepare_episode_features(daily_df, equip)

    episode_lengths_path = model_dir / f"modelo01_{equip.lower()}_episode_lengths.csv"
    cov_cov_path = model_dir / f"modelo01_{equip.lower()}_covariate_coverage_report.csv"
    km_plot_path = model_dir / f"modelo01_{equip.lower()}_km_plot.png"
    weibull_path = model_dir / f"modelo01_{equip.lower()}_weibull_results.txt"
    cox_a_path = model_dir / f"modelo01_{equip.lower()}_coxA_summary.txt"
    cox_a_temp_path = model_dir / f"modelo01_{equip.lower()}_coxA_temp_summary.txt"
    cox_a_clima_path = model_dir / f"modelo01_{equip.lower()}_coxA_clima_summary.txt"
    cox_b_path = model_dir / f"modelo01_{equip.lower()}_coxB_summary.txt"
    event_sample_path = model_dir / f"modelo01_{equip.lower()}_event_timeline_sample.csv"

    episode_lengths = ep_df[
        ["EQUIPAMENTO", "EPISODIO_ID", "DATA_INICIO", "DATA_FIM", "DURATION_DIAS", "EVENT_OBSERVED", "CENSORED"]
    ].copy()
    episode_lengths.to_csv(episode_lengths_path, sep=";", index=False, encoding="utf-8-sig")

    build_covariate_coverage_report(daily_df, cov_cov_path)
    create_km_plot(ep_df, equip, km_plot_path)
    weibull_metrics = fit_weibull(ep_df, equip, weibull_path)

    # Cox A (historico completo, sem dosadoras)
    ep_cox = ep_df.copy()
    ep_cox["LOG_CARGA_ACUM_EP"] = np.log1p(np.clip(ep_cox["CARGA_ACUM_EP"].astype(float), a_min=0, a_max=None))
    cox_a_features = [
        "LOG_CARGA_ACUM_EP",
        "MEDIA_CARGA_30D",
        "FLAG_EMENDA_7D",
        "FLAG_EMENDA_14D",
        "DIAS_DESDE_ULT_EMENDA",
    ]
    cox_a_metrics = run_cox(ep_cox, cox_a_features, cox_a_path, title="COX_A_HISTORICO_SEM_DOSADORAS", penalizer=0.15)

    # Cox A (historico completo, com temperatura operacional simulada)
    cox_a_temp_features = [
        "LOG_CARGA_ACUM_EP",
        "MEDIA_CARGA_30D",
        "FLAG_EMENDA_7D",
        "FLAG_EMENDA_14D",
        "DIAS_DESDE_ULT_EMENDA",
        "TEMP_PROCESSO_MEDIA_EP",
    ]
    cox_a_temp_metrics = run_cox(
        ep_cox,
        cox_a_temp_features,
        cox_a_temp_path,
        title="COX_A_HISTORICO_COM_TEMP_PROCESSO",
        penalizer=0.18,
    )

    # Cox A (historico completo, com clima)
    cox_a_clima_features = [
        "LOG_CARGA_ACUM_EP",
        "MEDIA_CARGA_30D",
        "FLAG_EMENDA_7D",
        "FLAG_EMENDA_14D",
        "DIAS_DESDE_ULT_EMENDA",
        "TEMP_PROCESSO_MEDIA_EP",
        "CLIMA_TEMP_MEDIA_EP",
        "CLIMA_UMIDADE_MEDIA_EP",
    ]
    cox_a_clima_metrics = run_cox(
        ep_cox,
        cox_a_clima_features,
        cox_a_clima_path,
        title="COX_A_HISTORICO_COM_CLIMA",
        penalizer=0.20,
    )

    # Cox B (2021+, com dosadoras)
    ep_cox_b = ep_cox[ep_cox["ANO_INICIO"] >= 2021].copy()
    cox_b_candidates = [
        "LOG_CARGA_ACUM_EP",
        "MEDIA_CARGA_30D",
        "DOS_VAZAO_MEDIA_EP",
        "DOS_VELOC_MEDIA_EP",
        "DOS_COBERTURA_MEDIA_EP",
        "CLIMA_TEMP_MEDIA_EP",
        "CLIMA_UMIDADE_MEDIA_EP",
        "TEMP_PROCESSO_MEDIA_EP",
    ]
    min_non_null = max(4, int(np.ceil(len(ep_cox_b) * 0.60))) if len(ep_cox_b) > 0 else 4
    cox_b_features = [c for c in cox_b_candidates if c in ep_cox_b.columns and ep_cox_b[c].notna().sum() >= min_non_null]
    if len(cox_b_features) == 0:
        cox_b_features = ["LOG_CARGA_ACUM_EP", "MEDIA_CARGA_30D"]

    cox_b_metrics = run_cox(
        ep_cox_b,
        cox_b_features,
        cox_b_path,
        title="COX_B_2021_COM_DOSADORAS",
        penalizer=0.25,
        min_rows=6,
    )
    if cox_b_metrics.get("status") != "OK":
        fallback_sets = [
            ["LOG_CARGA_ACUM_EP", "MEDIA_CARGA_30D", "DOS_VAZAO_MEDIA_EP"],
            ["LOG_CARGA_ACUM_EP", "MEDIA_CARGA_30D"],
            ["LOG_CARGA_ACUM_EP"],
        ]
        for fb in fallback_sets:
            fb_valid = [c for c in fb if c in ep_cox_b.columns and ep_cox_b[c].notna().sum() >= max(3, int(np.ceil(len(ep_cox_b) * 0.50)))]
            if len(fb_valid) == 0:
                continue
            cox_b_metrics = run_cox(
                ep_cox_b,
                fb_valid,
                cox_b_path,
                title="COX_B_2021_COM_DOSADORAS",
                penalizer=0.35,
                min_rows=6,
            )
            if cox_b_metrics.get("status") == "OK":
                break

    # sample around first event in 2021+, fallback first event
    work_daily = daily_df.copy()
    work_daily["DATA_DT"] = parse_date_series(work_daily["DATA_DIA"])
    events = work_daily[work_daily["EVENT_TROCA"] == 1].sort_values("DATA_DT")
    pivot = events[events["DATA_DT"] >= pd.Timestamp("2021-01-01")]
    if pivot.empty:
        pivot = events
    if not pivot.empty:
        d0 = pivot.iloc[0]["DATA_DT"]
        window = work_daily[(work_daily["DATA_DT"] >= (d0 - pd.Timedelta(days=60))) & (work_daily["DATA_DT"] <= (d0 + pd.Timedelta(days=10)))].copy()
        sample_cols = [
            "EQUIPAMENTO",
            "DATA_DIA",
            "EPISODIO_ID",
            "T_DIAS",
            "EVENT_TROCA",
            "EVENT_RECUP",
            "EVENTO_LISTA_DIA",
            "PRODUCAO_TONELADAS_DIA_LAG1",
            "CARGA_TON_30D_MEAN_LAG1",
            "FLAG_RECUP_7D_LAG1",
            "FLAG_RECUP_14D_LAG1",
            "DOS_VAZAO_MEDIA_LAG1",
            "DOS_VELOC_MEDIA_LAG1",
            "DOS_COBERTURA_24H_PCT_LAG1",
            "CLIMA_TEMP_MEDIA_C_LAG1",
            "CLIMA_UMIDADE_REL_MEDIA_PCT_LAG1",
        ]
        sample_cols = [c for c in sample_cols if c in window.columns]
        window = window[sample_cols].sort_values("DATA_DIA")
        window.to_csv(event_sample_path, sep=";", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["EQUIPAMENTO", "DATA_DIA"]).to_csv(event_sample_path, sep=";", index=False, encoding="utf-8-sig")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"modelo01_train_{equip.lower()}_{stamp}.log"
    log_lines = [
        "pipeline=modelo01_train",
        f"equipamento={equip}",
        f"input_dataset={ds_path}",
        f"rows_daily={len(daily_df)}",
        f"episodes={ep_df['EPISODIO_ID'].nunique()}",
        f"events={int(ep_df['EVENT_OBSERVED'].sum())}",
        f"weibull_beta={weibull_metrics.get('beta', np.nan):.6f}",
        f"weibull_eta={weibull_metrics.get('eta', np.nan):.6f}",
        f"cox_a_status={cox_a_metrics.get('status')}",
        f"cox_a_temp_status={cox_a_temp_metrics.get('status')}",
        f"cox_a_clima_status={cox_a_clima_metrics.get('status')}",
        f"cox_b_status={cox_b_metrics.get('status')}",
        f"episode_lengths={episode_lengths_path}",
        f"coverage_report={cov_cov_path}",
        f"km_plot={km_plot_path}",
        f"weibull_results={weibull_path}",
        f"coxA_summary={cox_a_path}",
        f"coxA_temp_summary={cox_a_temp_path}",
        f"coxA_clima_summary={cox_a_clima_path}",
        f"coxB_summary={cox_b_path}",
        f"event_timeline_sample={event_sample_path}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"MODELO01_TRAIN_EQUIPAMENTO: {equip}")
    print(f"EPISODES: {ep_df['EPISODIO_ID'].nunique()}")
    print(f"EVENTS: {int(ep_df['EVENT_OBSERVED'].sum())}")
    print(f"WEIBULL_BETA: {weibull_metrics.get('beta', np.nan):.6f}")
    print(f"WEIBULL_ETA: {weibull_metrics.get('eta', np.nan):.6f}")
    print(f"COX_A_STATUS: {cox_a_metrics.get('status')}")
    print(f"COX_A_TEMP_STATUS: {cox_a_temp_metrics.get('status')}")
    print(f"COX_A_CLIMA_STATUS: {cox_a_clima_metrics.get('status')}")
    print(f"COX_B_STATUS: {cox_b_metrics.get('status')}")
    print(f"EPISODE_LENGTHS: {episode_lengths_path}")
    print(f"COVERAGE_REPORT: {cov_cov_path}")
    print(f"KM_PLOT: {km_plot_path}")
    print(f"WEIBULL_RESULTS: {weibull_path}")
    print(f"COXA_SUMMARY: {cox_a_path}")
    print(f"COXA_TEMP_SUMMARY: {cox_a_temp_path}")
    print(f"COXA_CLIMA_SUMMARY: {cox_a_clima_path}")
    print(f"COXB_SUMMARY: {cox_b_path}")
    print(f"EVENT_TIMELINE_SAMPLE: {event_sample_path}")
    print(f"LOG: {log_path}")


if __name__ == "__main__":
    main()
