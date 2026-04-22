from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MIN_EVENT_GAP_DAYS = 8


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "project_config.json").exists():
            return candidate
    raise FileNotFoundError("project_config.json nao encontrado")


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar modulo: {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str).str.strip(), errors="coerce")


def to_num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def fmt_pt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def fmt_num(value: float | int, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{float(value):.{digits}f}".replace(".", ",")


@dataclass
class ScenarioSummary:
    scenario: str
    episodes: int
    events: int
    censored: int
    mean_days: float
    median_days: float
    std_days: float
    min_days: float
    max_days: float


def rebuild_d02_daily(daily_df: pd.DataFrame, min_gap_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = daily_df.copy()
    work["DATA_DT"] = parse_date(work["DATA_DIA"])
    work = work.sort_values("DATA_DT").reset_index(drop=True)

    if work.empty:
        raise ValueError("Dataset D02 vazio")

    work["EVENT_TROCA_ORIGINAL"] = to_num(work["EVENT_TROCA"]).fillna(0).astype(int)
    work["EVENT_OBSERVED_ORIGINAL"] = to_num(work["EVENT_OBSERVED"]).fillna(0).astype(int)

    seq = 1
    ep_start = work.loc[0, "DATA_DT"]
    ep_ids: list[str] = []
    ep_nums: list[int] = []
    t_dias: list[int] = []
    valid_events: list[int] = []
    invalid_short: list[int] = []
    invalid_rows: list[dict] = []

    for _, row in work.iterrows():
        current_date = row["DATA_DT"]
        t = int((current_date - ep_start).days)
        raw_event = int(row["EVENT_TROCA_ORIGINAL"]) == 1
        valid_event = int(raw_event and t >= min_gap_days)
        invalid_event = int(raw_event and t < min_gap_days)

        ep_ids.append(f"D02_MIN{min_gap_days:02d}_EP_{seq:04d}")
        ep_nums.append(seq)
        t_dias.append(t)
        valid_events.append(valid_event)
        invalid_short.append(invalid_event)

        if invalid_event:
            invalid_rows.append(
                {
                    "EPISODIO_EM_CURSO": seq,
                    "DATA_EVENTO": row["DATA_DIA"],
                    "T_DIAS_NO_EVENTO": t,
                    "EVENTO_ORIGINAL": "SUBSTITUICAO_COMPLETA",
                    "REGRA_APLICADA": f"IGNORAR_EVENTO_COM_DURACAO_MENOR_QUE_{min_gap_days}_DIAS",
                }
            )

        if raw_event and valid_event:
            seq += 1
            ep_start = current_date

    work["EPISODIO_ID"] = ep_ids
    work["EPISODIO_NUM"] = ep_nums
    work["T_DIAS"] = t_dias
    work["EVENT_TROCA"] = valid_events
    work["EVENT_TROCA_INVALIDO_CURTO"] = invalid_short
    work["EVENT_OBSERVED"] = valid_events
    work["CENSORED"] = 0
    if int(work.iloc[-1]["EVENT_OBSERVED"]) == 0:
        work.loc[work.index[-1], "CENSORED"] = 1

    if "PRODUCAO_TONELADAS_DIA_LAG1" in work.columns:
        carga_src = to_num(work["PRODUCAO_TONELADAS_DIA_LAG1"]).fillna(0.0)
        work["CARGA_TON_ACUM_EP_LAG1"] = carga_src.groupby(work["EPISODIO_ID"]).cumsum()

    episode_rows: list[dict] = []
    for eid, g in work.groupby("EPISODIO_ID", sort=True):
        g = g.sort_values("DATA_DT")
        episode_rows.append(
            {
                "EQUIPAMENTO": "D02",
                "EPISODIO_ID": str(eid),
                "EPISODIO_NUM": int(g["EPISODIO_NUM"].iloc[0]),
                "DATA_INICIO": g["DATA_DIA"].iloc[0],
                "DATA_FIM": g["DATA_DIA"].iloc[-1],
                "DURACAO_DIAS": int(g["T_DIAS"].max()),
                "EVENTO_OBSERVADO": int(g["EVENT_OBSERVED"].max()),
                "CENSORED": int(g["CENSORED"].max()),
                "REPAROS_EP": int(to_num(g["EVENT_RECUP"]).fillna(0).sum()) if "EVENT_RECUP" in g.columns else 0,
                "CARGA_ACUM_EP_TON": float(to_num(g["PRODUCAO_TONELADAS_DIA"]).fillna(0.0).sum())
                if "PRODUCAO_TONELADAS_DIA" in g.columns
                else np.nan,
                "EVENTOS_BRUTOS_NO_EP": int(g["EVENT_TROCA_ORIGINAL"].sum()),
                "EVENTOS_INVALIDOS_NO_EP": int(g["EVENT_TROCA_INVALIDO_CURTO"].sum()),
            }
        )

    ep_df = pd.DataFrame(episode_rows).sort_values("EPISODIO_NUM").reset_index(drop=True)
    invalid_df = pd.DataFrame(invalid_rows)
    if invalid_df.empty:
        invalid_df = pd.DataFrame(
            columns=["EPISODIO_EM_CURSO", "DATA_EVENTO", "T_DIAS_NO_EVENTO", "EVENTO_ORIGINAL", "REGRA_APLICADA"]
        )
    return work, ep_df, invalid_df


def summarize_observed(name: str, ep_df: pd.DataFrame) -> ScenarioSummary:
    obs = ep_df[ep_df["EVENTO_OBSERVADO"] == 1].copy()
    if obs.empty:
        return ScenarioSummary(name, int(len(ep_df)), 0, int((ep_df["CENSORED"] == 1).sum()), np.nan, np.nan, np.nan, np.nan, np.nan)
    s = to_num(obs["DURACAO_DIAS"])
    return ScenarioSummary(
        scenario=name,
        episodes=int(len(ep_df)),
        events=int(len(obs)),
        censored=int((ep_df["CENSORED"] == 1).sum()),
        mean_days=float(s.mean()),
        median_days=float(s.median()),
        std_days=float(s.std(ddof=1)),
        min_days=float(s.min()),
        max_days=float(s.max()),
    )


def classify_episodes(ep_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ep = ep_df.copy()
    ep["DATA_FIM_DT"] = parse_date(ep["DATA_FIM"])
    ep["DURACAO_DIAS"] = to_num(ep["DURACAO_DIAS"])
    ep["CARGA_ACUM_EP_TON"] = to_num(ep["CARGA_ACUM_EP_TON"])

    obs = ep[ep["EVENTO_OBSERVADO"] == 1].copy()
    mean_days = float(obs["DURACAO_DIAS"].mean())
    std_days = float(obs["DURACAO_DIAS"].std(ddof=1))
    lower_normal = max(float(MIN_EVENT_GAP_DAYS), mean_days - std_days)
    upper_normal = mean_days + std_days
    outlier_high = mean_days + 2 * std_days
    outlier_low = max(float(MIN_EVENT_GAP_DAYS), mean_days - 2 * std_days)

    ep["CLASSIFICACAO"] = np.where(ep["EVENTO_OBSERVADO"] == 0, "Censurado", "")
    ep.loc[(ep["EVENTO_OBSERVADO"] == 1) & (ep["DURACAO_DIAS"] > outlier_high), "CLASSIFICACAO"] = "Outlier"
    ep.loc[(ep["EVENTO_OBSERVADO"] == 1) & (ep["DURACAO_DIAS"] < outlier_low), "CLASSIFICACAO"] = "Outlier"
    ep.loc[
        (ep["EVENTO_OBSERVADO"] == 1) & (ep["CLASSIFICACAO"] == "") & (ep["DURACAO_DIAS"] < lower_normal),
        "CLASSIFICACAO",
    ] = "Desgaste acelerado"
    ep.loc[
        (ep["EVENTO_OBSERVADO"] == 1) & (ep["CLASSIFICACAO"] == "") & (ep["DURACAO_DIAS"] > upper_normal),
        "CLASSIFICACAO",
    ] = "Acima da média"
    ep.loc[(ep["EVENTO_OBSERVADO"] == 1) & (ep["CLASSIFICACAO"] == ""), "CLASSIFICACAO"] = "Dentro da média"

    non_outlier = ep[(ep["EVENTO_OBSERVADO"] == 1) & (ep["CLASSIFICACAO"] != "Outlier")].copy()
    non_outlier = non_outlier.sort_values(["DURACAO_DIAS", "CARGA_ACUM_EP_TON"], ascending=[False, False]).reset_index(drop=True)
    if len(non_outlier) >= 1:
        best1 = non_outlier.loc[0, "EPISODIO_ID"]
        ep.loc[ep["EPISODIO_ID"] == best1, "CLASSIFICACAO"] = "1ª melhor prática"
    if len(non_outlier) >= 2:
        best2 = non_outlier.loc[1, "EPISODIO_ID"]
        ep.loc[ep["EPISODIO_ID"] == best2, "CLASSIFICACAO"] = "2ª melhor prática"

    limits = {
        "mean_days": mean_days,
        "std_days": std_days,
        "lower_normal": lower_normal,
        "upper_normal": upper_normal,
        "outlier_low": outlier_low,
        "outlier_high": outlier_high,
    }
    return ep, limits


def plot_classificacao(
    ep_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
    out_png: Path,
    limits: dict[str, float],
    show_counts_in_legend: bool = False,
    annotate_episode_nums: bool = False,
) -> None:
    fig_height = 7.4 if show_counts_in_legend else 6.9
    fig, ax = plt.subplots(figsize=(11.8, fig_height))

    colors = {
        "Dentro da média": "#8c8c8c",
        "Acima da média": "#0a7f2e",
        "Desgaste acelerado": "#d62728",
        "Outlier": "#111111",
        "1ª melhor prática": "#f2c200",
        "2ª melhor prática": "#ffffff",
        "Censurado": "#111111",
    }
    markers = {
        "Dentro da média": "o",
        "Acima da média": "o",
        "Desgaste acelerado": "^",
        "Outlier": "o",
        "1ª melhor prática": "*",
        "2ª melhor prática": "*",
        "Censurado": "x",
    }
    sizes = {
        "Dentro da média": 28,
        "Acima da média": 28,
        "Desgaste acelerado": 38,
        "Outlier": 52,
        "1ª melhor prática": 110,
        "2ª melhor prática": 110,
        "Censurado": 42,
    }

    obs = ep_df[ep_df["EVENTO_OBSERVADO"] == 1].copy().sort_values("DATA_FIM_DT")
    ax.axhspan(
        limits["lower_normal"],
        limits["upper_normal"],
        facecolor="#9fd3c7",
        alpha=0.12,
        zorder=0,
    )
    if len(obs) > 1:
        ax.plot(obs["DATA_FIM_DT"], obs["DURACAO_DIAS"], color="#d9d9d9", linewidth=1.0, zorder=1)

    order = [
        "Dentro da média",
        "Acima da média",
        "Desgaste acelerado",
        "Outlier",
        "1ª melhor prática",
        "2ª melhor prática",
        "Censurado",
    ]
    for label in order:
        g = ep_df[ep_df["CLASSIFICACAO"] == label].copy()
        if g.empty:
            continue
        legend_label = label
        if show_counts_in_legend:
            legend_label = f"{label} (n={len(g)})"
        edge = "#222222" if label in {"1ª melhor prática", "2ª melhor prática", "Outlier"} else None
        ax.scatter(
            g["DATA_FIM_DT"],
            g["DURACAO_DIAS"],
            label=legend_label,
            color=colors[label],
            marker=markers[label],
            s=sizes[label],
            linewidths=1.0 if label in {"Outlier", "1ª melhor prática", "2ª melhor prática", "Censurado"} else 0.6,
            edgecolors=edge,
            zorder=3,
        )

    if not invalid_df.empty:
        invalid_plot = invalid_df.copy()
        invalid_plot["DATA_DT"] = parse_date(invalid_plot["DATA_EVENTO"])
        invalid_label = "Evento curto invalidado"
        if show_counts_in_legend:
            invalid_label = f"Evento curto invalidado (n={len(invalid_plot)})"
        ax.scatter(
            invalid_plot["DATA_DT"],
            invalid_plot["T_DIAS_NO_EVENTO"],
            label=invalid_label,
            color="#ff8c00",
            marker="s",
            s=36,
            edgecolors="#7a3e00",
            linewidths=0.6,
            zorder=4,
        )

    ax.axhline(
        limits["mean_days"],
        color="#3b6ea5",
        linestyle="--",
        linewidth=1.1,
        alpha=0.9,
        label="Média",
    )
    ax.axhline(
        limits["lower_normal"],
        color="#c96f00",
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
        label="Limite inferior",
    )
    ax.axhline(
        limits["upper_normal"],
        color="#2a9d8f",
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
        label="Limite superior",
    )

    # Numeric labels for reference lines on the main axis.
    line_labels = [
        (limits["mean_days"], "#3b6ea5", f"Média = {fmt_pt(float(limits['mean_days']), 1)} dias"),
        (
            limits["lower_normal"],
            "#c96f00",
            f"Limite inferior = {fmt_pt(float(limits['lower_normal']), 1)} dias",
        ),
        (
            limits["upper_normal"],
            "#2a9d8f",
            f"Limite superior = {fmt_pt(float(limits['upper_normal']), 1)} dias",
        ),
    ]
    x_text = ep_df["DATA_FIM_DT"].max()
    for y_val, color, label in line_labels:
        ax.annotate(
            label,
            xy=(x_text, y_val),
            xytext=(-6, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.85},
            zorder=6,
        )

    max_days = float(ep_df["DURACAO_DIAS"].astype(float).max())
    ax.set_ylim(-5, max_days + 40)
    ax.set_title("Classificação visual dos episódios da D02 com regra mínima de 8 dias")
    ax.set_xlabel("Data de encerramento do episódio")
    ax.set_ylabel("Duração do episódio (dias)")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.25)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3 if show_counts_in_legend else 4,
        frameon=True,
        fontsize=8,
    )
    if annotate_episode_nums:
        ann_df = ep_df.copy().sort_values("DATA_FIM_DT")
        low_cycle = [(-18, 12), (6, 18), (18, 10), (28, 18), (-22, -10), (8, -14), (22, -18)]
        cluster_anchor = None
        cluster_idx = 0
        for _, row in ann_df.iterrows():
            ep_num = int(row["EPISODIO_NUM"])
            label_txt = f"{ep_num}*" if str(row["CLASSIFICACAO"]) == "Outlier" else str(ep_num)
            y_val = float(row["DURACAO_DIAS"])
            dt = row["DATA_FIM_DT"]
            if y_val <= 100:
                if cluster_anchor is None or abs((dt - cluster_anchor).days) > 120:
                    cluster_anchor = dt
                    cluster_idx = 0
                else:
                    cluster_idx += 1
                x_offset, y_offset = low_cycle[cluster_idx % len(low_cycle)]
                ha = "left" if x_offset >= 0 else "right"
            elif y_val > 0.9 * max_days:
                x_offset, y_offset = (6, -12)
                ha = "left"
            else:
                x_offset = 4
                y_offset = 10 if ep_num % 2 else -12
                ha = "left"
            va = "bottom" if y_offset > 0 else "top"
            ax.annotate(
                label_txt,
                xy=(dt, row["DURACAO_DIAS"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=7,
                color="#222222",
                ha=ha,
                va=va,
                zorder=5,
            )
    if show_counts_in_legend:
        count_lines: list[str] = []
        for label in order:
            count = int((ep_df["CLASSIFICACAO"] == label).sum())
            if count > 0:
                count_lines.append(f"{label}: n={count}")
        if not invalid_df.empty:
            count_lines.append(f"Evento curto invalidado: n={len(invalid_df)}")
        count_text = " | ".join(count_lines)
        fig.text(
            0.5,
            0.02,
            count_text,
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#b5b5b5"},
        )
    fig.subplots_adjust(bottom=0.36 if show_counts_in_legend else 0.24, left=0.07, right=0.98, top=0.92)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def build_clean_dataset(daily_min8: pd.DataFrame, classified_eps: pd.DataFrame) -> pd.DataFrame:
    outlier_ids = set(
        classified_eps[
            (classified_eps["EVENTO_OBSERVADO"] == 1) & (classified_eps["CLASSIFICACAO"] == "Outlier")
        ]["EPISODIO_ID"].tolist()
    )
    clean = daily_min8[~daily_min8["EPISODIO_ID"].isin(outlier_ids)].copy()
    clean = clean.sort_values("DATA_DT").reset_index(drop=True)
    return clean


def prepare_original_episode_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    work = daily_df.copy()
    work["DATA_DT"] = parse_date(work["DATA_DIA"])
    rows: list[dict] = []
    for eid, g in work.groupby("EPISODIO_ID", sort=True):
        g = g.sort_values("DATA_DT")
        rows.append(
            {
                "EPISODIO_ID": str(eid),
                "EPISODIO_NUM": int(g["EPISODIO_NUM"].iloc[0]),
                "DATA_INICIO": g["DATA_DIA"].iloc[0],
                "DATA_FIM": g["DATA_DIA"].iloc[-1],
                "DURACAO_DIAS": float(to_num(g["T_DIAS"]).max()),
                "EVENTO_OBSERVADO": int(to_num(g["EVENT_OBSERVED"]).max()),
                "CENSORED": int(to_num(g["CENSORED"]).max()),
            }
        )
    return pd.DataFrame(rows).sort_values("EPISODIO_NUM").reset_index(drop=True)


def run_model_recalc(root: Path, out_dir: Path, clean_daily: pd.DataFrame) -> dict[str, str]:
    train_mod = load_module(root / "scripts" / "python" / "06_train_survival_models.py", "train_survival_mod")
    compare_mod = load_module(root / "scripts" / "python" / "16_compare_survival_models_d02.py", "compare_models_mod")

    ds_path = out_dir / "d02_survival_dataset_min8_sem_outliers.csv"
    ep_df = train_mod.prepare_episode_features(clean_daily, "D02")

    coverage_csv = out_dir / "d02_covariate_coverage_min8_sem_outliers.csv"
    km_png = out_dir / "d02_km_min8_sem_outliers.png"
    weibull_txt = out_dir / "d02_weibull_min8_sem_outliers.txt"
    cox_a_txt = out_dir / "d02_coxA_min8_sem_outliers.txt"

    train_mod.build_covariate_coverage_report(clean_daily, coverage_csv)
    train_mod.create_km_plot(ep_df, "D02 - sem outliers", km_png)
    weibull = train_mod.fit_weibull(ep_df, "D02", weibull_txt)

    ep_cox = ep_df.copy()
    ep_cox["LOG_CARGA_ACUM_EP"] = np.log1p(np.clip(ep_cox["CARGA_ACUM_EP"].astype(float), a_min=0, a_max=None))
    cox_features = [
        "LOG_CARGA_ACUM_EP",
        "MEDIA_CARGA_30D",
        "FLAG_EMENDA_7D",
        "FLAG_EMENDA_14D",
        "DIAS_DESDE_ULT_EMENDA",
    ]
    cox = train_mod.run_cox(
        ep_cox,
        cox_features,
        cox_a_txt,
        title="COX_A_D02_MIN8_SEM_OUTLIERS",
        penalizer=0.15,
    )

    compare_ep = compare_mod.build_episode_dataframe(clean_daily, equip="D02")
    folds = compare_mod.build_temporal_folds(len(compare_ep))

    compare_metrics = out_dir / "d02_model_compare_metrics_min8_sem_outliers.csv"
    compare_decision = out_dir / "d02_model_compare_decision_min8_sem_outliers.txt"
    model_rows: list[dict] = []

    if len(folds) > 0:
        specs = [
            compare_mod.ModelSpec(
                name="CoxPH",
                fitter_factory=lambda: compare_mod.CoxPHFitter(penalizer=0.15),
                features=["CARGA_ACUM_EP", "MEDIA_CARGA_30D", "FLAG_EMENDA_7D", "DIAS_DESDE_ULT_EMENDA"],
                penalizer=0.15,
            ),
            compare_mod.ModelSpec(
                name="WeibullAFT",
                fitter_factory=lambda: compare_mod.WeibullAFTFitter(),
                features=["CARGA_ACUM_EP", "MEDIA_CARGA_30D", "FLAG_EMENDA_7D", "DIAS_DESDE_ULT_EMENDA"],
            ),
            compare_mod.ModelSpec(
                name="LogNormalAFT",
                fitter_factory=lambda: compare_mod.LogNormalAFTFitter(),
                features=["CARGA_ACUM_EP", "MEDIA_CARGA_30D", "FLAG_EMENDA_7D", "DIAS_DESDE_ULT_EMENDA"],
            ),
            compare_mod.ModelSpec(
                name="LogLogisticAFT",
                fitter_factory=lambda: compare_mod.LogLogisticAFTFitter(),
                features=["CARGA_ACUM_EP", "MEDIA_CARGA_30D", "FLAG_EMENDA_7D", "DIAS_DESDE_ULT_EMENDA"],
            ),
        ]
        for fold_id, train_idx, test_idx in folds:
            train = compare_ep.iloc[train_idx].copy()
            test = compare_ep.iloc[test_idx].copy()
            t_test = compare_mod.to_num(test["DURATION_DIAS"]).values
            e_test = compare_mod.to_num(test["EVENT_OBSERVED"]).fillna(0).astype(int).values
            for spec in specs:
                row = {
                    "MODEL": spec.name,
                    "FOLD": fold_id,
                    "STATUS": "OK",
                    "ERROR": "",
                    "C_INDEX": np.nan,
                }
                try:
                    model = compare_mod.fit_model(spec, train)
                    x_test = compare_mod.prepare_test_matrix(spec, train, test)
                    score = compare_mod.predict_risk_score(spec, model, x_test)
                    if np.isfinite(score).sum() >= 3:
                        row["C_INDEX"] = float(compare_mod.concordance_index(t_test, score, e_test))
                except Exception as exc:
                    row["STATUS"] = "FAIL"
                    row["ERROR"] = f"{type(exc).__name__}: {exc}"
                model_rows.append(row)

    metrics_df = pd.DataFrame(model_rows)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=["MODEL", "FOLD", "STATUS", "ERROR", "C_INDEX"])
    metrics_df.to_csv(compare_metrics, sep=";", index=False, encoding="utf-8-sig")

    decision_lines = [
        "RECALCULO_D02_MIN8_SEM_OUTLIERS",
        f"DATASET={ds_path}",
        f"EPISODIOS={int(ep_df['EPISODIO_ID'].nunique())}",
        f"EVENTOS={int(ep_df['EVENT_OBSERVED'].sum())}",
        f"WEIBULL_BETA={weibull.get('beta', np.nan):.6f}" if np.isfinite(weibull.get("beta", np.nan)) else "WEIBULL_BETA=",
        f"WEIBULL_ETA={weibull.get('eta', np.nan):.6f}" if np.isfinite(weibull.get("eta", np.nan)) else "WEIBULL_ETA=",
        f"COX_A_STATUS={cox.get('status')}",
        f"COX_A_C_INDEX={cox.get('c_index', np.nan):.6f}" if np.isfinite(cox.get("c_index", np.nan)) else "COX_A_C_INDEX=",
    ]
    if not metrics_df.empty and (metrics_df["STATUS"] == "OK").any():
        ok = metrics_df[metrics_df["STATUS"] == "OK"].copy()
        res = ok.groupby("MODEL", as_index=False).agg(C_INDEX_MEAN=("C_INDEX", "mean"), FOLDS=("MODEL", "count"))
        res = res.sort_values("C_INDEX_MEAN", ascending=False).reset_index(drop=True)
        decision_lines.extend(["", "[COMPARE_MODELS_C_INDEX]"])
        for _, r in res.iterrows():
            decision_lines.append(f"{r['MODEL']}: C_INDEX_MEDIO={float(r['C_INDEX_MEAN']):.6f}; FOLDS={int(r['FOLDS'])}")
    compare_decision.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    return {
        "coverage_csv": str(coverage_csv),
        "km_png": str(km_png),
        "weibull_txt": str(weibull_txt),
        "cox_a_txt": str(cox_a_txt),
        "compare_metrics": str(compare_metrics),
        "compare_decision": str(compare_decision),
    }


def write_report(
    out_path: Path,
    current_summary: ScenarioSummary,
    min8_summary: ScenarioSummary,
    clean_summary: ScenarioSummary,
    invalid_df: pd.DataFrame,
    limits: dict[str, float],
    classified_eps: pd.DataFrame,
    generated: dict[str, str],
) -> None:
    outliers = classified_eps[(classified_eps["EVENTO_OBSERVADO"] == 1) & (classified_eps["CLASSIFICACAO"] == "Outlier")].copy()
    best = classified_eps[classified_eps["CLASSIFICACAO"].isin(["1ª melhor prática", "2ª melhor prática"])].copy()

    lines = [
        "RECALCULO_D02_COM_EVENTOS_CURTOS_INVALIDADOS_E_OUTLIERS",
        "",
        f"REGRA_1=Substituicao completa so encerra episodio se ocorrer com pelo menos {MIN_EVENT_GAP_DAYS} dias desde a ultima substituicao valida.",
        "REGRA_2=Eventos curtos invalidos permanecem rastreaveis no historico, mas nao reiniciam o ciclo analitico.",
        "REGRA_3=Outliers altos sao classificados por duracao acima de media + 2 desvios padrao da base ja ajustada pela regra minima.",
        "REGRA_4=Primeira e segunda melhores praticas sao escolhidas entre episodios validos nao classificados como outlier.",
        "",
        "[RESUMO_COMPARATIVO]",
        f"ATUAL: episodios={current_summary.episodes}; eventos={current_summary.events}; censurados={current_summary.censored}; media={fmt_num(current_summary.mean_days)}; mediana={fmt_num(current_summary.median_days)}; min={fmt_num(current_summary.min_days)}; max={fmt_num(current_summary.max_days)}",
        f"MIN8: episodios={min8_summary.episodes}; eventos={min8_summary.events}; censurados={min8_summary.censored}; media={fmt_num(min8_summary.mean_days)}; mediana={fmt_num(min8_summary.median_days)}; min={fmt_num(min8_summary.min_days)}; max={fmt_num(min8_summary.max_days)}",
        f"MIN8_SEM_OUTLIERS: episodios={clean_summary.episodes}; eventos={clean_summary.events}; censurados={clean_summary.censored}; media={fmt_num(clean_summary.mean_days)}; mediana={fmt_num(clean_summary.median_days)}; min={fmt_num(clean_summary.min_days)}; max={fmt_num(clean_summary.max_days)}",
        "",
        "[LIMITES_CLASSIFICACAO]",
        f"MEDIA_DIAS={fmt_num(limits['mean_days'])}",
        f"DESVIO_PADRAO_DIAS={fmt_num(limits['std_days'])}",
        f"LIMITE_INFERIOR_NORMAL={fmt_num(limits['lower_normal'])}",
        f"LIMITE_SUPERIOR_NORMAL={fmt_num(limits['upper_normal'])}",
        f"LIMIAR_OUTLIER_ALTO={fmt_num(limits['outlier_high'])}",
        "",
        "[EVENTOS_CURTOS_INVALIDADOS]",
    ]

    if invalid_df.empty:
        lines.append("Nenhum evento curto invalidado.")
    else:
        for _, r in invalid_df.iterrows():
            lines.append(f"{r['DATA_EVENTO']}: evento ignorado com {int(r['T_DIAS_NO_EVENTO'])} dias")

    lines.extend(["", "[OUTLIERS_CLASSIFICADOS]"])
    if outliers.empty:
        lines.append("Nenhum outlier classificado.")
    else:
        for _, r in outliers.iterrows():
            lines.append(
                f"Episodio {int(r['EPISODIO_NUM'])}: fim={r['DATA_FIM']}; duracao={int(r['DURACAO_DIAS'])} dias; carga={fmt_num(float(r['CARGA_ACUM_EP_TON'])/1000.0)} kton"
            )

    lines.extend(["", "[MELHORES_PRATICAS_VALIDAS]"])
    if best.empty:
        lines.append("Sem melhores praticas validas.")
    else:
        for _, r in best.sort_values("CLASSIFICACAO").iterrows():
            lines.append(
                f"{r['CLASSIFICACAO']}: episodio {int(r['EPISODIO_NUM'])}; fim={r['DATA_FIM']}; duracao={int(r['DURACAO_DIAS'])} dias; carga={fmt_num(float(r['CARGA_ACUM_EP_TON'])/1000.0)} kton"
            )

    lines.extend(
        [
            "",
            "[ARQUIVOS_GERADOS]",
            *[f"{k}={v}" for k, v in generated.items()],
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = find_project_root(Path(__file__).resolve().parent)
    cfg = load_cfg(root / "project_config.json")
    logs_dir = root / cfg["folders"]["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    src = root / '02_model_input' / 'modelo01' / 'modelo01_d02_survival_dataset.csv'
    if not src.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {src}")

    out_dir = root / '02_model_input' / 'modelo01_d02_min8_outliers'
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(src, sep=";", encoding="utf-8-sig", low_memory=False)

    original_ep = prepare_original_episode_summary(daily)
    current_summary = summarize_observed("ATUAL", original_ep)

    daily_min8, ep_min8, invalid_df = rebuild_d02_daily(daily, min_gap_days=MIN_EVENT_GAP_DAYS)
    min8_summary = summarize_observed("MIN8", ep_min8)
    classified_eps, limits = classify_episodes(ep_min8)
    clean_daily = build_clean_dataset(daily_min8, classified_eps)
    clean_ep = classified_eps[~classified_eps["CLASSIFICACAO"].eq("Outlier")].copy().reset_index(drop=True)
    clean_summary = summarize_observed("MIN8_SEM_OUTLIERS", clean_ep)

    # Persist core tables
    daily_min8.to_csv(out_dir / "d02_survival_dataset_min8.csv", sep=";", index=False, encoding="utf-8-sig")
    clean_daily.to_csv(out_dir / "d02_survival_dataset_min8_sem_outliers.csv", sep=";", index=False, encoding="utf-8-sig")
    ep_min8.to_csv(out_dir / "d02_episode_summary_min8.csv", sep=";", index=False, encoding="utf-8-sig")
    classified_eps.to_csv(out_dir / "d02_episode_classificacao_min8.csv", sep=";", index=False, encoding="utf-8-sig")
    invalid_df.to_csv(out_dir / "d02_eventos_curtos_invalidados.csv", sep=";", index=False, encoding="utf-8-sig")

    class_png = out_dir / "d02_classificacao_visual_min8.png"
    plot_classificacao(classified_eps, invalid_df, class_png, limits)

    generated = {
        "src_dataset": str(src),
        "daily_min8": str(out_dir / "d02_survival_dataset_min8.csv"),
        "daily_min8_clean": str(out_dir / "d02_survival_dataset_min8_sem_outliers.csv"),
        "episode_summary_min8": str(out_dir / "d02_episode_summary_min8.csv"),
        "episode_classificacao_min8": str(out_dir / "d02_episode_classificacao_min8.csv"),
        "eventos_curtos_invalidados": str(out_dir / "d02_eventos_curtos_invalidados.csv"),
        "fig_classificacao": str(class_png),
    }
    generated.update(run_model_recalc(root, out_dir, clean_daily))

    report_path = out_dir / "d02_recalculo_outliers_relatorio.txt"
    write_report(
        report_path,
        current_summary=current_summary,
        min8_summary=min8_summary,
        clean_summary=clean_summary,
        invalid_df=invalid_df,
        limits=limits,
        classified_eps=classified_eps,
        generated=generated,
    )
    generated["report"] = str(report_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"d02_recalculo_outliers_{stamp}.log"
    log_path.write_text(
        "\n".join(
            [
                "pipeline=d02_recalculo_outliers",
                f"src_dataset={src}",
                f"output_dir={out_dir}",
                f"episodes_original={current_summary.episodes}",
                f"episodes_min8={min8_summary.episodes}",
                f"episodes_clean={clean_summary.episodes}",
                f"events_original={current_summary.events}",
                f"events_min8={min8_summary.events}",
                f"events_clean={clean_summary.events}",
                f"report={report_path}",
                f"fig_classificacao={class_png}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"OUTPUT_DIR: {out_dir}")
    print(f"RELATORIO: {report_path}")
    print(f"FIGURA_CLASSIFICACAO: {class_png}")
    print(f"EPISODIOS_ATUAL: {current_summary.episodes}")
    print(f"EPISODIOS_MIN8: {min8_summary.episodes}")
    print(f"EPISODIOS_MIN8_SEM_OUTLIERS: {clean_summary.episodes}")
    print(f"EVENTOS_ATUAL: {current_summary.events}")
    print(f"EVENTOS_MIN8: {min8_summary.events}")
    print(f"EVENTOS_MIN8_SEM_OUTLIERS: {clean_summary.events}")
    print(f"LOG: {log_path}")


if __name__ == "__main__":
    main()
