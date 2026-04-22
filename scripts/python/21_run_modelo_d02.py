from __future__ import annotations

import json
import sys
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def find_root(start: Path) -> Path:
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


def to_num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def format_decimal_pt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def plot_trocas_por_ano(valid_daily: pd.DataFrame, out_png: Path) -> None:
    work = valid_daily.copy()
    work["DATA_DT"] = pd.to_datetime(work["DATA_DIA"], errors="coerce")
    ev = work[work["EVENT_TROCA"] == 1].copy()
    trocas = ev.groupby(ev["DATA_DT"].dt.year).size().reset_index(name="TROCAS")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(trocas["DATA_DT"].astype(int).astype(str), trocas["TROCAS"], color="#1f77b4")
    ax.set_title("Distribuição anual das substituições completas válidas na D02")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Número de substituições")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, float(trocas["TROCAS"].max()) + 0.8)
    for bar, value in zip(bars, trocas["TROCAS"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_duration_hist(ep_df: pd.DataFrame, out_png: Path) -> None:
    obs = ep_df[ep_df["EVENTO_OBSERVADO"] == 1].copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(obs["DURACAO_DIAS"].astype(float), bins=18, color="#17becf", edgecolor="black", alpha=0.85)
    ax.set_title("Distribuição da duração dos episódios modelados na D02")
    ax.set_xlabel("Duração do episódio (dias)")
    ax.set_ylabel("Frequência")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_weibull_vs_km(duration: pd.Series, events: pd.Series, wf, out_png: Path) -> None:
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter()
    km.fit(duration, event_observed=events, label="Curva observada (Kaplan-Meier)")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    km.plot_survival_function(ax=ax, ci_show=False)
    x = np.linspace(0, max(1.0, float(duration.max())), 300)
    y = wf.survival_function_at_times(x).values
    ax.plot(x, y, color="#d62728", linewidth=2.0, label="Curva ajustada (Weibull)")
    ax.set_title("Curva observada e ajuste Weibull na D02")
    ax.set_xlabel("Dias no episódio")
    ax.set_ylabel("Probabilidade de o episódio continuar sem substituição")
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_cox_a_only(cox_a: dict, out_png: Path) -> None:
    rows = cox_a.get("summary", pd.DataFrame()).copy()
    if cox_a.get("status") != "OK" or len(rows) == 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "Sem ajuste Cox A válido", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    label_map = {
        "LOG_CARGA_ACUM_EP": "Carga acumulada do episódio",
        "MEDIA_CARGA_30D": "Carga média nos últimos 30 dias",
        "FLAG_EMENDA_7D": "Emenda nos últimos 7 dias",
        "FLAG_EMENDA_14D": "Emenda nos últimos 14 dias",
        "DIAS_DESDE_ULT_EMENDA": "Dias desde a última emenda",
    }
    rows["LABEL"] = rows["COVARIAVEL"].astype(str).map(label_map).fillna(rows["COVARIAVEL"].astype(str))
    rows = rows.sort_values("exp(coef)", ascending=True).reset_index(drop=True)
    y = np.arange(len(rows))
    hr = rows["exp(coef)"].astype(float).values
    lo = rows["exp(coef) lower 95%"].astype(float).values
    hi = rows["exp(coef) upper 95%"].astype(float).values

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(rows) + 2)))
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", color="#1f77b4", ecolor="#7f7f7f", capsize=3)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["LABEL"].tolist())
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.50, 1.00, 2.00])
    ax.set_xticklabels(["0,25", "0,50", "1,00", "2,00"])
    ax.set_xlabel("Razão de risco (1 = sem efeito; escala log)")
    ax.set_title("Efeito estimado das variáveis sobre o risco de substituição")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_model_compare(compare_mod, ep: pd.DataFrame, out_model_dir: Path, out_plot_dir: Path) -> dict[str, str]:
    folds = compare_mod.build_temporal_folds(len(ep))
    if len(folds) == 0:
        raise ValueError("Amostra insuficiente para validacao temporal do modelo02")

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

    fold_rows: list[dict] = []
    calib_points_rows: list[dict] = []
    calib_sample_rows: list[dict] = []
    horizons = [180, 365]

    for fold_id, train_idx, test_idx in folds:
        train = ep.iloc[train_idx].copy()
        test = ep.iloc[test_idx].copy()
        t_test = compare_mod.to_num(test["DURATION_DIAS"]).values
        e_test = compare_mod.to_num(test["EVENT_OBSERVED"]).fillna(0).astype(int).values

        for spec in specs:
            row = {
                "MODEL": spec.name,
                "FOLD": fold_id,
                "TRAIN_ROWS": int(len(train)),
                "TEST_ROWS": int(len(test)),
                "TRAIN_EVENTS": int(compare_mod.to_num(train["EVENT_OBSERVED"]).fillna(0).sum()),
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
                model = compare_mod.fit_model(spec, train)
                x_test = compare_mod.prepare_test_matrix(spec, train, test)
                score = compare_mod.predict_risk_score(spec, model, x_test)
                if np.isfinite(score).sum() >= 3:
                    row["C_INDEX"] = float(compare_mod.concordance_index(t_test, score, e_test))

                for hz in horizons:
                    p_event = compare_mod.predict_event_prob_at_horizon(model, x_test, hz)
                    brier, mae, point_rows, sample_rows = compare_mod.eval_calibration(
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
    out_folds = out_model_dir / "modelo02_d02_model_compare_folds.csv"
    out_metrics = out_model_dir / "modelo02_d02_model_compare_metrics.csv"
    out_cond = out_model_dir / "modelo02_d02_model_compare_conditions.csv"
    out_calib_points = out_model_dir / "modelo02_d02_model_compare_calibration_points.csv"
    out_calib_samples = out_model_dir / "modelo02_d02_model_compare_calibration_samples.csv"
    out_decision = out_model_dir / "modelo02_d02_model_compare_decision.txt"
    folds_df.to_csv(out_folds, sep=";", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_metrics, sep=";", index=False, encoding="utf-8-sig")
    cond_df.to_csv(out_cond, sep=";", index=False, encoding="utf-8-sig")
    calib_points_df.to_csv(out_calib_points, sep=";", index=False, encoding="utf-8-sig")
    calib_sample_df.to_csv(out_calib_samples, sep=";", index=False, encoding="utf-8-sig")

    # Plot C-index
    cidx = metrics_df.sort_values("C_INDEX_MEAN", ascending=False).copy()
    cidx["MODEL_LABEL"] = cidx["MODEL"].map(compare_mod.model_display_name)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(cidx["MODEL_LABEL"], cidx["C_INDEX_MEAN"], yerr=cidx["C_INDEX_STD"].fillna(0.0), capsize=4)
    label_tops: list[float] = []
    for b, (_, row) in zip(bars, cidx.iterrows()):
        model = str(row["MODEL"])
        if model == "CoxPH":
            b.set_color("#1f77b4")
        else:
            b.set_color("#8a8a8a")
        cstd = float(row["C_INDEX_STD"]) if pd.notna(row["C_INDEX_STD"]) else 0.0
        label_y = float(row["C_INDEX_MEAN"]) + cstd + 0.012
        label_tops.append(label_y)
        ax.text(
            b.get_x() + b.get_width() / 2,
            label_y,
            compare_mod.format_decimal_pt(float(row["C_INDEX_MEAN"]), 3),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    upper = max(label_tops) + 0.02 if label_tops else 1.05
    ax.set_ylim(0.5, max(1.05, upper))
    ax.set_ylabel("C-index médio (quanto maior, melhor)")
    ax.set_title("Comparação entre modelos por capacidade de discriminação")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_cindex = out_plot_dir / "modelo02_model_compare_cindex.png"
    fig.savefig(plot_cindex, dpi=150)
    plt.close(fig)

    # Plot calibration MAE
    fig, ax = plt.subplots(figsize=(8.5, 5))
    mae_plot = metrics_df.sort_values("CAL_MAE_MEAN", ascending=True).copy()
    mae_plot["MODEL_LABEL"] = mae_plot["MODEL"].map(compare_mod.model_display_name)
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
            compare_mod.format_decimal_pt(float(row["CAL_MAE_MEAN"]), 3),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Erro médio de calibração (quanto menor, melhor)")
    ax.set_title("Comparação entre modelos por calibração")
    ax.set_ylim(0.0, float(mae_plot["CAL_MAE_MEAN"].max()) + 0.012)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_cal_mae = out_plot_dir / "modelo02_model_compare_calibration_mae.png"
    fig.savefig(plot_cal_mae, dpi=150)
    plt.close(fig)

    plot_cal_180 = out_plot_dir / "modelo02_model_compare_calibration_180d.png"
    plot_cal_365 = out_plot_dir / "modelo02_model_compare_calibration_365d.png"
    compare_mod.build_calibration_plot(calib_sample_df, 180, plot_cal_180)
    compare_mod.build_calibration_plot(calib_sample_df, 365, plot_cal_365)

    lines = [
        "COMPARACAO_OBJETIVA_MODELOS_SURVIVAL_D02_MODELO02",
        "",
        "REGRA_DE_DECISAO",
        "- Se modelo alternativo melhorar C-index de forma consistente e calibrar melhor, ele ganha.",
        "- Se ganho for pequeno ou instavel, manter Cox por interpretabilidade e robustez.",
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
    out_decision.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "compare_folds": str(out_folds),
        "compare_metrics": str(out_metrics),
        "compare_conditions": str(out_cond),
        "compare_calibration_points": str(out_calib_points),
        "compare_calibration_samples": str(out_calib_samples),
        "compare_decision": str(out_decision),
        "plot_cindex": str(plot_cindex),
        "plot_cal_mae": str(plot_cal_mae),
        "plot_cal_180": str(plot_cal_180),
        "plot_cal_365": str(plot_cal_365),
        "chosen_model": chosen_model,
        "decision_reason": decision_reason,
    }


def write_report(
    out_md: Path,
    current_summary,
    valid_summary,
    model_summary,
    limits: dict[str, float],
    cox_a: dict,
    cox_b: dict,
    compare_info: dict[str, str],
) -> None:
    lines = [
        "# RELATÓRIO TÉCNICO - D02",
        "",
        "## Regras do Modelo02",
        "- Regra mínima: substituição completa só encerra episódio se ocorrer com pelo menos 8 dias desde a última substituição válida.",
        "- Eventos curtos com 6 ou 7 dias permanecem registrados no histórico, mas não reiniciam o ciclo analítico.",
        "- Remoção para modelagem: episódios válidos classificados como outlier alto por duração acima de média + 2 desvios padrão são retirados do ajuste estatístico principal.",
        "- Melhor prática: primeira e segunda melhores práticas são selecionadas apenas entre episódios válidos não classificados como outlier.",
        "",
        "## Resumo numérico",
        f"- Base original D02: {current_summary.episodes} episódios, {current_summary.events} eventos observados, {current_summary.censored} censurado(s), média {format_decimal_pt(current_summary.mean_days,2)} dias e mediana {format_decimal_pt(current_summary.median_days,0)} dias.",
        f"- Base validada com regra mínima: {valid_summary.episodes} episódios, {valid_summary.events} eventos observados, {valid_summary.censored} censurado(s), média {format_decimal_pt(valid_summary.mean_days,2)} dias e mediana {format_decimal_pt(valid_summary.median_days,0)} dias.",
        f"- Base de modelagem do Modelo02: {model_summary.episodes} episódios, {model_summary.events} eventos observados, {model_summary.censored} censurado(s), média {format_decimal_pt(model_summary.mean_days,2)} dias e mediana {format_decimal_pt(model_summary.median_days,0)} dias.",
        "",
        "## Limites de classificação",
        f"- Média dos episódios válidos: {format_decimal_pt(limits['mean_days'],2)} dias.",
        f"- Desvio padrão: {format_decimal_pt(limits['std_days'],2)} dias.",
        f"- Limite inferior da faixa típica: {format_decimal_pt(limits['lower_normal'],2)} dias.",
        f"- Limite superior da faixa típica: {format_decimal_pt(limits['upper_normal'],2)} dias.",
        f"- Limiar de outlier alto: {format_decimal_pt(limits['outlier_high'],2)} dias.",
        "",
        "## Modelo de Cox",
        f"- Cox A status: {cox_a.get('status')}.",
        f"- Cox A C-index: {format_decimal_pt(float(cox_a.get('c_index', np.nan)),6) if np.isfinite(cox_a.get('c_index', np.nan)) else ''}.",
        f"- Cox B status: {cox_b.get('status')}.",
        "",
        "## Comparação entre modelos",
        f"- Modelo escolhido: {compare_info['chosen_model']}.",
        f"- Justificativa: {compare_info['decision_reason']}.",
        "",
        "## Arquivos principais",
        "- `02_model_input/modelo02/modelo02_d02_survival_dataset_validado.csv`",
        "- `02_model_input/modelo02/modelo02_d02_survival_dataset.csv`",
        "- `02_model_input/modelo02/modelo02_d02_episode_classificacao.csv`",
        "- `outputs/relatorio_graficos_modelo02/`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = find_root(Path(__file__).resolve().parent)
    cfg = load_cfg(root / "project_config.json")
    logs_dir = root / cfg["folders"]["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    model01_ds = root / '02_model_input' / 'modelo01' / 'modelo01_d02_survival_dataset.csv'
    if not model01_ds.exists():
        raise FileNotFoundError(f"Dataset fonte nao encontrado: {model01_ds}")

    out_model_dir = root / "02_model_input" / "modelo02"
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_plot_dir = root / "outputs" / "relatorio_graficos_modelo02"
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    mod20 = load_module(root / 'scripts' / 'python' / '20_d02_outlier_recalc.py', "mod20_d02")
    mod09 = load_module(root / "scripts" / "python" / "09_plot_substituicao_fatores.py", "mod09_plots")
    mod06 = load_module(root / "scripts" / "python" / "06_train_survival_models.py", "mod06_train")
    mod15 = load_module(root / 'scripts' / 'python' / '15_write_cox_utils.py', "mod15_report")
    mod16 = load_module(root / 'scripts' / 'python' / '16_compare_survival_models_d02.py', "mod16_compare")

    daily_src = pd.read_csv(model01_ds, sep=";", encoding="utf-8-sig", low_memory=False)
    current_ep = mod20.prepare_original_episode_summary(daily_src)
    current_summary = mod20.summarize_observed("ATUAL", current_ep)

    daily_valid, ep_valid, invalid_df = mod20.rebuild_d02_daily(daily_src, min_gap_days=8)
    valid_summary = mod20.summarize_observed("MIN8", ep_valid)
    classified_eps, limits = mod20.classify_episodes(ep_valid)
    clean_daily = mod20.build_clean_dataset(daily_valid, classified_eps)
    model_ep = classified_eps[classified_eps["CLASSIFICACAO"] != "Outlier"].copy().reset_index(drop=True)
    model_summary = mod20.summarize_observed("MODELO02", model_ep)

    # Persist datasets
    path_valid = out_model_dir / "modelo02_d02_survival_dataset_validado.csv"
    path_clean = out_model_dir / "modelo02_d02_survival_dataset.csv"
    path_episode_valid = out_model_dir / "modelo02_d02_episode_summary_validado.csv"
    path_class = out_model_dir / "modelo02_d02_episode_classificacao.csv"
    path_short = out_model_dir / "modelo02_d02_eventos_curtos_invalidados.csv"
    daily_valid.to_csv(path_valid, sep=";", index=False, encoding="utf-8-sig")
    clean_daily.to_csv(path_clean, sep=";", index=False, encoding="utf-8-sig")
    ep_valid.to_csv(path_episode_valid, sep=";", index=False, encoding="utf-8-sig")
    classified_eps.to_csv(path_class, sep=";", index=False, encoding="utf-8-sig")
    invalid_df.to_csv(path_short, sep=";", index=False, encoding="utf-8-sig")

    # Descriptive plots from validated base
    valid_daily = daily_valid.copy()
    valid_daily["DATA_DT"] = mod09.parse_date_series(valid_daily["DATA_DIA"])
    valid_daily["PRODUCAO_TONELADAS_DIA"] = mod09.to_num(valid_daily["PRODUCAO_TONELADAS_DIA"])
    valid_daily["EVENT_TROCA"] = mod09.to_num(valid_daily["EVENT_TROCA"]).fillna(0).astype(int)
    valid_daily["EVENT_RECUP"] = mod09.to_num(valid_daily["EVENT_RECUP"]).fillna(0).astype(int)
    valid_daily["EVENT_OBSERVED"] = mod09.to_num(valid_daily["EVENT_OBSERVED"]).fillna(0).astype(int)
    valid_daily["T_DIAS"] = mod09.to_num(valid_daily["T_DIAS"])

    valid_ep_plot = mod09.build_episode_summary(valid_daily)
    valid_sub_plot = valid_ep_plot[valid_ep_plot["EVENT_OBSERVED"] == 1].copy().sort_values("ORDEM_SUBSTITUICAO")
    summary_csv = out_model_dir / "modelo02_d02_substituicoes_resumo.csv"
    plot_carga_png = out_plot_dir / "modelo02_d02_substituicoes_carga_acum.png"
    plot_dur_rep_png = out_plot_dir / "modelo02_d02_substituicoes_duracao_reparos.png"
    plot_timeline_png = out_plot_dir / "modelo02_d02_timeline_carga_substituicoes.png"
    valid_out = valid_sub_plot.copy()
    valid_out["DATA_INICIO"] = valid_out["DATA_INICIO"].dt.strftime("%d/%m/%Y")
    valid_out["DATA_FIM"] = valid_out["DATA_FIM"].dt.strftime("%d/%m/%Y")
    valid_out["DATA_SUBSTITUICAO"] = pd.to_datetime(valid_out["DATA_SUBSTITUICAO"], errors="coerce").dt.strftime("%d/%m/%Y")
    valid_out.to_csv(summary_csv, sep=";", index=False, encoding="utf-8-sig")
    mod09.plot_carga_por_substituicao(valid_sub_plot, plot_carga_png)
    mod09.plot_duracao_reparos(valid_sub_plot, plot_dur_rep_png)
    outlier_episode_nums = (
        classified_eps.loc[classified_eps["CLASSIFICACAO"] == "Outlier", "EPISODIO_NUM"]
        .dropna()
        .astype(int)
        .tolist()
    )
    mod09.plot_timeline_carga_reparos(
        valid_daily,
        plot_timeline_png,
        outlier_episode_nums=outlier_episode_nums,
    )
    mod09.write_insights(valid_sub_plot, out_model_dir / "modelo02_d02_substituicoes_insights.txt")
    plot_trocas_png = out_plot_dir / "modelo02_d02_trocas_por_ano.png"
    plot_trocas_por_ano(valid_daily, plot_trocas_png)

    plot_class_png = out_plot_dir / "modelo02_d02_classificacao_visual.png"
    mod20.plot_classificacao(
        classified_eps,
        invalid_df,
        plot_class_png,
        limits,
        show_counts_in_legend=False,
        annotate_episode_nums=True,
    )

    # Modeling artifacts from clean base
    ep_df = mod06.prepare_episode_features(clean_daily, "D02")
    episode_lengths = ep_df[
        ["EQUIPAMENTO", "EPISODIO_ID", "DATA_INICIO", "DATA_FIM", "DURATION_DIAS", "EVENT_OBSERVED", "CENSORED"]
    ].copy()
    episode_lengths_path = out_model_dir / "modelo02_d02_episode_lengths.csv"
    episode_lengths.to_csv(episode_lengths_path, sep=";", index=False, encoding="utf-8-sig")

    coverage_path = out_model_dir / "modelo02_d02_covariate_coverage_report.csv"
    mod06.build_covariate_coverage_report(clean_daily, coverage_path)

    km_plot_path = out_plot_dir / "modelo02_d02_km_plot.png"
    mod06.create_km_plot(ep_df, "D02", km_plot_path)

    weibull_txt = out_model_dir / "modelo02_d02_weibull_results.txt"
    weibull_metrics = mod06.fit_weibull(ep_df, "D02", weibull_txt)

    # Additional plots from clean modeling base
    plot_duration_hist(model_ep, out_plot_dir / "modelo02_d02_duration_histogram.png")
    duration = ep_df["DURATION_DIAS"].astype(float)
    events = ep_df["EVENT_OBSERVED"].astype(int)
    from lifelines import WeibullFitter, KaplanMeierFitter

    wf = WeibullFitter()
    wf.fit(duration, event_observed=events, label="D02")
    plot_weibull_vs_km(duration, events, wf, out_plot_dir / "modelo02_d02_weibull_fit_plot.png")

    # KM by load and repair
    med_carga = float(ep_df["CARGA_ACUM_EP"].dropna().median()) if ep_df["CARGA_ACUM_EP"].notna().any() else 0.0
    ep_tmp = ep_df.copy()
    ep_tmp["GRUPO_CARGA"] = np.where(
        ep_tmp["CARGA_ACUM_EP"] >= med_carga,
        "Maior carga acumulada",
        "Menor carga acumulada",
    )
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for grp, g in ep_tmp.groupby("GRUPO_CARGA"):
        k = KaplanMeierFitter()
        k.fit(g["DURATION_DIAS"], g["EVENT_OBSERVED"], label=grp)
        k.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title("Curvas de sobrevivência por grupos de carga acumulada")
    ax.set_xlabel("Duração do episódio (dias)")
    ax.set_ylabel("Probabilidade de o episódio continuar sem substituição")
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "modelo02_d02_km_by_carga.png", dpi=150)
    plt.close(fig)

    rep_map = model_ep[["EPISODIO_ID", "REPAROS_EP"]].copy()
    ep_tmp = ep_df.merge(rep_map, on="EPISODIO_ID", how="left")
    ep_tmp["GRUPO_REPARO"] = np.where(ep_tmp["REPAROS_EP"].fillna(0) > 0, "Com reparo", "Sem reparo")
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for grp, g in ep_tmp.groupby("GRUPO_REPARO"):
        k = KaplanMeierFitter()
        k.fit(g["DURATION_DIAS"], g["EVENT_OBSERVED"], label=grp)
        k.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title("Kaplan-Meier por histórico de reparo - Modelo02 D02")
    ax.set_xlabel("Dias")
    ax.set_ylabel("Sobrevivência")
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "modelo02_d02_km_by_reparo.png", dpi=150)
    plt.close(fig)

    # Cox A and Cox B
    ep_cox = ep_df.copy()
    ep_cox["LOG_CARGA_ACUM_EP"] = np.log1p(np.clip(ep_cox["CARGA_ACUM_EP"].astype(float), a_min=0, a_max=None))
    cox_a_features = [
        "LOG_CARGA_ACUM_EP",
        "MEDIA_CARGA_30D",
        "FLAG_EMENDA_7D",
        "FLAG_EMENDA_14D",
        "DIAS_DESDE_ULT_EMENDA",
    ]
    cox_a_path = out_model_dir / "modelo02_d02_coxA_summary.txt"
    cox_a = mod06.run_cox(ep_cox, cox_a_features, cox_a_path, title="COX_A_MODELO02_D02", penalizer=0.15)

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
    cox_b_path = out_model_dir / "modelo02_d02_coxB_summary.txt"
    cox_b = mod06.run_cox(
        ep_cox_b,
        cox_b_features,
        cox_b_path,
        title="COX_B_MODELO02_D02",
        penalizer=0.25,
        min_rows=6,
    )
    if cox_b.get("status") != "OK":
        fallback_sets = [
            ["LOG_CARGA_ACUM_EP", "MEDIA_CARGA_30D", "DOS_VAZAO_MEDIA_EP"],
            ["LOG_CARGA_ACUM_EP", "MEDIA_CARGA_30D"],
            ["LOG_CARGA_ACUM_EP"],
        ]
        for fb in fallback_sets:
            fb_valid = [c for c in fb if c in ep_cox_b.columns and ep_cox_b[c].notna().sum() >= max(3, int(np.ceil(len(ep_cox_b) * 0.50)))]
            if len(fb_valid) == 0:
                continue
            cox_b = mod06.run_cox(
                ep_cox_b,
                fb_valid,
                cox_b_path,
                title="COX_B_MODELO02_D02",
                penalizer=0.35,
                min_rows=6,
            )
            if cox_b.get("status") == "OK":
                break

    mod15.write_cox(out_plot_dir / "modelo02_d02_cox_A_results.txt", cox_a)
    mod15.write_cox(out_plot_dir / "modelo02_d02_cox_B_results.txt", cox_b)
    plot_cox_a_only(cox_a, out_plot_dir / "modelo02_d02_hazard_ratios_cox.png")

    # Model comparison on clean base
    compare_ep = mod16.build_episode_dataframe(clean_daily, equip="D02")
    compare_info = run_model_compare(mod16, compare_ep, out_model_dir, out_plot_dir)

    # Report
    report_md = root / 'docs' / 'RELATORIO_TECNICO_D02.md'
    write_report(report_md, current_summary, valid_summary, model_summary, limits, cox_a, cox_b, compare_info)

    # Summary txt
    summary_txt = out_model_dir / "modelo02_d02_summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                "MODELO02_D02",
                f"INPUT_DATASET={model01_ds}",
                f"DATASET_VALIDADO={path_valid}",
                f"DATASET_MODELAGEM={path_clean}",
                f"EPISODIOS_VALIDOS={valid_summary.episodes}",
                f"EVENTOS_VALIDOS={valid_summary.events}",
                f"EPISODIOS_MODELAGEM={model_summary.episodes}",
                f"EVENTOS_MODELAGEM={model_summary.events}",
                f"MEDIANA_MODELAGEM_DIAS={model_summary.median_days:.2f}",
                f"MEDIA_MODELAGEM_DIAS={model_summary.mean_days:.2f}",
                f"WEIBULL_BETA={weibull_metrics.get('beta', np.nan):.6f}" if np.isfinite(weibull_metrics.get("beta", np.nan)) else "WEIBULL_BETA=",
                f"WEIBULL_ETA={weibull_metrics.get('eta', np.nan):.6f}" if np.isfinite(weibull_metrics.get("eta", np.nan)) else "WEIBULL_ETA=",
                f"COX_A_STATUS={cox_a.get('status')}",
                f"COX_A_CINDEX={cox_a.get('c_index', np.nan):.6f}" if np.isfinite(cox_a.get("c_index", np.nan)) else "COX_A_CINDEX=",
                f"COX_B_STATUS={cox_b.get('status')}",
                f"MODELO_ESCOLHIDO_COMPARE={compare_info['chosen_model']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"modelo02_d02_{stamp}.log"
    log_path.write_text(
        "\n".join(
            [
                "pipeline=modelo02_d02",
                f"input={model01_ds}",
                f"out_model_dir={out_model_dir}",
                f"out_plot_dir={out_plot_dir}",
                f"episodes_original={current_summary.episodes}",
                f"episodes_valid={valid_summary.episodes}",
                f"episodes_model={model_summary.episodes}",
                f"events_valid={valid_summary.events}",
                f"events_model={model_summary.events}",
                f"report={report_md}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"MODELO02_DIR: {out_model_dir}")
    print(f"PLOTS_DIR: {out_plot_dir}")
    print(f"REPORT: {report_md}")
    print(f"EPISODIOS_VALIDOS: {valid_summary.episodes}")
    print(f"EVENTOS_VALIDOS: {valid_summary.events}")
    print(f"EPISODIOS_MODELAGEM: {model_summary.episodes}")
    print(f"EVENTOS_MODELAGEM: {model_summary.events}")
    print(f"COX_A_STATUS: {cox_a.get('status')}")
    print(f"COX_B_STATUS: {cox_b.get('status')}")
    print(f"MODELO_ESCOLHIDO_COMPARE: {compare_info['chosen_model']}")
    print(f"LOG: {log_path}")


if __name__ == "__main__":
    main()
