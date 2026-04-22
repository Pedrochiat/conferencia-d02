from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "project_config.json").exists():
            return candidate
    raise FileNotFoundError("project_config.json nao encontrado em nenhum diretorio pai")


def parse_date_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    d1 = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    d2 = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    d3 = pd.to_datetime(text, errors="coerce")
    return d1.fillna(d2).fillna(d3)


def to_num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def build_episode_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    work = daily_df.copy().sort_values(["EPISODIO_NUM", "DATA_DT"])
    rows = []
    for eid, g in work.groupby("EPISODIO_ID", sort=True):
        g = g.sort_values("DATA_DT")
        has_event = int(g["EVENT_OBSERVED"].max())
        sub_dates = g.loc[g["EVENT_TROCA"] == 1, "DATA_DT"]
        sub_date = sub_dates.iloc[0] if len(sub_dates) > 0 else pd.NaT
        rows.append(
            {
                "EQUIPAMENTO": str(g["EQUIPAMENTO"].iloc[0]),
                "EPISODIO_ID": str(eid),
                "EPISODIO_NUM": int(g["EPISODIO_NUM"].iloc[0]),
                "DATA_INICIO": g["DATA_DT"].iloc[0],
                "DATA_FIM": g["DATA_DT"].iloc[-1],
                "DATA_SUBSTITUICAO": sub_date,
                "DURACAO_DIAS": float(g["T_DIAS"].max()),
                "EVENT_OBSERVED": has_event,
                "N_REPAROS_EP": int(g["EVENT_RECUP"].fillna(0).sum()),
                "TRANSPORTE_ACUM_EP_TON": float(g["PRODUCAO_TONELADAS_DIA"].fillna(0).sum()),
            }
        )

    ep = pd.DataFrame(rows).sort_values("EPISODIO_NUM").reset_index(drop=True)
    ep["ORDEM_SUBSTITUICAO"] = np.nan
    event_mask = ep["EVENT_OBSERVED"] == 1
    ep.loc[event_mask, "ORDEM_SUBSTITUICAO"] = np.arange(1, int(event_mask.sum()) + 1)
    return ep


def plot_carga_por_substituicao(sub_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = sub_df["ORDEM_SUBSTITUICAO"].astype(int)
    y = sub_df["TRANSPORTE_ACUM_EP_TON"].astype(float)
    ax.bar(x, y, color="#1f77b4")
    ax.set_title("Transporte Acumulado Ate Cada Substituicao")
    ax.set_xlabel("Ordem da Substituicao")
    ax.set_ylabel("Transporte Acumulado no Episodio (ton)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_duracao_reparos(sub_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    work = sub_df.copy()
    work["N_REPAROS_EP"] = work["N_REPAROS_EP"].fillna(0).astype(int)
    work["DURACAO_DIAS"] = work["DURACAO_DIAS"].astype(float)

    x_plot = np.zeros(len(work), dtype=float)
    for rep_value, idx in work.groupby("N_REPAROS_EP").groups.items():
        idx = list(idx)
        if len(idx) == 1:
            offsets = np.array([0.0])
        else:
            if int(rep_value) == 0:
                offsets = np.linspace(0.00, 0.18, len(idx))
            else:
                offsets = np.linspace(-0.10, 0.10, len(idx))
        x_plot[idx] = rep_value + offsets

    ax.scatter(
        x_plot,
        work["DURACAO_DIAS"],
        s=58,
        color="#1f77b4",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.9,
        label="Episódios válidos",
        zorder=3,
    )

    max_rep = int(work["N_REPAROS_EP"].max()) if len(work) else 0
    ax.set_xlim(-0.05, max_rep + 0.3)
    ax.set_xticks(np.arange(0, max_rep + 1, 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Número de reparos intermediários no episódio")
    ax.set_ylabel("Duração do episódio (dias)")
    ax.set_title("Relação entre reparos intermediários e duração dos episódios na D02")
    ax.grid(alpha=0.28)
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


def plot_timeline_carga_reparos(
    daily_df: pd.DataFrame,
    out_png: Path,
    outlier_episode_nums: list[int] | None = None,
) -> None:
    work = daily_df.copy().sort_values(["EPISODIO_NUM", "DATA_DT"])
    outlier_set = set(outlier_episode_nums or [])
    work["CARGA_ACUM_EP_TON"] = (
        work.groupby("EPISODIO_ID", sort=True)["PRODUCAO_TONELADAS_DIA"]
        .transform(lambda s: s.fillna(0).cumsum())
    )
    work["CARGA_ACUM_EP_KTON"] = work["CARGA_ACUM_EP_TON"] / 1000.0

    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    first_line = True
    for _, g in work.groupby("EPISODIO_ID", sort=True):
        g = g.sort_values("DATA_DT")
        ax.plot(
            g["DATA_DT"],
            g["CARGA_ACUM_EP_KTON"],
            color="#1f77b4",
            linewidth=1.5,
            alpha=0.9,
            label="Linha azul = 1 episódio" if first_line else None,
        )
        first_line = False

    subs = work[work["EVENT_TROCA"] == 1]
    if not subs.empty:
        subs_regular = subs[~subs["EPISODIO_NUM"].astype(int).isin(outlier_set)].copy()
        subs_outlier = subs[subs["EPISODIO_NUM"].astype(int).isin(outlier_set)].copy()

        if not subs_regular.empty:
            ax.scatter(
                subs_regular["DATA_DT"],
                subs_regular["CARGA_ACUM_EP_KTON"],
                color="#d62728",
                s=48,
                label="Substituições",
                zorder=3,
            )
        if not subs_outlier.empty:
            ax.scatter(
                subs_outlier["DATA_DT"],
                subs_outlier["CARGA_ACUM_EP_KTON"],
                color="#111111",
                s=52,
                label="Outlier",
                zorder=4,
            )
        for _, row in subs.iterrows():
            ep_num = int(row["EPISODIO_NUM"])
            y_val = float(row["CARGA_ACUM_EP_KTON"])
            if ep_num in {8, 9, 10, 11}:
                custom_offsets = {
                    8: (-14, 10),
                    9: (-16, 20),
                    10: (6, -22),
                    11: (10, 10),
                }
                x_offset, y_offset = custom_offsets[ep_num]
            elif y_val > 0.9 * float(work["CARGA_ACUM_EP_KTON"].max()):
                x_offset, y_offset = (4, -18)
            elif y_val < 35:
                x_offset, y_offset = (4, 12)
            else:
                x_offset, y_offset = (4, 12 if ep_num % 2 else -20)

            if y_val > 0.9 * float(work["CARGA_ACUM_EP_KTON"].max()):
                y_offset = -18
            va = "bottom" if y_offset > 0 else "top"
            ax.annotate(
                f"{ep_num}*" if ep_num in outlier_set else str(ep_num),
                xy=(row["DATA_DT"], row["CARGA_ACUM_EP_KTON"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=7,
                color="#222222",
                ha="left",
                va=va,
            )

    reps = work[work["EVENT_RECUP"] == 1]
    if not reps.empty:
        ax.scatter(
            reps["DATA_DT"],
            reps["CARGA_ACUM_EP_KTON"],
            color="#ff8c00",
            marker="s",
            s=42,
            edgecolors="#7a3e00",
            linewidths=0.6,
            alpha=0.9,
            label="Reparos",
            zorder=3,
        )

    censored = work[work["CENSORED"] == 1].sort_values("DATA_DT")
    if not censored.empty:
        row = censored.iloc[-1]
        ax.scatter(
            [row["DATA_DT"]],
            [row["CARGA_ACUM_EP_KTON"]],
            color="#111111",
            marker="x",
            s=50,
            linewidths=1.2,
            label="Censurado",
            zorder=4,
        )
        ax.annotate(
            f"{int(row['EPISODIO_NUM'])}",
            xy=(row["DATA_DT"], row["CARGA_ACUM_EP_KTON"]),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=7,
            color="#111111",
            ha="left",
            va="bottom",
        )

    ax.set_title("Linha do tempo da D02: carga acumulada em cada episódio")
    ax.set_xlabel("Data")
    ax.set_ylabel("Carga acumulada dentro do episódio (kton)")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=5,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.24)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_insights(sub_df: pd.DataFrame, out_txt: Path) -> None:
    carga = sub_df["TRANSPORTE_ACUM_EP_TON"].astype(float)
    dur = sub_df["DURACAO_DIAS"].astype(float)
    rep = sub_df["N_REPAROS_EP"].astype(float)

    corr_rep_dur = np.nan
    corr_rep_carga = np.nan
    if len(sub_df) >= 3:
        corr_rep_dur = float(rep.corr(dur))
        corr_rep_carga = float(rep.corr(carga))

    lines = [
        "INSIGHTS_SUBSTITUICOES_MODELO01",
        f"SUBSTITUICOES_TOTAL={len(sub_df)}",
        f"TRANSPORTE_ACUM_MEDIO_TON={float(carga.mean()):.2f}",
        f"TRANSPORTE_ACUM_MEDIANO_TON={float(carga.median()):.2f}",
        f"TRANSPORTE_ACUM_P95_TON={float(carga.quantile(0.95)):.2f}",
        f"DURACAO_MEDIA_DIAS={float(dur.mean()):.2f}",
        f"DURACAO_MEDIANA_DIAS={float(dur.median()):.2f}",
        f"REPAROS_MEDIOS_POR_EP={float(rep.mean()):.2f}",
        f"CORR_REPAROS_DURACAO={corr_rep_dur:.4f}" if np.isfinite(corr_rep_dur) else "CORR_REPAROS_DURACAO=",
        f"CORR_REPAROS_TRANSPORTE={corr_rep_carga:.4f}" if np.isfinite(corr_rep_carga) else "CORR_REPAROS_TRANSPORTE=",
        "",
        "[NOTA_METODOLOGICA]",
        "- Alvo principal de previsao: SUBSTITUICAO COMPLETA de correia.",
        "- Reparos intermediarios entram como indicativo de estado de desgaste, nao como alvo final.",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plots de substituicao e desgaste (modelo01)")
    parser.add_argument("--equipamento", default="D02", help="Equipamento alvo (default: D02)")
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

    daily = pd.read_csv(ds_path, sep=";", encoding="utf-8-sig", low_memory=False)
    required_cols = ["EQUIPAMENTO", "EPISODIO_ID", "EPISODIO_NUM", "DATA_DIA", "T_DIAS", "EVENT_OBSERVED", "EVENT_TROCA", "EVENT_RECUP", "PRODUCAO_TONELADAS_DIA"]
    miss = [c for c in required_cols if c not in daily.columns]
    if miss:
        raise KeyError(f"Colunas obrigatorias ausentes no survival dataset: {','.join(miss)}")

    daily["DATA_DT"] = parse_date_series(daily["DATA_DIA"])
    daily["PRODUCAO_TONELADAS_DIA"] = to_num(daily["PRODUCAO_TONELADAS_DIA"])
    daily["EVENT_TROCA"] = to_num(daily["EVENT_TROCA"]).fillna(0).astype(int)
    daily["EVENT_RECUP"] = to_num(daily["EVENT_RECUP"]).fillna(0).astype(int)
    daily["EVENT_OBSERVED"] = to_num(daily["EVENT_OBSERVED"]).fillna(0).astype(int)
    daily["T_DIAS"] = to_num(daily["T_DIAS"])

    ep = build_episode_summary(daily)
    sub = ep[ep["EVENT_OBSERVED"] == 1].copy()
    if sub.empty:
        raise ValueError(f"Nenhum episodio com substituicao observado para {equip}")
    sub = sub.sort_values("ORDEM_SUBSTITUICAO").reset_index(drop=True)

    summary_csv = model_dir / f"modelo01_{equip.lower()}_substituicoes_resumo.csv"
    plot_carga_png = model_dir / f"modelo01_{equip.lower()}_plot_substituicoes_carga_acum.png"
    plot_dur_rep_png = model_dir / f"modelo01_{equip.lower()}_plot_substituicoes_duracao_reparos.png"
    plot_timeline_png = model_dir / f"modelo01_{equip.lower()}_plot_timeline_carga_substituicoes.png"
    insights_txt = model_dir / f"modelo01_{equip.lower()}_substituicoes_insights.txt"

    out = sub.copy()
    out["DATA_INICIO"] = out["DATA_INICIO"].dt.strftime("%d/%m/%Y")
    out["DATA_FIM"] = out["DATA_FIM"].dt.strftime("%d/%m/%Y")
    out["DATA_SUBSTITUICAO"] = pd.to_datetime(out["DATA_SUBSTITUICAO"], errors="coerce").dt.strftime("%d/%m/%Y")
    out.to_csv(summary_csv, sep=";", index=False, encoding="utf-8-sig")

    plot_carga_por_substituicao(sub, plot_carga_png)
    plot_duracao_reparos(sub, plot_dur_rep_png)
    plot_timeline_carga_reparos(daily, plot_timeline_png)
    write_insights(sub, insights_txt)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"modelo01_plots_{equip.lower()}_{stamp}.log"
    log_lines = [
        "pipeline=modelo01_plots_substituicao",
        f"equipamento={equip}",
        f"input_dataset={ds_path}",
        f"substituicoes={len(sub)}",
        f"summary_csv={summary_csv}",
        f"plot_carga={plot_carga_png}",
        f"plot_duracao_reparos={plot_dur_rep_png}",
        f"plot_timeline={plot_timeline_png}",
        f"insights={insights_txt}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"MODELO01_PLOTS_EQUIPAMENTO: {equip}")
    print(f"SUBSTITUICOES: {len(sub)}")
    print(f"SUMMARY: {summary_csv}")
    print(f"PLOT_CARGA: {plot_carga_png}")
    print(f"PLOT_DURACAO_REPAROS: {plot_dur_rep_png}")
    print(f"PLOT_TIMELINE: {plot_timeline_png}")
    print(f"INSIGHTS: {insights_txt}")
    print(f"LOG: {log_path}")


if __name__ == "__main__":
    main()
