"""
Geração de gráficos e PDFs para comparação de estimativas.

Gera:
  - Um PDF por método (barras + distribuição de erros + box plot)
  - Um PDF consolidado com todos os métodos
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # sem display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

_METRICS = ["average_speed", "percentage_of_track", "average_vehicles", "km_traveled"]
_METRIC_LABELS = {
    "average_speed": "Velocidade Média (m/s)",
    "percentage_of_track": "Ocupação (%)",
    "average_vehicles": "Veículos Médios",
    "km_traveled": "Km Trafegados",
}
_METHOD_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
                  "#F44336", "#00BCD4", "#795548", "#607D8B"]


# ---------------------------------------------------------------------------
# Helpers de dado
# ---------------------------------------------------------------------------

def _agg_values(all_results):
    """Organiza dados para gráficos: {metric: {method: value}}."""
    agg = {m: {} for m in _METRICS}
    for method, res in all_results.items():
        for m in _METRICS:
            agg[m][method] = res["aggregate"]["error_pct"][m]
    return agg


def _mape_values(all_results):
    mape = {m: {} for m in _METRICS}
    for method, res in all_results.items():
        for m in _METRICS:
            mape[m][method] = res["per_series_error"][m]["mape"]
    return mape


def _box_data(all_results, metric):
    """Para box plot: lista de erros por série (estimados como dist stats)."""
    # Recria distribuição aproximada usando p25, median, p75, min, max
    data = {}
    for method, res in all_results.items():
        ps = res["per_series_error"][metric]
        data[method] = {
            "whislo": ps["min"],
            "q1": ps["p25"],
            "med": ps["median"],
            "q3": ps["p75"],
            "whishi": ps["max"],
        }
    return data


# ---------------------------------------------------------------------------
# Figuras individuais
# ---------------------------------------------------------------------------

def _fig_bar_errors(all_results, title_suffix=""):
    """Barras de erro % agregado por método para cada métrica."""
    methods = list(all_results.keys())
    n_metrics = len(_METRICS)
    x = np.arange(n_metrics)
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = [all_results[method]["aggregate"]["error_pct"][m] for m in _METRICS]
        bars = ax.bar(x + i * width - (len(methods) - 1) * width / 2,
                      vals, width * 0.9,
                      label=method, color=_METHOD_COLORS[i % len(_METHOD_COLORS)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:+.1f}%", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS[m] for m in _METRICS], rotation=15, ha="right")
    ax.set_ylabel("Erro % (estimado vs real)")
    ax.set_title(f"Erro Percentual Agregado (gmean){title_suffix}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_mape_bars(all_results, title_suffix=""):
    """Barras de MAPE por método para cada métrica."""
    methods = list(all_results.keys())
    n_metrics = len(_METRICS)
    x = np.arange(n_metrics)
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = [all_results[method]["per_series_error"][m]["mape"] for m in _METRICS]
        bars = ax.bar(x + i * width - (len(methods) - 1) * width / 2,
                      vals, width * 0.9,
                      label=method, color=_METHOD_COLORS[i % len(_METHOD_COLORS)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS[m] for m in _METRICS], rotation=15, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"MAPE por Método{title_suffix}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_box_per_metric(all_results, metric, title_suffix=""):
    """Box plot de distribuição de erros por série para uma métrica."""
    methods = list(all_results.keys())
    box_data = _box_data(all_results, metric)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 5))
    bplot = ax.bxp(
        [box_data[m] for m in methods],
        positions=range(len(methods)),
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(bplot["boxes"], _METHOD_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Erro % por série")
    ax.set_title(f"Distribuição de Erros — {_METRIC_LABELS[metric]}{title_suffix}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_radar(all_results, title_suffix=""):
    """Radar chart comparando MAPE de todos os métodos."""
    methods = list(all_results.keys())
    labels = [_METRIC_LABELS[m] for m in _METRICS]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, method in enumerate(methods):
        vals = [all_results[method]["per_series_error"][m]["mape"] for m in _METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.5,
                color=_METHOD_COLORS[i % len(_METHOD_COLORS)], label=method)
        ax.fill(angles, vals, alpha=0.05,
                color=_METHOD_COLORS[i % len(_METHOD_COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_title(f"MAPE por Métrica{title_suffix}", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return fig


def _fig_coverage(all_results, title_suffix=""):
    """Barras de cobertura (séries estimadas/reais) por método."""
    methods = list(all_results.keys())
    vals = [all_results[m]["coverage_pct"] for m in methods]

    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 1.2), 4))
    bars = ax.bar(methods, vals,
                  color=_METHOD_COLORS[:len(methods)])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.1f}%", ha="center", va="bottom")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Cobertura (%)")
    ax.set_title(f"Cobertura — Séries Estimadas vs Reais{title_suffix}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_pca_variance(pca_models, title_suffix=""):
    """Variância explicada pelo PCA por cluster."""
    cluster_ids = sorted(pca_models.keys())
    variances = [pca_models[c]["explained_variance_pct"] for c in cluster_ids]
    n_components = [pca_models[c]["n_components"] for c in cluster_ids]
    sizes = [pca_models[c]["size"] for c in cluster_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Variância explicada
    bars = axes[0].bar(
        [f"C{c}" for c in cluster_ids], variances,
        color="#2196F3", alpha=0.8
    )
    for bar, v in zip(bars, variances):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[0].axhline(90, color="red", linewidth=0.8, linestyle="--", label="90%")
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel("Variância Explicada (%)")
    axes[0].set_title(f"Variância Explicada por Cluster{title_suffix}")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Tamanho dos clusters + n_components
    x = np.arange(len(cluster_ids))
    w = 0.4
    axes[1].bar(x - w/2, sizes, w, label="Séries no cluster", color="#4CAF50", alpha=0.8)
    axes[1].bar(x + w/2, n_components, w, label="Componentes PCA", color="#FF9800", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"C{c}" for c in cluster_ids])
    axes[1].set_ylabel("Quantidade")
    axes[1].set_title(f"Séries por Cluster vs Componentes PCA{title_suffix}")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Exportação
# ---------------------------------------------------------------------------

def save_individual_pdfs(all_results, output_dir, prefix="compare", title_suffix="", pca_models=None):
    """Salva um PDF por método com todas as suas visualizações."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for method, res in all_results.items():
        path = os.path.join(output_dir, f"{prefix}_{method}.pdf")
        single = {method: res}
        with PdfPages(path) as pdf:
            fig = _fig_bar_errors(single, title_suffix)
            pdf.savefig(fig); plt.close(fig)
            fig = _fig_mape_bars(single, title_suffix)
            pdf.savefig(fig); plt.close(fig)
            for metric in _METRICS:
                fig = _fig_box_per_metric(single, metric, title_suffix)
                pdf.savefig(fig); plt.close(fig)
            # Gráfico de variância PCA para o método pca
            if method == "pca" and pca_models:
                fig = _fig_pca_variance(pca_models, title_suffix)
                pdf.savefig(fig); plt.close(fig)
        saved.append(path)
        print(f"  → PDF individual: {path}")

    return saved


def save_pca_pdf(pca_models, output_dir, prefix="pca", title_suffix=""):
    """Salva PDF dedicado à análise do modelo PCA (variância, clusters)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{prefix}_analysis.pdf")
    with PdfPages(path) as pdf:
        fig = _fig_pca_variance(pca_models, title_suffix)
        pdf.savefig(fig); plt.close(fig)
    print(f"  → PDF PCA: {path}")
    return path


def save_consolidated_pdf(all_results, output_dir, prefix="compare", title_suffix="", pca_models=None):
    """Salva um único PDF consolidado com todos os métodos."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{prefix}_consolidated.pdf")

    with PdfPages(path) as pdf:
        # Página 1: erro % agregado
        fig = _fig_bar_errors(all_results, title_suffix)
        pdf.savefig(fig); plt.close(fig)

        # Página 2: MAPE
        fig = _fig_mape_bars(all_results, title_suffix)
        pdf.savefig(fig); plt.close(fig)

        # Página 3: radar
        if len(all_results) > 1:
            fig = _fig_radar(all_results, title_suffix)
            pdf.savefig(fig); plt.close(fig)

        # Página 4: cobertura
        fig = _fig_coverage(all_results, title_suffix)
        pdf.savefig(fig); plt.close(fig)

        # Páginas 5+: box plot por métrica
        for metric in _METRICS:
            fig = _fig_box_per_metric(all_results, metric, title_suffix)
            pdf.savefig(fig); plt.close(fig)

        # Última página: variância PCA (se disponível)
        if pca_models:
            fig = _fig_pca_variance(pca_models, title_suffix)
            pdf.savefig(fig); plt.close(fig)

    print(f"  → PDF consolidado: {path}")
    return path


# ---------------------------------------------------------------------------
# Sweep — progressão do erro por cenário parcial
# ---------------------------------------------------------------------------

def _sweep_all_methods(sweep_results):
    """Retorna lista ordenada de todos os métodos encontrados nos resultados."""
    methods = []
    for sdata in sweep_results.values():
        for m in sdata["methods"]:
            if m not in methods:
                methods.append(m)
    return methods


def _fig_sweep_line(sweep_results, metric, title_suffix=""):
    """Gráfico de linhas: X=cenários, Y=MAPE para uma métrica, uma linha por método."""
    scenarios = list(sweep_results.keys())
    methods = _sweep_all_methods(sweep_results)

    fig, ax = plt.subplots(figsize=(max(7, len(scenarios) * 1.4), 5))

    for idx, method in enumerate(methods):
        values = []
        labels_x = []
        for label in scenarios:
            sdata = sweep_results[label]
            if method in sdata["methods"]:
                mape = sdata["methods"][method]["per_series_error"][metric]["mape"]
                values.append(mape if mape == mape else None)  # NaN → None
                labels_x.append(label)

        # filtra Nones para plotagem
        xs = [l for l, v in zip(labels_x, values) if v is not None]
        ys = [v for v in values if v is not None]

        color = _METHOD_COLORS[idx % len(_METHOD_COLORS)]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=7,
                label=method, color=color)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=color)

    metric_label = {
        "average_speed": "Velocidade Média",
        "percentage_of_track": "Ocupação",
        "average_vehicles": "Veículos Médios",
        "km_traveled": "Km Trafegados",
    }.get(metric, metric)

    ax.set_xlabel("Cenário Parcial")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"Progressão MAPE — {metric_label}{title_suffix}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_sweep_grouped_bars(sweep_results, metric, title_suffix=""):
    """Barras agrupadas: para cada cenário, uma barra por método."""
    scenarios = list(sweep_results.keys())
    methods = _sweep_all_methods(sweep_results)

    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.5), 5))

    for idx, method in enumerate(methods):
        values = []
        for label in scenarios:
            sdata = sweep_results[label]
            if method in sdata["methods"]:
                mape = sdata["methods"][method]["per_series_error"][metric]["mape"]
                values.append(mape if mape == mape else 0.0)
            else:
                values.append(0.0)

        offset = (idx - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=method, color=_METHOD_COLORS[idx % len(_METHOD_COLORS)],
                      alpha=0.85)
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom", fontsize=6)

    metric_label = {
        "average_speed": "Velocidade Média",
        "percentage_of_track": "Ocupação",
        "average_vehicles": "Veículos Médios",
        "km_traveled": "Km Trafegados",
    }.get(metric, metric)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_xlabel("Cenário Parcial")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"MAPE por Cenário — {metric_label}{title_suffix}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_sweep_heatmap(sweep_results, title_suffix=""):
    """Heatmap: linhas=métodos, colunas=cenários, cor=MAPE (velocidade média)."""
    scenarios = list(sweep_results.keys())
    methods = _sweep_all_methods(sweep_results)

    data = np.full((len(methods), len(scenarios)), np.nan)
    for j, label in enumerate(scenarios):
        for i, method in enumerate(methods):
            sdata = sweep_results[label]
            if method in sdata["methods"]:
                data[i, j] = sdata["methods"][method]["per_series_error"]["average_speed"]["mape"]

    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 1.5), max(4, len(methods) * 0.9)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="MAPE (%)")

    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)

    for i in range(len(methods)):
        for j in range(len(scenarios)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9,
                        color="black" if v < np.nanmax(data) * 0.7 else "white")

    ax.set_title(f"Heatmap MAPE — Velocidade Média{title_suffix}")
    fig.tight_layout()
    return fig


def _fig_sweep_coverage(sweep_results, title_suffix=""):
    """Cobertura de séries estimadas vs reais para cada cenário e método."""
    scenarios = list(sweep_results.keys())
    methods = _sweep_all_methods(sweep_results)

    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.5), 4))

    for idx, method in enumerate(methods):
        values = []
        for label in scenarios:
            sdata = sweep_results[label]
            cov = sdata["methods"].get(method, {}).get("coverage_pct", 0.0)
            values.append(cov)

        offset = (idx - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, values, width,
               label=method, color=_METHOD_COLORS[idx % len(_METHOD_COLORS)],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.axhline(100, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 115)
    ax.set_xlabel("Cenário Parcial")
    ax.set_ylabel("Cobertura (%)")
    ax.set_title(f"Cobertura por Cenário{title_suffix}")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def save_sweep_pdfs(sweep_results, output_dir, prefix="sweep", title_suffix=""):
    """Salva PDFs de progressão do sweep:
    - {prefix}_progression.pdf  — consolidado com todos os gráficos
    - {prefix}_heatmap.pdf      — heatmap rápido
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # PDF consolidado de progressão
    path = os.path.join(output_dir, f"{prefix}_progression.pdf")
    with PdfPages(path) as pdf:
        # Heatmap visão geral
        fig = _fig_sweep_heatmap(sweep_results, title_suffix)
        pdf.savefig(fig); plt.close(fig)

        # Cobertura
        fig = _fig_sweep_coverage(sweep_results, title_suffix)
        pdf.savefig(fig); plt.close(fig)

        # Linhas de progressão — uma página por métrica
        for metric in _METRICS:
            fig = _fig_sweep_line(sweep_results, metric, title_suffix)
            pdf.savefig(fig); plt.close(fig)

        # Barras agrupadas — uma página por métrica
        for metric in _METRICS:
            fig = _fig_sweep_grouped_bars(sweep_results, metric, title_suffix)
            pdf.savefig(fig); plt.close(fig)

    saved.append(path)
    print(f"  → PDF progressão: {path}")

    return saved
