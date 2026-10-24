"""
Estágios PCA para o pipeline SimEDaPE.

Método alternativo ao DTW: em vez de warping paths, usa K-Shape clustering +
PCA especializado por cluster + mínimos quadrados para reconstruir a série
completa a partir de dados parciais.

Fluxo:
    DataLoadStage → ProcessDatasetStage → ClusteringStage
    → PCATrainStage      (treina PCA por cluster no 100%)
    → PCAEstimationStage (estima a partir do parcial)
    → CompareStage       (compara com new_simulation.json)

Referência: TimeSeriesPCAEstimator em simedape/simedape/estimation/pca_estimator.py
"""
import os
import time
import numpy as np
import joblib
from sklearn.decomposition import PCA

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.source import read_scenario, training_location
from pipeline.io.writer import save_json, ensure_dir
from pipeline.domain.processor import process_dataset, pre_processing_estimation
from pipeline.domain.metrics import calc_metrics
from pipeline.stages.simulation_points import ensure_clustering_loaded


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _denormalize(X, mean, std):
    return (X * std + mean).squeeze()  # (n, size)


def _assign_cluster(partial_series, cluster_centers):
    """Associa série parcial ao cluster mais próximo pelo centroide truncado."""
    plen = len(partial_series)
    dists = []
    for center in cluster_centers:
        c = np.array(center).ravel()[:plen]
        dists.append(np.linalg.norm(partial_series - c))
    return int(np.argmin(dists))


def _reconstruct(partial_series, cluster_model, target_length, use_eig=False):
    """Reconstrói série completa via mínimos quadrados no espaço PCA.

    use_eig : bool
        Se True, aplica filtro de Wiener espectral nos pesos antes de reconstruir.
    """
    pca = cluster_model["pca"]
    complete_mean = cluster_model["mean"]   # (target_length,)
    plen = len(partial_series)

    partial_mean = complete_mean[:plen]
    centered_partial = partial_series - partial_mean

    # P_complete: (target_length, n_components)
    P_complete = pca.components_.T
    P_partial = P_complete[:plen, :]

    weights, _, _, _ = np.linalg.lstsq(P_partial, centered_partial, rcond=None)
    if use_eig:
        weights = _apply_eigenvalue_filter(weights, cluster_model)

    if target_length != len(complete_mean):
        if target_length > len(complete_mean):
            ext_mean = np.concatenate([
                complete_mean,
                np.full(target_length - len(complete_mean), complete_mean[-1]),
            ])
            ext_P = np.vstack([
                P_complete,
                np.zeros((target_length - len(complete_mean), P_complete.shape[1])),
            ])
        else:
            ext_mean = complete_mean[:target_length]
            ext_P = P_complete[:target_length, :]
    else:
        ext_mean = complete_mean
        ext_P = P_complete

    return ext_mean + ext_P @ weights


def _pruning(series):
    return np.where(np.array(series) < 0, 0, np.array(series))


def _apply_eigenvalue_filter(weights, cluster_model):
    """Filtro de Wiener espectral: atenua componentes PCA de baixa variância.

    Para cada componente j:
        w_j_filt = (λ_j / (λ_j + σ²)) · w_j

    σ² = menor eigenvalor retido (proxy do nível de ruído residual).
    Componentes de alta variância (λ_j >> σ²) passam quase intactos;
    componentes de baixa variância (ruído) são atenuados.
    """
    eigenvalues = cluster_model["pca"].explained_variance_  # (n_components,)
    sigma2 = float(eigenvalues[-1])  # menor eigenvalor como proxy de ruído
    filter_coefs = eigenvalues / (eigenvalues + sigma2)
    return weights * filter_coefs


def _stretch_to_length(series, target_length, method="linear"):
    """Interpola (estira) uma série parcial até ``target_length``.

    Métodos disponíveis
    -------------------
    'linear'      : interpolação linear trecho a trecho (scipy.interp1d).
    'poly'        : ajuste polinomial grau 3 por mínimos quadrados (np.polyfit).
    'lagrange'    : polinômio de Lagrange com até 15 pontos de controle.
    'newton'      : interpolação de Newton (KroghInterpolator) com até 10 pontos.
    'vandermonde' : matriz de Vandermonde com até 12 pontos (lstsq).
    'spline'      : spline cúbica natural (scipy.CubicSpline).

    Para os métodos polinomiais de alto grau (lagrange, newton, vandermonde),
    usa subamostras para evitar o fenômeno de Runge.
    """
    src_len = len(series)
    if src_len == target_length:
        return series.copy()

    x_src = np.linspace(0.0, 1.0, src_len)
    x_tgt = np.linspace(0.0, 1.0, target_length)

    if method == "linear":
        from scipy.interpolate import interp1d
        f = interp1d(x_src, series, kind="linear", fill_value="extrapolate")
        return f(x_tgt)

    elif method == "poly":
        # Grau 3 por mínimos quadrados — suave, global, não passa por todos os pontos
        degree = min(3, src_len - 1)
        coeffs = np.polyfit(x_src, series, degree)
        return np.polyval(coeffs, x_tgt)

    elif method == "lagrange":
        from scipy.interpolate import lagrange
        n_ctrl = min(15, src_len)
        idx = np.round(np.linspace(0, src_len - 1, n_ctrl)).astype(int)
        poly = lagrange(x_src[idx], series[idx])
        return np.asarray(poly(x_tgt), dtype=float)

    elif method == "newton":
        from scipy.interpolate import KroghInterpolator
        n_ctrl = min(10, src_len)
        idx = np.round(np.linspace(0, src_len - 1, n_ctrl)).astype(int)
        ki = KroghInterpolator(x_src[idx], series[idx])
        return np.asarray(ki(x_tgt), dtype=float)

    elif method == "vandermonde":
        n_ctrl = min(12, src_len)
        idx = np.round(np.linspace(0, src_len - 1, n_ctrl)).astype(int)
        x_ctrl = x_src[idx]
        y_ctrl = series[idx]
        V = np.vander(x_ctrl, N=n_ctrl, increasing=True)
        coeffs, _, _, _ = np.linalg.lstsq(V, y_ctrl, rcond=None)
        V_tgt = np.vander(x_tgt, N=n_ctrl, increasing=True)
        return (V_tgt @ coeffs).astype(float)

    elif method == "spline":
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x_src, series)
        return np.asarray(cs(x_tgt), dtype=float)

    else:
        raise ValueError(f"Método de interpolação desconhecido: '{method}'. "
                         f"Use: 'linear', 'poly', 'lagrange', 'newton', 'vandermonde', 'spline'.")


def _reconstruct_stretched(partial_series, cluster_model, target_length, interp_method="linear", use_eig=False):
    """Reconstrói série completa via PCA após esticar a série parcial.

    Fluxo:
        1. Interpola partial_series → série de target_length pontos (método ``interp_method``)
        2. Projeta no espaço PCA completo (sem truncar loadings):  weights = Pᵀ(x − μ)
        3. Reconstrói via manifold PCA:  x̂ = μ + P · weights

    Parâmetros
    ----------
    interp_method : str
        Método de interpolação passado para ``_stretch_to_length``.
        Valores: 'linear', 'poly', 'lagrange', 'newton', 'vandermonde', 'spline'.
    use_eig : bool
        Se True, aplica filtro de Wiener espectral nos pesos antes de reconstruir.
    """
    pca = cluster_model["pca"]
    complete_mean = cluster_model["mean"]
    n = len(complete_mean)

    stretched = _stretch_to_length(partial_series, n, method=interp_method)
    centered = stretched - complete_mean

    # Projeção completa: usa todos os loadings (sem truncagem)
    P = pca.components_.T   # (n, n_components)
    weights = P.T @ centered  # equiv. a pca.transform sem StandardScaler
    if use_eig:
        weights = _apply_eigenvalue_filter(weights, cluster_model)

    reconstructed = complete_mean + P @ weights

    if target_length == n:
        return reconstructed
    elif target_length > n:
        ext_mean = np.concatenate([complete_mean, np.full(target_length - n, complete_mean[-1])])
        ext_P = np.vstack([P, np.zeros((target_length - n, P.shape[1]))])
        return ext_mean + ext_P @ weights
    else:
        return reconstructed[:target_length]


# ---------------------------------------------------------------------------
# PCATrainStage
# ---------------------------------------------------------------------------

class PCATrainStage(Stage):
    """Treina modelos K-Shape + PCA por cluster usando o dataset completo (100%).

    Requer que ClusteringStage já tenha sido executado (usa os clusters existentes).
    Salva modelos como .joblib para reuso.

    Parâmetros
    ----------
    n_components : int
        Número de componentes PCA por cluster. Default: 5.
    models_file : str
        Nome do arquivo .joblib de saída (sem extensão). Default: 'pca_models'.
    """

    name = "pca_train"

    def __init__(self, n_components: int = 5, models_file: str = "pca_models",
                 load_from_disk: bool = False):
        self.n_components = n_components
        self.models_file = models_file
        self.load_from_disk = load_from_disk

    def execute(self, ctx: PipelineContext) -> None:
        ensure_clustering_loaded(ctx)
        ensure_dir(ctx.output_dir)

        models_path = os.path.join(ctx.output_dir, f"{self.models_file}.joblib")

        # Carrega do disco se solicitado ou se arquivo existe e load_from_disk=True
        if self.load_from_disk and os.path.exists(models_path):
            print(f"[{self.name}] Carregando modelos do disco: {models_path}")
            data = joblib.load(models_path)
            ctx.pca_models = data["pca_models"]
            ctx.pca_models_path = models_path
            print(f"[{self.name}] {len(ctx.pca_models)} clusters carregados.")
            # Computa train_weights se modelo antigo não os tiver
            _ensure_train_weights(ctx.pca_models, ctx)
            return

        if ctx.X is None or ctx.mean is None or ctx.std is None:
            raise RuntimeError("Execute ProcessDatasetStage antes de PCATrainStage.")

        clustering = ctx.clustering_result
        n_clusters = clustering["number_of_clusters"]
        clusters = np.array(clustering["clusters"])

        # Denormaliza séries
        X_denorm = _denormalize(ctx.X, ctx.mean, ctx.std)  # (n, size)

        print(f"[{self.name}] Treinando PCA por cluster")
        print(f"  n_clusters={n_clusters}, n_components={self.n_components}")
        print(f"  Dataset shape: {X_denorm.shape}")

        pca_models = {}
        t0 = time.time()

        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            cluster_data = X_denorm[mask]
            n_series = len(cluster_data)

            if n_series == 0:
                print(f"  cluster {cluster_id:>2}: vazio, pulando")
                continue

            cluster_mean = cluster_data.mean(axis=0)
            n_comp = min(self.n_components, n_series)
            pca = PCA(n_components=n_comp)
            pca.fit(cluster_data - cluster_mean)

            explained = float(np.sum(pca.explained_variance_ratio_) * 100)

            # Pré-computa pesos de treino por série: w_i = Pᵀ(xᵢ − μ)
            P = pca.components_.T   # (T, n_comp)
            cluster_labels = [ctx.time_series_labels[idx] for idx in np.where(clusters == cluster_id)[0]]
            train_weights_by_label = {
                lbl: (P.T @ (x - cluster_mean)).tolist()
                for lbl, x in zip(cluster_labels, cluster_data)
            }

            # Pesos do SP no treino
            sp_indexs = getattr(ctx, "simulation_points_indexs", None)
            if sp_indexs is not None and cluster_id < len(sp_indexs):
                sp_lbl = ctx.time_series_labels[sp_indexs[cluster_id]]
                sp_weights_train = train_weights_by_label.get(sp_lbl)
            else:
                sp_weights_train = None

            pca_models[cluster_id] = {
                "pca": pca,
                "mean": cluster_mean,
                "size": n_series,
                "n_components": n_comp,
                "explained_variance_pct": round(explained, 2),
                "train_weights_by_label": train_weights_by_label,
                "sp_weights_train": sp_weights_train,
            }
            print(f"  cluster {cluster_id:>2}: {n_series:>4} séries, "
                  f"{n_comp} componentes, variância explicada={explained:.1f}%")

        elapsed = time.time() - t0
        print(f"[{self.name}] Treinamento concluído em {elapsed:.1f}s")

        # Salva modelos
        models_path = os.path.join(ctx.output_dir, f"{self.models_file}.joblib")
        joblib.dump({
            "pca_models": pca_models,
            "n_clusters": n_clusters,
            "n_components": self.n_components,
            "cluster_centers": clustering["centroids"],
        }, models_path)
        ctx.pca_models = pca_models
        ctx.pca_models_path = models_path
        print(f"[{self.name}] Modelos salvos em: {models_path}")


# ---------------------------------------------------------------------------
# PCAEstimationStage
# ---------------------------------------------------------------------------

def _weights_from_sp(sp_partial_series, cluster_model):
    """Calcula os pesos W a partir da série parcial do Simulation Point.

    Os pesos capturam o 'estado' do cluster no cenário parcial e são
    aplicados para reconstruir TODAS as séries daquele cluster.

    W = lstsq(P_parcial, SP_parcial - μ_parcial)
    """
    pca = cluster_model["pca"]
    complete_mean = cluster_model["mean"]
    plen = len(sp_partial_series)

    partial_mean = complete_mean[:plen]
    centered_partial = sp_partial_series - partial_mean

    P_complete = pca.components_.T          # (n_timesteps, n_components)
    P_partial = P_complete[:plen, :]        # truncado ao trecho observado

    weights, _, _, _ = np.linalg.lstsq(P_partial, centered_partial, rcond=None)
    return weights, P_complete, complete_mean


def _weights_from_sp_stretched(sp_partial_series, cluster_model, interp_method="linear"):
    """Calcula pesos W a partir do SP após esticá-lo ao comprimento completo.

    Diferença em relação a ``_weights_from_sp``: em vez de truncar os loadings
    (lstsq com P_parcial), interpola o SP até o comprimento do cluster e faz
    a projeção completa no espaço PCA.

    W = Pᵀ · (SP_esticado − μ)

    Parâmetros
    ----------
    interp_method : str
        Método de interpolação para ``_stretch_to_length``.
    """
    pca = cluster_model["pca"]
    complete_mean = cluster_model["mean"]
    n = len(complete_mean)

    stretched = _stretch_to_length(sp_partial_series, n, method=interp_method)
    P_complete = pca.components_.T          # (n_timesteps, n_components)
    weights = P_complete.T @ (stretched - complete_mean)
    return weights, P_complete, complete_mean


def _reconstruct_from_weights(weights, P_complete, complete_mean, target_length):
    """Reconstrói uma série completa com pesos já calculados."""
    n = len(complete_mean)
    if target_length == n:
        return complete_mean + P_complete @ weights
    elif target_length > n:
        ext_mean = np.concatenate([complete_mean,
                                   np.full(target_length - n, complete_mean[-1])])
        ext_P = np.vstack([P_complete,
                           np.zeros((target_length - n, P_complete.shape[1]))])
    else:
        ext_mean = complete_mean[:target_length]
        ext_P = P_complete[:target_length, :]
    return ext_mean + ext_P @ weights


# ---------------------------------------------------------------------------
# Delta modes — pesos do treino + deslocamento do SP
# ---------------------------------------------------------------------------

def _ensure_train_weights(pca_models, ctx):
    """Garante que cada modelo contenha 'train_weights_by_label' e 'sp_weights_train'.

    Chamado automaticamente pelo PCATrainStage e pelos métodos delta no sweep.
    Se os dados não estiverem nos modelos (modelos antigos), os computa a partir
    de ctx.X (requer que ProcessDatasetStage já tenha rodado).

    Estrutura adicionada a cada cluster_model:
        train_weights_by_label : dict {label: list[float]}  — w_i = Pᵀ(xᵢ − μ)
        sp_weights_train       : list[float]                — w_SP = Pᵀ(s_SP − μ)
    """
    sample = next(iter(pca_models.values()), {})
    if "train_weights_by_label" in sample:
        return  # já computado

    if ctx.X is None or ctx.mean is None or ctx.std is None:
        raise RuntimeError(
            "train_weights_by_label ausente no modelo e ctx.X não disponível.\n"
            "Execute PCATrainStage com load_from_disk: false para retreinar."
        )

    clustering = ctx.clustering_result
    clusters = np.array(clustering["clusters"])
    X_denorm = _denormalize(ctx.X, ctx.mean, ctx.std)
    labels = ctx.time_series_labels
    sp_indexs = getattr(ctx, "simulation_points_indexs", None)

    for cluster_id, model in pca_models.items():
        mask = clusters == cluster_id
        cluster_indices = np.where(mask)[0]
        P = model["pca"].components_.T   # (T, r)
        mean = model["mean"]

        tw = {}
        for idx in cluster_indices:
            lbl = labels[idx]
            w = P.T @ (X_denorm[idx] - mean)
            tw[lbl] = w.tolist()
        model["train_weights_by_label"] = tw

        # SP do treino
        if sp_indexs is not None:
            sp_idx = sp_indexs[cluster_id]
            sp_lbl = labels[sp_idx]
            if sp_lbl in tw:
                model["sp_weights_train"] = tw[sp_lbl]
            else:
                w_sp = P.T @ (X_denorm[sp_idx] - mean)
                model["sp_weights_train"] = w_sp.tolist()
        else:
            model["sp_weights_train"] = None


def _reconstruct_delta(label, cluster_model, sp_weights_new, delta_mode,
                        percentage, use_eig, target_length):
    """Reconstrói série via combinação dos pesos do treino com delta do SP.

    Análogo ao DTW: pré-computa o padrão de deformação no treino (w_train_i)
    e aplica o deslocamento observado no novo cenário (Δw = w_SP_new − w_SP_train).

    delta_mode
    ----------
    'base'  : ŵᵢ = wᵢᵗʳᵃⁱⁿ                  (ignora novo cenário — baseline puro)
    'delta' : ŵᵢ = wᵢᵗʳᵃⁱⁿ + Δw              (deslocamento completo)
    'avg'   : ŵᵢ = wᵢᵗʳᵃⁱⁿ + Δw/2            (deslocamento médio)
    'wavg'  : ŵᵢ = wᵢᵗʳᵃⁱⁿ + (p/100)·Δw     (deslocamento ponderado pelo %)

    Retorna None se o label não estiver no conjunto de treino.
    """
    tw = cluster_model.get("train_weights_by_label", {})
    w_train_i = tw.get(label)
    if w_train_i is None:
        return None  # série não estava no treino

    w_train_i = np.array(w_train_i, dtype=float)

    if delta_mode == "base":
        w_final = w_train_i.copy()
    else:
        sp_w_train = cluster_model.get("sp_weights_train")
        if sp_w_train is None or sp_weights_new is None:
            return None
        delta_w = np.array(sp_weights_new, dtype=float) - np.array(sp_w_train, dtype=float)

        if delta_mode == "delta":
            w_final = w_train_i + delta_w
        elif delta_mode == "avg":
            w_final = w_train_i + delta_w / 2.0
        elif delta_mode == "wavg":
            p = percentage / 100.0
            w_final = w_train_i + p * delta_w
        else:
            raise ValueError(f"delta_mode desconhecido: '{delta_mode}'")

    if use_eig:
        w_final = _apply_eigenvalue_filter(w_final, cluster_model)

    P = cluster_model["pca"].components_.T
    mean = cluster_model["mean"]
    return _reconstruct_from_weights(w_final, P, mean, target_length)


class PCAEstimationStage(Stage):
    """Estima métricas a partir de um cenário parcial usando modelos PCA treinados.

    Três modos de operação controlados por ``mode``:

    mode='sp' (padrão — comparável ao DTW):
        Para cada cluster, carrega a série parcial do Simulation Point (SP),
        calcula os pesos W via mínimos quadrados e usa esses mesmos pesos para
        reconstruir TODAS as séries do cluster.

    mode='ind':
        Cada série é reconstruída individualmente a partir de seus próprios
        dados parciais (lstsq com loadings truncados).

    mode='stretched':
        Cada série parcial é interpolada linearmente até o comprimento completo,
        projetada no espaço PCA completo e reconstruída. Combina o esticamento
        do DTW com a suavização pelo manifold PCA.

    Parâmetros
    ----------
    second_scenario_dir : str
    models_file : str
    output_file : str
    mode : str
        'sp', 'ind' ou 'stretched'. Default: 'sp'.
    use_sp : bool
        Deprecated. Se fornecido, sobrescreve: True → 'sp', False → 'ind'.
    """

    name = "pca_estimation"

    def __init__(
        self,
        second_scenario_dir: str = "",
        models_file: str = "pca_models",
        output_file: str = "result_pca",
        mode: str = "sp",
        use_sp: bool = None,
    ):
        self.second_scenario_dir = second_scenario_dir
        self.models_file = models_file
        self.output_file = output_file
        # backward compat: use_sp sobrescreve mode se passado explicitamente
        if use_sp is not None:
            self.mode = "sp" if use_sp else "ind"
        else:
            self.mode = mode

    def execute(self, ctx: PipelineContext) -> None:
        ensure_clustering_loaded(ctx)
        cfg = ctx.config
        ensure_dir(ctx.output_dir)

        # Carrega modelos PCA
        pca_models = getattr(ctx, "pca_models", None)
        if pca_models is None:
            models_path = os.path.join(ctx.output_dir, f"{self.models_file}.joblib")
            if not os.path.exists(models_path):
                raise RuntimeError(
                    f"Modelos PCA não encontrados: '{models_path}'\n"
                    "Execute PCATrainStage primeiro."
                )
            data = joblib.load(models_path)
            pca_models = data["pca_models"]
            cluster_centers = data["cluster_centers"]
            print(f"[{self.name}] Modelos PCA carregados de: {models_path}")
        else:
            cluster_centers = ctx.clustering_result["centroids"]

        # Carrega cenário parcial
        second_dir = self.second_scenario_dir or training_location(cfg)
        print(f"[{self.name}] Carregando cenário parcial: {second_dir}")
        _, ss2, _, _, labels_2, _ = read_scenario(cfg, second_dir)

        clustering = ctx.clustering_result
        centroid_size = len(clustering["centroids"]["0"])
        X_2, mean_2, std_2 = process_dataset(ss2, centroid_size)
        X_2_denorm = _denormalize(X_2, mean_2, std_2)  # (n, size)
        _, labels_2_dict = pre_processing_estimation(labels_2)

        n_clusters = clustering["number_of_clusters"]
        centers_np = [
            np.array(cluster_centers[str(i)]).ravel()
            for i in range(n_clusters)
        ]

        mode_str = {"sp": "SP (Simulation Point)", "ind": "individual por série",
                    "stretched": "stretched (interpola → PCA completo)"}.get(self.mode, self.mode)
        print(f"[{self.name}] Modo: {mode_str}")
        print(f"[{self.name}] Séries no cenário parcial: {len(labels_2)}")

        metrics = {
            "average_speed": [],
            "percentage_of_track": [],
            "average_vehicles": [],
            "km_traveled": [],
            "missing_time_series": 0,
        }

        t0 = time.time()
        processed = 0

        if self.mode == "sp":
            # ── Modo SP: um W por cluster, derivado do Simulation Point ──
            if ctx.simulation_points_indexs is None or ctx.time_series_labels is None:
                raise RuntimeError(
                    "Simulation Points não encontrados no contexto.\n"
                    "Execute SimulationPointsStage antes de PCAEstimationStage."
                )
            sp_indexs = ctx.simulation_points_indexs       # lista: sp_indexs[i] → índice global do SP do cluster i
            ts_labels_first = ctx.time_series_labels       # labels completas (100%)
            ts_indexs_first, ts_dict_first = pre_processing_estimation(ts_labels_first)
            ts_indexs_array = np.array(ts_indexs_first)
            clusters_array = np.array(clustering["clusters"])

            # Pré-calcula W para cada cluster a partir do SP parcial
            cluster_weights = {}
            for i in range(n_clusters):
                if i not in pca_models:
                    print(f"  cluster {i:>2}: sem modelo PCA — pulando")
                    continue

                sp_label = ts_labels_first[sp_indexs[i]]
                if sp_label not in labels_2_dict:
                    print(f"  cluster {i:>2}: SP '{sp_label}' ausente no parcial — pulando cluster")
                    continue

                sp_partial = X_2_denorm[labels_2_dict[sp_label]]
                weights, P_complete, complete_mean = _weights_from_sp(sp_partial, pca_models[i])
                cluster_weights[i] = (weights, P_complete, complete_mean)
                print(f"  cluster {i:>2}: pesos calculados a partir de '{sp_label}' "
                      f"({len(sp_partial)} pontos)")

            # Reconstrói todas as séries usando o W do respectivo cluster
            for k, label in enumerate(ts_labels_first):
                link_id = label.split("-")[0]
                if link_id not in (ctx.streets_info or {}):
                    metrics["missing_time_series"] += 1
                    continue

                cluster_id = int(clusters_array[k])
                if cluster_id not in cluster_weights:
                    metrics["missing_time_series"] += 1
                    continue

                weights, P_complete, complete_mean = cluster_weights[cluster_id]
                estimated = _reconstruct_from_weights(weights, P_complete, complete_mean, centroid_size)
                estimated = _pruning(estimated)

                calc = calc_metrics(estimated, link_id, ctx.streets_info)
                metrics["average_speed"].append(calc["average_speed"])
                metrics["percentage_of_track"].append(calc["percentage_of_track"])
                metrics["average_vehicles"].append(calc["average_vehicles"])
                metrics["km_traveled"].append(calc["km_traveled"])
                processed += 1

        elif self.mode == "stretched":
            # ── Modo stretched: interpola série parcial → comprimento completo → PCA ──
            cluster_centers_raw = ctx.clustering_result["centroids"]
            centers_np_str = [
                np.array(cluster_centers_raw[str(k)]).ravel()
                for k in sorted(int(k) for k in cluster_centers_raw)
            ]
            for idx, label in enumerate(labels_2):
                link_id = label.split("-")[0]
                if link_id not in (ctx.streets_info or {}):
                    metrics["missing_time_series"] += 1
                    continue
                if label not in labels_2_dict:
                    metrics["missing_time_series"] += 1
                    continue

                partial_series = X_2_denorm[labels_2_dict[label]]
                cluster_id = _assign_cluster(partial_series, centers_np_str)

                if cluster_id not in pca_models:
                    metrics["missing_time_series"] += 1
                    continue

                estimated = _reconstruct_stretched(partial_series, pca_models[cluster_id], centroid_size)
                estimated = _pruning(estimated)

                calc = calc_metrics(estimated, link_id, ctx.streets_info)
                metrics["average_speed"].append(calc["average_speed"])
                metrics["percentage_of_track"].append(calc["percentage_of_track"])
                metrics["average_vehicles"].append(calc["average_vehicles"])
                metrics["km_traveled"].append(calc["km_traveled"])
                processed += 1

        else:
            # ── Modo individual (ind): W calculado série a série com lstsq truncado ──
            for idx, label in enumerate(labels_2):
                link_id = label.split("-")[0]
                if link_id not in (ctx.streets_info or {}):
                    metrics["missing_time_series"] += 1
                    continue
                if label not in labels_2_dict:
                    metrics["missing_time_series"] += 1
                    continue

                partial_series = X_2_denorm[labels_2_dict[label]]
                cluster_id = _assign_cluster(partial_series, centers_np)

                if cluster_id not in pca_models:
                    metrics["missing_time_series"] += 1
                    continue

                estimated = _reconstruct(partial_series, pca_models[cluster_id], centroid_size)
                estimated = _pruning(estimated)

                calc = calc_metrics(estimated, link_id, ctx.streets_info)
                metrics["average_speed"].append(calc["average_speed"])
                metrics["percentage_of_track"].append(calc["percentage_of_track"])
                metrics["average_vehicles"].append(calc["average_vehicles"])
                metrics["km_traveled"].append(calc["km_traveled"])
                processed += 1

        elapsed = time.time() - t0

        from scipy.stats import gmean
        def _sg(v): return float(gmean([x for x in v if x and x > 0])) if v else float("nan")

        print(f"[{self.name}] Estimação concluída em {elapsed:.1f}s")
        print(f"  Séries estimadas: {processed} | ausentes: {metrics['missing_time_series']}")
        print(f"  Velocidade Média (gmean): {_sg(metrics['average_speed']):.4f}")
        print(f"  Ocupação        (gmean): {_sg(metrics['percentage_of_track']):.4f}")
        print(f"  Veículos        (gmean): {_sg(metrics['average_vehicles']):.4f}")
        print(f"  Km Trafegados   (gmean): {_sg(metrics['km_traveled']):.4f}")

        out_path = os.path.join(ctx.output_dir, self.output_file)
        save_json(out_path, metrics)
        ctx.estimation_results["pca"] = metrics
        print(f"[{self.name}] Resultados salvos em: {out_path}.json")
