import numpy as np
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
from pipeline.domain.scaler import TimeSeriesScalerSimEDaPE


def process_dataset(data, size):
    """Converte lista de séries para formato tslearn, reamostrada para `size`
    pontos e normalizada (Z-score por série).

    Retorna
    -------
    X : ndarray (n_series, size, 1)  - séries normalizadas
    mean : ndarray (n_series, 1, 1) - média original por série
    std  : ndarray (n_series, 1, 1) - desvio padrão original por série
    """
    import time
    print(f"  [1/3] Convertendo {len(data)} séries para formato tslearn...")
    t0 = time.time()
    X = to_time_series_dataset(data)
    print(f"        Shape original: {X.shape}  ({time.time()-t0:.1f}s)")

    print(f"  [2/3] Reamostrando para {int(size)} pontos (TimeSeriesResampler)...")
    t0 = time.time()
    X = TimeSeriesResampler(int(size)).fit_transform(X)
    print(f"        Shape reamostrado: {X.shape}  ({time.time()-t0:.1f}s)")

    print(f"  [3/3] Normalizando (Z-score por série)...")
    t0 = time.time()
    X, mean, std = TimeSeriesScalerSimEDaPE().fit_transform(X)
    print(f"        Normalização concluída  ({time.time()-t0:.1f}s)")
    return X, mean, std


def process_dataset_interp(data, size, interp_method="linear"):
    """Como ``process_dataset``, mas usa ``_stretch_to_length`` para reamostrar.

    Permite testar diferentes métodos de interpolação (linear, poly, lagrange,
    newton, vandermonde, spline) no lugar do ``TimeSeriesResampler`` padrão.
    """
    from pipeline.stages.pca import _stretch_to_length

    stretched = []
    for s in data:
        arr = np.array(s).ravel().astype(float)
        stretched.append(_stretch_to_length(arr, int(size), method=interp_method))

    X = to_time_series_dataset(stretched)
    X, mean, std = TimeSeriesScalerSimEDaPE().fit_transform(X)
    return X, mean, std


def pre_processing_estimation(time_series_labels):
    """Cria duas estruturas de indexação para lookup rápido por label:

    - indexs: lista [0, 1, 2, ...] (índices inteiros)
    - label_dict: dict {label -> index}
    """
    indexs = list(range(len(time_series_labels)))
    label_dict = {label: i for i, label in enumerate(time_series_labels)}
    return indexs, label_dict


def process_dataset_multimodal(data, size, mode_column=None):
    """Processa um dataset que pode conter séries temporais de múltiplos modos
    de transporte.

    Parameters
    ----------
    data : list[list[float]] or pandas.DataFrame
        Séries temporais como lista de listas (formato original) **ou** um
        DataFrame com uma coluna ``mode`` opcional.  Quando um DataFrame é
        fornecido, as séries são agrupadas por modo antes de processar; a
        função retorna um dict por modo.
    size : int
        Tamanho alvo para reamostrar cada série.
    mode_column : str, optional
        Nome da coluna de modo no DataFrame (padrão ``"mode"``).
        Ignorado quando ``data`` não é um DataFrame.

    Returns
    -------
    Se ``data`` é uma lista:
        Igual a :func:`process_dataset` — ``(X, mean, std)``.
    Se ``data`` é um DataFrame com coluna de modo:
        dict ``{mode_str: (X, mean, std)}`` — uma entrada por modo presente.
    """
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        # Fallback: same behaviour as original process_dataset
        return process_dataset(data, size)

    col = mode_column or "mode"
    if col not in data.columns:
        # No mode column — treat entire DataFrame as single series collection
        series_list = _dataframe_to_series(data)
        return process_dataset(series_list, size)

    modes = data[col].unique().tolist()
    results: dict = {}
    for mode in modes:
        subset = data[data[col] == mode]
        series_list = _dataframe_to_series(subset)
        if not series_list:
            continue
        print(f"  [multimodal] Processing mode='{mode}': {len(series_list)} series")
        results[str(mode)] = process_dataset(series_list, size)
    return results


def _dataframe_to_series(df) -> list:
    """Converts a DataFrame with per-tick records into a list of time-series.

    The DataFrame is expected to have at least a numeric value column.
    Recognised value columns (in priority order): ``count``, ``speed``,
    ``travel_time``.  Rows are grouped by ``link_id`` / ``agent_id`` when
    those columns exist; otherwise the entire column is returned as one series.

    This is a best-effort helper — callers that need precise grouping should
    prepare ``data`` as a list of lists themselves.
    """
    value_col = None
    for candidate in ("count", "speed", "travel_time"):
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        # Use first numeric column as fallback
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
        else:
            return []

    group_col = None
    for candidate in ("link_id", "agent_id"):
        if candidate in df.columns:
            group_col = candidate
            break

    if group_col is not None:
        series_list = []
        for _, group in df.groupby(group_col, sort=False):
            s = group[value_col].tolist()
            if len(s) > 1:
                series_list.append(s)
        return series_list

    return [df[value_col].tolist()]
