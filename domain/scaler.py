import numpy
from sklearn.base import TransformerMixin
from tslearn.bases import TimeSeriesMixin
from tslearn.utils import to_time_series_dataset, ts_size


class TimeSeriesScalerSimEDaPE(TransformerMixin, TimeSeriesMixin):
    """Z-score scaler for time series. Normalizes each series individually
    to zero mean and unit standard deviation. Stores per-series mean and std
    so the transformation can be reverted.

    Parameters
    ----------
    mu : float (default: 0.)
        Target mean of the output time series.
    std : float (default: 1.)
        Target standard deviation of the output time series.
    """

    def __init__(self, mu=0., std=1.):
        self.mu = mu
        self.std = std

    def fit(self, X, y=None, **kwargs):
        return self

    def fit_transform(self, X, y=None, **kwargs):
        n = X.shape[0]
        sz = X.shape[1]
        d = X.shape[2]

        means = numpy.zeros((n, 1, d))
        stds = numpy.zeros((n, 1, d))
        X_normalized = numpy.zeros((n, sz, d))

        for i in range(n):
            for dim in range(d):
                series = X[i, :ts_size(X[i]), dim]
                mu_i = numpy.nanmean(series)
                std_i = numpy.nanstd(series)
                means[i, 0, dim] = mu_i
                stds[i, 0, dim] = std_i if std_i != 0 else 1.0
                X_normalized[i, :, dim] = (X[i, :, dim] - mu_i) / stds[i, 0, dim]

        return X_normalized, means, stds

    def transform(self, X, y=None, **kwargs):
        normalized, means, stds = self.fit_transform(X)
        return normalized, means, stds
