import numpy as np


def delete_outlier(residual, outlier_mask, threshold=5, method='sigma', min_absolute_threshold=0):
    inner_bias = residual[~outlier_mask]
    if method == 'sigma':
        sigma = inner_bias.std()
        mean_bias = inner_bias.mean()
        tsigma = max(threshold * sigma, min_absolute_threshold)
        upper_limit = mean_bias + tsigma
        lower_limit = mean_bias - tsigma
    elif method == 'iqr':
        Q1 = np.percentile(inner_bias, 25)
        Q3 = np.percentile(inner_bias, 75)
        IQR = Q3 - Q1
        tIQR = max(threshold * IQR, min_absolute_threshold)
        lower_limit = Q1 - tIQR
        upper_limit = Q3 + tIQR
    else:
        raise ValueError("Method must be either 'sigma' or 'iqr'.")

    return (residual < lower_limit) | (residual > upper_limit)


def iterative_delete_outliers(residual, **kwargs):
    outlier_mask = np.zeros_like(residual, dtype=bool)
    for i in range(100):
        new_outlier_mask = delete_outlier(residual, outlier_mask, **kwargs)
        new_outlier_mask = np.logical_or(outlier_mask, new_outlier_mask)

        # Calculate the difference between the new mask and the old mask
        if np.array_equal(new_outlier_mask, outlier_mask):
            # print('No more outliers detected.')
            return outlier_mask
        else:
            # print(f"Detected {new_outlier_mask.sum() - outlier_mask.sum()} new outliers, total {new_outlier_mask.sum()}")
            outlier_mask = new_outlier_mask
    else:
        raise OverflowError("Too many iterations without convergence.")