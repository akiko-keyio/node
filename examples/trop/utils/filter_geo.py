import numpy as np
import pandas as pd


def filter_points_sphere(data,
                         min_sep_deg: float = 1.0,
                         lon_col: str = 'lon',
                         lat_col: str = 'lat',
                         randomize: bool = True) -> np.ndarray:
    """
    过滤球面点，使任意两点夹角 > min_sep_deg（默认 1°）

    Parameters
    ----------
    data : pd.DataFrame 或 (N,2) ndarray
        - DataFrame：需包含经度列 `lon_col` 与纬度列 `lat_col`
        - ndarray：形状 (N,2)，列顺序需为 [lat, lon]（与旧版保持兼容）
    min_sep_deg : float, optional
        最小允许角距离（度）
    lon_col, lat_col : str, optional
        DataFrame 对应列名
    randomize : bool, optional
        是否在筛选前随机打乱顺序，减少顺序偏差

    Returns
    -------
    mask : np.ndarray(bool)
        与输入等长的布尔数组，True 表示该点被保留
    """
    # ---------- 解析输入 ----------
    if min_sep_deg is None:
        return np.ones(len(data), dtype=bool)
    if isinstance(data, pd.DataFrame):
        lon = data[lon_col].to_numpy(dtype=float)
        lat = data[lat_col].to_numpy(dtype=float)
        lat_lon_deg = np.column_stack((lat, lon))  # 转成旧版顺序 [lat, lon]
    else:  # 假定为 ndarray
        lat_lon_deg = np.asarray(data, dtype=float)
        if lat_lon_deg.shape[1] != 2:
            raise ValueError("ndarray 输入必须形如 (N,2) 且列顺序 [lat, lon]")

    # ---------- 转三维单位向量 ----------
    lat_r = np.deg2rad(lat_lon_deg[:, 0])
    lon_r = np.deg2rad(lat_lon_deg[:, 1])
    vecs = np.column_stack((np.cos(lat_r) * np.cos(lon_r),
                            np.cos(lat_r) * np.sin(lon_r),
                            np.sin(lat_r)))

    # ---------- 贪心筛选 ----------
    cos_min = np.cos(np.deg2rad(min_sep_deg))
    idx = np.arange(len(vecs))
    if randomize:
        np.random.shuffle(idx)

    kept_idx = []
    for i in idx:
        v = vecs[i]
        if kept_idx and np.any(np.dot(vecs[kept_idx], v) > cos_min):
            continue
        kept_idx.append(i)

    # ---------- 构造 bool 掩码 ----------
    mask = np.zeros(len(vecs), dtype=bool)
    mask[kept_idx] = True
    return mask
import numpy as np

def vce_two_components(sig_g_sq, sig_n_sq, d, *,
                       max_iter=50, tol=1e-8,
                       return_cov=False):
    k_g2, k_n2 = 1.0, 1.0
    for _ in range(max_iter):
        var_d = k_g2*sig_g_sq + k_n2*sig_n_sq
        w     = 1.0 / (2.0 * var_d**2)

        # 正规方程
        r_g = 0.5 * np.sum((d**2 - var_d) * sig_g_sq * w)
        r_n = 0.5 * np.sum((d**2 - var_d) * sig_n_sq * w)
        N_gg = 0.5 * np.sum(sig_g_sq**2       * w)
        N_nn = 0.5 * np.sum(sig_n_sq**2       * w)
        N_gn = 0.5 * np.sum(sig_g_sq*sig_n_sq * w)

        det   = N_gg*N_nn - N_gn**2
        dk_g2 = ( r_g*N_nn - r_n*N_gn) / det
        dk_n2 = (-r_g*N_gn + r_n*N_gg) / det

        k_g2_new, k_n2_new = k_g2+dk_g2, k_n2+dk_n2
        if max(abs(dk_g2), abs(dk_n2)) < tol:
            k_g2, k_n2 = k_g2_new, k_n2_new
            break
        k_g2, k_n2 = k_g2_new, k_n2_new

    if not return_cov:
        return k_g2, k_n2

    # ---------- 协方差矩阵 ----------
    var_d = k_g2*sig_g_sq + k_n2*sig_n_sq
    inv_v2 = 1.0/var_d**2
    N_gg = 0.5*np.sum(sig_g_sq**2       * inv_v2)
    N_nn = 0.5*np.sum(sig_n_sq**2       * inv_v2)
    N_gn = 0.5*np.sum(sig_g_sq*sig_n_sq * inv_v2)
    N = np.array([[N_gg, N_gn],
                  [N_gn, N_nn]])
    cov = np.linalg.inv(N)
    return k_g2, k_n2, cov

