import pandas as pd
import numpy as np


def apply_loo_bias_correction(
        df,
        time_col='time',
        site_col='site',
        target_col='res',
        window_back='15D',
        window_fwd='15D',
        min_periods=1
):
    """
    对每个测站进行基于滑动窗口的偏差校正（使用留一法 LOO-CMA）。

    参数:
    ----------
    df : pd.DataFrame
        包含数据的 DataFrame。
    time_col : str
        时间列名 (需要是 datetime 格式)。
    site_col : str
        测站列名。
    target_col : str
        需要计算偏差的目标列名 (这里通常是 'res')。
    window_back : str
        向过去回溯的时间窗口 (例如 '15D')。
    window_fwd : str
        向未来前瞻的时间窗口 (例如 '15D')。
    min_periods : int
        窗口内计算偏差所需的最小样本数 (不含当前点)。

    返回:
    ----------
    df_out : pd.DataFrame
        在原 df 基础上增加了 'bias' 和 'res_cor' 两列。
    """

    # 1. 准备数据：确保不修改原数据，并设置时间索引
    df_work = df.copy()
    if not np.issubdtype(df_work[time_col].dtype, np.datetime64):
        df_work[time_col] = pd.to_datetime(df_work[time_col])

    # 设置索引为时间，以便进行基于时间的 rolling
    # 注意：为了 transform 能正常工作，我们暂不set_index，而是在内部处理

    print(f"开始计算偏差校正...")
    print(f"  窗口设置: -{window_back} 到 +{window_fwd}")
    print(f"  策略: 留一法 (Leave-One-Out)")

    # 定义核心计算逻辑 (针对单个测站的时间序列)
    def calculate_loo_bias(series):
        # series 的索引必须是时间
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)

        # 排序以防万一
        series = series.sort_index()

        # --- A. 计算向后的窗口 [t - back, t] ---
        # closed='both' 表示包含边界 t
        roll_back = series.rolling(window=window_back, min_periods=0, closed='both')
        sum_back = roll_back.sum()
        cnt_back = roll_back.count()

        # --- B. 计算向前的窗口 [t, t + fwd] ---
        # 技巧：将序列时间倒序，然后做 rolling，再倒序回来
        # 这样原先的“未来”对于 rolling 来说就变成了“过去”
        series_rev = series.sort_index(ascending=False)
        roll_fwd = series_rev.rolling(window=window_fwd, min_periods=0, closed='both')

        # 注意：结果需要按索引重新排序回正常时间顺序
        sum_fwd = roll_fwd.sum().sort_index()
        cnt_fwd = roll_fwd.count().sort_index()

        # --- C. 合并窗口 [t - back, t + fwd] ---
        # 因为 closed='both'，当前时刻 t 在 sum_back 和 sum_fwd 中都被算了一次
        # 所以 Total = Back + Fwd - Current
        total_sum = sum_back + sum_fwd - series
        total_cnt = cnt_back + cnt_fwd - 1

        # --- D. 执行留一法 (Exclude Current) ---
        # 从总和中减去当前时刻的值
        loo_sum = total_sum - series
        loo_cnt = total_cnt - 1

        # 计算均值 (处理除以0的情况)
        # 如果 loo_cnt < min_periods，则 bias 设为 NaN
        bias = loo_sum / loo_cnt
        bias[loo_cnt < min_periods] = np.nan

        return bias

    # 2. 分组应用
    # 我们需要先将 index 设置为 time，再 groupby，这样 transform 传入的就是带时间索引的 Series
    df_work = df_work.set_index(time_col).sort_index()

    # 使用 transform 可以直接返回与原 DataFrame 长度一致的序列
    # 这里的 x 是每个 site 的 res 列 (带时间索引)
    bias_series = df_work.groupby(site_col)[target_col].transform(calculate_loo_bias)

    # 3. 整理结果
    # 因为做过 set_index 和 sort_index，需要还原顺序以匹配原 df (如果需要的话)
    # 这里我们直接将结果赋回去，pandas 会自动对齐索引

    # 恢复索引以便合并
    df_work = df_work.reset_index()

    # 将计算出的 bias 赋值回原数据 (注意索引对齐)
    # 为了保险，我们使用 merge 或者直接通过索引赋值（如果 df_work index 没乱）
    df_out = df.copy()

    # 必须保证 df_out 和 bias_series 的索引对齐逻辑。
    # 最安全的方法是 merge on key，或者确信 index 未变。
    # 这里我们利用 pandas 的自动索引对齐功能：

    # 给 df_work 加上 bias
    df_work['bias'] = bias_series.values  # 此时 df_work 已经是排好序的

    # 计算 res_cor
    df_work['res_cor'] = df_work[target_col] - df_work['bias']

    # 4. 返回用户需要的列，保持原 df 的顺序（可选，视需求而定）
    # 这里我们返回排序后的 df_work，通常时序数据分析需要按时间排序
    df_work['ztd_nwm_cor'] = df_work['ztd_nwm'] - df_work['bias']
    return df_work

