# 文件名: spread_skill_tools.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from trop.vce.vce import estimate_vce_generalized

# --------------------
# 1. 全局视觉配置
# --------------------
COL_WITH = "#1f77b4"  # 蓝色
COL_WITHOUT = "#ff7f0e"  # 橙色
COL_DIAG = "#9e9e9e"  # 灰色对角线


# --------------------
# 2. 核心算法
# --------------------

def _ols_seed_for_sd_fit(x, y):
    """非线性拟合的初值猜测"""
    X, Y = x ** 2, y ** 2
    if len(X) < 2 or np.allclose(X.var(), 0):
        return 1.0, max(float(np.nanmin(y)), 1e-6)
    A, B = np.polyfit(X, Y, 1)
    a0 = float(np.sqrt(max(A, 1e-12)))
    b0 = float(np.sqrt(max(B, 1e-12)))
    return a0, b0


def _gn_fit_sd(x, y, a0, b0, max_iter=50, lam=1e-3, tol=1e-6):
    """高斯-牛顿法求解非线性最小二乘"""
    a, b = float(a0), float(b0)
    x2 = x ** 2
    for _ in range(max_iter):
        yhat = np.sqrt(np.maximum((a * a) * x2 + b * b, 1e-12))
        r = yhat - y
        denom = np.maximum(yhat, 1e-12)
        J1 = (a * x2) / denom
        J2 = (b) / denom
        JTJ_11 = np.sum(J1 * J1) + lam
        JTJ_12 = np.sum(J1 * J2)
        JTJ_22 = np.sum(J2 * J2) + lam
        JTr_1 = np.sum(J1 * r)
        JTr_2 = np.sum(J2 * r)
        det = JTJ_11 * JTJ_22 - JTJ_12 * JTJ_12
        if det <= 0: break
        da = (-JTJ_22 * JTr_1 + JTJ_12 * JTr_2) / det
        db = (JTJ_12 * JTr_1 - JTJ_11 * JTr_2) / det
        a_new = max(a + da, 1e-8)
        b_new = max(b + db, 1e-8)
        if abs(da) < tol * max(1.0, a) and abs(db) < tol * max(1.0, b):
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return a, b


def fit_rmse(x, y, linear: bool):
    """拟合接口"""
    if linear:
        a, b = np.polyfit(x, y, 1)
    else:
        a0, b0 = _ols_seed_for_sd_fit(x, y)
        a, b = _gn_fit_sd(x, y, a0, b0)
    return float(a), float(b)


def bin_and_aggregate(g: pd.DataFrame, mean_col: str, xaxis: str,
                      n_bins=20, samples_per_bin=None, min_per_bin=2) -> pd.DataFrame:
    """
    分箱并计算 Spread 与 Skill
    :param samples_per_bin: 若设置(>0)，则忽略 n_bins，根据样本量动态计算箱数
    """
    # 基础校验
    total_valid = g[xaxis].notna().sum()
    if total_valid < max(2, min_per_bin):
        return pd.DataFrame(columns=["RMS Spread", 'RMS Error', 'bin_count', 'bin_median_spread'])

    gg = g.copy()
    gg['e'] = gg[mean_col]-gg['ztd_gnss']
    gg['s2'] = gg[xaxis] ** 2

    # ---- 分箱逻辑修改开始 ----
    if samples_per_bin is not None and samples_per_bin > 0:
        # 动态计算 n_bins
        # 比如 50000 数据，20000 一箱 -> 2.5 -> 3箱
        calculated_bins = int(np.ceil(total_valid / samples_per_bin))
        # 至少保证有1箱，如果数据极少，至少也是1箱
        n_bins_to_use = max(1, calculated_bins)
        # 如果计算出 bins 太多导致每箱少于 min_per_bin，这里暂不处理，由后面 qcut 处理
    else:
        n_bins_to_use = n_bins

    try:
        q = pd.qcut(gg[xaxis], q=n_bins_to_use, labels=False, duplicates='drop')
    except ValueError:
        uniq = gg[xaxis].nunique(dropna=True)
        if uniq < 2:
            return pd.DataFrame(columns=["RMS Spread", 'RMS Error', 'bin_count', 'bin_median_spread'])
        q = pd.qcut(gg[xaxis], q=min(int(uniq), n_bins_to_use), labels=False, duplicates='drop')

    gg['bin_id'] = q
    # ---- 分箱逻辑修改结束 ----

    rows = []
    N_target = 50
    factor = (N_target + 1) / N_target

    for _, gbin in gg.groupby('bin_id'):
        if len(gbin) < min_per_bin: continue

        # ERMSE
        mean_s2 = gbin['s2'].mean()
        ermse = np.sqrt(factor * mean_s2)
        # RMSE
        err = gbin['e'].to_numpy()
        rmse = np.sqrt(np.mean(err ** 2))

        rows.append({
            "RMS Spread": ermse,
            'RMS Error': rmse,
            'bin_count': len(gbin),
            'bin_median_spread': np.median(gbin[xaxis])
        })

    if not rows:
        return pd.DataFrame(columns=["RMS Spread", 'RMS Error', 'bin_count', 'bin_median_spread'])

    agg = pd.DataFrame(rows)
    return agg.sort_values('bin_median_spread')[["RMS Spread", 'RMS Error', 'bin_count', 'bin_median_spread']]


def pretty_label(x: str) -> str:
    if x.startswith('With '):    return "<b>With</b> Bias Corrections"
    if x.startswith('Without '): return "<b>Without</b> Bias Corrections"
    return x


# --------------------
# 3. 绘图主函数
# --------------------

def plot_spread_skill_matrix(df_all,
                              col_var,  # col_var 仍为必填
                              row_var=None,  # row_var 改为可选
                              template='simple_white',
                              xaxis='ztd_nwm_sigma',
                              row_order=None, col_order=None,
                              include_global=True,
                              fit_linear=True,
                              n_bins=20,
                              samples_per_bin=None,  # 默认按样本量分箱
                              ax_min=0.0,
                              width=1200, height=970):
    # --- A. 数据准备 ---
    _base = df_all.copy()
    if not np.issubdtype(_base['time'].dtype, np.datetime64):
        _base['time'] = pd.to_datetime(_base['time'])
    _base = _base[np.isfinite(_base[xaxis])].copy()

    # 扩展 Global
    if include_global:
        _global = _base.copy()
        _global[col_var] = 'Global'
        if col_order and 'Global' not in col_order:
            col_order.append('Global')
        _base = pd.concat([_base, _global], ignore_index=True)

    # 处理 row_var 为 None 的情况
    # 为了复用 groupby([row, col]) 逻辑，我们创建一个虚拟列
    internal_row_var = row_var
    if row_var is None:
        internal_row_var = '__SingleRow__'
        _base[internal_row_var] = ' '  # 空格作为标签，避免显示太突兀
        if row_order is None:
            row_order = [' ']

    need = ['time', 'ztd_gnss', 'ztd_nwm', 'ztd_nwm_cor', xaxis, internal_row_var, col_var]
    _base = _base.dropna(subset=need)

    # 过滤顺序
    if row_order:
        _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
    if col_order:
        _base = _base[_base[col_var].astype(str).isin(col_order)]

    # --- B. 聚合计算 ---
    agg_list = []
    targets = [('ztd_nwm_cor', 'With Bias Corrections'),
               ('ztd_nwm', 'Without Bias Corrections')]

    for (r_val, c_val), g in _base.groupby([internal_row_var, col_var], observed=True):
        for mean_col, mean_name in targets:
            # 传入新的 samples_per_bin 参数
            res = bin_and_aggregate(g, mean_col, xaxis, n_bins=n_bins, samples_per_bin=samples_per_bin)
            if not res.empty:
                res[internal_row_var] = r_val
                res[col_var] = c_val
                res['MeanType'] = mean_name
                agg_list.append(res)

    if not agg_list:
        print("No valid data.")
        return None

    agg_all = pd.concat(agg_list, ignore_index=True)
    agg_all['MeanLabel'] = agg_all['MeanType'].map(pretty_label)

    # 计算坐标轴上限
    max_val = agg_all[["RMS Spread", 'RMS Error']].max().max()
    AX_MAX = np.ceil(max_val*1.1)

    # --- C. 对角线数据 ---
    diag_rows = []
    use_rows = row_order if row_order else sorted(agg_all[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(agg_all[col_var].unique())

    for r in use_rows:
        for c in use_cols:
            diag_rows.append({
                "RMS Spread": ax_min, 'RMS Error': ax_min,
                internal_row_var: r, col_var: c, 'MeanLabel': 'y = x'
            })
            diag_rows.append({
                "RMS Spread": AX_MAX, 'RMS Error': AX_MAX,
                internal_row_var: r, col_var: c, 'MeanLabel': 'y = x'
            })

    plot_df = pd.concat([agg_all, pd.DataFrame(diag_rows)], ignore_index=True)

    # --- D. 拟合计算 ---
    fit_rows = []
    curve_rows = []
    mask_real = (plot_df['MeanLabel'] != 'y = x') & plot_df['bin_count'].notna()

    for (r, c, lbl), g in plot_df[mask_real].groupby([internal_row_var, col_var, 'MeanLabel']):
        x_dat = g["RMS Spread"].to_numpy()
        y_dat = g['RMS Error'].to_numpy()
        if len(x_dat) < 2: continue

        a, b = fit_rmse(x_dat, y_dat, fit_linear)
        fit_rows.append({'row': r, 'col': c, 'label': lbl, 'a': a, 'b': b})

        xs = np.linspace(ax_min, AX_MAX, 160)
        if fit_linear:
            ys = a * xs + b
        else:
            ys = np.sqrt(np.maximum((a * xs) ** 2 + b * b, 0.0))

        tmp = pd.DataFrame({"RMS Spread": xs, 'RMS Error': ys})
        tmp[internal_row_var] = r;
        tmp[col_var] = c;
        tmp['MeanLabel'] = lbl
        curve_rows.append(tmp)

    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    # --- E. 绘图 ---
    color_map = {
        "<b>With</b> Bias Corrections": COL_WITH,
        "<b>Without</b> Bias Corrections": COL_WITHOUT,
        "y = x": COL_DIAG
    }

    # 决定是否使用 facet_row
    # 如果 row_var 为 None，则不启用 facet_row，只显示一行
    actual_facet_row = internal_row_var if row_var is not None else None

    # 如果没有 row_var，需要手动调整高度，防止图太高
    if row_var is None and height == 970:
        height = 450  # 给个合理的默认高度给单行图

    # 1. 散点图
    fig = px.scatter(
        plot_df, x="RMS Spread", y="RMS Error",
        template=template,
        color="MeanLabel", color_discrete_map=color_map,
        facet_col=col_var,
        facet_row=actual_facet_row,  # 只有当 row_var 存在时才传入
        facet_col_spacing=0.01, facet_row_spacing=0.02,
        category_orders={col_var: use_cols, internal_row_var: use_rows,
                         "MeanLabel": ["<b>With</b> Bias Corrections",
                                       "<b>Without</b> Bias Corrections", "y = x"]},
        hover_data={"bin_count": True, "RMS Spread": ':.1f', "RMS Error": ':.1f'},
    )

    # 2. 散点与对角线样式
    for tr in fig.data:
        tr.showlegend = False
        if tr.name == 'y = x':
            tr.mode = 'lines'
            tr.line.update(color=COL_DIAG, width=1.0, dash='dot')
        else:
            tr.mode = 'markers'
            tr.marker.update(size=4)

    # 3. 拟合曲线
    if not curve_df.empty:
        fig_curve = px.line(
            curve_df, x="RMS Spread", y="RMS Error",
            color="MeanLabel", color_discrete_map=color_map,
            facet_col=col_var,
            facet_row=actual_facet_row,
            category_orders={col_var: use_cols, internal_row_var: use_rows}
        )
        for tr in fig_curve.data:
            tr.showlegend = False
            tr.mode = 'lines'
            tr.line.update(width=1.0)
            tr.marker.update(size=0, opacity=0)
            tr.hoverinfo = 'skip'
            fig.add_trace(tr)

    # --- F. 标注与布局 ---

    # 1. 公式标注
    fit_df = pd.DataFrame(fit_rows)

    if not fit_df.empty:
        # 遍历每一个分面 (Row, Col)
        # 如果 row_var 是 None，use_rows 只有一个元素 ' '，index 为 0，annot_row = 1
        for (r_val, c_val), g in fit_df.groupby(['row', 'col']):
            try:
                row_idx = use_rows.index(r_val)
                col_idx = use_cols.index(c_val)

                # facet_row 的顺序是反的，所以要翻转行号
                annot_row = len(use_rows) - row_idx
                annot_col = col_idx + 1

            except ValueError:
                continue

            # (A) 处理 With Bias Corrections
            gw = g[g['label'].str.contains("With")]
            if not gw.empty:
                r0 = gw.iloc[0]
                a, b = r0['a'], r0['b']
                sign = "+" if b >= 0 else "-"
                val_b = abs(b)

                if fit_linear:
                    txt = rf"$\mathrm{{Error}} = {a:.1f}\,\mathrm{{Spread}} {sign} {val_b:.1f}$"
                else:
                    txt = rf"$\mathrm{{Error}}^2 = ({a:.1f}\,\mathrm{{Spread}})^2 {sign} {val_b:.1f}^2$"
                if fit_linear:
                    txt = (
                        f"Error = "
                        f"<b>{a:.1f}</b> · Spread {sign} <b>{val_b:.1f}</b>"
                    )
                else:
                    txt = (
                        f"Error² = "
                        f"(<b>{a:.1f}</b> · Spread)² {sign} <b>{val_b:.1f}</b>²"
                    )

                fig.add_annotation(
                    text=txt, xref="x domain", yref="y domain",
                    x=0.2, y=0.05, showarrow=False,
                    font=dict(size=14, color=COL_WITH),#,family="Times New Roman"),
                    row=annot_row, col=annot_col,
                )

            # (B) 处理 Without Bias Corrections
            gwo = g[g['label'].str.contains("Without")]
            if not gwo.empty:
                r1 = gwo.iloc[0]
                a, b = r1['a'], r1['b']
                sign = "+" if b >= 0 else "-"
                val_b = abs(b)

                if fit_linear:
                    txt = rf"$\mathrm{{Error}} = {a:.1f}\,\mathrm{{Spread}} {sign} {val_b:.1f}$"
                else:
                    txt = rf"$\mathrm{{Error}}^2 = ({a:.1f}\,\mathrm{{Spread}})^2 {sign} {val_b:.1f}^2$"
                if fit_linear:
                    txt = (
                        f"Error = "
                        f"<b>{a:.1f}</b> · Spread {sign} <b>{val_b:.1f}</b>"
                    )
                else:
                    txt = (
                        f"Error² = "
                        f"(<b>{a:.1f}</b> · Spread)² {sign} <b>{val_b:.1f}</b>²"
                    )

                fig.add_annotation(
                    text=txt, xref="x domain", yref="y domain",
                    x=0.2, y=0.15, showarrow=False,
                    font=dict(size=14, color=COL_WITHOUT),#,family="Times New Roman"),
                    row=annot_row, col=annot_col
                )

    # 2. 清理 Facet 标题
    def _strip_facet(t):
        # 清理 row_var, col_var 以及内部临时变量
        t = t.replace(f"{col_var}=", "")
        if row_var:
            t = t.replace(f"{row_var}=", "")
        # 如果是临时行变量，整个替换为空（或者保留空格）
        if internal_row_var == '__SingleRow__':
            if '__SingleRow__=' in t or t == ' ':
                return ''  # 隐藏侧边标题
        return t

    fig.for_each_annotation(lambda a: a.update(text=_strip_facet(a.text))
    if isinstance(a.text, str) else None)

    # 3. 布局更新
    fig.update_layout(title=None, width=width, height=height)

    # 4. 轴样式
    fig.update_xaxes(zeroline=False, tickmode='linear', dtick=5, mirror=True, showline=True)
    fig.update_yaxes(zeroline=False, tickmode='linear', dtick=5, mirror=True, showline=True)

    # Range and Scale
    for k in fig.layout:
        if k.startswith('xaxis'):
            fig.layout[k].update(range=[ax_min, AX_MAX])
        if k.startswith('yaxis'):
            x_token = k.replace('yaxis', 'x')
            fig.layout[k].update(range=[ax_min, AX_MAX], scaleanchor=x_token, scaleratio=1.0)

    # 拉伸 y=x 对角线
    for tr in fig.data:
        if tr.name == 'y = x':
            tr.x = [ax_min, AX_MAX]
            tr.y = [ax_min, AX_MAX]

    # 5. 轴标题逻辑
    # Clear internal titles
    for k in fig.layout:
        if k.startswith('xaxis'): fig.layout[k].title.text = ''
        if k.startswith('yaxis'): fig.layout[k].title.text = ''

    x_domains = {k: fig.layout[k].domain for k in fig.layout if k.startswith('xaxis')}
    y_domains = {k: fig.layout[k].domain for k in fig.layout if k.startswith('yaxis')}

    rows = {}
    for yk, dom in y_domains.items():
        rows.setdefault(tuple(dom), []).append(yk)

    for ydom, y_axes in rows.items():
        xs = []
        for yk in y_axes:
            for xk in x_domains:
                if getattr(fig.layout[xk], 'anchor', None) == yk.replace('yaxis', 'y'):
                    xs.append(xk)
        if xs:
            left_x = min(xs, key=lambda k: x_domains[k][0])
            left_y = next(
                (yk for yk in y_axes if getattr(fig.layout[left_x], 'anchor', None) == yk.replace('yaxis', 'y')), None)
            if left_y:
                fig.layout[left_y].title.text = "RMS Error (mm)"

    if rows:
        bottom_ydom = min(rows.keys(), key=lambda d: d[0])
        for yk in rows[bottom_ydom]:
            for xk in x_domains:
                if getattr(fig.layout[xk], 'anchor', None) == yk.replace('yaxis', 'y'):
                    fig.layout[xk].title.text = "RMS Spread (mm)"

    # 6. 伪图例 (Legend)
    # 逻辑：放在第一行第一列
    if use_rows and use_cols:
        top_r_idx = 1
        top_c_idx = 1

        fig.add_annotation(
            text=(f"<span style='color:{COL_WITHOUT}; font-weight:bold'>Raw</span> / <span style='color:{COL_WITH}; font-weight:bold'>BC</span>"),
            xref="x domain", yref="y domain",
            x=0.06, y=0.96, xanchor="left", yanchor="top",
            showarrow=False, font=dict(size=15),
            row=top_r_idx, col=top_c_idx
        )

    # 返回最终图形
    return fig

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from scipy.special import erf, erfinv

# 上面这一段在文件中应该已经有 import 的就不用重复




def plot_qq_normresid_matrix(
    df_all: pd.DataFrame,
    col_var: str,
    row_var: str | None = None,
    template="simple_white",
    xaxis: str = "ztd_nwm_sigma",
    row_order=None,
    col_order=None,
    include_global: bool = True,
    N: int = 50,       # 有限样本修正的参考 N
    step: float = 0.1,    # (Legacy)
    L: float = 2.5,    # 轴范围 [-L, L]
    width: int = 1200,
    height: int = 1000,
):
    """
    绘制归一化残差的 Q-Q 图矩阵。

    - 残差: e = mean_col - ztd_gnss
    - 归一化: z = e / (alpha * sigma), 其中 alpha = sqrt((N+1)/N)
      (对于 BC + Obs，sigma 使用包含了 GNSS 观测误差的 combined spread)
    - 经验分位: z_q = quantile(z, p_grid)
    - 理论分位: q_norm = Phi^{-1}(p_grid) = N(0,1) 分布的分位数

    注意坐标轴定义（已调整）:
        x 轴 = q_norm  → Expected normalized residual
        y 轴 = z_q     → Observed normalized residual
    """

    # --------------------
    # A. 数据准备
    # --------------------
    _base = df_all.copy()

    if not np.issubdtype(_base["time"].dtype, np.datetime64):
        _base["time"] = pd.to_datetime(_base["time"])

    # 确保主 spread 列有效
    _base = _base[np.isfinite(_base[xaxis])].copy()

    # Pre-calculate Adjusted Combined Sigma for 'BC + Obs'
    # 逻辑同 plot_spread_skill_ex: total_variance = spread^2 + inv_factor * obs_err^2
    # 这里我们构造一个 combined spread 用于归一化残差
    # inv_factor = N / (N + 1)  (注意这里是观测误差的修正项)
    inv_factor = N / (N + 1)
    if 'ztd_gnss_sigma' not in _base.columns:
        gnss_sig_sq = 0.0
    else:
        gnss_sig_sq = _base['ztd_gnss_sigma'].fillna(0) ** 2

    # Constructed combined spread
    _base['ztd_combined_sigma_pre'] = np.sqrt(
        _base[xaxis]**2 + inv_factor * gnss_sig_sq
    )

    # include_global: 构造一个全局面板
    if include_global:
        _global = _base.copy()
        _global[col_var] = "Global"
        if col_order and "Global" not in col_order:
            col_order.append("Global")
        _base = pd.concat([_base, _global], ignore_index=True)

    # 仿照 plot_spread_skill_matrix 的内部行变量逻辑
    internal_row_var = row_var
    if row_var is None:
        internal_row_var = "__SingleRow__"
        _base[internal_row_var] = " "
        if row_order is None:
            row_order = [" "]

    need = [
        "time",
        "ztd_gnss",
        "ztd_nwm",
        "ztd_nwm_cor",
        xaxis,
        internal_row_var,
        col_var,
    ]
    _base = _base.dropna(subset=need)

    # 过滤 row/col 顺序
    if row_order:
        _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
    if col_order:
        _base = _base[_base[col_var].astype(str).isin(col_order)]

    # 有效行/列列表（后面 facet/对角线会用到）
    use_rows = row_order if row_order else sorted(_base[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(_base[col_var].unique())

    # 有限样本修正系数 alpha (用于归一化残差的分母修正)
    alpha = np.sqrt((N + 1.0) / N)
    # --- Uniform Sampling on X-axis (Step 0.5) ---
    q_norm_grid = np.arange(-L, L + 0.001, step)
    p_grid = 0.5 * (1 + erf(q_norm_grid / np.sqrt(2)))
    p_grid = np.clip(p_grid, 1e-6, 1 - 1e-6)

    # --------------------
    # B. 逐面板构造 Q-Q 数据
    # --------------------
    qq_rows_all: list[pd.DataFrame] = []
    
    # 绘图目标配置: (残差来源列, 图例显示名, 分母spread列)
    # 绘图目标配置: (残差来源列, 图例显示名, 分母spread列)
    targets = [
        ("ztd_nwm",     "Raw",       xaxis),                  # Orange
        ("ztd_nwm_cor", "BC",        xaxis),                  # Blue
        ("ztd_nwm_cor", "BC + Obs",  'ztd_combined_sigma_pre') # Green
    ]

    for (r_val, c_val), g_rc in _base.groupby(
        [internal_row_var, col_var], observed=True
    ):
        for mean_col, mean_name, spread_col in targets:
            g = g_rc.copy()
            # 过滤 spread 有效
            g = g[np.isfinite(g[spread_col]) & (g[spread_col] > 0)].copy()
            if len(g) < 5:
                continue

            # sigma
            g["sigma"] = g[spread_col]

            # 残差 & 归一化残差
            # e = mean - truth
            # z = e / (alpha * sigma)
            g["e"] = g[mean_col] - g["ztd_gnss"]
            g["z"] = g["e"] / (alpha * g["sigma"])
            
            g = g[np.isfinite(g["z"])].copy()
            if len(g) < 5:
                continue

            z = g["z"].to_numpy()
            # 经验分位
            z_q = np.quantile(z, p_grid)
            # 理论 N(0,1) 分位
            q_norm = q_norm_grid

            if mean_name == "BC + Obs":
                mean_label = "BC + Obs"
            else:
                mean_label = mean_name # 直接使用 "Raw" / "BC"

            rows = []
            for p, z0, q0 in zip(p_grid, z_q, q_norm):
                rows.append(
                    {
                        internal_row_var: r_val,
                        col_var: c_val,
                        "MeanLabel": mean_label,
                        "p": float(p),
                        "z_q": float(z0),      # Observed
                        "q_norm": float(q0),   # Expected
                    }
                )

            qq_rows_all.append(pd.DataFrame(rows))

    if not qq_rows_all:
        print("No valid data for Q-Q plot.")
        return None

    qq_df = pd.concat(qq_rows_all, ignore_index=True)

    # --------------------
    # C. 添加对角线 y = x
    # --------------------
    diag_rows = []
    for r in use_rows:
        for c in use_cols:
            diag_rows.append(
                {
                    internal_row_var: r,
                    col_var: c,
                    "MeanLabel": "y = x",
                    "p": np.nan,
                    "z_q": -L,
                    "q_norm": -L,
                }
            )
            diag_rows.append(
                {
                    internal_row_var: r,
                    col_var: c,
                    "MeanLabel": "y = x",
                    "p": np.nan,
                    "z_q": L,
                    "q_norm": L,
                }
            )

    qq_plot_df = pd.concat([qq_df, pd.DataFrame(diag_rows)], ignore_index=True)

    # 为 hover 添加更直观的列名
    qq_plot_df["Expected"] = qq_plot_df["q_norm"]
    qq_plot_df["Observed"] = qq_plot_df["z_q"]

    # --------------------
    # D. 绘图
    # --------------------
    # 颜色映射
    color_map = {
        "Raw":      '#FF7F0E', # Orange
        "BC":       '#1F77B4', # Blue
        "BC + Obs": '#2CA02C', # Green
        "y = x":    COL_DIAG,
    }

    actual_facet_row = internal_row_var if row_var is not None else None
    if row_var is None and height == 1000:
        height = 450  # 单行时默认矮一点

    fig = px.scatter(
        qq_plot_df,
        x="q_norm",       # Expected normalized residual
        y="z_q",          # Observed normalized residual
        color="MeanLabel",
        color_discrete_map=color_map,
        facet_col=col_var,
        facet_row=actual_facet_row,
        facet_col_spacing=0.01,
        facet_row_spacing=0.02,
        category_orders={
            col_var: use_cols,
            internal_row_var: use_rows,
            "MeanLabel": ["Raw", "BC", "BC + Obs", "y = x"],
        },
        hover_data={"p": ':.3f', "Expected": ':.2f', "Observed": ':.2f'},
        template=template,
    )

    # 线型设置
    for tr in fig.data:
        if tr.name == "y = x":
            tr.mode = "lines"
            tr.line.update(color=COL_DIAG, width=1)
            # tr.showlegend = False
        else:
            tr.mode = "lines+markers"
            tr.marker.update(size=4)
            tr.line.update(width=1)
            # tr.showlegend = False

    # 轴范围、比例
    fig.update_layout(
        title=None, width=width, height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, title_text=""),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    for k in fig.layout:
        if k.startswith("xaxis"):
            fig.layout[k].update(
                range=[-L, L],
                zeroline=False,
                mirror=True,
                showline=True,
                tickmode="linear",
                dtick=1,
            )
        if k.startswith("yaxis"):
            x_token = k.replace("yaxis", "x")
            fig.layout[k].update(
                range=[-L, L],
                zeroline=False,
                mirror=True,
                showline=True,
                tickmode="linear",
                dtick=1,
                scaleanchor=x_token,
                scaleratio=1.0,
            )

    # 去掉 facet 标题里的 "row_var=" / "col_var=" 等前缀
    def _strip_facet(t: str) -> str:
        t = t.replace(f"{col_var}=", "")
        if row_var:
            t = t.replace(f"{row_var}=", "")
        if internal_row_var == "__SingleRow__":
            if "__SingleRow__=" in t or t == " ":
                return ""
        return t

    fig.for_each_annotation(
        lambda a: a.update(text=_strip_facet(a.text))
        if isinstance(a.text, str)
        else None
    )

    # 清空各子图轴标题
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    # 全局 x/y 轴标题
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    fig.add_annotation(
        text="Theoretical Quantiles",
        x=0.5, y=-0.2,
        xref="paper", yref="paper",
        showarrow=False,
        font_size=16,
    )
    fig.add_annotation(
        text="Sample Quantiles",
        x=-0.06, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        textangle=-90,
        font_size=16,
    )

    # (Custom legend removed)

    return fig


def plot_qq_normresid_matrix_multi(
    data_list: list[tuple[pd.DataFrame, int]],
    col_var: str,
    compare_target: str = "BC",  # "Raw", "BC", "BC + Obs"
    row_var: str | None = None,
    template="simple_white",
    xaxis: str = "ztd_nwm_sigma",
    row_order=None,
    col_order=None,
    include_global: bool = True,
    step: float = 0.25,
    L: float = 3.0,
    width: int = 1200,
    height: int = 1000,
    fixed_aspect_ratio: bool = True,
):
    """
    对比不同 Ensemble Size (N) 下的 Q-Q 图差异。
    只绘制 compare_target 指定的那一种类型。
    """
    
    # --- 统一的 X 轴采样 ---
    q_norm_grid = np.arange(-L, L + 0.001, step)
    p_grid = 0.5 * (1 + erf(q_norm_grid / np.sqrt(2)))
    p_grid = np.clip(p_grid, 1e-6, 1 - 1e-6)
    
    qq_rows_all = []
    use_rows = set()
    use_cols = set()

    # 遍历每个 (df, N) 对
    for df_in, N_val in data_list:
        _base = df_in.copy()
        
        if not np.issubdtype(_base["time"].dtype, np.datetime64):
            _base["time"] = pd.to_datetime(_base["time"])
            
        _base = _base[np.isfinite(_base[xaxis])].copy()

        # 根据 N 计算参数
        alpha = np.sqrt((N_val + 1.0) / N_val)
        inv_factor = N_val / (N_val + 1)
        
        # 准备数据列
        # 如果需要 BC + Obs，构造 combined sigma
        if compare_target == "BC + Obs":
            if 'ztd_gnss_sigma' not in _base.columns:
                gnss_sig_sq = 0.0
            else:
                gnss_sig_sq = _base['ztd_gnss_sigma'].fillna(0) ** 2
            
            _base['ztd_combined_sigma_multi'] = np.sqrt(
                _base[xaxis]**2 + inv_factor * gnss_sig_sq
            )
            target_mean_col = "ztd_nwm_cor"
            target_spread_col = "ztd_combined_sigma_multi"
        elif compare_target == "BC":
            target_mean_col = "ztd_nwm_cor"
            target_spread_col = xaxis
        else: # Raw
            target_mean_col = "ztd_nwm"
            target_spread_col = xaxis
            
        # Global Logic
        if include_global:
            _global = _base.copy()
            _global[col_var] = "Global"
            _base = pd.concat([_base, _global], ignore_index=True)

        # Row Logic
        internal_row_var = row_var
        if row_var is None:
            internal_row_var = "__SingleRow__"
            _base[internal_row_var] = " "

        # 过滤
        need = ["ztd_gnss", target_mean_col, target_spread_col, internal_row_var, col_var]
        _base = _base.dropna(subset=need)
        
        if row_order:
            _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
        if col_order:
            _base = _base[_base[col_var].astype(str).isin(col_order)]

        # 收集用到的行列 (用于后面画对角线)
        current_rows = row_order if row_order else sorted(_base[internal_row_var].unique())
        current_cols = col_order if col_order else sorted(_base[col_var].unique())
        use_rows.update(current_rows)
        use_cols.update(current_cols)

        # Groupby 计算
        label_str = f"{compare_target} (N={N_val})"
        
        for (r_val, c_val), g_rc in _base.groupby([internal_row_var, col_var], observed=True):
             g = g_rc.copy()
             g = g[np.isfinite(g[target_spread_col]) & (g[target_spread_col] > 0)].copy()
             if len(g) < 5: continue
             
             g["sigma"] = g[target_spread_col]
             g["e"] = g[target_mean_col] - g["ztd_gnss"]
             # 核心：使用该数据集对应的 alpha
             g["z"] = g["e"] / (alpha * g["sigma"])
             
             g = g[np.isfinite(g["z"])].copy()
             if len(g) < 5: continue
             
             z = g["z"].to_numpy()
             z_q = np.quantile(z, p_grid)
             q_norm = q_norm_grid
             
             rows = []
             for p, z0, q0 in zip(p_grid, z_q, q_norm):
                 rows.append({
                     internal_row_var: r_val,
                     col_var: c_val,
                     "MeanLabel": label_str,
                     "p": float(p),
                     "z_q": float(z0),
                     "q_norm": float(q0)
                 })
             qq_rows_all.append(pd.DataFrame(rows))

    if not qq_rows_all:
        print("No valid data for Q-Q plot.")
        return None
        
    qq_df = pd.concat(qq_rows_all, ignore_index=True)
    
    # 构造对角线
    diag_rows = []
    final_rows = sorted(list(use_rows))
    final_cols = sorted(list(use_cols))
    
    # 如果原先是没传 row_var 的，确保 "__SingleRow__" -> " " 逻辑一致
    if row_var is None and not row_order:
         final_rows = [" "]

    for r in final_rows:
        for c in final_cols:
            diag_rows.append({
                internal_row_var: r, col_var: c, 
                "MeanLabel": "y = x", "z_q": -L, "q_norm": -L, "p": np.nan
            })
            diag_rows.append({
                internal_row_var: r, col_var: c, 
                "MeanLabel": "y = x", "z_q": L, "q_norm": L, "p": np.nan
            })
            
    qq_plot_df = pd.concat([qq_df, pd.DataFrame(diag_rows)], ignore_index=True)
    
    # 绘图
    # 动态生成颜色：只是为了区分不同 N，简单起见用 Plotly 默认色板即可
    # 只要 MeanLabel 不同，颜色就会不同
    
    actual_facet_row = internal_row_var if row_var is not None else None
    if row_var is None and height == 1000:
        height = 450

    fig = px.scatter(
        qq_plot_df,
        x="q_norm",
        y="z_q",
        color="MeanLabel",
        facet_col=col_var,
        facet_row=actual_facet_row,
        facet_col_spacing=0.01,
        facet_row_spacing=0.02,
        category_orders={
            col_var: final_cols,
            internal_row_var: final_rows,
            # 这里不对 MeanLabel 做强制排序，让它自然排列(通常 N 小到大)
        },
        hover_data={"p": ':.3f', "q_norm": ':.2f', "z_q": ':.2f'},
        template=template,
    )
    
    # 样式
    for tr in fig.data:
        if tr.name == "y = x":
            tr.mode = "lines"
            tr.line.update(color=COL_DIAG, width=1)
            # tr.showlegend = False
        else:
            tr.mode = "lines+markers"
            tr.marker.update(size=4, opacity=0.8)
            tr.line.update(width=1)
            
    # Layout
    fig.update_layout(
        title=None, width=width, height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, title_text=""),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    for k in fig.layout:
        if k.startswith("xaxis"):
            fig.layout[k].update(
                range=[-L, L], zeroline=False, mirror=True, showline=True, 
                showgrid=True, gridcolor='#EAEAEA', tickmode="linear", dtick=1
            )
        if k.startswith("yaxis"):
            x_token = k.replace("yaxis", "x")
            # 是否固定比例
            y_update = dict(
                range=[-L, L], zeroline=False, mirror=True, showline=True,
                showgrid=True, gridcolor='#EAEAEA', tickmode="linear", dtick=1
            )
            if fixed_aspect_ratio:
                y_update.update(scaleanchor=x_token, scaleratio=1.0)
            fig.layout[k].update(y_update)

    # Clean facet titles
    def _strip_facet(t: str) -> str:
        t = t.replace(f"{col_var}=", "")
        if row_var: t = t.replace(f"{row_var}=", "")
        if internal_row_var == "__SingleRow__":
            if "__SingleRow__=" in t or t == " ": return ""
        return t

    fig.for_each_annotation(lambda a: a.update(text=_strip_facet(a.text)) if isinstance(a.text, str) else None)
    
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    fig.add_annotation(
        text="Theoretical Quantiles",
        x=0.5, y=-0.2, xref="paper", yref="paper", showarrow=False, font_size=16,
    )
    fig.add_annotation(
        text="Sample Quantiles",
        x=-0.06, y=0.5, xref="paper", yref="paper", showarrow=False, textangle=-90, font_size=16,
    )
    
    return fig


def plot_error_spread_matrix(df_all,
                             col_var,
                             row_var=None,
                             template='simple_white',
                             xaxis='ztd_nwm_sigma',  # 仍然需要指定 sigma 列名用于计算 Y 轴
                             row_order=None, col_order=None,
                             include_global=True,
                             fit_linear=True,
                             n_bins=20,
                             samples_per_bin=None,
                             N=50,  # 修正因子参数
                             ax_min=0.0,
                             width=1200, height=970):
    """
    绘制 Error-Spread 矩阵图 (反向 Spread-Skill)。
    X 轴: RMS Error (基于绝对误差分箱)
    Y 轴: RMS Spread (应用 (N+1)/N 修正)
    """

    # --- A. 数据准备 ---
    _base = df_all.copy()
    if not np.issubdtype(_base['time'].dtype, np.datetime64):
        _base['time'] = pd.to_datetime(_base['time'])

    # 确保有 sigma 数据
    _base = _base[np.isfinite(_base[xaxis])].copy()

    # 扩展 Global
    if include_global:
        _global = _base.copy()
        _global[col_var] = 'Global'
        if col_order and 'Global' not in col_order:
            col_order.append('Global')
        _base = pd.concat([_base, _global], ignore_index=True)

    # 处理 row_var
    internal_row_var = row_var
    if row_var is None:
        internal_row_var = '__SingleRow__'
        _base[internal_row_var] = ' '
        if row_order is None:
            row_order = [' ']

    need = ['time', 'ztd_gnss', 'ztd_nwm', 'ztd_nwm_cor', xaxis, internal_row_var, col_var]
    _base = _base.dropna(subset=need)

    # 过滤顺序
    if row_order:
        _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
    if col_order:
        _base = _base[_base[col_var].astype(str).isin(col_order)]

    # 修正因子
    factor = (N + 1.0) / N

    # --- B. 聚合计算 (自定义逻辑) ---
    agg_list = []
    # (数据列名, 显示标签)
    targets = [('ztd_nwm_cor', 'With Bias Corrections'),
               ('ztd_nwm', 'Without Bias Corrections')]

    for (r_val, c_val), g_rc in _base.groupby([internal_row_var, col_var], observed=True):

        for mean_col, mean_name in targets:
            # 1. 准备当前组数据
            # 计算原始误差
            raw_err = g_rc[mean_col] - g_rc['ztd_gnss']
            # 用于分箱的依据：绝对误差
            sort_key = raw_err.abs()

            # 提取需要的列
            df_tmp = pd.DataFrame({
                'err': raw_err,
                'abs_err': sort_key,
                's2': g_rc[xaxis] ** 2  # sigma squared
            })

            total_valid = len(df_tmp)
            if total_valid < 2:
                continue

            # 2. 动态计算箱数
            if samples_per_bin is not None and samples_per_bin > 0:
                calculated_bins = int(np.ceil(total_valid / samples_per_bin))
                n_bins_to_use = max(1, calculated_bins)
            else:
                n_bins_to_use = n_bins

            # 3. 分箱 (按绝对误差)huachulai
            try:
                q = pd.qcut(df_tmp['abs_err'], q=n_bins_to_use, labels=False, duplicates='drop')
            except ValueError:
                # 如果数据极少或重复值极多，尝试减少箱数
                uniq = df_tmp['abs_err'].nunique()
                if uniq < 2: continue
                q = pd.qcut(df_tmp['abs_err'], q=min(int(uniq), n_bins_to_use), labels=False, duplicates='drop')

            df_tmp['bin_id'] = q

            # 4. 聚合
            rows = []
            for _, gbin in df_tmp.groupby('bin_id'):
                if len(gbin) < 2: continue  # 忽略过小的箱

                # X轴: RMS Error
                # 注意：虽然是按绝对误差分箱，但标准图示 X 轴通常是 RMSE
                rmse_val = np.sqrt(np.mean(gbin['err'] ** 2))

                # Y轴: RMS Spread (带修正因子)
                spread_val = np.sqrt(factor * gbin['s2'].mean())

                rows.append({
                    internal_row_var: r_val,
                    col_var: c_val,
                    "MeanType": mean_name,
                    "RMS Error": rmse_val,  # X Axis
                    "RMS Spread": spread_val,  # Y Axis
                    "bin_count": len(gbin)
                })

            if rows:
                agg_list.append(pd.DataFrame(rows))

    if not agg_list:
        print("No valid data for Error-Spread plot.")
        return None

    agg_all = pd.concat(agg_list, ignore_index=True)
    agg_all['MeanLabel'] = agg_all['MeanType'].map(pretty_label)

    # 计算坐标轴上限
    max_val = agg_all[["RMS Spread", 'RMS Error']].max().max()
    AX_MAX = np.ceil(max_val * 1.1)

    # --- C. 对角线数据 ---
    diag_rows = []
    use_rows = row_order if row_order else sorted(agg_all[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(agg_all[col_var].unique())

    for r in use_rows:
        for c in use_cols:
            # 对角线: y = x
            diag_rows.append({
                "RMS Error": ax_min, "RMS Spread": ax_min,
                internal_row_var: r, col_var: c, 'MeanLabel': 'y = x'
            })
            diag_rows.append({
                "RMS Error": AX_MAX, "RMS Spread": AX_MAX,
                internal_row_var: r, col_var: c, 'MeanLabel': 'y = x'
            })

    plot_df = pd.concat([agg_all, pd.DataFrame(diag_rows)], ignore_index=True)

    # --- D. 拟合计算 (Spread = f(Error)) ---
    fit_rows = []
    curve_rows = []
    mask_real = (plot_df['MeanLabel'] != 'y = x') & plot_df['bin_count'].notna()

    for (r, c, lbl), g in plot_df[mask_real].groupby([internal_row_var, col_var, 'MeanLabel']):
        # 注意：这里交换了输入，X=Error, Y=Spread
        x_dat = g["RMS Error"].to_numpy()
        y_dat = g["RMS Spread"].to_numpy()

        if len(x_dat) < 2: continue

        # 调用之前的拟合函数 (复用 fit_rmse)
        # 拟合目标: Spread ~ a * Error + b
        a, b = fit_rmse(x_dat, y_dat, fit_linear)
        fit_rows.append({'row': r, 'col': c, 'label': lbl, 'a': a, 'b': b})

        # 生成曲线点
        xs = np.linspace(ax_min, AX_MAX, 160)
        if fit_linear:
            ys = a * xs + b
        else:
            ys = np.sqrt(np.maximum((a * xs) ** 2 + b * b, 0.0))

        tmp = pd.DataFrame({"RMS Error": xs, "RMS Spread": ys})
        tmp[internal_row_var] = r
        tmp[col_var] = c
        tmp['MeanLabel'] = lbl
        curve_rows.append(tmp)

    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    # --- E. 绘图 ---
    color_map = {
        "<b>With</b> Bias Corrections": COL_WITH,
        "<b>Without</b> Bias Corrections": COL_WITHOUT,
        "y = x": COL_DIAG
    }

    actual_facet_row = internal_row_var if row_var is not None else None
    if row_var is None and height == 970:
        height = 450

    # 1. 散点图 (X=RMS Error, Y=RMS Spread)
    fig = px.scatter(
        plot_df, x="RMS Error", y="RMS Spread",
        template=template,
        color="MeanLabel", color_discrete_map=color_map,
        facet_col=col_var,
        facet_row=actual_facet_row,
        facet_col_spacing=0.01, facet_row_spacing=0.02,
        category_orders={col_var: use_cols, internal_row_var: use_rows,
                         "MeanLabel": ["<b>With</b> Bias Corrections",
                                       "<b>Without</b> Bias Corrections", "y = x"]},
        hover_data={"bin_count": True, "RMS Spread": ':.2f', "RMS Error": ':.2f'},
    )

    # 2. 样式调整
    for tr in fig.data:
        tr.showlegend = False
        if tr.name == 'y = x':
            tr.mode = 'lines'
            tr.line.update(color=COL_DIAG, width=1.0, dash='dot')
        else:
            tr.mode = 'markers'
            tr.marker.update(size=4)

    # 3. 拟合曲线
    if not curve_df.empty:
        fig_curve = px.line(
            curve_df, x="RMS Error", y="RMS Spread",
            color="MeanLabel", color_discrete_map=color_map,
            facet_col=col_var,
            facet_row=actual_facet_row,
            category_orders={col_var: use_cols, internal_row_var: use_rows}
        )
        for tr in fig_curve.data:
            tr.showlegend = False
            tr.mode = 'lines'
            tr.line.update(width=1.0)
            tr.marker.update(size=0, opacity=0)
            tr.hoverinfo = 'skip'
            fig.add_trace(tr)

    # --- F. 标注与布局 ---

    # 1. 公式标注 (更新为 Spread = ... Error)
    fit_df = pd.DataFrame(fit_rows)
    if not fit_df.empty:
        for (r_val, c_val), g in fit_df.groupby(['row', 'col']):
            try:
                row_idx = use_rows.index(r_val)
                col_idx = use_cols.index(c_val)
                annot_row = len(use_rows) - row_idx
                annot_col = col_idx + 1
            except ValueError:
                continue

            # 定义标注生成辅助函数
            def _add_eqn(sub_g, y_pos, color):
                if sub_g.empty: return
                r0 = sub_g.iloc[0]
                a, b = r0['a'], r0['b']
                sign = "+" if b >= 0 else "-"
                val_b = abs(b)

                # 文本逻辑: Spread = a * Error + b
                if fit_linear:
                    # txt = rf"$\mathrm{{Spread}} = {a:.1f}\,\mathrm{{Error}} {sign} {val_b:.1f}$"
                    txt = (f"Spread = <b>{a:.1f}</b> · Error {sign} <b>{val_b:.1f}</b>")
                else:
                    # txt = rf"$\mathrm{{Spread}}^2 = ({a:.1f}\,\mathrm{{Error}})^2 {sign} {val_b:.1f}^2$"
                    txt = (f"Spread² = (<b>{a:.1f}</b> · Error)² {sign} <b>{val_b:.1f}</b>²")

                fig.add_annotation(
                    text=txt, xref="x domain", yref="y domain",
                    x=0.05, y=y_pos, showarrow=False, xanchor='left',
                    font=dict(size=14, color=color),
                    row=annot_row, col=annot_col,
                )

            _add_eqn(g[g['label'].str.contains("With")], 0.95, COL_WITH)
            _add_eqn(g[g['label'].str.contains("Without")], 0.85, COL_WITHOUT)

    # 2. 清理 Facet 标题
    def _strip_facet(t):
        t = t.replace(f"{col_var}=", "")
        if row_var: t = t.replace(f"{row_var}=", "")
        if internal_row_var == '__SingleRow__':
            if '__SingleRow__=' in t or t == ' ': return ''
        return t

    fig.for_each_annotation(lambda a: a.update(text=_strip_facet(a.text))
    if isinstance(a.text, str) else None)

    # 3. 轴与布局
    fig.update_layout(title=None, width=width, height=height)

    # 统一设置所有轴
    # common_axis_config = dict(
    #     zeroline=False, tickmode='linear', dtick=5, mirror=True, showline=True, range=[ax_min, AX_MAX]
    # )
    # fig.update_xaxes(**common_axis_config)
    # fig.update_yaxes(**common_axis_config)

    # 强制比例 1:1
    # for k in fig.layout:
    #     if k.startswith('yaxis'):
    #         x_token = k.replace('yaxis', 'x')
    #         fig.layout[k].update(scaleanchor=x_token, scaleratio=1.0)

    # 拉伸对角线
    for tr in fig.data:
        if tr.name == 'y = x':
            tr.x = [ax_min, AX_MAX]
            tr.y = [ax_min, AX_MAX]

    # 4. 统一轴标题
    # 清空所有子图标题
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    # 全局 X 轴标题
    fig.add_annotation(
        text="RMS Error (mm)",
        x=0.5, y=0, xref="paper", yref="paper",
        yshift=-50, showarrow=False, font=dict(size=16)
    )
    # 全局 Y 轴标题
    fig.add_annotation(
        text="RMS Spread (mm)",
        x=0, y=0.5, xref="paper", yref="paper",
        xshift=-60, textangle=-90, showarrow=False, font=dict(size=16)
    )

    # 5. 伪图例
    if use_rows and use_cols:
        fig.add_annotation(
            text=(f"<span style='color:{COL_WITHOUT}; font-weight:bold'>Raw</span> / "
                  f"<span style='color:{COL_WITH}; font-weight:bold'>BC</span>"),
            xref="x domain", yref="y domain",
            x=0.96, y=0.04, xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=15),
            row=1, col=1
        )

    return fig


def plot_spread_skill_proxy_binning(df_all,
                                    col_var,
                                    proxy_col='ztd_volatility',  # 分箱依据：真实波动性
                                    sigma_col='ztd_nwm_sigma',  # 计算 X 轴依据：模型 Spread
                                    row_var=None,
                                    template='simple_white',
                                    row_order=None, col_order=None,
                                    include_global=True,
                                    fit_linear=True,
                                    n_bins=20,
                                    samples_per_bin=None,
                                    N=50,
                                    ax_min=0.0,
                                    width=1200, height=970):
    """
    基于 Proxy (如波动性) 分箱的 Spread-Skill 图。

    逻辑：
    1. 根据 proxy_col (如 ztd_volatility) 对数据进行排序分箱。
    2. 对每个箱，计算 RMS Spread (X轴) 和 RMS Error (Y轴)。
    3. 绘制散点和拟合曲线。

    目的：
    检验当真实不确定性(Proxy)增加时，模型的 Spread 是否随之增加(向右移动)以匹配 Error 的增加(向上移动)。
    """

    # --- A. 数据准备 ---
    _base = df_all.copy()
    if not np.issubdtype(_base['time'].dtype, np.datetime64):
        _base['time'] = pd.to_datetime(_base['time'])

    # 确保 Proxy 和 Sigma 都有值
    _base = _base[np.isfinite(_base[proxy_col]) & np.isfinite(_base[sigma_col])].copy()

    # Global 扩展
    if include_global:
        _global = _base.copy()
        _global[col_var] = 'Global'
        if col_order and 'Global' not in col_order:
            col_order.append('Global')
        _base = pd.concat([_base, _global], ignore_index=True)

    # Row Var 处理
    internal_row_var = row_var
    if row_var is None:
        internal_row_var = '__SingleRow__'
        _base[internal_row_var] = ' '
        if row_order is None: row_order = [' ']

    need = ['ztd_gnss', 'ztd_nwm', 'ztd_nwm_cor', proxy_col, sigma_col, internal_row_var, col_var]
    _base = _base.dropna(subset=need)

    # 过滤
    if row_order: _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
    if col_order: _base = _base[_base[col_var].astype(str).isin(col_order)]

    factor = (N + 1.0) / N

    # --- B. 聚合计算 (核心修改) ---
    agg_list = []
    targets = [('ztd_nwm_cor', 'With Bias Corrections'),
               ('ztd_nwm', 'Without Bias Corrections')]

    for (r_val, c_val), g_rc in _base.groupby([internal_row_var, col_var], observed=True):

        # 1. 准备排序键 (Proxy)
        sort_key = g_rc[proxy_col]
        total_valid = len(g_rc)
        if total_valid < 2: continue

        # 2. 动态计算箱数
        if samples_per_bin is not None and samples_per_bin > 0:
            n_bins_to_use = max(1, int(np.ceil(total_valid / samples_per_bin)))
        else:
            n_bins_to_use = n_bins

        # 3. 分箱 (按 Proxy)
        try:
            q = pd.qcut(sort_key, q=n_bins_to_use, labels=False, duplicates='drop')
        except ValueError:
            uniq = sort_key.nunique()
            if uniq < 2: continue
            q = pd.qcut(sort_key, q=min(int(uniq), n_bins_to_use), labels=False, duplicates='drop')

        g_rc = g_rc.copy()
        g_rc['bin_id'] = q

        # 4. 计算每个箱的坐标 (X=Spread, Y=Error)
        for mean_col, mean_name in targets:
            rows = []
            for _, gbin in g_rc.groupby('bin_id'):
                if len(gbin) < 2: continue

                # --- X 轴: RMS Spread ---
                # 注意：这里取的是 sigma_col 的均方根
                s2 = gbin[sigma_col] ** 2
                rms_spread = np.sqrt(factor * s2.mean())

                # --- Y 轴: RMS Error ---
                err = gbin[mean_col] - gbin['ztd_gnss']
                rmse = np.sqrt(np.mean(err ** 2))

                # (可选) 记录 Proxy 的均值，方便 tooltip 查看这个箱对应多大的波动性
                proxy_mean = gbin[proxy_col].mean()

                rows.append({
                    internal_row_var: r_val,
                    col_var: c_val,
                    "MeanType": mean_name,
                    "RMS Spread": rms_spread,  # X
                    "RMS Error": rmse,  # Y
                    "Proxy Mean": proxy_mean,  # 用于 Hover
                    "bin_count": len(gbin)
                })

            if rows:
                agg_list.append(pd.DataFrame(rows))

    if not agg_list:
        print("No valid data.")
        return None

    agg_all = pd.concat(agg_list, ignore_index=True)
    agg_all['MeanLabel'] = agg_all['MeanType'].map(pretty_label)

    # --- C. 计算坐标轴范围 ---
    max_val = agg_all[["RMS Spread", 'RMS Error']].max().max()
    AX_MAX = np.ceil(max_val * 1.1)

    # --- D. 拟合与对角线 ---
    # 1. 对角线
    diag_rows = []
    use_rows = row_order if row_order else sorted(agg_all[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(agg_all[col_var].unique())

    for r in use_rows:
        for c in use_cols:
            diag_rows.append({
                "RMS Spread": ax_min, "RMS Error": ax_min,
                internal_row_var: r, col_var: c, 'MeanLabel': 'y = x'
            })
            diag_rows.append({
                "RMS Spread": AX_MAX, "RMS Error": AX_MAX,
                internal_row_var: r, col_var: c, 'MeanLabel': 'y = x'
            })

    plot_df = pd.concat([agg_all, pd.DataFrame(diag_rows)], ignore_index=True)

    # 2. 拟合曲线 (Error = f(Spread))
    fit_rows = []
    curve_rows = []
    mask_real = (plot_df['MeanLabel'] != 'y = x') & plot_df['bin_count'].notna()

    for (r, c, lbl), g in plot_df[mask_real].groupby([internal_row_var, col_var, 'MeanLabel']):
        x_dat = g["RMS Spread"].to_numpy()
        y_dat = g["RMS Error"].to_numpy()
        if len(x_dat) < 2: continue

        a, b = fit_rmse(x_dat, y_dat, fit_linear)
        fit_rows.append({'row': r, 'col': c, 'label': lbl, 'a': a, 'b': b})

        xs = np.linspace(ax_min, AX_MAX, 160)
        if fit_linear:
            ys = a * xs + b
        else:
            ys = np.sqrt(np.maximum((a * xs) ** 2 + b * b, 0.0))

        tmp = pd.DataFrame({"RMS Spread": xs, "RMS Error": ys})
        tmp[internal_row_var] = r
        tmp[col_var] = c
        tmp['MeanLabel'] = lbl
        curve_rows.append(tmp)

    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    # --- E. 绘图 ---
    color_map = {
        "<b>With</b> Bias Corrections": COL_WITH,
        "<b>Without</b> Bias Corrections": COL_WITHOUT,
        "y = x": COL_DIAG
    }

    actual_facet_row = internal_row_var if row_var is not None else None
    if row_var is None and height == 970: height = 500

    fig = px.scatter(
        plot_df, x="RMS Spread", y="RMS Error",
        template=template,
        color="MeanLabel", color_discrete_map=color_map,
        facet_col=col_var,
        facet_row=actual_facet_row,
        facet_col_spacing=0.01, facet_row_spacing=0.02,
        category_orders={col_var: use_cols, internal_row_var: use_rows,
                         "MeanLabel": ["<b>With</b> Bias Corrections", "<b>Without</b> Bias Corrections", "y = x"]},
        # Hover 中增加 Proxy Mean 的显示，这是该图的特色
        hover_data={"bin_count": True, "RMS Spread": ':.2f', "RMS Error": ':.2f', "Proxy Mean": ':.2f'}
    )

    # 样式
    for tr in fig.data:
        tr.showlegend = False
        if tr.name == 'y = x':
            tr.mode = 'lines'
            tr.line.update(color=COL_DIAG, width=1.0, dash='dot')
        else:
            tr.mode = 'markers'
            tr.marker.update(size=5)  # 稍微大一点，看清分布

    # 添加拟合线
    if not curve_df.empty:
        fig_curve = px.line(
            curve_df, x="RMS Spread", y="RMS Error",
            color="MeanLabel", color_discrete_map=color_map,
            facet_col=col_var, facet_row=actual_facet_row,
            category_orders={col_var: use_cols, internal_row_var: use_rows}
        )
        for tr in fig_curve.data:
            tr.showlegend = False
            tr.mode = 'lines'
            tr.line.update(width=1.0)
            tr.marker.update(size=0, opacity=0)
            tr.hoverinfo = 'skip'
            fig.add_trace(tr)

    # --- F. 布局与标注 ---
    # 1. 轴
    common_axis = dict(zeroline=False, tickmode='linear', dtick=5, mirror=True, showline=True, range=[ax_min, AX_MAX])
    fig.update_xaxes(**common_axis)
    fig.update_yaxes(**common_axis)

    for k in fig.layout:
        if k.startswith('yaxis'):
            x_token = k.replace('yaxis', 'x')
            fig.layout[k].update(scaleanchor=x_token, scaleratio=1.0)

    # 2. 对角线拉伸
    for tr in fig.data:
        if tr.name == 'y = x':
            tr.x = [ax_min, AX_MAX]
            tr.y = [ax_min, AX_MAX]

    # 3. Facet 标题清理
    def _strip_facet(t):
        t = t.replace(f"{col_var}=", "")
        if row_var: t = t.replace(f"{row_var}=", "")
        if internal_row_var == '__SingleRow__':
            if '__SingleRow__=' in t or t == ' ': return ''
        return t

    fig.for_each_annotation(lambda a: a.update(text=_strip_facet(a.text)) if isinstance(a.text, str) else None)

    # 4. 全局标题
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    fig.add_annotation(text="RMS Spread (mm)", x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
                       font_size=16)
    fig.add_annotation(text="RMS Error (mm)", x=-0.06, y=0.5, xref="paper", yref="paper", textangle=-90,
                       showarrow=False, font_size=16)

    # 5. 顶部说明（解释分箱依据）
    fig.add_annotation(
        text=f"Binned by: <b>{proxy_col}</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.05, showarrow=False, font=dict(size=14, color="gray")
    )

    # 6. 伪图例
    if use_rows and use_cols:
        fig.add_annotation(
            text=(
                f"<span style='color:{COL_WITHOUT}; font-weight:bold'>Raw</span> / <span style='color:{COL_WITH}; font-weight:bold'>BC</span>"),
            xref="x domain", yref="y domain",
            x=0.06, y=0.96, xanchor="left", yanchor="top",
            showarrow=False, font=dict(size=15),
            row=1, col=1
        )

    fig.update_layout(title=None, width=width, height=height)

    return fig


from plotly.subplots import make_subplots
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 确保上下文中有颜色定义，如果没有则在此处定义
try:
    COL_WITH
    COL_WITHOUT
    COL_DIAG
except NameError:
    COL_WITH = "#1f77b4"
    COL_WITHOUT = "#ff7f0e"
    COL_DIAG = "#9e9e9e"


def plot_spread_skill_analysis(df_all,
                               bin_by_col,  # str 或 list[str]
                               metric='both',  # 'diff', 'ratio', 或 'both'
                               bin_method='quantile',  # 'quantile' 或 'uniform'
                               sigma_col='ztd_nwm_sigma',
                               template='simple_white',
                               n_bins=20,
                               samples_per_bin=None,
                               N=50,
                               width=None, height=500):
    """
    Spread-Skill 分析图 (支持单变量或多变量并排对比)。
    """

    # --- 1. 参数校验与初始化 ---
    if isinstance(bin_by_col, str):
        bin_vars = [bin_by_col]
    else:
        bin_vars = bin_by_col

    if metric not in ['diff', 'ratio', 'both']:
        raise ValueError("metric must be 'diff', 'ratio', or 'both'")

    if bin_method not in ['quantile', 'uniform']:
        raise ValueError("bin_method must be 'quantile' or 'uniform'")

    num_vars = len(bin_vars)
    if width is None:
        width = 800 if num_vars == 1 else 400 * num_vars  # 稍微调窄一点默认宽度

    fig = make_subplots(
        rows=1, cols=num_vars,
        # 移除 subplot_titles，因为 X 轴标签已包含信息
        subplot_titles=None,
        shared_yaxes=True,
        # 缩小子图间距
        horizontal_spacing=0.015
    )

    factor = (N + 1.0) / N

    # 定义图例标签逻辑
    if metric == 'both':
        label_bc = "RMS Error (BC)"
        label_raw = "RMS Error (Raw)"
    else:
        label_bc = "With Bias Corrections"
        label_raw = "Without Bias Corrections"

    targets = [
        ('ztd_nwm_cor', label_bc, COL_WITH)
    ]

    # --- 2. 循环处理每个变量 ---
    for i, var_name in enumerate(bin_vars):
        col_idx = i + 1

        # A. 数据清洗
        required_cols = ['ztd_gnss', 'ztd_nwm', 'ztd_nwm_cor', sigma_col, var_name]
        try:
            _base = df_all.dropna(subset=required_cols).copy()
        except KeyError:
            print(f"Skipping {var_name}: Column not found.")
            continue

        if len(_base) < 2: continue

        # B. 确定箱数
        total_valid = len(_base)
        if samples_per_bin is not None and samples_per_bin > 0:
            n_bins_use = max(1, int(np.ceil(total_valid / samples_per_bin)))
        else:
            n_bins_use = n_bins

        # C. 执行分箱
        sort_key = _base[var_name]
        if bin_method == 'quantile':
            try:
                q = pd.qcut(sort_key, q=n_bins_use, labels=False, duplicates='drop')
            except ValueError:
                uniq = sort_key.nunique()
                if uniq < 2: continue
                q = pd.qcut(sort_key, q=min(int(uniq), n_bins_use), labels=False, duplicates='drop')
        else:
            q = pd.cut(sort_key, bins=n_bins_use, labels=False)

        _base['bin_id'] = q

        # D. 聚合计算
        plot_data = []
        for bin_id, gbin in _base.groupby('bin_id'):
            if len(gbin) < 2: continue

            x_val = gbin[var_name].median()
            s2 = gbin[sigma_col] ** 2
            rms_spread = np.sqrt(factor * s2.mean())

            # --- metric='both' 特殊处理 ---
            if metric == 'both':
                plot_data.append({
                    "x": x_val, "y": rms_spread,
                    "label": "RMS Spread",
                    "color": "black",  # 显式设为黑色
                    "type": "Spread", "count": len(gbin)
                })

            # --- 计算各个指标 ---
            for col_val, label, color in targets:
                err = gbin[col_val] - gbin['ztd_gnss']
                rmse = np.sqrt(np.mean(err ** 2))

                if metric == 'diff':
                    val = rmse - rms_spread
                    plot_data.append({
                        "x": x_val, "y": val, "label": label, "color": color,
                        "type": "Line", "rmse": rmse, "spread": rms_spread, "count": len(gbin)
                    })
                elif metric == 'ratio':
                    val = rmse / max(rms_spread, 1e-6)
                    plot_data.append({
                        "x": x_val, "y": val, "label": label, "color": color,
                        "type": "Line", "rmse": rmse, "spread": rms_spread, "count": len(gbin)
                    })
                else:  # metric == 'both' -> 仅添加 RMSE
                    plot_data.append({
                        "x": x_val, "y": rmse, "label": label, "color": color,
                        "type": "RMSE", "count": len(gbin)
                    })

        if not plot_data: continue
        df_plot = pd.DataFrame(plot_data).sort_values('x')

        # E. 绘图 (Add Traces)
        unique_labels = list(reversed([l for l in df_plot['label'].unique() if l != "RMS Spread"]))
        if metric == 'both':
            unique_labels.append("RMS Spread")

        for label in unique_labels:
            group_df = df_plot[df_plot['label'] == label]
            if group_df.empty: continue

            color = group_df['color'].iloc[0]
            trace_type = group_df['type'].iloc[0]

            show_legend = (i == 0)

            # --- 样式配置 ---
            if metric == 'both':
                if trace_type == 'Spread':
                    # 黑色实线，实心方形
                    line_style = dict(color='black', width=2, dash='solid')
                    marker_style = dict(symbol='square', size=6, color='black')
                    hover_tmpl = "<b>Spread</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>N: %{text}<extra></extra>"
                else:
                    # RMSE: 实线，颜色区分，实心圆
                    line_style = dict(color=color, width=2, dash='solid')
                    marker_style = dict(symbol='circle', size=6, color=color)
                    hover_tmpl = f"<b>{label}</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>N: %{{text}}<extra></extra>"
            else:
                # Diff/Ratio
                line_style = dict(color=color, width=2)
                marker_style = dict(size=6, opacity=0.8)
                hover_tmpl = f"<b>{label}</b><br>x: %{{x:.2f}}<br>y: %{{y:.3f}}<br>RMSE: %{{customdata[0]:.2f}}<br>Spread: %{{customdata[1]:.2f}}<br>N: %{{text}}<extra></extra>"

            trace = go.Scatter(
                x=group_df['x'], y=group_df['y'],
                mode='lines+markers',
                name=label,
                line=line_style,
                marker=marker_style,
                legendgroup=label,
                showlegend=show_legend,
                text=group_df['count'],
                customdata=group_df[['rmse', 'spread']] if metric != 'both' else None,
                hovertemplate=hover_tmpl
            )
            fig.add_trace(trace, row=1, col=col_idx)

        # 设置 X 轴标题 (替代了 Subplot Title)
        fig.update_xaxes(title_text=var_name, row=1, col=col_idx)

    # --- 3. 全局样式 ---
    if metric == 'diff':
        ref_line = 0
        y_title = "RMS Error - RMS Spread (mm)"
        txt_top, txt_bot = "Under-dispersive", "Over-dispersive"
    elif metric == 'ratio':
        ref_line = 1
        y_title = "Ratio (Error / Spread)"
        txt_top, txt_bot = "Under-dispersive", "Over-dispersive"
    else:  # both
        ref_line = None
        y_title = "RMS Error and Spread (mm)"
        txt_top, txt_bot = "", ""

    if ref_line is not None:
        fig.add_hline(y=ref_line, line_dash="dash", line_color=COL_DIAG, line_width=1.5, opacity=0.7, row='all',
                      col='all')

    fig.update_layout(
        template=template,
        width=width, height=height,
        margin=dict(l=80, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )

    # Y轴标题
    fig.add_annotation(
        text=y_title,
        x=-0.06 if num_vars > 1 else -0.10,  # 微调位置
        y=0.5, xref="paper", yref="paper", textangle=-90,
        showarrow=False,font_size=16,
    )

    if metric != 'both':
        fig.add_annotation(
            text=txt_top, xref="x domain", yref="y domain",
            x=0.05, y=0.95, showarrow=False, xanchor="left",
            font=dict(size=10, color="gray"), row=1, col=1
        )
        fig.add_annotation(
            text=txt_bot, xref="x domain", yref="y domain",
            x=0.05, y=0.05, showarrow=False, xanchor="left",
            font=dict(size=10, color="gray"), row=1, col=1
        )

    fig.update_xaxes(zeroline=False, mirror=True, showline=True)
    fig.update_yaxes(zeroline=False, mirror=True, showline=True)

    return fig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def plot_spread_skill_analysis2(df_all,
                               bin_by_col=None,  # 如果启用 dist_col模式，此参数可忽略或设为 None
                               dist_col=None,  # 【新增参数】指定距离列名，若不为None，则启用三段式Regime模式
                               metric='diff',  # 'diff', 'ratio', 或 'both'
                               bin_method='quantile',
                               sigma_col='ztd_nwm_sigma',
                               template='simple_white',
                               n_bins=20,
                               samples_per_bin=None,
                               N=50,
                               width=None, height=500):
    """
    Spread-Skill 分析图。

    修改说明：
    新增 dist_col 参数。如果提供了 dist_col (例如 'dist_nearest')，
    则忽略 bin_by_col，强制生成 3 个子图 (High/Medium/Sparse Density)，
    每个子图内部按照 dist_col 进行分箱统计。
    """

    # --- 1. 参数校验与模式定义 ---

    # 定义绘图任务列表：[(Subplot Title, Binning Variable, Filter Function)]
    plot_tasks = []

    if dist_col is not None:
        # === 启用 Regime 模式 ===
        # 定义三个区间
        regimes = [
            ("High-Density (< 25 km)", lambda df: df[df[dist_col] < 25]),
            ("Medium-Density (25~50 km)", lambda df: df[(df[dist_col] >= 25) & (df[dist_col] <= 50)]),
            ("Sparse-Network (> 50 km)", lambda df: df[df[dist_col] > 50])
        ]

        for name, filter_func in regimes:
            # 这里的 var_name 统一为 dist_col，即每个子图X轴都是距离
            plot_tasks.append({
                "title": name,
                "var_name": dist_col,
                "filter_func": filter_func
            })
    else:
        # === 保持原有通用模式 ===
        if bin_by_col is None:
            raise ValueError("Must provide either 'bin_by_col' or 'dist_col'.")

        if isinstance(bin_by_col, str):
            bin_vars = [bin_by_col]
        else:
            bin_vars = bin_by_col

        for var in bin_vars:
            plot_tasks.append({
                "title": var,  # 原逻辑标题通常直接用变量名或空
                "var_name": var,
                "filter_func": lambda df: df  # 不做筛选
            })

    if metric not in ['diff', 'ratio', 'both']:
        raise ValueError("metric must be 'diff', 'ratio', or 'both'")

    if bin_method not in ['quantile', 'uniform']:
        raise ValueError("bin_method must be 'quantile' or 'uniform'")

    num_vars = len(plot_tasks)
    if width is None:
        width = 800 if num_vars == 1 else 400 * num_vars

    # 提取子图标题 (Regime模式下显示区间名，普通模式下显示变量名)
    subplot_titles = [t["title"] for t in plot_tasks] if dist_col is not None else None

    fig = make_subplots(
        rows=1, cols=num_vars,
        subplot_titles=subplot_titles,  # Regime模式下启用标题
        shared_yaxes=True,
        horizontal_spacing=0.03  # 稍微调大一点间距以免挤压
    )

    factor = (N + 1.0) / N

    if metric == 'both':
        label_bc = "RMS Error (BC)"
        label_raw = "RMS Error (Raw)"
    else:
        label_bc = "With Bias Corrections"
        label_raw = "Without Bias Corrections"

    targets = [
        ('ztd_nwm_cor', label_bc, COL_WITH),
        ('ztd_nwm', label_raw, COL_WITHOUT)
    ]

    # --- 2. 循环处理每个任务 (Regime 或 Variable) ---
    for i, task in enumerate(plot_tasks):
        col_idx = i + 1
        var_name = task["var_name"]
        filter_func = task["filter_func"]

        # A. 数据筛选与清洗
        # 先根据 Regime 筛选数据
        try:
            df_filtered = filter_func(df_all)
        except Exception as e:
            print(f"Error filtering data for {task['title']}: {e}")
            continue

        # 再清洗空值
        required_cols = ['ztd_gnss', 'ztd_nwm', 'ztd_nwm_cor', sigma_col, var_name]
        try:
            _base = df_filtered.dropna(subset=required_cols).copy()
        except KeyError:
            print(f"Skipping {task['title']}: Column not found.")
            continue

        if len(_base) < 2:
            # 如果某区间没数据，跳过绘图但保留坐标轴
            print(f"Warning: Not enough data for {task['title']}")
            continue

        # B. 确定箱数
        total_valid = len(_base)
        if samples_per_bin is not None and samples_per_bin > 0:
            n_bins_use = max(1, int(np.ceil(total_valid / samples_per_bin)))
        else:
            n_bins_use = n_bins

        # C. 执行分箱
        sort_key = _base[var_name]

        # 安全分箱逻辑
        try:
            if bin_method == 'quantile':
                # 处理数据量极少导致分位重复的情况
                unique_vals = sort_key.nunique()
                if unique_vals < n_bins_use:
                    actual_bins = unique_vals
                    if actual_bins < 2: actual_bins = 2
                else:
                    actual_bins = n_bins_use

                try:
                    q = pd.qcut(sort_key, q=actual_bins, labels=False, duplicates='drop')
                except ValueError:
                    q = pd.cut(sort_key, bins=actual_bins, labels=False)  # 降级为均匀分箱
            else:
                q = pd.cut(sort_key, bins=n_bins_use, labels=False)
        except Exception as e:
            print(f"Binning failed for {task['title']}: {e}")
            continue

        _base['bin_id'] = q

        # D. 聚合计算
        plot_data = []
        for bin_id, gbin in _base.groupby('bin_id'):
            if len(gbin) < 2: continue

            x_val = gbin[var_name].median()
            s2 = gbin[sigma_col] ** 2
            rms_spread = np.sqrt(factor * s2.mean())

            # --- metric='both' 特殊处理 ---
            if metric == 'both':
                plot_data.append({
                    "x": x_val, "y": rms_spread,
                    "label": "RMS Spread",
                    "color": "black",
                    "type": "Spread", "count": len(gbin)
                })

            # --- 计算各个指标 ---
            for col_val, label, color in targets:
                err = gbin[col_val] - gbin['ztd_gnss']
                rmse = np.sqrt(np.mean(err ** 2))

                if metric == 'diff':
                    val = rmse - rms_spread
                    plot_data.append({
                        "x": x_val, "y": val, "label": label, "color": color,
                        "type": "Line", "rmse": rmse, "spread": rms_spread, "count": len(gbin)
                    })
                elif metric == 'ratio':
                    val = rmse / max(rms_spread, 1e-6)
                    plot_data.append({
                        "x": x_val, "y": val, "label": label, "color": color,
                        "type": "Line", "rmse": rmse, "spread": rms_spread, "count": len(gbin)
                    })
                else:  # metric == 'both'
                    plot_data.append({
                        "x": x_val, "y": rmse, "label": label, "color": color,
                        "type": "RMSE", "count": len(gbin)
                    })

        if not plot_data: continue
        df_plot = pd.DataFrame(plot_data).sort_values('x')

        # E. 绘图 (Add Traces)
        unique_labels = list(reversed([l for l in df_plot['label'].unique() if l != "RMS Spread"]))
        if metric == 'both':
            unique_labels.append("RMS Spread")

        for label in unique_labels:
            group_df = df_plot[df_plot['label'] == label]
            if group_df.empty: continue

            color = group_df['color'].iloc[0]
            trace_type = group_df['type'].iloc[0]

            # 仅在第一个子图显示图例，避免重复
            show_legend = (i == 0)

            # --- 样式配置 ---
            if metric == 'both':
                if trace_type == 'Spread':
                    line_style = dict(color='black', width=2, dash='solid')
                    marker_style = dict(symbol='square', size=6, color='black')
                    hover_tmpl = "<b>Spread</b><br>Dist: %{x:.1f} km<br>y: %{y:.2f}<br>N: %{text}<extra></extra>"
                else:
                    line_style = dict(color=color, width=2, dash='solid')
                    marker_style = dict(symbol='circle', size=6, color=color)
                    hover_tmpl = f"<b>{label}</b><br>Dist: %{{x:.1f}} km<br>y: %{{y:.2f}}<br>N: %{{text}}<extra></extra>"
            else:
                line_style = dict(color=color, width=2)
                marker_style = dict(size=6, opacity=0.8)
                hover_tmpl = f"<b>{label}</b><br>Dist: %{{x:.1f}} km<br>y: %{{y:.3f}}<br>RMSE: %{{customdata[0]:.2f}}<br>Spread: %{{customdata[1]:.2f}}<br>N: %{{text}}<extra></extra>"

            trace = go.Scatter(
                x=group_df['x'], y=group_df['y'],
                mode='lines+markers',
                name=label,
                line=line_style,
                marker=marker_style,
                legendgroup=label,
                showlegend=show_legend,
                text=group_df['count'],
                customdata=group_df[['rmse', 'spread']] if metric != 'both' else None,
                hovertemplate=hover_tmpl
            )
            fig.add_trace(trace, row=1, col=col_idx)

        # 设置 X 轴标题
        # 如果是 dist_col 模式，虽然 Subplot Title 已经说明了区间，但 X 轴仍应显示变量名和单位
        x_label_text = var_name if dist_col is None else f"{dist_col} (km)"
        fig.update_xaxes(title_text=x_label_text, row=1, col=col_idx)

    # --- 3. 全局样式 ---
    if metric == 'diff':
        ref_line = 0
        y_title = "RMS Error - RMS Spread (mm)"
        txt_top, txt_bot = "Under-dispersive", "Over-dispersive"
    elif metric == 'ratio':
        ref_line = 1
        y_title = "Ratio (Error / Spread)"
        txt_top, txt_bot = "Under-dispersive", "Over-dispersive"
    else:  # both
        ref_line = None
        y_title = "RMS Error and Spread (mm)"
        txt_top, txt_bot = "", ""

    if ref_line is not None:
        fig.add_hline(y=ref_line, line_dash="dash", line_color=COL_DIAG, line_width=1.5, opacity=0.7, row='all',
                      col='all')

    fig.update_layout(
        template=template,
        width=width, height=height,
        margin=dict(l=80, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1)
    )

    fig.add_annotation(
        text=y_title,
        x=-0.06 if num_vars > 1 else -0.12,
        y=0.5, xref="paper", yref="paper", textangle=-90,
        showarrow=False, font_size=16,
    )

    # 仅在 Metric 不为 both 且是第一个图时添加说明性文字
    if metric != 'both':
        fig.add_annotation(
            text=txt_top, xref="x domain", yref="y domain",
            x=0.05, y=0.95, showarrow=False, xanchor="left",
            font=dict(size=10, color="gray"), row=1, col=1
        )
        fig.add_annotation(
            text=txt_bot, xref="x domain", yref="y domain",
            x=0.05, y=0.05, showarrow=False, xanchor="left",
            font=dict(size=10, color="gray"), row=1, col=1
        )

    fig.update_xaxes(zeroline=False, mirror=True, showline=True)
    fig.update_yaxes(zeroline=False, mirror=True, showline=True)

    return fig


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# 假设上下文已有 fit_rmse, pretty_label, COL_WITH, COL_WITHOUT, COL_DIAG 等辅助变量/函数

def bin_and_aggregate2(g: pd.DataFrame, mean_col: str, xaxis: str,
                       n_bins=20, min_per_bin=2,
                       metric='rmse',
                       bin_method='quantile',
                       min_bin_points=0,
                       bin_spacing=None  # 控制分箱间隔
                       ) -> pd.DataFrame:
    """
    分箱并计算 Spread 与 Skill (重构版)
    :param metric: 'rmse' 或 'mse'
    :param bin_method: 'quantile' 或 'uniform'
    :param bin_spacing: uniform模式下的步长。
                        注意：若 metric='mse'，此步长作用于 spread^2 (方差) 空间；
                        若 metric='rmse'，此步长作用于 spread (标准差) 空间。
    """
    # 基础校验
    total_valid = g[xaxis].notna().sum()
    if total_valid < max(2, min_per_bin):
        return pd.DataFrame()

    gg = g.copy()
    gg['e'] = gg[mean_col] - gg['ztd_gnss']
    gg['s2'] = gg[xaxis] ** 2

    # ---- 分箱逻辑 (修改后) ----
    if bin_method == 'uniform':
        # 等宽分箱 (Uniform)
        if bin_spacing is not None and bin_spacing > 0:
            # 模式 A: 指定间隔，从 0 开始

            # 关键修改：根据 metric 决定对谁进行切分，以保证最终图表X轴点的视觉间距尽可能相等
            if metric == 'mse':
                target_series = gg['s2']  # 对方差切分
            else:
                target_series = gg[xaxis]  # 对标准差切分

            max_val = target_series.max()
            # 生成边界 [0, spacing, 2*spacing, ...]
            bins = np.arange(0, max_val + bin_spacing, bin_spacing)

            # 边界检查
            if len(bins) < 2:
                bins = np.array([0, max(max_val, bin_spacing)])

            # 对 target_series 进行切分
            gg['bin_id'] = pd.cut(target_series, bins=bins, labels=False, include_lowest=True)

        else:
            # 模式 B: 指定箱数，自动计算间隔 (fallback)
            # 这里同样应该考虑 metric，但为了简化 fallback 逻辑，这里暂且对原始 xaxis 分箱
            # 或者如果非常严格，这里也可以根据 metric 切换，但在没有指定 spacing 时，quantile 是更好的选择
            min_x, max_x = gg[xaxis].min(), gg[xaxis].max()
            if min_x == max_x:
                return pd.DataFrame()
            bins = np.linspace(min_x, max_x, n_bins + 1)
            gg['bin_id'] = pd.cut(gg[xaxis], bins=bins, labels=False, include_lowest=True)
    else:
        # 等频分箱 (Quantile - 默认)
        try:
            gg['bin_id'] = pd.qcut(gg[xaxis], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            uniq = gg[xaxis].nunique(dropna=True)
            if uniq < 2: return pd.DataFrame()
            gg['bin_id'] = pd.qcut(gg[xaxis], q=min(int(uniq), n_bins), labels=False, duplicates='drop')

    # ---- 聚合计算 ----
    rows = []
    # 偏差修正系数
    N_target = 50
    factor = (N_target + 1) / N_target

    for _, gbin in gg.groupby('bin_id'):
        count = len(gbin)
        if count < min_per_bin:
            continue

        # 最小样本过滤
        if min_bin_points > 0 and count < min_bin_points:
            continue

        # 核心统计
        mean_s2 = gbin['s2'].mean()  # E[sigma^2]
        mse_val = np.mean(gbin['e'] ** 2)  # MSE

        # 应用偏差修正
        spread_sq_corrected = factor * mean_s2

        if metric == 'mse':
            val_x = spread_sq_corrected
            val_y = mse_val
        else:  # rmse
            val_x = np.sqrt(spread_sq_corrected)
            val_y = np.sqrt(mse_val)

        rows.append({
            "spread_val": val_x,
            "error_val": val_y,
            "bin_count": count,
            "bin_median_spread": np.median(gbin[xaxis])
        })

    if not rows:
        return pd.DataFrame()

    agg = pd.DataFrame(rows)
    # 按 spread 大小排序
    # 注意：如果是 mse 模式且按 s2 分箱，这里排序依然有效，因为 s 与 s^2 单调性一致
    return agg.sort_values('spread_val')[["spread_val", "error_val", "bin_count"]]


def plot_spread_skill_matrix2(df_all,
                              col_var,
                              row_var=None,
                              template='simple_white',
                              xaxis='ztd_nwm_sigma',
                              row_order=None, col_order=None,
                              include_global=True,
                              fit_line='linear',  # 修改: "linear" or None
                              n_bins=20,
                              metric='rmse',  # 'rmse' | 'mse'
                              bin_method='quantile',  # 'quantile' | 'uniform'
                              bin_spacing=None,  # uniform 模式下的步长
                              min_bin_points=0,  # 最小样本过滤
                              ax_min=0.0,
                              width=1200, height=970):
    # --- 0. 配置标签与常量 ---
    is_mse = (metric == 'mse')

    if is_mse:
        col_x = "Mean Squared Spread"
        col_y = "Mean Squared Error"
        unit_label = "(mm²)"
    else:
        col_x = "RMS Spread"
        col_y = "RMS Error"
        unit_label = "(mm)"

    # --- A. 数据准备 ---
    _base = df_all.copy()
    if not np.issubdtype(_base['time'].dtype, np.datetime64):
        _base['time'] = pd.to_datetime(_base['time'])
    _base = _base[np.isfinite(_base[xaxis])].copy()

    # 扩展 Global
    if include_global:
        _global = _base.copy()
        _global[col_var] = 'Global'
        if col_order and 'Global' not in col_order:
            col_order.append('Global')
        _base = pd.concat([_base, _global], ignore_index=True)

    internal_row_var = row_var
    if row_var is None:
        internal_row_var = '__SingleRow__'
        _base[internal_row_var] = ' '
        if row_order is None:
            row_order = [' ']

    need = ['time', 'ztd_gnss', 'ztd_nwm', 'ztd_nwm_cor', xaxis, internal_row_var, col_var]
    _base = _base.dropna(subset=need)

    if row_order:
        _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
    if col_order:
        _base = _base[_base[col_var].astype(str).isin(col_order)]

    # --- B. 聚合计算 ---
    agg_list = []
    targets = [('ztd_nwm_cor', 'With Bias Corrections'),
               ('ztd_nwm', 'Without Bias Corrections')]

    for (r_val, c_val), g in _base.groupby([internal_row_var, col_var], observed=True):
        for mean_col, mean_name in targets:
            res = bin_and_aggregate2(g, mean_col, xaxis,
                                     n_bins=n_bins,
                                     metric=metric,
                                     bin_method=bin_method,
                                     bin_spacing=bin_spacing,
                                     min_bin_points=min_bin_points)

            if not res.empty:
                res = res.rename(columns={'spread_val': col_x, 'error_val': col_y})
                res[internal_row_var] = r_val
                res[col_var] = c_val
                res['MeanType'] = mean_name
                agg_list.append(res)

    if not agg_list:
        print("No valid data.")
        return None

    agg_all = pd.concat(agg_list, ignore_index=True)
    agg_all['MeanLabel'] = agg_all['MeanType'].map(pretty_label)

    # 计算坐标轴上限
    max_val = agg_all[[col_x, col_y]].max().max()
    AX_MAX = np.ceil(max_val * 1.1)

    # --- C. 对角线数据 ---
    diag_rows = []
    use_rows = row_order if row_order else sorted(agg_all[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(agg_all[col_var].unique())

    for r in use_rows:
        for c in use_cols:
            diag_rows.append({
                col_x: ax_min, col_y: ax_min,
                internal_row_var: r, col_var: c, 'MeanLabel': 'y = x'
            })
            diag_rows.append({
                col_x: AX_MAX, col_y: AX_MAX,
                internal_row_var: r, col_var: c, 'MeanLabel': 'y = x'
            })

    plot_df = pd.concat([agg_all, pd.DataFrame(diag_rows)], ignore_index=True)

    # --- D. 拟合计算 (仅当 fit_line 不为 None 时) ---
    fit_rows = []
    curve_rows = []

    # 判断是否进行拟合
    do_fit = (fit_line == 'linear')

    if do_fit:
        mask_real = (plot_df['MeanLabel'] != 'y = x') & plot_df['bin_count'].notna()

        for (r, c, lbl), g in plot_df[mask_real].groupby([internal_row_var, col_var, 'MeanLabel']):
            x_dat = g[col_x].to_numpy()
            y_dat = g[col_y].to_numpy()
            if len(x_dat) < 2: continue

            # 使用 fit_rmse 进行计算 (假定 fit_rmse 支持线性回归)
            # 无论 metric 是 RMSE 还是 MSE，只要 fit_line='linear'，我们都拟合 y = ax + b
            a, b = fit_rmse(x_dat, y_dat, linear=True)
            fit_rows.append({'row': r, 'col': c, 'label': lbl, 'a': a, 'b': b})

            xs = np.linspace(ax_min, AX_MAX, 160)
            ys = a * xs + b  # 线性公式

            tmp = pd.DataFrame({col_x: xs, col_y: ys})
            tmp[internal_row_var] = r
            tmp[col_var] = c
            tmp['MeanLabel'] = lbl
            curve_rows.append(tmp)

        curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    # --- E. 绘图 ---
    color_map = {
        "<b>With</b> Bias Corrections": COL_WITH,
        "<b>Without</b> Bias Corrections": COL_WITHOUT,
        "y = x": COL_DIAG
    }

    actual_facet_row = internal_row_var if row_var is not None else None
    if row_var is None and height == 970:
        height = 450

    fig = px.scatter(
        plot_df, x=col_x, y=col_y,
        template=template,
        color="MeanLabel", color_discrete_map=color_map,
        facet_col=col_var,
        facet_row=actual_facet_row,
        facet_col_spacing=0.01, facet_row_spacing=0.02,
        category_orders={col_var: use_cols, internal_row_var: use_rows,
                         "MeanLabel": ["<b>With</b> Bias Corrections",
                                       "<b>Without</b> Bias Corrections", "y = x"]},
        hover_data={"bin_count": True, col_x: ':.2f', col_y: ':.2f'},
    )

    for tr in fig.data:
        tr.showlegend = False
        if tr.name == 'y = x':
            tr.mode = 'lines'
            tr.line.update(color=COL_DIAG, width=1.0, dash='dot')
        else:
            tr.mode = 'markers'
            tr.marker.update(size=4)

    # 绘制拟合线
    if do_fit and not curve_df.empty:
        fig_curve = px.line(
            curve_df, x=col_x, y=col_y,
            color="MeanLabel", color_discrete_map=color_map,
            facet_col=col_var,
            facet_row=actual_facet_row,
            category_orders={col_var: use_cols, internal_row_var: use_rows}
        )
        for tr in fig_curve.data:
            tr.showlegend = False
            tr.mode = 'lines'
            tr.line.update(width=1.0)
            tr.marker.update(size=0, opacity=0)
            tr.hoverinfo = 'skip'
            fig.add_trace(tr)

    # --- F. 标注与布局 ---
    fit_df = pd.DataFrame(fit_rows)
    # 只有 fit_df 不为空 (意味着 do_fit=True 且有足够数据) 才绘制标注
    if not fit_df.empty:
        for (r_val, c_val), g in fit_df.groupby(['row', 'col']):
            try:
                row_idx = use_rows.index(r_val)
                col_idx = use_cols.index(c_val)
                annot_row = len(use_rows) - row_idx
                annot_col = col_idx + 1
            except ValueError:
                continue

            def add_fit_annot(sub_df, y_pos, color_code):
                if sub_df.empty: return
                r0 = sub_df.iloc[0]
                a, b = r0['a'], r0['b']
                sign = "+" if b >= 0 else "-"
                val_b = abs(b)

                # 生成公式文本
                if is_mse:
                    # MSE 线性公式
                    txt = (
                        f"MSE = "
                        f"<b>{a:.2f}</b> · MSS {sign} <b>{val_b:.2f}</b>"
                    )
                else:
                    # RMSE 线性公式
                    txt = (
                        f"Error = "
                        f"<b>{a:.1f}</b> · Spread {sign} <b>{val_b:.1f}</b>"
                    )

                fig.add_annotation(
                    text=txt, xref="x domain", yref="y domain",
                    x=0.05, y=y_pos, showarrow=False,
                    font=dict(size=14, color=color_code),
                    row=annot_row, col=annot_col,
                )

            add_fit_annot(g[g['label'].str.contains("With")], 0.05, COL_WITH)
            add_fit_annot(g[g['label'].str.contains("Without")], 0.15, COL_WITHOUT)

    def _strip_facet(t):
        t = t.replace(f"{col_var}=", "")
        if row_var:
            t = t.replace(f"{row_var}=", "")
        if internal_row_var == '__SingleRow__':
            if '__SingleRow__=' in t or t == ' ':
                return ''
        return t

    fig.for_each_annotation(lambda a: a.update(text=_strip_facet(a.text))
    if isinstance(a.text, str) else None)

    fig.update_layout(title=None, width=width, height=height)

    fig.update_xaxes(zeroline=False, tickmode='linear',  mirror=True, showline=True)
    fig.update_yaxes(zeroline=False, tickmode='linear',  mirror=True, showline=True)

    for k in fig.layout:
        if k.startswith('xaxis'):
            fig.layout[k].update(range=[ax_min, AX_MAX])
        if k.startswith('yaxis'):
            x_token = k.replace('yaxis', 'x')
            fig.layout[k].update(range=[ax_min, AX_MAX], scaleanchor=x_token, scaleratio=1.0)

    for tr in fig.data:
        if tr.name == 'y = x':
            tr.x = [ax_min, AX_MAX]
            tr.y = [ax_min, AX_MAX]

    for k in fig.layout:
        if k.startswith('xaxis'): fig.layout[k].title.text = ''
        if k.startswith('yaxis'): fig.layout[k].title.text = ''

    x_domains = {k: fig.layout[k].domain for k in fig.layout if k.startswith('xaxis')}
    y_domains = {k: fig.layout[k].domain for k in fig.layout if k.startswith('yaxis')}

    rows = {}
    for yk, dom in y_domains.items():
        rows.setdefault(tuple(dom), []).append(yk)

    for ydom, y_axes in rows.items():
        xs = []
        for yk in y_axes:
            for xk in x_domains:
                if getattr(fig.layout[xk], 'anchor', None) == yk.replace('yaxis', 'y'):
                    xs.append(xk)
        if xs:
            left_x = min(xs, key=lambda k: x_domains[k][0])
            left_y = next(
                (yk for yk in y_axes if getattr(fig.layout[left_x], 'anchor', None) == yk.replace('yaxis', 'y')), None)
            if left_y:
                fig.layout[left_y].title.text = f"{col_y} {unit_label}"

    if rows:
        bottom_ydom = min(rows.keys(), key=lambda d: d[0])
        for yk in rows[bottom_ydom]:
            for xk in x_domains:
                if getattr(fig.layout[xk], 'anchor', None) == yk.replace('yaxis', 'y'):
                    fig.layout[xk].title.text = f"{col_x} {unit_label}"

    if use_rows and use_cols:
        top_r_idx = 1
        top_c_idx = 1
        fig.add_annotation(
            text=(
                f"<span style='color:{COL_WITHOUT}; font-weight:bold'>Raw</span> / <span style='color:{COL_WITH}; font-weight:bold'>BC</span>"),
            xref="x domain", yref="y domain",
            x=0.06, y=0.96, xanchor="left", yanchor="top",
            showarrow=False, font=dict(size=15),
            row=top_r_idx, col=top_c_idx
        )

    return fig


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 假设上下文中已有的颜色常量，如果没有则定义默认值
COL_RAW = "#e74c3c"  # Red for Raw
COL_BC = "#3498db"  # Blue for Bias-Corrected
COL_VCE = "#2ecc71"  # Green for BC + VCE Prediction
COL_DIAG = "#95a5a6"  # Gray for diagonal


def _solve_vce_for_plot(subset, res_col, spread_col, formal_col, member_count=50):
    """
    内部轻量级 VCE 求解器，用于绘图时的参数估计。
    返回 alpha, beta, gamma 以及计算出的 background_noise_sq (用于画曲线)
    """
    # 简单的迭代求解逻辑 (简化版 strict_lsvce)
    y_sq = subset[res_col].values ** 2
    scale = (member_count + 1) / member_count
    q1 = scale * (subset[spread_col].values ** 2)
    q2 = subset[formal_col].values ** 2
    q3 = np.ones_like(q1)

    s1, s2, s3 = 1.0, 1.0, np.var(subset[res_col]) * 0.1

    # 快速迭代 20 次
    for _ in range(20):
        v = s1 * q1 + s2 * q2 + s3 * q3
        v = np.maximum(v, 1e-8)
        w = 1.0 / (v ** 2)

        # 简化版矩阵构建 (只列出需要的)
        n11 = np.sum(w * q1 * q1);
        n12 = np.sum(w * q1 * q2);
        n13 = np.sum(w * q1 * q3)
        n22 = np.sum(w * q2 * q2);
        n23 = np.sum(w * q2 * q3);
        n33 = np.sum(w * q3 * q3)
        l1 = np.sum(w * y_sq * q1);
        l2 = np.sum(w * y_sq * q2);
        l3 = np.sum(w * y_sq * q3)

        try:
            N_mat = np.array([[n11, n12, n13], [n12, n22, n23], [n13, n23, n33]])
            l_vec = np.array([l1, l2, l3])
            s_new = np.linalg.solve(N_mat, l_vec)
            s1, s2, s3 = np.maximum(s_new, 1e-6)  # 非负约束
        except:
            break  # 奇异，保持旧值

    # 计算背景噪声平均值 (用于画图：y = sqrt(alpha^2*x^2 + C))
    # C = mean(beta^2 * Formal^2 + gamma^2)
    bg_noise_sq = np.mean(s2 * q2 + s3)

    return np.sqrt(s1), np.sqrt(s2), np.sqrt(s3), bg_noise_sq


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# 假设 estimate_vce_generalized 和 bin_and_aggregate2 已定义
# from trop.vce.vce import estimate_vce_generalized
# from ... import bin_and_aggregate2

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_spread_skill_vce_matrix(df_all,
                                 col_var,
                                 row_var=None,
                                 template='simple_white',
                                 # --- 核心数据列配置 ---
                                 spread_col='ztd_nwm_sigma',  # NWM Spread
                                 nwm_raw_col='ztd_nwm',  # NWM 原始值
                                 nwm_cor_col='ztd_nwm_cor',  # NWM 校正值 (预留，虽然下面强制用了 res_cor)
                                 gnss_val_col='ztd_gnss',  # GNSS 观测值
                                 gnss_formal_col='ztd_gnss_sigma',  # GNSS 形式误差

                                 row_order=None, col_order=None,
                                 include_global=True,

                                 # --- VCE 配置 ---
                                 member_count=50,
                                 fixed_beta=None,

                                 # --- 分箱与绘图配置 ---
                                 n_bins=20,
                                 metric='rmse',
                                 bin_method='quantile',
                                 bin_spacing=None,
                                 min_bin_points=50,
                                 ax_min=0.0,
                                 width=1200, height=970):
    # --- 0. 配置标签与常量 ---
    is_mse = (metric == 'mse')

    NAME_RAW = "Raw"
    NAME_BC = "BC"
    NAME_VCE = "BC+VCE"

    COL_RAW = "#999999"
    COL_BC = "#1f77b4"
    COL_VCE = "#d62728"
    COL_DIAG = "black"

    if is_mse:
        col_x = "Estimated Mean Squared Error"
        col_y = "Observed Mean Squared Error"
        unit_label = "(mm²)"
    else:
        col_x = "Estimated RMS Error"
        col_y = "Observed RMS Error"
        unit_label = "(mm)"

    # --- A. 数据准备 ---
    _base = df_all.copy()
    if not np.issubdtype(_base['time'].dtype, np.datetime64):
        _base['time'] = pd.to_datetime(_base['time'])

    # 确保 'res_cor' 存在 (如果不存在尝试计算，但最好由外部传入)
    if 'res_cor' not in _base.columns:
        if nwm_cor_col in _base.columns and gnss_val_col in _base.columns:
            _base['res_cor'] = _base[nwm_cor_col] - _base[gnss_val_col]
        else:
            # 如果实在没有，假设用户已经准备好了，或者报错
            pass

            # 扩展 Global
    if include_global:
        _global = _base.copy()
        _global[col_var] = 'Global'
        if col_order and 'Global' not in col_order:
            col_order.append('Global')
        _base = pd.concat([_base, _global], ignore_index=True)

    internal_row_var = row_var
    if row_var is None:
        internal_row_var = '__SingleRow__'
        _base[internal_row_var] = ' '
        if row_order is None:
            row_order = [' ']

    # 筛选
    if row_order:
        _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
    if col_order:
        _base = _base[_base[col_var].astype(str).isin(col_order)]

    # --- B. 核心循环：VCE 计算 + 数据分箱 ---
    agg_list = []
    vce_results_map = {}

    scale_factor = (member_count + 1) / member_count

    # 按子图分组循环
    for (r_val, c_val), g in _base.groupby([internal_row_var, col_var], observed=True):
        if len(g) < max(min_bin_points, 20): continue

        # === 1. VCE 参数估计 ===
        g_vce = g.copy()
        g_vce['__dummy_group__'] = 1  # 避免 groupby(None) 错误

        try:
            _, df_params = estimate_vce_generalized(
                g_vce,
                group_col='__dummy_group__',
                fixed_beta=fixed_beta,
                res_col='res_cor',  # 这里还是用参数名传递给计算函数
                nwm_spread_col=spread_col,
                gnss_formal_col=gnss_formal_col,
                member_count=member_count,
                max_iter=30
            )
            p = df_params.iloc[0]
            alpha, beta, gamma = p['alpha'], p['beta'], p['gamma']
            vce_results_map[(r_val, c_val)] = (alpha, beta, gamma)

        except Exception as e:
            print(f"VCE Failed for {r_val}-{c_val}: {e}")
            alpha, beta, gamma = np.nan, np.nan, np.nan
            continue

        # === 2. 准备绘图数据 ===

        # VCE 预测的总标准差 (X轴 for VCE)
        term1 = (alpha ** 2) * scale_factor * (g[spread_col] ** 2)
        term2 = (beta ** 2) * (g[gnss_formal_col] ** 2)
        term3 = gamma ** 2
        vce_predicted_sigma = np.sqrt(term1 + term2 + term3)

        # 原始 Spread (X轴 for Raw/BC)
        raw_spread_inflated = np.sqrt(scale_factor) * g[spread_col]

        # --- Scenario A: Raw ---
        # 原始残差: ztd_nwm - ztd_gnss
        raw_resid = g[nwm_raw_col] - g[gnss_val_col]

        df_raw = pd.DataFrame({'x': raw_spread_inflated, 'y': raw_resid, 'ztd_gnss': 0.0})
        res_raw_bin = bin_and_aggregate2(df_raw, 'y', 'x', n_bins=n_bins, metric=metric,
                                         bin_method=bin_method, min_bin_points=min_bin_points)
        if not res_raw_bin.empty:
            res_raw_bin['MeanType'] = NAME_RAW
            res_raw_bin[internal_row_var] = r_val
            res_raw_bin[col_var] = c_val
            agg_list.append(res_raw_bin)

        # --- Scenario B: BC ---
        # [要求4] 强制使用 g['res_cor']
        df_bc = pd.DataFrame({'x': raw_spread_inflated, 'y': g['res_cor'], 'ztd_gnss': 0.0})
        res_bc_bin = bin_and_aggregate2(df_bc, 'y', 'x', n_bins=n_bins, metric=metric,
                                        bin_method=bin_method, min_bin_points=min_bin_points)
        if not res_bc_bin.empty:
            res_bc_bin['MeanType'] = NAME_BC
            res_bc_bin[internal_row_var] = r_val
            res_bc_bin[col_var] = c_val
            agg_list.append(res_bc_bin)

        # --- Scenario C: BC+VCE ---
        # [要求4] 强制使用 g['res_cor']
        df_vce = pd.DataFrame({'x': vce_predicted_sigma, 'y': g['res_cor'], 'ztd_gnss': 0.0})
        res_vce_bin = bin_and_aggregate2(df_vce, 'y', 'x', n_bins=n_bins, metric=metric,
                                         bin_method=bin_method, min_bin_points=min_bin_points)
        if not res_vce_bin.empty:
            res_vce_bin['MeanType'] = NAME_VCE
            res_vce_bin[internal_row_var] = r_val
            res_vce_bin[col_var] = c_val
            agg_list.append(res_vce_bin)

    if not agg_list:
        print("No valid data.")
        return None

    agg_all = pd.concat(agg_list, ignore_index=True)

    # 列重命名
    agg_all = agg_all.rename(columns={'spread_val': col_x, 'error_val': col_y})

    # 坐标轴上限
    max_val = agg_all[[col_x, col_y]].max().max()
    AX_MAX = np.ceil(max_val * 1.1)
    if AX_MAX <= ax_min: AX_MAX = 1.0

    # --- C. 对角线 ---
    diag_rows = []
    use_rows = row_order if row_order else sorted(agg_all[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(agg_all[col_var].unique())

    for r in use_rows:
        for c in use_cols:
            diag_rows.append({
                col_x: ax_min, col_y: ax_min,
                internal_row_var: r, col_var: c, 'MeanType': 'y = x'
            })
            diag_rows.append({
                col_x: AX_MAX, col_y: AX_MAX,
                internal_row_var: r, col_var: c, 'MeanType': 'y = x'
            })

    plot_df = pd.concat([agg_all, pd.DataFrame(diag_rows)], ignore_index=True)

    # --- D. 绘图 ---
    color_map = {
        NAME_RAW: COL_RAW,
        NAME_BC: COL_BC,
        NAME_VCE: COL_VCE,
        "y = x": COL_DIAG
    }

    actual_facet_row = internal_row_var if row_var is not None else None
    if row_var is None and height == 970:
        height = 500

    fig = px.scatter(
        plot_df, x=col_x, y=col_y,
        template=template,
        color="MeanType", color_discrete_map=color_map,
        facet_col=col_var,
        facet_row=actual_facet_row,
        facet_col_spacing=0.015, facet_row_spacing=0.04,  # 稍微增加行间距
        category_orders={col_var: use_cols, internal_row_var: use_rows,
                         "MeanType": [NAME_RAW, NAME_BC, NAME_VCE, "y = x"]},
    )

    # 图例去重与样式
    names_seen = set()
    for tr in fig.data:
        if tr.name == 'y = x':
            tr.mode = 'lines'
            tr.line.update(color=COL_DIAG, width=1.0, dash='dash')
        else:
            tr.mode = 'markers'
            if tr.name == NAME_VCE:
                tr.marker.update(size=6, opacity=0.9, symbol='circle')
            elif tr.name == NAME_BC:
                tr.marker.update(size=5, opacity=0.7, symbol='diamond')
            elif tr.name == NAME_RAW:
                tr.marker.update(size=4, opacity=0.4, symbol='cross')

        if tr.name in names_seen:
            tr.showlegend = False
        else:
            tr.showlegend = True
            names_seen.add(tr.name)

    # --- E. 标注 VCE 参数 (无框公式) ---
    if vce_results_map:
        for (r_val, c_val), (alpha, beta, gamma) in vce_results_map.items():
            try:
                row_idx = use_rows.index(r_val)
                col_idx = use_cols.index(c_val)
                annot_row = row_idx + 1
                annot_col = col_idx + 1
            except ValueError:
                continue

            # [要求3] 无框公式
            tex_str = (
                r"$\hat{\sigma} = \sqrt{"
                rf"({alpha:.2f} S)^2 + "
                rf"({beta:.2f} \sigma_o)^2 + "
                rf"{gamma:.2f}^2"
                r"}$"
            )

            fig.add_annotation(
                text=tex_str,
                row=annot_row, col=annot_col,
                x=0.05, y=0.95,
                xref="x domain", yref="y domain",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.7)",
                # bordercolor="black", # [FIX] 移除边框
                # borderwidth=1,       # [FIX] 移除边框宽度
                font=dict(size=11, color="black")
            )

    # --- F. 布局美化与修复 ---

    # [要求1] 修复 col_var 注释丢失
    # 逻辑：只移除 "col_var=" 这样的前缀，保留值。同时保护 LaTeX 公式。
    def _clean_annotation(t):
        if not isinstance(t, str): return t
        # 保护公式
        if "hat" in t or "sigma" in t: return t

        # 移除 facet 前缀
        if col_var and f"{col_var}=" in t:
            t = t.replace(f"{col_var}=", "")
        if row_var and f"{row_var}=" in t:
            t = t.replace(f"{row_var}=", "")

        # 处理 Dummy Row
        if internal_row_var == '__SingleRow__':
            if '__SingleRow__=' in t or t == ' ':
                return ''
        return t

    fig.for_each_annotation(lambda a: a.update(text=_clean_annotation(a.text)))

    # 轴范围
    fig.update_xaxes(range=[ax_min, AX_MAX], zeroline=False, mirror=True, showline=True)
    fig.update_yaxes(range=[ax_min, AX_MAX], zeroline=False, mirror=True, showline=True, scaleanchor="x", scaleratio=1)

    # 移除子图轴标题
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)

    # [要求2] 增加底部 Margin，防止轴标签与 tick 重合
    fig.update_layout(
        width=width, height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            title=None,
            itemsizing='constant'
        ),
        # 增加 b (bottom) margin
        margin=dict(l=60, r=40, t=80, b=90)
    )

    # 全局轴标题 (位置微调)
    fig.add_annotation(
        text=f"{col_x} {unit_label}",
        xref="paper", yref="paper",
        # [FIX] 将 y 坐标调得更低，适配增加的 margin
        x=0.5, y=-0.08,
        showarrow=False,
        font=dict(size=16)
    )
    fig.add_annotation(
        text=f"{col_y} {unit_label}",
        xref="paper", yref="paper",
        x=-0.04, y=0.5, textangle=-90, showarrow=False,
        font=dict(size=16)
    )

    return fig


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# 假设上下文已有的辅助变量，如果环境中没有，请自行定义
# COL_WITH = "red"
# COL_WITHOUT = "blue"
# COL_DIAG = "gray"
# fit_rmse = ... (你的拟合函数)
# pretty_label = ... (你的标签映射函数)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# 假设上下文已有的辅助变量（保持你原有的定义）
# COL_WITH = "red"
# COL_WITHOUT = "blue"
# COL_DIAG = "gray"
# fit_rmse = ...
# pretty_label = ...

def bin_and_aggregate3(g: pd.DataFrame, mean_col: str, xaxis: str,
                       n_bins=20, min_per_bin=2,
                       metric='rmse',
                       bin_method='quantile',
                       min_bin_points=0,
                       bin_spacing=None) -> pd.DataFrame:
    total_valid = g[xaxis].notna().sum()
    if total_valid < max(2, min_per_bin):
        return pd.DataFrame()

    gg = g.copy()
    gg['e'] = gg[mean_col] - gg['ztd_gnss']
    gg['s2'] = gg[xaxis] ** 2

    # ---- 分箱逻辑 ----
    if bin_method == 'uniform':
        if bin_spacing is not None and bin_spacing > 0:
            if metric == 'mse':
                target_series = gg['s2']
            else:
                target_series = gg[xaxis]

            max_val = target_series.max()
            # 确保bins覆盖最大值
            bins = np.arange(0, max_val + bin_spacing, bin_spacing)
            if len(bins) < 2: bins = np.array([0, max(max_val, bin_spacing)])

            gg['bin_id'] = pd.cut(target_series, bins=bins, labels=False, include_lowest=True)
        else:
            # 如果没有提供 spacing，回退到 linspace
            min_x, max_x = gg[xaxis].min(), gg[xaxis].max()
            bins = np.linspace(min_x, max_x, n_bins + 1)
            gg['bin_id'] = pd.cut(gg[xaxis], bins=bins, labels=False, include_lowest=True)
    else:
        # Quantile
        try:
            gg['bin_id'] = pd.qcut(gg[xaxis], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            uniq = gg[xaxis].nunique(dropna=True)
            if uniq < 2: return pd.DataFrame()
            gg['bin_id'] = pd.qcut(gg[xaxis], q=min(int(uniq), n_bins), labels=False, duplicates='drop')

    # ---- 聚合计算 ----
    rows = []
    N_target = 50
    factor = (N_target + 1) / N_target

    for bid, gbin in gg.groupby('bin_id'):
        count = len(gbin)
        if count < min_per_bin: continue
        if min_bin_points > 0 and count < min_bin_points: continue

        mean_s2 = gbin['s2'].mean()
        mse_val = np.mean(gbin['e'] ** 2)
        spread_sq_corrected = factor * mean_s2

        if metric == 'mse':
            val_x = spread_sq_corrected
            val_y = mse_val
        else:
            val_x = np.sqrt(spread_sq_corrected)
            val_y = np.sqrt(mse_val)

        rows.append({
            "bin_id": bid,  # 【关键】保存 bin_id
            "spread_val": val_x,
            "error_val": val_y,
            "bin_count": count,
            "bin_median_spread": np.median(gbin[xaxis])
        })

    if not rows: return pd.DataFrame()
    agg = pd.DataFrame(rows)
    return agg.sort_values('spread_val')


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 假设上下文已有的辅助变量
# COL_WITH = "red"
# COL_WITHOUT = "blue"
# COL_DIAG = "gray"
# fit_rmse = ...
# pretty_label = ...
# bin_and_aggregate3 = ... (使用你之前定义好的函数)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 假设环境已有 bin_and_aggregate3, fit_rmse 等函数
# 如果没有，请确保将之前的辅助函数也包含在内

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_spread_skill_final_v8(df_all,
                               col_var,
                               row_var=None,
                               template='simple_white',
                               xaxis='ztd_nwm_sigma',
                               row_order=None, col_order=None,
                               include_global=True,
                               fit_line='linear',
                               n_bins=25,
                               metric='rmse',
                               bin_method='quantile',
                               bin_spacing=None,
                               min_bin_points=0,
                               ax_min=0.0,
                               width=1200, height=None):
    # --- 1. 基础配置 ---
    is_mse = (metric == 'mse')
    col_x_label = "Mean Squared Spread (mm²)" if is_mse else "RMS Spread (mm)"
    col_y_label = "Mean Squared Error (mm²)" if is_mse else "RMS Error (mm)"

    # 计算 dtick (如果 bin_spacing 存在)
    scatter_dtick = int(bin_spacing * 5) if (bin_spacing is not None and bin_spacing > 0) else None

    # --- 2. 数据准备 ---
    _base = df_all.copy()
    if not np.issubdtype(_base['time'].dtype, np.datetime64):
        _base['time'] = pd.to_datetime(_base['time'])
    _base = _base[np.isfinite(_base[xaxis])].copy()

    if include_global:
        _global = _base.copy()
        _global[col_var] = 'Global'
        if col_order and 'Global' not in col_order:
            col_order.append('Global')
        _base = pd.concat([_base, _global], ignore_index=True)

    internal_row_var = row_var if row_var else '__SingleRow__'
    if row_var is None:
        _base[internal_row_var] = ' '
        if row_order is None: row_order = [' ']

    if row_order: _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
    if col_order: _base = _base[_base[col_var].astype(str).isin(col_order)]

    use_rows = row_order if row_order else sorted(_base[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(_base[col_var].unique())
    n_rows = len(use_rows)
    n_cols = len(use_cols)

    if height is None:
        height = 400 * n_rows

    # --- 3. 布局设置 ---
    row_heights = []
    for _ in range(n_rows):
        row_heights.extend([0.70, 0.30])

    fig = make_subplots(
        rows=n_rows * 2,
        cols=n_cols,
        shared_xaxes=True,
        vertical_spacing=0.0,
        horizontal_spacing=0.04,
        row_heights=row_heights,
        column_titles=use_cols
    )

    global_max_xy = 0
    global_max_count = 0

    # --- 4. 循环绘图 ---
    for r_idx, r_val in enumerate(use_rows):
        for c_idx, c_val in enumerate(use_cols):

            scatter_row = r_idx * 2 + 1
            hist_row = r_idx * 2 + 2
            col_idx = c_idx + 1

            g_panel = _base[(_base[internal_row_var] == r_val) & (_base[col_var] == c_val)]
            if g_panel.empty: continue

            targets = [
                ('ztd_nwm_cor', 'BC', '#1F77B4'),
                ('ztd_nwm', 'Raw', '#FF7F0E')
            ]

            hist_plotted = False

            for mean_col, mean_name, color_code in targets:
                agg = bin_and_aggregate3(g_panel, mean_col, xaxis,
                                         n_bins=n_bins, metric=metric,
                                         bin_method=bin_method, bin_spacing=bin_spacing,
                                         min_bin_points=min_bin_points)
                if agg.empty: continue

                curr_max = max(agg['spread_val'].max(), agg['error_val'].max())
                global_max_xy = max(global_max_xy, curr_max)

                curr_max_cnt = agg['bin_count'].max()
                if not np.isnan(curr_max_cnt):
                    global_max_count = max(global_max_count, curr_max_cnt)

                # A. 散点图 (使用数据真实均值)
                show_legend = (r_idx == 0 and c_idx == 0)

                fig.add_trace(
                    go.Scatter(
                        x=agg['spread_val'], y=agg['error_val'],
                        mode='markers',
                        marker=dict(color=color_code, size=6, opacity=0.9,
                                    line=dict(width=0.5, color='white')),
                        name=mean_name,
                        showlegend=show_legend,
                        legendgroup=mean_name,
                        hovertemplate='Spread: %{x:.2f}<br>Error: %{y:.2f}<extra></extra>'
                    ),
                    row=scatter_row, col=col_idx
                )

                # B. 直方图 (使用理论箱子中心，解决不等宽问题)
                if not hist_plotted:

                    # --- 核心修改：决定直方图的 x 和 width ---
                    if bin_method == 'uniform' and bin_spacing is not None:
                        # 理论模式：完全均匀
                        # x = (bin_id * spacing) + (spacing / 2)
                        # width = spacing
                        bar_x = (agg['bin_id'] * bin_spacing) + (bin_spacing / 2.0)
                        bar_width = [bin_spacing] * len(agg)
                    else:
                        # 原始模式：基于数据分布计算（用于 Quantile 分箱）
                        x_vals = agg['spread_val'].to_numpy()
                        bar_x = x_vals
                        if len(x_vals) > 1:
                            inner_widths = (x_vals[2:] - x_vals[:-2]) / 2
                            w_first = x_vals[1] - x_vals[0]
                            w_last = x_vals[-1] - x_vals[-2]
                            bar_width = np.concatenate(([w_first], inner_widths, [w_last]))
                        else:
                            bar_width = [1]

                    fig.add_trace(
                        go.Bar(
                            x=bar_x,  # 使用计算后的理论X
                            y=agg['bin_count'],
                            width=bar_width,  # 使用计算后的宽度
                            marker=dict(color='#bbbbbb', opacity=0.6, line_width=0),
                            name='Count',
                            showlegend=False,
                            hoverinfo='y',
                        ),
                        row=hist_row, col=col_idx
                    )
                    hist_plotted = True

                # C. 拟合线
                if fit_line == 'linear' and len(agg) > 1:
                    a, b = fit_rmse(agg['spread_val'], agg['error_val'], linear=True)
                    xs = np.linspace(ax_min, curr_max * 1.5, 100)
                    ys = a * xs + b

                    fig.add_trace(
                        go.Scatter(
                            x=xs, y=ys, mode='lines',
                            line=dict(color=color_code, width=2),
                            showlegend=False, hoverinfo='skip'
                        ),
                        row=scatter_row, col=col_idx
                    )

                    y_pos = 0.92 if 'Without' in mean_name else 0.82
                    sign = "+" if b >= 0 else "-"
                    txt = f"{a:.2f}x {sign} {abs(b):.2f}"

                    fig.add_annotation(
                        text=txt,
                        xref="x domain", yref="y domain",
                        x=0.03, y=y_pos, showarrow=False,
                        font=dict(color=color_code),
                        row=scatter_row, col=col_idx
                    )

            # D. 对角线
            fig.add_trace(
                go.Scatter(
                    x=[ax_min, 3000], y=[ax_min, 3000],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False, hoverinfo='skip'
                ),
                row=scatter_row, col=col_idx
            )

            # E. 行标签
            if row_var and c_idx == n_cols - 1:
                fig.add_annotation(
                    text=f"<b>{r_val}</b>",
                    xref="paper", yref="paper",
                    x=1.02, y=0.5,
                    showarrow=False, textangle=90,
                    row=scatter_row, col=col_idx
                )

    # --- 5. 全局样式调整 ---
    AX_MAX = np.ceil(global_max_xy * 1.05)
    hist_log_max = np.log10(global_max_count * 1.5) if global_max_count > 0 else 1

    fig.update_layout(
        template=template,
        width=width, height=height,
        bargap=0,
        margin=dict(l=80, r=50, t=60, b=60),
        # 图例略微上移 (1.10)
        legend=dict(
            orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5
        ),
    )

    # --- 6. 坐标轴精细化控制 ---

    # 6.1 散点图轴 (奇数行)
    for r in range(1, n_rows * 2 + 1, 2):
        fig.update_yaxes(
            range=[ax_min, AX_MAX],
            dtick=scatter_dtick,
            zeroline=False, mirror=True, showline=True,
            linecolor='black', gridcolor='#EAEAEA',
            row=r
        )
        fig.update_xaxes(
            range=[ax_min, AX_MAX],
            dtick=scatter_dtick,
            showticklabels=False,
            zeroline=False, mirror=True, showline=True,
            linecolor='black', gridcolor='#EAEAEA',
            row=r
        )

    # 6.2 直方图轴 (偶数行)
    for r_idx in range(n_rows):
        hist_row = r_idx * 2 + 2
        for c_idx in range(n_cols):
            col_idx = c_idx + 1
            y_title = "Count" if c_idx == 0 else None

            fig.update_yaxes(
                # type="log",
                # # 统一 Y 轴范围
                # range=[np.log10(min_bin_points*0.5), hist_log_max],
                # exponentformat="power",
                # dtick=1,

                # range=[0, global_max_count*1.1],

                showticklabels=True,
                ticks="inside",
                # ticklen=4,


                title_text=y_title,
                showgrid=True,
                gridcolor='#F0F0F0',
                zeroline=False,
                mirror=True, showline=True, linecolor='black',
                row=hist_row, col=col_idx
            )

            fig.update_xaxes(
                range=[ax_min, AX_MAX],
                dtick=scatter_dtick,
                showticklabels=True,
                zeroline=False, mirror=False, showline=True,
                linecolor='black', gridcolor='#EAEAEA',
                row=hist_row, col=col_idx
            )

    # --- 7. 轴标题 ---
    for r_idx in range(n_rows):
        fig.update_yaxes(
            title_text=col_y_label,
            row=r_idx * 2 + 1, col=1
        )

    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text=col_x_label,
            row=n_rows * 2, col=c
        )

    return fig


def plot_spread_skill_ex2(df_all,
                         col_var,
                         row_var=None,
                         template='simple_white',
                         xaxis='ztd_nwm_sigma',
                         row_order=None, col_order=None,
                         include_global=True,
                         fit_line='linear',
                         n_bins=25,
                         metric='rmse',
                         bin_method='quantile',
                         bin_spacing=None,
                         min_bin_points=0,
                         ax_min=0.0,
                         width=1200, height=None,
                         # New toggles
                         show_hist=True,
                         show_bc_obs=True):
    """
    Revised version of plot_spread_skill_final_v8 (Style-Matched).
    Changes:
      1. Labels: 'RMS Error' -> 'Observed RMSE', 'RMS Spread' -> 'Predicted RMSE' (or MSE).
      2. 'BC+Consider Obs Error' renamed to 'BC + Obs'.
      3. Histogram Y-axis title: 'Bin Size'. Range: [0, local_max * 1.05].
      4. Toggles: show_hist, show_bc_obs.
      5. Histogram Visibility: Added to legend to clarify 'Raw/BC' (Standard) vs 'BC+Obs'.
    """
    # --- 1. Basic Config ---
    is_mse = (metric == 'mse')
    col_x_label = "Predicted MSE (mm²)" if is_mse else "Predicted RMSE (mm)"
    col_y_label = "Observed MSE (mm²)" if is_mse else "Observed RMSE (mm)"
    hist_y_label = "Bin size"

    # Calculate dtick (if bin_spacing exists and > 0)
    scatter_dtick = int(bin_spacing * 5) if (bin_spacing is not None and bin_spacing > 0) else None

    # --- 2. Data Preparation ---
    _base = df_all.copy()
    if not np.issubdtype(_base['time'].dtype, np.datetime64):
        _base['time'] = pd.to_datetime(_base['time'])
    
    # Filter valid base on primary xaxis
    _base = _base[np.isfinite(_base[xaxis])].copy()

    # --- Pre-calculate Adjusted Combined Sigma for 'BC + Obs' ---
    N_target = 50 
    inv_factor = N_target / (N_target + 1)
    
    if 'ztd_gnss_sigma' not in _base.columns:
        gnss_sig_sq = 0.0
    else:
        gnss_sig_sq = _base['ztd_gnss_sigma'].fillna(0) ** 2

    # Constructed combined spread
    _base['ztd_combined_sigma_pre'] = np.sqrt(
        _base[xaxis]**2 + inv_factor * gnss_sig_sq
    )

    if include_global:
        _global = _base.copy()
        _global[col_var] = 'Global'
        if col_order and 'Global' not in col_order:
            col_order.append('Global')
        _base = pd.concat([_base, _global], ignore_index=True)

    internal_row_var = row_var if row_var else '__SingleRow__'
    if row_var is None:
        _base[internal_row_var] = ' '
        if row_order is None: row_order = [' ']

    if row_order: _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
    if col_order: _base = _base[_base[col_var].astype(str).isin(col_order)]

    use_rows = row_order if row_order else sorted(_base[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(_base[col_var].unique())
    n_rows = len(use_rows)
    n_cols = len(use_cols)

    # Note: If showing histograms, we need 2x rows. If not, 1x.
    layout_rows = n_rows * 2 if show_hist else n_rows

    if height is None:
        height = 400 * n_rows 

    # --- 3. Layout ---
    row_heights = []
    for _ in range(n_rows):
        if show_hist:
            row_heights.extend([0.70, 0.30])
        else:
            row_heights.append(1.0)

    fig = make_subplots(
        rows=layout_rows,
        cols=n_cols,
        shared_xaxes=True if show_hist else False, # Only share if hist is below
        vertical_spacing=0.0 if show_hist else 0.05,
        horizontal_spacing=0.04,
        row_heights=row_heights,
        column_titles=use_cols
    )

    global_max_xy = 0
    subplot_max_counts = {}

    # --- 4. Loop ---
    original_xaxis = xaxis
    for r_idx, r_val in enumerate(use_rows):
        for c_idx, c_val in enumerate(use_cols):

            if show_hist:
                scatter_row = r_idx * 2 + 1
                hist_row = r_idx * 2 + 2
            else:
                scatter_row = r_idx + 1
                hist_row = None

            col_idx = c_idx + 1

            g_panel = _base[(_base[internal_row_var] == r_val) & (_base[col_var] == c_val)]
            if g_panel.empty: continue

            # Targets configuration
            targets = [
                ('ztd_nwm',      'Raw',       '#FF7F0E',  original_xaxis),
                ('ztd_nwm_cor',  'BC',        '#1F77B4',  original_xaxis)
            ]
            if show_bc_obs:
                targets.append(('ztd_nwm_cor',  'BC + Obs',  '#2CA02C',  'ztd_combined_sigma_pre'))

            hist_plotted_standard = False
            hist_plotted_combined = False
            local_max_cnt = 0

            for mean_col, mean_name, color_code, x_col in targets:
                agg = bin_and_aggregate3(g_panel, mean_col, x_col,
                                         n_bins=n_bins, metric=metric,
                                         bin_method=bin_method, bin_spacing=bin_spacing,
                                         min_bin_points=min_bin_points)
                if agg.empty: continue

                curr_max = max(agg['spread_val'].max(), agg['error_val'].max())
                global_max_xy = max(global_max_xy, curr_max)

                # Track max count for histogram scaling
                if show_hist:
                    curr_max_cnt = agg['bin_count'].max()
                    if not np.isnan(curr_max_cnt):
                        local_max_cnt = max(local_max_cnt, curr_max_cnt)*1.05

                # A. Scatter
                show_legend = (r_idx == 0 and c_idx == 0)

                fig.add_trace(
                    go.Scatter(
                        x=agg['spread_val'], y=agg['error_val'],
                        mode='lines+markers',
                        marker=dict(color=color_code, size=6, opacity=0.9,
                                    line=dict(width=0.5, color='white')),
                        line=dict(color=color_code, width=1),
                        name=mean_name,
                        showlegend=show_legend,
                        legendgroup=mean_name,
                        hovertemplate='Pred RMSE: %{x:.2f}<br>Obs RMSE: %{y:.2f}<extra></extra>'
                    ),
                    row=scatter_row, col=col_idx
                )

                # B. Histogram
                if show_hist:
                    is_standard_dist = (x_col == original_xaxis)
                    should_plot = False
                    
                    # Style Config
                    # User requested: Blue for Standard (BC), Green for Combined (BC+Obs).
                    # No borders, no legend.
                    if is_standard_dist:
                        # Standard (Raw/BC share same spread) -> Use BC color (Blue)
                        bar_color = '#1F77B4' 
                        bar_opacity = 0.4
                        bar_name = 'Bin size (BC)' # Hidden from legend
                        
                        if not hist_plotted_standard:
                            should_plot = True
                            hist_plotted_standard = True
                            
                    else:
                        # Combined (Obs) -> Use BC+Obs color (Green)
                        bar_color = '#2CA02C'
                        bar_opacity = 0.4
                        bar_name = 'Bin ize (BC+Obs)' # Hidden from legend
                        
                        if not hist_plotted_combined:
                            should_plot = True
                            hist_plotted_combined = True
                    
                    if should_plot:
                        if bin_method == 'uniform' and bin_spacing is not None:
                            bar_x = (agg['bin_id'] * bin_spacing) + (bin_spacing / 2.0)
                            bar_width = [bin_spacing] * len(agg)
                        else:
                            x_vals = agg['spread_val'].to_numpy()
                            bar_x = x_vals
                            if len(x_vals) > 1:
                                inner_widths = (x_vals[2:] - x_vals[:-2]) / 2
                                w_first = x_vals[1] - x_vals[0]
                                w_last = x_vals[-1] - x_vals[-2]
                                bar_width = np.concatenate(([w_first], inner_widths, [w_last]))
                            else:
                                bar_width = [1]

                        fig.add_trace(
                            go.Bar(
                                x=bar_x,
                                y=agg['bin_count'],
                                width=bar_width,
                                marker=dict(
                                    color=bar_color, 
                                    opacity=bar_opacity, 
                                    line=dict(width=0) # No border
                                ),
                                name=bar_name,
                                showlegend=False, # No legend for hist
                                hoverinfo='y',
                            ),
                            row=hist_row, col=col_idx
                        )

                # C. Fit Line
                if fit_line == 'linear' and len(agg) > 1:
                    a, b = fit_rmse(agg['spread_val'], agg['error_val'], linear=True)
                    xs = np.linspace(ax_min, curr_max * 1.5, 100)
                    ys = a * xs + b

                    fig.add_trace(
                        go.Scatter(
                            x=xs, y=ys, mode='lines',
                            line=dict(color=color_code, width=2, dash='dash'),
                            showlegend=False, hoverinfo='skip'
                        ),
                        row=scatter_row, col=col_idx
                    )

                    y_pos = 0.92
                    if 'Raw' in mean_name: y_pos = 0.82
                    if 'Obs' in mean_name: y_pos = 0.72 

                    sign = "+" if b >= 0 else "-"
                    txt = f"{a:.2f}x {sign} {abs(b):.2f}"

                    fig.add_annotation(
                        text=txt,
                        xref="x domain", yref="y domain",
                        x=0.03, y=y_pos, showarrow=False,
                        font=dict(color=color_code),
                        row=scatter_row, col=col_idx
                    )
            
            # Store local max per subplot for later scaling
            if show_hist:
                subplot_max_counts[(r_idx, c_idx)] = local_max_cnt

            # D. Diagonal
            fig.add_trace(
                go.Scatter(
                    x=[ax_min, 3000], y=[ax_min, 3000],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False, hoverinfo='skip'
                ),
                row=scatter_row, col=col_idx
            )

            # E. Row Label (Restored from v8)
            if row_var and c_idx == n_cols - 1:
                fig.add_annotation(
                    text=f"<b>{r_val}</b>",
                    xref="paper", yref="paper",
                    x=1.02, y=0.5,
                    showarrow=False, textangle=90,
                    row=scatter_row, col=col_idx
                )

    # --- 5. Global Range Setting ---
    AX_MAX = np.ceil(global_max_xy * 1.05) if global_max_xy > 0 else 1.0
    
    fig.update_layout(
        template=template,
        width=width, height=height,
        bargap=0,
        margin=dict(l=80, r=50, t=60, b=60), # MATCH V8 MARGINS
        legend=dict(
            orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5
        ),
        barmode='overlay' # Crucial for overlapping histograms
    )

    # --- 6. Axis Styling ---
    for r_idx in range(n_rows):
        if show_hist:
            scatter_row = r_idx * 2 + 1
            hist_row = r_idx * 2 + 2
        else:
            scatter_row = r_idx + 1
            hist_row = None
        
        for c_idx in range(n_cols):
            col_idx = c_idx + 1
            
            # Scatter Axes
            fig.update_yaxes(
                range=[ax_min, AX_MAX],
                dtick=scatter_dtick,
                zeroline=False, mirror=True, showline=True,
                linecolor='black', gridcolor='#EAEAEA',
                row=scatter_row, col=col_idx
            )
            # Remove X ticks/labels for scatter if hist is present
            show_scatter_x = True if not show_hist else False
            fig.update_xaxes(
                range=[ax_min, AX_MAX],
                dtick=scatter_dtick,
                showticklabels=show_scatter_x,
                zeroline=False, mirror=True, showline=True,
                linecolor='black', gridcolor='#EAEAEA',
                row=scatter_row, col=col_idx
            )

            # Histogram Axes
            if show_hist and hist_row:
                y_title_text = hist_y_label if c_idx == 0 else None
                local_max = subplot_max_counts.get((r_idx, c_idx), 10)
                # 1.05 multiplier
                hist_y_limit = local_max * 1.05 if local_max > 0 else 10

                fig.update_yaxes(
                    range=[0, hist_y_limit],
                    showticklabels=True,
                    title_text=y_title_text,
                    showgrid=True,
                    gridcolor='#F0F0F0',
                    zeroline=False,
                    mirror=True, showline=True, linecolor='black',
                    row=hist_row, col=col_idx
                )
                fig.update_xaxes(
                    range=[ax_min, AX_MAX],
                    dtick=scatter_dtick,
                    showticklabels=True,
                    zeroline=False, mirror=False, showline=True,
                    linecolor='black', gridcolor='#EAEAEA',
                    row=hist_row, col=col_idx
                )

    # --- 7. Axis Titles ---
    # Global Y title
    for r_idx in range(n_rows):
        row_idx = (r_idx * 2 + 1) if show_hist else (r_idx + 1)
        fig.update_yaxes(
            title_text=col_y_label,
            row=row_idx, col=1
        )

    # Global X title
    last_row_offset = (n_rows * 2) if show_hist else n_rows
    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text=col_x_label,
            row=last_row_offset, col=c
        )

    return fig


def plot_qq_normresid_combined(
    data_list: list[tuple[pd.DataFrame, int]],
    col_var: str,
    row_var: str | None = None,
    template="simple_white",
    xaxis: str = "ztd_nwm_sigma",
    row_order=None,
    col_order=None,
    include_global: bool = True,
    step: float = 0.25,
    L: float = 3,
    width: int = 1200,
    height: int = 1000,
):
    """
    绘制混合 N 的 QQ 图 (Combined View):
    Input: data_list = [(df_10, 10), (df_50, 50)] (Order does not matter)
    
    Logic:
      1. Identify Max N as 'Ref' (e.g. 50), Min N as 'Target' (e.g. 10).
      2. Ref (N=50): Plot 'Raw' and 'BC'.
      3. Target (N=10): Plot 'BC (Small)' and 'BC + Obs'.
    """
    
    # --------------------
    # 0. 解析 Input
    # --------------------
    if len(data_list) < 1:
        print("data_list is empty.")
        return None
        
    # 按 N 排序: small -> large
    sorted_data = sorted(data_list, key=lambda x: x[1])
    
    # Target (Small N)
    df_tgt, N_tgt = sorted_data[0]
    # Ref (Large N) -> 如果只有一个，就都用它
    if len(sorted_data) > 1:
        df_ref, N_ref = sorted_data[-1]
    else:
        df_ref, N_ref = df_tgt, N_tgt

    # --------------------
    # A. 统一的数据预处理函数
    # --------------------
    def _prepare_df(df_in, current_N, is_ref_set=False):
        _base = df_in.copy()
        if not np.issubdtype(_base["time"].dtype, np.datetime64):
            _base["time"] = pd.to_datetime(_base["time"])
        _base = _base[np.isfinite(_base[xaxis])].copy()

        # Global Logic
        if include_global:
            _global = _base.copy()
            _global[col_var] = "Global"
            _base = pd.concat([_base, _global], ignore_index=True)

        # Row Var Logic
        internal_row_var = row_var
        if row_var is None:
            internal_row_var = "__SingleRow__"
            _base[internal_row_var] = " "
        
        # BC + Obs Logic (Needed for Ref set now, based on user request)
        spread_col_obs = None
        # Always calculate it if possible? Or strictly follow request.
        # User wants BC+Obs(N=50) -> Ref set needs it.
        # Target set (BC N=10) doesn't strictly need it, but good to have helper.
        
        inv_factor_obs = current_N / (current_N + 1)
        if 'ztd_gnss_sigma' not in _base.columns:
            gnss_sig_sq = 0.0
        else:
            gnss_sig_sq = _base['ztd_gnss_sigma'].fillna(0) ** 2
        
        spread_col_obs = 'ztd_combined_sigma_obs'
        _base[spread_col_obs] = np.sqrt(
            _base[xaxis]**2 + inv_factor_obs * gnss_sig_sq
        )

        # Dropna
        need = ["ztd_gnss", "ztd_nwm", "ztd_nwm_cor", xaxis, internal_row_var, col_var, spread_col_obs]
        _base = _base.dropna(subset=need)

        # Order Filter
        if row_order:
            _base = _base[_base[internal_row_var].astype(str).isin(row_order)]
        if col_order:
            _base = _base[_base[col_var].astype(str).isin(col_order)]
            
        return _base, internal_row_var, spread_col_obs

    # Prepare Ref and Target DFs
    df_ref_clean, ref_row_var, ref_spread_obs = _prepare_df(df_ref, N_ref, is_ref_set=True)
    df_tgt_clean, tgt_row_var, tgt_spread_obs = _prepare_df(df_tgt, N_tgt, is_ref_set=False)
    
    # 确保 row variable name 一致
    internal_row_var = ref_row_var 

    # --- Uniform Sampling on X-axis ---
    q_norm_grid = np.arange(-L, L + 0.001, step)
    p_grid = 0.5 * (1 + erf(q_norm_grid / np.sqrt(2)))
    p_grid = np.clip(p_grid, 1e-6, 1 - 1e-6)
    
    qq_rows_all = []

    # --------------------
    # B. Process Ref (Max N) -> Raw, BC, BC+Obs
    # --------------------
    # Configs: (MeanCol, Label, SpreadCol)
    ref_configs = [
        ("ztd_nwm",     "Raw",       xaxis), 
        ("ztd_nwm_cor", "BC",        xaxis),
        ("ztd_nwm_cor", "BC + Obs",  ref_spread_obs),
    ]
    alpha_ref = np.sqrt((N_ref + 1.0) / N_ref)

    for (r_val, c_val), g_rc in df_ref_clean.groupby([internal_row_var, col_var], observed=True):
        for mean_col, base_label, spread_col in ref_configs:
            g = g_rc.copy()
            g = g[g[spread_col] > 0]
            if len(g) < 5: continue
            
            g["e"] = g[mean_col] - g["ztd_gnss"]
            g["z"] = g["e"] / (alpha_ref * g[spread_col])
            g = g[np.isfinite(g["z"])].copy()
            if len(g) < 5: continue
            
            z_q = np.quantile(g["z"].to_numpy(), p_grid)
            
            # Generate labels with N notation
            if base_label == "Raw":
                mean_label = f"Raw (N={N_ref})"
            elif base_label == "BC":
                mean_label = f"BC (N={N_ref})"
            else: # BC + Obs
                mean_label = f"BC + Obs (N={N_ref})"
            
            rows = pd.DataFrame({
                internal_row_var: r_val,
                col_var: c_val,
                "MeanLabel": mean_label, 
                "p": p_grid,
                "z_q": z_q,
                "q_norm": q_norm_grid
            })
            qq_rows_all.append(rows)

    # --------------------
    # C. Process Target (Min N) -> BC
    # --------------------
    # Configs: (MeanCol, Label, SpreadCol)
    # Only BC
    tgt_configs = [
        ("ztd_nwm_cor", f"BC (N={N_tgt})", xaxis),
    ]
    alpha_tgt = np.sqrt((N_tgt + 1.0) / N_tgt)

    for (r_val, c_val), g_rc in df_tgt_clean.groupby([internal_row_var, col_var], observed=True):
        for mean_col, label, spread_col in tgt_configs:
            g = g_rc.copy()
            g = g[g[spread_col] > 0]
            if len(g) < 5: continue
            
            g["e"] = g[mean_col] - g["ztd_gnss"]
            g["z"] = g["e"] / (alpha_tgt * g[spread_col])
            g = g[np.isfinite(g["z"])].copy()
            if len(g) < 5: continue
            
            z_q = np.quantile(g["z"].to_numpy(), p_grid)
            
            rows = pd.DataFrame({
                internal_row_var: r_val,
                col_var: c_val,
                "MeanLabel": label,
                "p": p_grid,
                "z_q": z_q,
                "q_norm": q_norm_grid
            })
            qq_rows_all.append(rows)

    if not qq_rows_all:
        print("No valid Q-Q data.")
        return None
        
    qq_df = pd.concat(qq_rows_all, ignore_index=True)

    # --------------------
    # D. Add Diagonal and Plot
    # --------------------
    use_rows = row_order if row_order else sorted(qq_df[internal_row_var].unique())
    use_cols = col_order if col_order else sorted(qq_df[col_var].unique())
    
    diag_rows = []
    for r in use_rows:
        for c in use_cols:
            diag_rows.append({internal_row_var: r, col_var: c, "MeanLabel": "y = x", "q_norm": -L, "z_q": -L})
            diag_rows.append({internal_row_var: r, col_var: c, "MeanLabel": "y = x", "q_norm": L,  "z_q": L})
            
    qq_plot_df = pd.concat([qq_df, pd.DataFrame(diag_rows)], ignore_index=True)
    qq_plot_df["Expected"] = qq_plot_df["q_norm"]
    qq_plot_df["Observed"] = qq_plot_df["z_q"]

    # Color Map
    raw_lbl = f"Raw (N={N_ref})"
    bc_lbl = f"BC (N={N_ref})"
    bc_obs_lbl = f"BC + Obs (N={N_ref})"
    bc_small_lbl = f"BC (N={N_tgt})"
    
    color_map = {
        raw_lbl: px.colors.diverging.Portland[4],      
        bc_small_lbl: px.colors.diverging.Portland[3],   
        bc_lbl:  px.colors.diverging.Portland[2],       
        bc_obs_lbl: px.colors.diverging.Portland[1],    
        "y = x": COL_DIAG
    }
    
    actual_facet_row = internal_row_var if row_var is not None else None
    if row_var is None and height == 1000: height = 450

    fig = px.scatter(
        qq_plot_df, x="q_norm", y="z_q",
        color="MeanLabel", color_discrete_map=color_map,
        facet_col=col_var, facet_row=actual_facet_row,
        facet_col_spacing=0.01, facet_row_spacing=0.02,
        category_orders={
            col_var: use_cols,
            internal_row_var: use_rows,
            "MeanLabel": [raw_lbl, bc_small_lbl, bc_lbl, bc_obs_lbl, "y = x"]
        },
        hover_data={"Expected": ':.2f', "Observed": ':.2f'},
        template=template,
    )
    
    # Styles
    for tr in fig.data:
        if tr.name == "y = x":
            tr.mode = "lines"
            tr.line.update(color=COL_DIAG, width=1)
        else:
            tr.mode = "lines+markers"
            tr.marker.update(size=4)
            tr.line.update(width=1)

    # Layout
    fig.update_layout(
        title=None, width=width, height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, title_text=""),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    for k in fig.layout:
        if k.startswith("xaxis"):
            fig.layout[k].update(range=[-L, L], zeroline=False, mirror=True, showline=True, dtick=1)
        if k.startswith("yaxis"):
            x_token = k.replace("yaxis", "x")
            fig.layout[k].update(range=[-L, L], zeroline=False, mirror=True, showline=True, dtick=1, scaleanchor=x_token, scaleratio=1.0)

    # Strip Facet Titles
    def _strip_facet(t: str) -> str:
        t = t.replace(f"{col_var}=", "")
        if row_var: t = t.replace(f"{row_var}=", "")
        if internal_row_var == "__SingleRow__" and ("__SingleRow__=" in t or t == " "): return ""
        return t

    fig.for_each_annotation(lambda a: a.update(text=_strip_facet(a.text)) if isinstance(a.text, str) else None)
    
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    
    fig.add_annotation(text="Theoretical Quantiles", x=0.5, y=-0.2, xref="paper", yref="paper", showarrow=False, font_size=16)
    fig.add_annotation(text="Sample Quantiles", x=-0.06, y=0.5, xref="paper", yref="paper", showarrow=False, textangle=-90, font_size=16)

    return fig


from scipy.stats import norm

def plot_spread_skill_analysis_z(df_all,
                                 bin_by_col,  # str 或 list[str]
                                 sigma_col='ztd_nwm_sigma',
                                 template='simple_white',
                                 n_bins=20,
                                 samples_per_bin=None,
                                 N=50,
                                 width=None, height=500):
    """
    对每个分箱，画出对于 z (z=error/spread) 理论上 z 应该为 +-0/1/2/3 的对应的分位数对应的实际 z 值为多少。
    只展示 BC (ztd_nwm_cor) 的结果。

    理论分位点:
    Theoretical Z in [-3, -2, -1, 0, 1, 2, 3]
    对应概率 P = Phi(Z)
    Observed Z = Quartile(Data_Z, P)

    Lines: 7条线 (-3sigma to +3sigma)
    """
    # 局部引用 make_subplots 以防万一
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    import plotly.express as px

    # --- 1. 参数校验与初始化 ---
    if isinstance(bin_by_col, str):
        bin_vars = [bin_by_col]
    else:
        bin_vars = bin_by_col

    num_vars = len(bin_vars)
    if width is None:
        width = 800 if num_vars == 1 else 400 * num_vars

    fig = make_subplots(
        rows=1, cols=num_vars,
        subplot_titles=None,
        shared_yaxes=True,
        horizontal_spacing=0.015
    )

    # 有限样本修正系数 (Spread 调整)
    # z = e / (alpha * sigma)
    alpha = np.sqrt((N + 1.0) / N)

    # 理论 Z 值及其对应概率
    target_sigmas = np.array([-3, -2, -1, 0, 1, 2, 3])
    target_probs = norm.cdf(target_sigmas)
    
    # 颜色映射: Use plotly.colors.diverging.Portland
    # Indices: 0(3sigma), 1(2sigma), 2(1sigma), 4(0sigma)
    p_colors = px.colors.diverging.Portland
    
    # 0: Index 4
    # +/-1: Index 2
    # +/-2: Index 1
    # +/-3: Index 0
    sigma_colors = {
        0: p_colors[0],
        1: p_colors[1], -1: p_colors[1],
        2: p_colors[2], -2: p_colors[2],
        3: p_colors[4], -3: p_colors[4]
    }

    # --- 2. 循环处理每个变量 ---
    for i, var_name in enumerate(bin_vars):
        col_idx = i + 1

        # A. 数据清洗
        # 只需 BC: ztd_nwm_cor
        required_cols = ['ztd_gnss', 'ztd_nwm_cor', sigma_col, var_name]
        try:
            _base = df_all.dropna(subset=required_cols).copy()
        except KeyError:
            print(f"Skipping {var_name}: Column not found.")
            continue

        if len(_base) < 10: continue

        # B. 确定箱数
        total_valid = len(_base)
        if samples_per_bin is not None and samples_per_bin > 0:
            n_bins_use = max(1, int(np.ceil(total_valid / samples_per_bin)))
        else:
            n_bins_use = n_bins

        # C. 执行分箱 (使用 pd.qcut)
        sort_key = _base[var_name]
        try:
            # 尝试 qcut
            q = pd.qcut(sort_key, q=n_bins_use, labels=False, duplicates='drop')
        except ValueError:
            uniq = sort_key.nunique()
            if uniq < 2: continue
            q = pd.qcut(sort_key, q=min(int(uniq), n_bins_use), labels=False, duplicates='drop')
        
        _base['bin_id'] = q

        # D. 预计算 Z-score
        # z = (BC - GNSS) / (sigma * alpha)
        # 注意: sigma 必须 > 0
        valid_sigma_mask = _base[sigma_col] > 1e-6
        _base = _base[valid_sigma_mask].copy()
        
        err = _base['ztd_nwm_cor'] - _base['ztd_gnss']
        spread_adj = _base[sigma_col] * alpha
        _base['z_score'] = err / spread_adj

        # E. 聚合计算
        plot_data = []
        for bin_id, gbin in _base.groupby('bin_id'):
            if len(gbin) < 5: continue

            x_val = gbin[var_name].median()
            z_vals = gbin['z_score'].to_numpy()
            
            # 计算分位数
            obs_quantiles = np.quantile(z_vals, target_probs)
            
            for k_sigma, obs_q in zip(target_sigmas, obs_quantiles):
                plot_data.append({
                    "x": x_val,
                    "y": obs_q,
                    "sigma": k_sigma,
                    "count": len(gbin)
                })

        if not plot_data: continue
        df_plot = pd.DataFrame(plot_data).sort_values('x')

        # F. 绘图
        # 分别画7条线
        for k_sigma in sorted(target_sigmas):
            sub_df = df_plot[df_plot['sigma'] == k_sigma]
            if sub_df.empty: continue

            label = f"{k_sigma}σ" if k_sigma != 0 else "0σ (Median)"

            show_legend = (i == 0) # 只在第一幅图显示图例

            trace = go.Scatter(
                x=sub_df['x'], y=sub_df['y'],
                mode='lines+markers',
                name=label,
                line=dict(color=sigma_colors[k_sigma], width=2),
                marker=dict(size=4, color=sigma_colors[k_sigma]),
                showlegend=show_legend,
                legendgroup=str(k_sigma),
                hovertemplate=f"<b>Theoretical {k_sigma}σ</b><br>bin_center: %{{x:.2f}}<br>Observed Z: %{{y:.2f}}<br>N: %{{text}}<extra></extra>",
                text=sub_df['count']
            )
            fig.add_trace(trace, row=1, col=col_idx)

        # X轴标题
        fig.update_xaxes(title_text=var_name, row=1, col=col_idx)

    # --- 3. 全局样式 ---
    fig.update_layout(
        template=template,
        width=width, height=height,
        margin=dict(l=60, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title=None
    )
    
    # Y轴标题
    fig.add_annotation(
        text="Observed Z-Score",
        x=-0.06 if num_vars > 1 else -0.10,
        y=0.5, xref="paper", yref="paper", textangle=-90,
        showarrow=False, font_size=16
    )
    
    # 轴样式
    fig.update_yaxes(zeroline=False, mirror=True, showline=True, showgrid=True, dtick=1)
    fig.update_xaxes(zeroline=False, mirror=True, showline=True, showgrid=True)

    return fig

