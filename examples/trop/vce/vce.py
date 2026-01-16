import numpy as np
import pandas as pd

def estimate_vce_generalized(
    df,
    group_col,
    fixed_beta=None,                 # None: 解3分量；数值: 固定beta求2分量
    res_col='res_cor',               # 线性化后的“常数项/残差”(y)，不是平方
    nwm_spread_col='ztd_nwm_sigma',  # Q1 的尺度列（会平方）
    gnss_formal_col='ztd_gnss_sigma',# Q2 的尺度列（会平方）
    member_count=50,                 # Q1 的缩放因子用到
    max_iter=30,
    tol=1e-6,
    # === 关键新增：为了与 LS-VCE 一致，需要能形成 P_perp ===
    design_cols=None,                # 可选，给出设计矩阵 A 的列名列表；若 None 则无法重算 e
    fit_intercept=False,             # 若没有 design_cols，可选是否仅消常数（A=1列）
    mode='auto',                     # 'bique' | 'irls' | 'auto'（有 A/截距则 bique，否则 irls）
    enforce_nonneg=True,
    verbose=False,
):
    """
    与上文一致的 LS-VCE（BIQUE, κ=0, 对角情形）实现。
    - 若提供 design_cols 或 fit_intercept=True，就按 BIQUE-对角的公式：
        * R = diag(1 / v_i), P_perp = I - A (A^T R A)^-1 A^T R
        * e = P_perp y
        * s_i = (e_i * r_i)^2
        * d_k = q_k ⊙ r
        * r_k = 0.5 * sum_i s_i * q_{k,i}            (假定 Q0=0)
        * N_kl = 0.5 * d_k^T (P_perp ∘ P_perp^T) d_l
        * 解 N σ = r 得 σ=(α^2, β^2, γ^2)
      这与 TAS 2008/AS 2007 在“独立对角”的化简完全一致。
    - 否则退化到近似 IRLS（把 y^2 当观测），给出可用但近似的结果。
    """

    def _make_A(sub_df):
        # 构造设计矩阵 A（可选）
        cols = []
        if design_cols is not None and len(design_cols) > 0:
            cols = [np.asarray(sub_df[c].values, float) for c in design_cols]
        A = None
        if cols:
            A = np.vstack(cols).T  # shape (m, p)
        if fit_intercept:
            one = np.ones((len(sub_df), 1), float)
            A = one if A is None else np.hstack([one, A])
        return A  # None 表示不做投影

    def _proj_perp(A, r):  # r 是权向量对角的对角元 (1/v_i)
        # 返回 P_perp（按 W=diag(r) 的斜投影）
        m = len(r)
        if A is None or A.size == 0:
            return np.eye(m)
        W_half = np.sqrt(r)[:, None]     # 便于数值稳定
        Aw = W_half * A                  # = W^{1/2} A
        # P_orth on weighted space: I - Aw (Aw^T Aw)^-1 Aw^T
        try:
            N = Aw.T @ Aw
            N_inv = np.linalg.pinv(N)
        except np.linalg.LinAlgError:
            N_inv = np.linalg.pinv(Aw.T @ Aw)
        P_orth = np.eye(m) - (Aw @ (N_inv @ Aw.T))
        # 回到原空间的 P_perp = W^{-1/2} P_orth W^{1/2}
        # 但在二次型里我们只需要 P_perp 本身；下面给出等价表达：
        # 用 R=diag(r)，P = A (A^T R A)^{-1} A^T R，P_perp = I - P
        # 直接构造（数值上与上面等价）：
        R = np.diag(r)
        try:
            N2 = A.T @ (R @ A)
            N2_inv = np.linalg.pinv(N2)
        except np.linalg.LinAlgError:
            N2_inv = np.linalg.pinv(A.T @ (R @ A))
        P = A @ (N2_inv @ (A.T @ R))
        return np.eye(m) - P

    def _solve_bique_diag(y, q1, q2, q3, beta_fixed=None):
        # y: (m,) 线性化后的常数项/残差；qk: (m,)，都是非负
        # 变量：s1=α^2, s2=β^2, s3=γ^2
        s1, s2, s3 = 1.0, 1.0, max(np.var(y), 1e-3)
        if beta_fixed is not None:
            s2 = float(beta_fixed) ** 2

        A = _make_A(pd.DataFrame(index=range(len(y))))  # 按组构造一次 A

        converged = False
        for it in range(max_iter):
            s1_old, s2_old, s3_old = s1, s2, s3

            v = s1 * q1 + s2 * q2 + s3 * q3
            v = np.maximum(v, 1e-12)
            r = 1.0 / v                         # R 的对角元
            P_perp = _proj_perp(A, r)           # 用当前权形成 P_perp
            e = P_perp @ y                      # 残差（投影后）
            s_vec = (e * r)**2                  # s_i = (e_i r_i)^2

            # Hadamard 核 M = P_perp ∘ P_perp^T
            M = P_perp * P_perp.T               # 按元素积

            d1, d2, d3 = q1 * r, q2 * r, q3 * r

            # 右端向量（假定 Q0=0）：r_k = 0.5 * Σ s_i q_{k,i}
            rhs = np.array([
                0.5 * float(s_vec @ q1),
                0.5 * float(s_vec @ q2),
                0.5 * float(s_vec @ q3),
            ])

            # 法矩阵：N_kl = 0.5 * d_k^T M d_l
            def quad(dk, dl):
                return 0.5 * float(dk @ (M @ dl))
            N = np.array([
                [quad(d1, d1), quad(d1, d2), quad(d1, d3)],
                [quad(d2, d1), quad(d2, d2), quad(d2, d3)],
                [quad(d3, d1), quad(d3, d2), quad(d3, d3)],
            ])

            # 固定 beta 的 2×2 子系统
            if beta_fixed is not None:
                # 取 (1,3) 分量
                N_sub = N[np.ix_([0, 2], [0, 2])]
                rhs_sub = rhs[[0, 2]]
                try:
                    s1s3 = np.linalg.solve(N_sub, rhs_sub)
                except np.linalg.LinAlgError:
                    s1s3 = np.linalg.lstsq(N_sub, rhs_sub, rcond=None)[0]
                s1, s3 = map(float, s1s3)
            else:
                try:
                    s_new = np.linalg.solve(N, rhs)
                except np.linalg.LinAlgError:
                    s_new = np.linalg.lstsq(N, rhs, rcond=None)[0]
                s1, s2, s3 = map(float, s_new)

            if enforce_nonneg:
                s1 = max(s1, 1e-12)
                if beta_fixed is None:
                    s2 = max(s2, 1e-12)
                s3 = max(s3, 0.0)

            if abs(s1 - s1_old) + abs(s2 - s2_old) + abs(s3 - s3_old) < tol:
                converged = True
                break

        return s1, s2, s3, converged

    def _solve_irls(y, q1, q2, q3, beta_fixed=None):
        # 近似 IRLS：把 y^2 当观测，权重 ~ 1/(2 v^2)
        s1, s2, s3 = 1.0, 1.0, max(np.var(y), 1e-3)
        if beta_fixed is not None:
            s2 = float(beta_fixed) ** 2

        y_sq = y**2
        converged = False
        for _ in range(max_iter):
            s1o, s2o, s3o = s1, s2, s3
            v = s1 * q1 + s2 * q2 + s3 * q3
            v = np.maximum(v, 1e-12)
            w = 1.0 / (2.0 * v**2)  # 1/Var(y^2)=1/(2 v^2)

            # 正规方程
            Q = np.vstack([q1, q2, q3]).T
            if beta_fixed is not None:
                # 只解 α^2 和 γ^2
                Q = np.vstack([q1, q3]).T
            WQ = w[:, None] * Q
            N = Q.T @ WQ
            rhs = Q.T @ (w * y_sq)
            try:
                s = np.linalg.solve(N, rhs)
            except np.linalg.LinAlgError:
                s = np.linalg.lstsq(N, rhs, rcond=None)[0]

            if beta_fixed is not None:
                s1, s3 = map(float, s)
            else:
                s1, s2, s3 = map(float, s)

            if enforce_nonneg:
                s1 = max(s1, 1e-12)
                if beta_fixed is None:
                    s2 = max(s2, 1e-12)
                s3 = max(s3, 0.0)

            if abs(s1 - s1o) + abs(s2 - s2o) + abs(s3 - s3o) < tol:
                converged = True
                break

        return s1, s2, s3, converged

    # -------- 分组应用 --------
    if verbose:
        print(f"[VCE] group={group_col} | fixed_beta={fixed_beta} | mode={mode}")

    scale = (member_count + 1) / member_count

    def _apply(sub):
        sub = sub[[res_col, nwm_spread_col, gnss_formal_col] + (design_cols or [])].dropna()
        m = len(sub)
        if m < 5:
            return pd.Series({'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan,
                              'converged': False, 'n_samples': m})

        y = sub[res_col].to_numpy(float)             # 注意：这里是 y，不是 y^2
        q1 = scale * (sub[nwm_spread_col].to_numpy(float) ** 2)
        q2 = (sub[gnss_formal_col].to_numpy(float) ** 2)
        q3 = np.ones_like(q1)

        use_bique = (mode == 'bique') or (mode == 'auto' and (design_cols or fit_intercept))
        if use_bique:
            s1, s2, s3, conv = _solve_bique_diag(y, q1, q2, q3, beta_fixed=fixed_beta)
        else:
            s1, s2, s3, conv = _solve_irls(y, q1, q2, q3, beta_fixed=fixed_beta)

        return pd.Series({
            'alpha': np.sqrt(s1),
            'beta':  np.sqrt(s2),
            'gamma': np.sqrt(s3),
            'converged': conv,
            'n_samples': m
        })

    df_params = df.groupby(group_col, group_keys=False).apply(_apply)

    # 合并回去
    df_out = df.copy()
    cols_to_drop = ['alpha', 'beta', 'gamma', 'ztd_sigma_vce', 'converged', 'n_samples']
    df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns], errors='ignore')
    df_out = df_out.merge(df_params, left_on=group_col, right_index=True, how='left')

    # 预测总标准差
    term1 = (df_out['alpha'] ** 2) * scale * (df_out[nwm_spread_col] ** 2)
    term2 = (df_out['beta']  ** 2) * (df_out[gnss_formal_col] ** 2)
    term3 = (df_out['gamma'] ** 2) * 1.0
    df_out['ztd_sigma_vce'] = np.sqrt(term1 + term2 + term3)

    return df_out, df_params
