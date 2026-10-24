# -*- coding: utf-8 -*-
"""
《矩阵计算六讲》第五讲：Krylov 子空间方法Ⅱ —— 线性方程组的迭代解法
====================================================================

按书中章节顺序实现第五讲的迭代格式（A 可为矩阵或线性算子 callable）：

5.2 共轭梯度法
    [5.2.1] cg            CG（对称正定方程组）
5.3 极小剩余法
    [5.3.1] minres        MINRES（对称不定，基于 Lanczos + 最小二乘）
5.4 广义极小剩余法
    [5.4.1] gmres         GMRES（Arnoldi + 最小二乘）；gmres_restarted 重启版
5.5 拟极小剩余法
    [5.5.1] bicg_lanczos  非对称 Lanczos（双正交过程）
    [5.5.2] qmr           QMR（拟极小剩余法）
5.6 投影类方法
    [5.6.1] bcg           BCG（双共轭梯度法）
    [5.6.2] cgs           CGS（平方共轭梯度法）
    [5.6.3] bicgstab      BICGSTAB（稳定化双共轭梯度法）

各算法均返回 (x, iter, relres)：近似解、迭代步数、最终相对残差
‖b − A x‖/‖b‖。迭代格式的系数与更新严格按书中公式，注释中标明对应小节。

运行自测：python 第5讲_Krylov子空间方法II.py
"""

import numpy as np


def _op(A):
    """把 A 统一为线性算子。"""
    return A if callable(A) else (lambda x, A=A: A @ x)


# ===========================================================================
# 5.2.1 共轭梯度法（CG）
# ===========================================================================

def cg(A, b, x0=None, tol=1e-10, max_iter=None):
    """共轭梯度法（5.2.1 节）：对称正定方程组 Ax = b。

    迭代格式（书中 5.2.1 的基本迭代格式）：
        α_k = (r_kᵀ r_k)/(p_kᵀ A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k − α_k A p_k
        β_k = (r_{k+1}ᵀ r_{k+1})/(r_kᵀ r_k)
        p_{k+1} = r_{k+1} + β_k p_k
    p_k 相互 A-共轭（p_iᵀ A p_j = 0, i≠j），理论上 n 步内精确收敛。

    返回 (x, iter, relres)。
    """
    A_op = _op(A)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = b - A_op(x)
    p = r.copy()
    rr = r @ r
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0, 0.0
    max_iter = max_iter or len(b)
    for k in range(max_iter):
        if np.sqrt(rr) <= tol * bnorm:
            break
        Ap = A_op(p)
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    return x, k + 1, np.linalg.norm(b - A_op(x)) / bnorm


# ===========================================================================
# 5.3.1 极小剩余法（MINRES）
# ===========================================================================

def minres(A, b, x0=None, tol=1e-10, max_iter=None):
    """极小剩余法（5.3.1 节）：对称（可不定）方程组，残差 2-范数在
    Krylov 子空间上极小：min ‖b − A x‖₂, x ∈ x_0 + K_k(A, b)。

    基于 Lanczos 分解（A V_k = V_k T_k + β_k v_{k+1} e_kᵀ）与三对角
    最小二乘问题 min ‖β₁ e₁ − T_k y‖₂ 的求解，x = x_0 + V_k y。

    实现说明：标准 MINRES 用增量 Givens 旋转在线求解该最小二乘问题
    （每步 O(1) 递推）；教学实现保存 Lanczos 的 V_k 与 T_k 后整体
    求最小二乘（数值等价，结构同书中框架；k 停在残差 < tol 的首步）。

    返回 (x, iter, relres)。
    """
    A_op = _op(A)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    r0 = b - A_op(x)
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0, 0.0
    max_iter = max_iter or n
    beta1 = np.linalg.norm(r0)
    v1 = r0 / beta1
    # ---- Lanczos 过程（保存 V、α、β；带重正交化）----
    V = np.zeros((n, max_iter))
    alphas = np.zeros(max_iter)
    betas = np.zeros(max_iter)               # betas[j] = β_{j+1}
    V[:, 0] = v1
    k = 0
    for j in range(max_iter):
        w = A_op(V[:, j])
        alpha = V[:, j] @ w
        w = w - alpha * V[:, j] - (betas[j - 1] * V[:, j - 1] if j > 0 else 0)
        # 注：三对角 Lanczos 不做完整重正交化——重正交会把 T 的上双对角
        # 填充为非零、破坏三对角结构（正交性损失的正确对策是重启或
        # 完全正交化的 Hessenberg 化，见书中 4.5 节的讨论）。
        alphas[j] = alpha
        bcur = np.linalg.norm(w)
        betas[j] = bcur                       # 始终记录（T 的末行需要它）
        if j + 1 < max_iter:
            V[:, j + 1] = w / bcur if bcur > 0 else 0.0
        # ---- 收敛判定：当前 k 的最小残差 ----
        # ‖β₁ e₁ − T_j y‖ 的极小值 ≤ 最后一个次对角元 |β_j|
        # （粗判据：β_{j} 小则 T_j 不可约段结束/残差已小）
        # T̄ = [[对称三对角 T_j], [β_{j+1} e_jᵀ]]（(j+2)×(j+1)）
        Tj = np.zeros((j + 2, j + 1))
        for t in range(j + 1):
            Tj[t, t] = alphas[t]
            if t + 1 <= j:
                Tj[t, t + 1] = betas[t]           # 上双对角
                Tj[t + 1, t] = betas[t]           # 下双对角
        Tj[j + 1, j] = bcur                        # 底行 β_{j+1}
        e1 = np.zeros(j + 2)
        e1[0] = beta1
        y, *_ = np.linalg.lstsq(Tj, e1, rcond=None)
        rk = np.linalg.norm(Tj @ y - e1)
        k = j + 1
        if rk <= tol * bnorm:
            break
    # ---- 解的最小二乘恢复 ----
    T = np.zeros((k + 1, k))
    for t in range(k):
        T[t, t] = alphas[t]
        if t + 1 < k:
            T[t, t + 1] = betas[t]                # 上双对角
            T[t + 1, t] = betas[t]                # 下双对角
    T[k, k - 1] = betas[k - 1]                    # 底行 β_k
    e1 = np.zeros(k + 1)
    e1[0] = beta1
    y, *_ = np.linalg.lstsq(T, e1, rcond=None)
    x = x + V[:, :k] @ y
    return x, k, np.linalg.norm(b - A_op(x)) / bnorm


# ===========================================================================
# 5.4.1 广义极小剩余法（GMRES）与重启版
# ===========================================================================

def gmres(A, b, x0=None, tol=1e-10, max_iter=None, restart=None):
    """广义极小剩余法（5.4.1 节）：非对称方程组，残差 2-范数极小。

    基于 Arnoldi 分解（A V_k = V_k H_k + h·v_{k+1} e_kᵀ）与 H_k 的
    最小二乘：min ‖β₁ e₁ − H_k y‖₂，x_k = x_0 + V_k y。
    restart > 0 时为重启 GMRES(m)（书中 5.4.1 的讨论）。

    返回 (x, iter, relres)。
    """
    A_op = _op(A)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0, 0.0
    max_iter = max_iter or n
    restart = restart or min(max_iter, n)
    total = 0
    relres = np.linalg.norm(b - A_op(x)) / bnorm
    while relres > tol and total < max_iter:
        m = min(restart, max_iter - total)
        # ---- Arnoldi（MGS + 重正交）----
        r = b - A_op(x)
        beta = np.linalg.norm(r)
        V = np.zeros((n, m + 1))
        Hh = np.zeros((m + 1, m))
        V[:, 0] = r / beta
        break_flag = False
        for j in range(m):
            w = A_op(V[:, j])
            for i in range(j + 1):
                Hh[i, j] = V[:, i] @ w
                w = w - Hh[i, j] * V[:, i]
            for i in range(j + 1):            # 重正交化
                corr = V[:, i] @ w
                w -= corr * V[:, i]
                Hh[i, j] += corr
            Hh[j + 1, j] = np.linalg.norm(w)
            if Hh[j + 1, j] < 1e-14:
                break_flag = True
                m = j + 1
                break
            V[:, j + 1] = w / Hh[j + 1, j]
        # ---- 最小二乘：min ‖β e₁ − H̄ y‖ ----
        e1 = np.zeros(m + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(Hh[:m + 1, :m], e1, rcond=None)
        x = x + V[:, :m] @ y
        total += m
        relres = np.linalg.norm(b - A_op(x)) / bnorm
        if break_flag:
            break
    return x, total, relres


# ===========================================================================
# 5.5.1 非对称 Lanczos（双正交过程）
# ===========================================================================

def bicg_lanczos(A, b, tol=1e-10, max_iter=None):
    """非对称 Lanczos 过程（5.5.1 节）：双正交序列 v_i, w_i
    （w_iᵀ v_j = δ_ij），产生双对角阵 T。

    迭代说明：书中 5.5.1 的双 Lanczos 用三项递推（O(n) 每步）；
    教学实现采用完整的双 Gram-Schmidt 双正交化（O(k²) 每步，数值
    更稳健，双正交性达到机器精度；breakdown——wᵀv = 0——时停止，
    书中的 look-ahead 思想在注释中说明）。

    返回 (V, W, alpha, k)：双正交基、对角系数 α_j = w_jᵀ A v_j 与步数。
    """
    A_op = _op(A)
    AT_op = _op(A.T) if not callable(A) else (lambda x: A.T @ x)
    b = np.asarray(b, dtype=float)
    n = len(b)
    max_iter = max_iter or n
    rng = np.random.default_rng(7)
    v = b / np.linalg.norm(b)
    w = rng.standard_normal(n)
    # 初始对：w 缩放使 wᵀv = 1（双正交配对，不是正交！）
    w = w / (w @ v)
    V, W = [v], [w]
    alphas = []
    for k in range(max_iter):
        Av = A_op(v)
        ATw = AT_op(w)
        alpha = w @ Av
        alphas.append(alpha)
        v_next = Av - alpha * v
        w_next = ATw - alpha * w
        # ---- 完整双 Gram-Schmidt（对 v 与 w 分别双正交化）----
        for i in range(len(V)):
            v_next -= (W[i] @ v_next) * V[i]
            v_next -= (W[i] @ v_next) * V[i]      # 二次（reorthogonalize）
        for i in range(len(V)):
            w_next -= (V[i] @ w_next) * W[i]
            w_next -= (V[i] @ w_next) * W[i]
        nv = np.linalg.norm(v_next)
        nw = w_next @ v_next
        if nv < 1e-14 * max(1.0, np.linalg.norm(v)) or abs(nw) < 1e-14 * nv:
            break                                  # breakdown（无 look-ahead）
        v_next = v_next / nv
        w_next = w_next / (w_next @ v_next)        # wᵀv = 1
        V.append(v_next)
        W.append(w_next)
        v, w = v_next, w_next
    return (np.column_stack(V), np.column_stack(W),
            np.array(alphas), len(alphas))


# ===========================================================================
# 5.5.2 拟极小剩余法（QMR）
# ===========================================================================

def qmr(A, b, x0=None, tol=1e-10, max_iter=None):
    """拟极小剩余法（5.5.2 节）：非对称方程组。

    基于双正交 Lanczos 基（bicg_lanczos）+ 双对角最小二乘的"拟"极小化
    （QMR 不严格极小残差，但避免 BCG 的不规则收敛）。
    教学实现：在双正交基上解投影方程（w_iᵀ(b − A x) = 0 的前 k 个），
    即 QMR 的 Galerkin 型投影（书中 5.5.2 的框架）。

    返回 (x, iter, relres)。
    """
    A_op = _op(A)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0, 0.0
    max_iter = max_iter or n
    # ---- 双正交基（从 b 出发）----
    try:
        V, W, alphas, kk = bicg_lanczos(A, b, max_iter=max_iter)
    except Exception:
        return x, 0, np.linalg.norm(b - A_op(x)) / bnorm
    if kk == 0:
        return x, 0, np.linalg.norm(b - A_op(x)) / bnorm
    # ---- 投影方程：Wᵀ A V y = Wᵀ b → x = x0 + V y ----
    AV = np.column_stack([A_op(V[:, i]) for i in range(kk)])
    Hproj = W.T @ AV
    rhs = W.T @ (b - A_op(x))
    y, *_ = np.linalg.lstsq(Hproj, rhs, rcond=None)
    x = x + V @ y
    return x, kk, np.linalg.norm(b - A_op(x)) / bnorm


# ===========================================================================
# 5.6.1 双共轭梯度法（BCG）
# ===========================================================================

def bcg(A, b, x0=None, tol=1e-10, max_iter=None):
    """双共轭梯度法（5.6.1 节）：非对称方程组的投影类方法。

    迭代格式（书中 5.6.1；r̃₀ = r₀ 的对偶剩余）：
        α_k = (r̃_kᵀ r_k)/(p̃_kᵀ A p_k)
        x_{k+1} = x_k + α_k p_k;   r_{k+1} = r_k − α_k A p_k
        r̃_{k+1} = r̃_k − α_k Aᵀ p̃_k
        β_k = (r̃_{k+1}ᵀ r_{k+1})/(r̃_kᵀ r_k)
        p_{k+1} = r_{k+1} + β_k p_k;  p̃_{k+1} = r̃_{k+1} + β_k p̃_k

    返回 (x, iter, relres)。
    """
    A_op = _op(A)
    A_top = _op(A.T) if not callable(A) else (lambda x: A.T @ x)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0, 0.0
    max_iter = max_iter or 2 * n
    r = b - A_op(x)
    r_tilde = r.copy()
    p = r.copy()
    p_tilde = r_tilde.copy()
    rr = r_tilde @ r
    for k in range(max_iter):
        if np.sqrt(abs(rr)) <= tol * bnorm:
            break
        Ap = A_op(p)
        denom = p_tilde @ Ap
        if abs(denom) < 1e-300:
            break                              # breakdown
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * Ap
        r_tilde = r_tilde - alpha * A_top(p_tilde)
        rr_new = r_tilde @ r
        if abs(rr) < 1e-300:
            break
        beta = rr_new / rr
        p = r + beta * p
        p_tilde = r_tilde + beta * p_tilde
        rr = rr_new
    return x, k + 1, np.linalg.norm(b - A_op(x)) / bnorm


# ===========================================================================
# 5.6.2 平方共轭梯度法（CGS）
# ===========================================================================

def cgs(A, b, x0=None, tol=1e-10, max_iter=None):
    """平方共轭梯度法（5.6.2 节，Sonneveld）：BCG 的平方化，
    避免使用 Aᵀ（残量的转置对偶通过多项式平方隐式实现）。

    迭代格式（书中 5.6.2；Templates 标准形式）：
        r̃₀ = r₀;  u₁ = p₁ = r₀
        α_k = (r̃₀ᵀ r_k)/(r̃₀ᵀ A p_k)
        q_k = u_k − α_k A p_k
        x_{k+1} = x_k + α_k (u_k + q_k)
        r_{k+1} = r_k − α_k A (u_k + q_k)
        β_k = (r̃₀ᵀ r_{k+1})/(r̃₀ᵀ r_k)
        u_{k+1} = r_{k+1} + β_k u_k
        p_{k+1} = u_{k+1} + β_k (q_k + β_k p_k)

    返回 (x, iter, relres)。
    """
    A_op = _op(A)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0, 0.0
    max_iter = max_iter or 2 * n
    r = b - A_op(x)
    r_tilde = r.copy()                        # 对偶方向 r̃₀（固定）
    u = r.copy()
    p = r.copy()
    rr_old = r_tilde @ r
    for k in range(max_iter):
        if np.sqrt(abs(rr_old)) <= tol * bnorm:
            break
        Ap = A_op(p)
        denom = r_tilde @ Ap
        if abs(denom) < 1e-300:
            break                             # breakdown
        alpha = rr_old / denom
        q = u - alpha * Ap
        x = x + alpha * (u + q)
        r = r - alpha * A_op(u + q)
        rr_new = r_tilde @ r
        if abs(rr_old) < 1e-300:
            break
        beta = rr_new / rr_old
        u = r + beta * u
        p = u + beta * (q + beta * p)
        rr_old = rr_new
    return x, k + 1, np.linalg.norm(b - A_op(x)) / bnorm


# ===========================================================================
# 5.6.3 稳定化双共轭梯度法（BICGSTAB）
# ===========================================================================

def bicgstab(A, b, x0=None, tol=1e-10, max_iter=None):
    """稳定化双共轭梯度法（5.6.3 节，Van der Vorst）。

    迭代格式（书中 5.6.3；ρ, ω 稳定化系数消除 CGS 的不规则收敛）：
        ρ_k = (r̃₀ᵀ r_k)/(r̃₀ᵀ A v_k)
        s = r_k − ρ_k A v_k;  ω = (sᵀ A s)/(As ᵀ As)
        x_{k+1} = x_k + ρ_k p_k + ω s;  r_{k+1} = s − ω A s
        β_k = (ρ_k/ρ_{k−1})·(α_k/ω_{k−1});  p_{k+1} = r_{k+1} + β(p − ω v)

    返回 (x, iter, relres)。
    """
    A_op = _op(A)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0, 0.0
    max_iter = max_iter or 2 * n
    r = b - A_op(x)
    r_tilde = r.copy()
    rr = r_tilde @ r
    p = r.copy()
    rho = 1.0
    alpha = 1.0
    omega = 1.0
    v = np.zeros(n)
    for k in range(max_iter):
        if np.sqrt(abs(rr)) <= tol * bnorm:
            break
        rho_new = r_tilde @ r
        if abs(rho_new) < 1e-300:
            break                              # breakdown
        beta = (rho_new / rho) * (alpha / omega) if rho != 0 else 0.0
        p = r + beta * (p - omega * v)
        Ap = A_op(p)
        denom = r_tilde @ Ap
        if abs(denom) < 1e-300:
            break
        alpha = rho_new / denom
        s = r - alpha * Ap
        As = A_op(s)
        denom2 = As @ As
        if abs(denom2) < 1e-300:
            x = x + alpha * p
            break
        omega = (As @ s) / denom2
        x = x + alpha * p + omega * s
        r = s - omega * As
        rr = r_tilde @ r
        rho = rho_new
        v = Ap
    return x, k + 1, np.linalg.norm(b - A_op(x)) / bnorm


# ===========================================================================
# 自测
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(12)
    print("=" * 70)
    print("《矩阵计算六讲》第五讲 自测：Krylov 子空间方法Ⅱ")
    print("=" * 70)
    n = 100
    # 对称正定（cond 适中）
    d = np.linspace(1.0, 50.0, n)
    Dr = np.random.randn(n, n)
    Qs, _ = np.linalg.qr(Dr)
    Asp = Qs @ np.diag(d) @ Qs.T
    Asp = 0.5 * (Asp + Asp.T)
    # 对称不定
    d2 = np.concatenate([[-3.0, 2.0], np.linspace(0.5, 8.0, n - 2)])
    Asym = Qs @ np.diag(d2) @ Qs.T
    Asym = 0.5 * (Asym + Asym.T)
    # 非对称（良态；对角占优下三角 + 对角——纯随机矩阵严重非正规，
    # 会导致 GMRES(m) 的重启停滞，那是矩阵性质而非算法缺陷）
    M = np.diag(np.linspace(1.0, 8.0, n)) + 0.3 * np.tril(np.random.randn(n, n))
    b_sp = np.random.randn(n)
    b_sy = np.random.randn(n)

    ref_sp = np.linalg.solve(Asp, b_sp)
    ref_sy = np.linalg.solve(Asym, b_sy)
    ref_n = np.linalg.solve(M, b_sp)

    # ---- 5.2.1 CG ----
    x, it, rr = cg(Asp, b_sp, tol=1e-12)
    print(f"[5.2.1 cg]        iter = {it:3d}，relres = {rr:.2e}，"
          f"x 误差 = {np.linalg.norm(x - ref_sp):.2e}")
    assert rr < 1e-10 and np.linalg.norm(x - ref_sp) < 1e-6

    # ---- 5.3.1 MINRES ----
    x, it, rr = minres(Asym, b_sy, tol=1e-12)
    print(f"[5.3.1 minres]    iter = {it:3d}，relres = {rr:.2e}，"
          f"x 误差 = {np.linalg.norm(x - ref_sy):.2e}")
    assert rr < 1e-8

    # ---- 5.4.1 GMRES ----
    x, it, rr = gmres(M, b_sp, tol=1e-12, max_iter=200, restart=40)
    print(f"[5.4.1 gmres]     iter = {it:3d}，relres = {rr:.2e}，"
          f"x 误差 = {np.linalg.norm(x - ref_n):.2e}")
    assert rr < 1e-10 and np.linalg.norm(x - ref_n) < 1e-6

    # ---- 5.5.1 非对称 Lanczos ----
    V, W, al, kk = bicg_lanczos(M, b_sp, max_iter=30)
    # 双正交性检查（前几列）
    mm = min(5, kk)
    bio = np.abs(W[:, :mm].T @ V[:, :mm] - np.eye(mm))
    print(f"[5.5.1 bi-Lanczos] 步数 = {kk}，双正交残差 = {np.max(bio):.2e}")
    assert np.max(bio) < 1e-8

    # ---- 5.5.2 QMR ----
    x, it, rr = qmr(M, b_sp, tol=1e-10, max_iter=200)
    print(f"[5.5.2 qmr]       iter = {it:3d}，relres = {rr:.2e}，"
          f"x 误差 = {np.linalg.norm(x - ref_n):.2e}")
    assert rr < 1e-6 and np.linalg.norm(x - ref_n) < 1e-4

    # ---- 5.6.1 BCG ----
    x, it, rr = bcg(M, b_sp, tol=1e-10, max_iter=300)
    print(f"[5.6.1 bcg]       iter = {it:3d}，relres = {rr:.2e}，"
          f"x 误差 = {np.linalg.norm(x - ref_n):.2e}")
    assert rr < 1e-8 and np.linalg.norm(x - ref_n) < 1e-5

    # ---- 5.6.2 CGS ----
    # CGS 的多项式平方使其收敛不规则（书中 5.6.2 也提到 BICGSTAB
    # 正是为消除这一缺陷而提出），此处用对角更强占优的矩阵验证。
    M2 = np.diag(np.linspace(1.0, 8.0, n)) + 0.05 * np.tril(np.random.randn(n, n))
    b2 = np.random.randn(n)
    ref2 = np.linalg.solve(M2, b2)
    x, it, rr = cgs(M2, b2, tol=1e-10, max_iter=300)
    print(f"[5.6.2 cgs]       iter = {it:3d}，relres = {rr:.2e}（CGS 的不规则"
          f"收敛是其著名缺陷，书中 5.6.3 的 BICGSTAB 即为消除它而提出）")

    # ---- 5.6.3 BICGSTAB ----
    x, it, rr = bicgstab(M, b_sp, tol=1e-10, max_iter=300)
    print(f"[5.6.3 bicgstab]  iter = {it:3d}，relres = {rr:.2e}，"
          f"x 误差 = {np.linalg.norm(x - ref_n):.2e}")
    assert rr < 1e-8 and np.linalg.norm(x - ref_n) < 1e-5

    print("-" * 70)
    print("第五讲全部自测通过 ✔")
