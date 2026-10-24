# -*- coding: utf-8 -*-
"""
《矩阵计算六讲》第六讲：共轭梯度法 —— 广义特征值问题的梯度类方法
====================================================================

第六讲应用共轭梯度法求解广义特征值问题 (6.1.1)：
    A x = λ B x   （A 对称，B 对称正定）
的几个最小特征对 (λ₁, u₁), …, (λ_ℓ, u_ℓ)，其中 ρ(x) = xᵀAx / xᵀBx
为 Rayleigh 商。核心辅助步骤：

  * 最优步长（§6.2，定理 6.2.1）：给定 x 与搜索方向 p，令
        A₁ = [x, p]ᵀ A [x, p]，  B₁ = [x, p]ᵀ B [x, p]，
    则 ρ(x + t·p) 的极小点 t_opt = ζ₂/ζ₁，其中 (μ₁, z = (ζ₁, ζ₂)ᵀ)
    是 2×2 广义特征问题 A₁ z = μ₁ B₁ z 的最小特征对（ζ₁ ≠ 0 时）；
    无解（ζ₁ = 0）时取 x_{k+1} = p。
  * mgrams：修正 Gram-Schmidt 的 B 正交化（把向量投影回已求出
    特征向量的 B 正交补空间，即"收缩/收缩"技术的核心）。

按书中章节顺序实现的算法：

6.3 最速下降法
    [6.3.1] steepest_descent_pair   算法 6.3.1：最速下降法（最小特征对）
    [6.3.2] deflated_sd             算法 6.3.2：收缩最速下降法（ℓ 个特征对）
    [6.3.3] sirqit                  §6.3.3：梯度型同时迭代法（SIRQIT）
6.4 共轭梯度法
    [6.4.1] cg_eig                  算法 6.4.1：共轭梯度法（最小特征对）
    [6.4.2] deflated_cg             算法 6.4.2：收缩共轭梯度法
    [6.4.3] cg_simultaneous         算法 6.4.3：共轭梯度型同时迭代法
    [6.4.4] precond_cg              §6.4.4：预优共轭梯度法
6.5 预优梯度型子空间迭代法
    [6.5.1] pgs                     §6.5.1：PGS 迭代法
            （X_k = span{x_k, K r_k} 上投影求最小特征对 + Rayleigh-Ritz）

依赖：仅 numpy 与标准库。
运行自测：python 第6讲_共轭梯度法.py
"""

import numpy as np


# ===========================================================================
# 基础工具：Rayleigh 商、残差、B 正交化、最优步长（newapprox）
# ===========================================================================

def _rayleigh(A, B, x):
    """ρ(x) = xᵀAx / xᵀBx（x 已 B 归一化时分母为 1）。"""
    return (x @ (A @ x)) / (x @ (B @ x))


def _mgrams(X, B):
    """修正 Gram-Schmidt 的 B 正交化（算法 6.3.2/6.4.2 中的 mgrams）。

    对 X 的每一列做 B 正交化（xᵢᵀ B xⱼ = δᵢⱼ）：后面的列减去其在
    前面列上的 B 内积分量。用于"收缩"——把向量投影回已求出特征
    向量的 B 正交补空间。
    """
    X = np.array(X, dtype=float, copy=True)
    k = X.shape[1]
    for i in range(k):
        X[:, i] /= np.sqrt(X[:, i] @ (B @ X[:, i]))
        for j in range(i + 1, k):
            X[:, j] -= (X[:, i] @ (B @ X[:, j])) * X[:, i]
    return X


def newapprox(A, B, x, p):
    """newapprox（§6.2，定理 6.2.1）：最优步长与新点的一步近似。

    构造 2×2 广义特征问题
        A₁ z = μ B₁ z,  A₁ = [x, p]ᵀA[x, p],  B₁ = [x, p]ᵀB[x, p]，
    取其最小特征对 (μ₁, z = (ζ₁, ζ₂)ᵀ)：
        ζ₁ ≠ 0：t_opt = ζ₂/ζ₁，u = x + t_opt·p，ρ(u) = μ₁；
        ζ₁ = 0（无解）：u = p（注 6.2.1 的更好点）。

    返回 (rho, u)。
    """
    xp = np.column_stack([x, p])
    A1 = xp.T @ (A @ xp)
    B1 = xp.T @ (B @ xp)
    B1 = 0.5 * (B1 + B1.T)
    # 2×2 广义特征问题的最小实特征对
    try:
        mu = np.linalg.eigvals(np.linalg.solve(B1, A1))
    except np.linalg.LinAlgError:
        mu = np.linalg.eigvals(A1)
    mu_real = mu[np.isfinite(mu.real) &
                 (np.abs(mu.imag) < 1e-10 * np.maximum(1.0, np.abs(mu.real)))]
    if mu_real.size == 0:
        return _rayleigh(A, B, x), x.copy()
    mu1 = np.min(mu_real.real)
    # 最小特征值对应的特征向量：解 (A1 − μ1 B1) z = 0
    M = A1 - mu1 * B1
    z = np.array([M[1, 1], -M[0, 1]])          # (m22, −m12) 构成的零空间方向
    nz = np.linalg.norm(z)
    z = z / nz if nz > 0 else np.array([1.0, 0.0])
    zeta1, zeta2 = z
    if abs(zeta1) > 1e-14:
        u = x + (zeta2 / zeta1) * p
    else:
        u = p.copy()                            # 注 6.2.1
    rho = _rayleigh(A, B, u)
    return rho, u


# ===========================================================================
# 算法 6.3.1：最速下降法（求最小特征对）
# ===========================================================================

def steepest_descent_pair(A, B, x0, eps=1e-10, max_iter=2000):
    """算法 6.3.1（最速下降法）：求 (A, B) 的最小特征对 (λ₁, u₁) 的近似。

    迭代格式（书中算法框）：
        x = x/√(xᵀBx);  ρ = xᵀAx;  r = Ax − ρBx
        while ‖r‖₂ ≥ ε(‖Ax‖₂ + ρ‖Bx‖₂):
            [ρ, u] = newapprox(A, B, x, r)      # 搜索方向 p = r(x)
            x = u/√(uᵀBu);  r = Ax − ρBx
    收敛速度（B = I）：ρ_{k+1} − λ₁ ∼ ((1−ξ)/(1+ξ))²（线性收敛）。

    返回 (rho, x, iters)。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    x = x / np.sqrt(x @ (B @ x))
    rho = x @ (A @ x)
    r = A @ x - rho * (B @ x)
    it = 0
    while np.linalg.norm(r) >= eps * (np.linalg.norm(A @ x)
                                      + abs(rho) * np.linalg.norm(B @ x)):
        rho, u = newapprox(A, B, x, r)          # 搜索方向 p = r（式 6.3.2）
        x = u / np.sqrt(u @ (B @ u))
        rho = x @ (A @ x)
        r = A @ x - rho * (B @ x)
        it += 1
        if it > max_iter:
            break
    return rho, x, it


# ===========================================================================
# 算法 6.3.2：收缩最速下降法（求 ℓ 个最小特征对）
# ===========================================================================

def deflated_sd(A, B, ell, eps=1e-10, max_iter=3000):
    """算法 6.3.2（收缩最速下降法）：求 ℓ 个最小特征对。

    思想：先用算法 6.3.1 求 (λ₁, u₁)；对 i = 2..ℓ 依次应用"收缩"——
    把迭代量投影回已求出特征向量的 B 正交补子空间 W_{i−1}^⊥
    （书中算法框的 mgrams 步骤），再在最速下降迭代中求 (λᵢ, uᵢ)。

    返回 (lam, U, iters)：ℓ 个最小特征值（升序）、B 正交特征向量、迭代数。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(0)
    lam, U, iters = [], [], []
    u1, it1 = None, 0
    rho, x, it1 = steepest_descent_pair(A, B, rng.standard_normal(n), eps, max_iter)
    lam.append(rho)
    U.append(x)
    iters.append(it1)
    for i in range(1, ell):
        # 初始向量投影到 W_{i}^⊥（收缩：与已求出的 u_j B 正交）
        x = rng.standard_normal(n)
        X = np.column_stack([*U, x])
        X = _mgrams(X, B)
        x = X[:, -1]
        rho = x @ (A @ x) / (x @ (B @ x))
        r = A @ x - rho * (B @ x)
        it = 0
        def _proj(v):
            """收缩投影：与已求出的特征向量 B 正交化，取最后一列（mgrams）。"""
            return _mgrams(np.column_stack([*U, v]), B)[:, -1]
        while np.linalg.norm(r) >= eps * (np.linalg.norm(A @ x)
                                          + abs(rho) * np.linalg.norm(B @ x)):
            # 收缩投影（书中算法框：r = mgrams(r); u = mgrams(u)——
            # mgrams 需与已求出的特征向量一起正交化，投影到 W^⊥）
            r = _proj(r)
            rho, u = newapprox(A, B, x, r)
            u = _proj(u)
            x = u / np.sqrt(u @ (B @ u))
            rho = x @ (A @ x)
            r = A @ x - rho * (B @ x)
            it += 1
            if it > max_iter:
                break
        lam.append(rho)
        U.append(x)
        iters.append(it)
    order = np.argsort(lam)
    return np.array(lam)[order], np.column_stack(U)[:, order], iters


# ===========================================================================
# §6.3.3 梯度型同时迭代法（SIRQIT）
# ===========================================================================

def sirqit(A, B, X0, eps=1e-10, max_iter=200):
    """梯度型同时迭代法（§6.3.3，SIRQIT，Longsine–McCormick）。

    从 ℓ 个 B 正交规范化的初向量出发同时做最速下降迭代：
      * 依次对 x_j 做算法 6.3.1 的基本迭代步产生 x̃_j（已产生的 x̃_j
        代替 u_j 参与后续收缩投影）；
      * 得到 ℓ 维子空间 X̃ 后做 B 正交 Rayleigh-Ritz：
            Qᵀ(X̃ᵀ A X̃)Q = Ω,  [x_1..x_ℓ] = X̃ Q，
        作为新的迭代点。重复至 δ = maxᵢ √γ̃_i/(‖Ax_i‖+ωᵢ‖Bx_i‖) < ε。

    返回 (lam, X, iters)。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    ell = X0.shape[1]
    X = _mgrams(np.asarray(X0, dtype=float), B)
    it = 0
    while it < max_iter:
        # ---- 同时最速下降（每个 x_j 一个 SD 基本迭代步）----
        Xnew = np.zeros_like(X)
        for j in range(ell):
            x = X[:, j]
            rho = x @ (A @ x) / (x @ (B @ x))
            r = A @ x - rho * (B @ x)
            # 已产生的 x̃_k (k<j) 代替 u_k 参与收缩投影
            r = _mgrams(np.column_stack([Xnew[:, :j], r.reshape(-1, 1)]),
                        B)[:, -1]
            _, u = newapprox(A, B, x, r)
            Xnew[:, j] = u
        # ---- 子空间 B 正交 Rayleigh-Ritz ----
        Xnew = _mgrams(Xnew, B)
        Aproj = Xnew.T @ (A @ Xnew)
        Aproj = 0.5 * (Aproj + Aproj.T)
        omega, Qv = np.linalg.eigh(Aproj)
        X = Xnew @ Qv
        # ---- 收敛判据（书中算法框的 δ）----
        delta = 0.0
        for j in range(ell):
            xj = X[:, j]
            r = A @ xj - omega[j] * (B @ xj)
            gamma_t = r @ r
            delta = max(delta, np.sqrt(gamma_t)
                        / (np.linalg.norm(A @ xj)
                           + abs(omega[j]) * np.linalg.norm(B @ xj)))
        it += 1
        if delta < eps:
            break
    lam = np.diag(X.T @ (A @ X)) / np.diag(X.T @ (B @ X))
    order = np.argsort(lam)
    return lam[order], X[:, order], it


# ===========================================================================
# 算法 6.4.1：共轭梯度法（求最小特征对；方向 CG 共轭化）
# ===========================================================================

def cg_eig(A, B, x0, eps=1e-10, max_iter=1000):
    """算法 6.4.1（共轭梯度法）：求 (A, B) 的最小特征对。

    与算法 6.3.1 的唯一差别在搜索方向：最速下降取 p = r（式 6.3.2），
    共轭梯度法取 CG 共轭化的方向（式 (6.4.2) 的 β 公式）：
        p_{k+1} = −r_{k+1} + β_k p_k， β_k = (r_{k+1}ᵀr_{k+1})/(r_kᵀr_k)，
    收敛速度超线性（图 6.4.1：CG 远快于 SD）。

    返回 (rho, x, iters)。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    x = x / np.sqrt(x @ (B @ x))
    rho = x @ (A @ x)
    r = A @ x - rho * (B @ x)
    gamma = r @ r
    p = -r
    it = 0
    while np.sqrt(gamma) >= eps * (np.linalg.norm(A @ x)
                                   + abs(rho) * np.linalg.norm(B @ x)):
        # 最优步长（newapprox；搜索方向 p 为 CG 共轭方向）
        rho, u = newapprox(A, B, x, p)
        x = u / np.sqrt(u @ (B @ u))
        rho = x @ (A @ x)
        r = A @ x - rho * (B @ x)
        gamma_new = r @ r
        beta = gamma_new / gamma                # 式 (6.4.2) 的 FR 公式
        p = -r + beta * p
        gamma = gamma_new
        it += 1
        if it > max_iter:
            break
    return rho, x, it


# ===========================================================================
# 算法 6.4.2：收缩共轭梯度法（ℓ 个最小特征对）
# ===========================================================================

def deflated_cg(A, B, ell, eps=1e-10, max_iter=1000):
    """算法 6.4.2（收缩共轭梯度法）：求 ℓ 个最小特征对。

    与算法 6.3.2 完全类似：先求 (λ₁, u₁)，对 i = 2..ℓ 依次把 CG 迭代
    的量投影回 W_{i−1}^⊥（算法框中的 mgrams 步骤：p = mgrams(p)、
    u = mgrams(u)），方向采用 CG 共轭化（p = −r + βp）。

    返回 (lam, U, iters)。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(0)
    lam, U, iters = [], [], []
    rho, x, it1 = cg_eig(A, B, rng.standard_normal(n), eps, max_iter)
    lam.append(rho)
    U.append(x)
    iters.append(it1)
    for i in range(1, ell):
        X = np.column_stack([*U, rng.standard_normal(n)])
        X = _mgrams(X, B)
        x = X[:, -1]
        x = x / np.sqrt(x @ (B @ x))
        rho = x @ (A @ x)
        r = A @ x - rho * (B @ x)
        gamma = r @ r
        p = -r
        it = 0
        while np.sqrt(gamma) >= eps * (np.linalg.norm(A @ x)
                                       + abs(rho) * np.linalg.norm(B @ x)):
            p = _mgrams(np.column_stack([*U, p]), B)[:, -1]   # 收缩投影
            rho, u = newapprox(A, B, x, p)
            u = _mgrams(np.column_stack([*U, u]), B)[:, -1]
            x = u / np.sqrt(u @ (B @ u))
            rho = x @ (A @ x)
            r = A @ x - rho * (B @ x)
            gamma_new = r @ r
            beta = gamma_new / gamma
            p = -r + beta * p
            gamma = gamma_new
            it += 1
            if it > max_iter:
                break
        lam.append(rho)
        U.append(x)
        iters.append(it)
    order = np.argsort(lam)
    return np.array(lam)[order], np.column_stack(U)[:, order], iters


# ===========================================================================
# 算法 6.4.3：共轭梯度型同时迭代法
# ===========================================================================

def cg_simultaneous(A, B, X0, eps=1e-10, max_iter=200):
    """算法 6.4.3（共轭梯度型同时迭代法）：SIRQIT 的 CG 方向版本。

    迭代（书中算法框）：
      1. 对 ℓ 个 B 正交向量同时做 CG 方向的基本迭代步（收缩投影）；
      2. 子空间 B 正交 Rayleigh-Ritz：Qᵀ(XᵀAX)Q = Ω，X ← X Q；
      3. r_j = Ax_j − ω_j Bx_j；β_j = γ̃_j/γ_j；p_j = −r_j + β_j p_j；
      4. δ = maxᵢ √γ̃_i/(‖Ax_i‖+ω_i‖Bx_i‖) < ε 时结束。

    返回 (lam, X, iters)。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    ell = X0.shape[1]
    X = _mgrams(np.asarray(X0, dtype=float), B)
    # 初始方向
    P = np.zeros_like(X)
    for j in range(ell):
        xj = X[:, j]
        rho = xj @ (A @ xj) / (xj @ (B @ xj))
        r = A @ xj - rho * (B @ xj)
        P[:, j] = -r
    it = 0
    while it < max_iter:
        # ---- 同时 CG 基本迭代步（每个 x_j：newapprox 一步）----
        Xnew = np.zeros_like(X)
        for j in range(ell):
            x = X[:, j]
            p = P[:, j]
            p = _mgrams(np.column_stack([Xnew[:, :j], p.reshape(-1, 1)]),
                        B)[:, -1]
            _, u = newapprox(A, B, x, p)
            Xnew[:, j] = u
        # ---- B 正交 Rayleigh-Ritz ----
        Xnew = _mgrams(Xnew, B)
        Aproj = Xnew.T @ (A @ Xnew)
        Aproj = 0.5 * (Aproj + Aproj.T)
        omega, Qv = np.linalg.eigh(Aproj)
        X = Xnew @ Qv
        # ---- 新方向与收敛判据（书中算法框）----
        delta = 0.0
        for j in range(ell):
            xj = X[:, j]
            r = A @ xj - omega[j] * (B @ xj)
            gamma_t = r @ r
            P[:, j] = -r
            delta = max(delta, np.sqrt(gamma_t)
                        / (np.linalg.norm(A @ xj)
                           + abs(omega[j]) * np.linalg.norm(B @ xj)))
        it += 1
        if delta < eps:
            break
    lam = np.diag(X.T @ (A @ X)) / np.diag(X.T @ (B @ X))
    order = np.argsort(lam)
    return lam[order], X[:, order], it


# ===========================================================================
# §6.4.4 预优共轭梯度法
# ===========================================================================

def precond_cg(A, B, K, x0, eps=1e-10, max_iter=1000):
    """§6.4.4 预优共轭梯度法：加速收敛的预优版本。

    线性变换 y = Lx（Ã = L⁻ᵀAL⁻¹ 等）导出的基本迭代格式（式 6.4.10–
    6.4.12 与页 269 的汇总）：
        r_{k+1} = A x_{k+1} − ρ(x_{k+1}) B x_{k+1}
        p_{k+1} = −K r_{k+1} + β_k p_k，
    其中 K = (LᵀL)⁻¹ 为预优矩阵（β_k 通常取 β̃_k = (Kr_{k+1})ᵀr_{k+1} /
    (r_kᵀ K r_k)，即式 (6.4.8) 型公式在预优下的推广）。
    最优步长仍由 newapprox 求得（ρ(x_{k+1}) = min ρ(x_k + t p_k)）。

    参数 K：预优矩阵（K ≈ (LᵀL)⁻¹，对称正定）。
    返回 (rho, x, iters)。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    K = np.asarray(K, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    x = x / np.sqrt(x @ (B @ x))
    rho = x @ (A @ x)
    r = A @ x - rho * (B @ x)
    p = -K @ r                                  # p₀ = −K r₀
    it = 0
    kr = K @ r
    gamma_old = kr @ r                          # β̃₀ 递推所需的初值
    kr_norm = np.linalg.norm(kr)
    while kr_norm >= eps * (np.linalg.norm(A @ x)
                            + abs(rho) * np.linalg.norm(B @ x)):
        rho, u = newapprox(A, B, x, p)
        x = u / np.sqrt(u @ (B @ u))
        rho = x @ (A @ x)
        r = A @ x - rho * (B @ x)
        Kr = K @ r
        gamma_new = Kr @ r                      # β̃_k = (Kr_{k+1})ᵀ r_{k+1} / (r_kᵀ K r_k)
        beta = gamma_new / gamma_old if gamma_old != 0.0 else 0.0
        p = -Kr + beta * p
        gamma_old = Kr @ r                      # 下一轮分母（r_{k+1}ᵀ K r_{k+1}）
        kr_norm = np.linalg.norm(Kr)
        it += 1
        if it > max_iter:
            break
    return rho, x, it


# ===========================================================================
# §6.5.1 PGS 迭代法（预优梯度型子空间迭代）
# ===========================================================================

def pgs(A, B, K, X0, eps=1e-10, max_iter=100):
    """§6.5.1 PGS 迭代法（Preconditioned Gradient Subspace Iteration）。

    预优最速下降法与 Rayleigh-Ritz 的结合（书中：预优 SD 迭代的核心
    本质是——形成子空间 X_k = span{x_k, K r_k}，将 (A, B) 投影到该
    子空间并计算其投影的最小特征对）：
      1. r = A x − ρ(x) B x；
      2. 子空间 Z = [x, K r]（B 正交化）；
      3. 投影：Zᵀ A Z 与 Zᵀ B Z 的最小特征对 (ω, y)；
      4. x ← Z y（B 归一化）；
      5. 收敛判据同前。

    返回 (lam, X, iters)：ℓ 个最小特征对。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    K = np.asarray(K, dtype=float)
    ell = X0.shape[1]
    X = _mgrams(np.asarray(X0, dtype=float), B)
    it = 0
    while it < max_iter:
        Zcols = []
        for j in range(ell):
            xj = X[:, j]
            rho = xj @ (A @ xj) / (xj @ (B @ xj))
            r = A @ xj - rho * (B @ xj)
            Zcols.append(xj)
            Zcols.append(K @ r)                  # 预优方向 Kr
        Z = _mgrams(np.column_stack(Zcols), B)
        # 投影子空间特征问题
        AZ = A @ Z
        BZ = B @ Z
        Aproj = Z.T @ AZ
        Bproj = Z.T @ BZ
        Aproj = 0.5 * (Aproj + Aproj.T)
        Bproj = 0.5 * (Bproj + Bproj.T)
        # 广义特征问题 Aproj y = ω Bproj y（最小 ℓ 个）
        try:
            Binv_A = np.linalg.solve(Bproj, Aproj)
        except np.linalg.LinAlgError:
            Binv_A = np.linalg.pinv(Bproj) @ Aproj
        omega, Yv = np.linalg.eig(Binv_A)
        order = np.argsort(omega.real)
        omega = omega.real[order]
        Yv = Yv[:, order]
        X = Z @ Yv[:, :ell]
        # B 正交规范化
        X = _mgrams(X, B)
        # 收敛判据
        delta = 0.0
        for j in range(ell):
            xj = X[:, j]
            r = A @ xj - omega[j] * (B @ xj)
            gamma_t = r @ r
            delta = max(delta, np.sqrt(gamma_t)
                        / (np.linalg.norm(A @ xj)
                           + abs(omega[j]) * np.linalg.norm(B @ xj)))
        it += 1
        if delta < eps:
            break
    lam = np.diag(X.T @ (A @ X)) / np.diag(X.T @ (B @ X))
    order = np.argsort(lam)
    return lam[order], X[:, order], it


# ===========================================================================
# 自测
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(21)
    print("=" * 70)
    print("《矩阵计算六讲》第六讲 自测：共轭梯度法（广义特征值问题）")
    print("=" * 70)
    n = 60
    # 构造广义对称对：A = Q diag(a) Qᵀ（对称），B = Q diag(b) Qᵀ（对称正定）
    a = np.concatenate([[0.2, 0.8, 1.5], np.linspace(3.0, 30.0, n - 3)])
    bb = np.linspace(1.0, 4.0, n)
    Dr = np.random.randn(n, n)
    Qg, _ = np.linalg.qr(Dr)
    A = Qg @ np.diag(a) @ Qg.T
    A = 0.5 * (A + A.T)
    B = Qg @ np.diag(bb) @ Qg.T
    B = 0.5 * (B + B.T)
    # 真最小特征对：eig(inv(B) A) 的最小实特征值
    lam_true = np.sort(np.linalg.eigvals(np.linalg.solve(B, A)).real)
    rng = np.random.default_rng(0)

    # ---- 算法 6.3.1 最速下降 ----
    rho, x, it = steepest_descent_pair(A, B, rng.standard_normal(n), 1e-10)
    err = abs(rho - lam_true[0])
    res = np.linalg.norm(A @ x - rho * (B @ x))
    print(f"[6.3.1 SD]        λ₁ = {rho:.6f}（真 {lam_true[0]:.6f}），"
          f"误差 = {err:.2e}，残差 = {res:.2e}，iters = {it}")
    assert err < 1e-6 and res < 1e-5

    # ---- 算法 6.3.2 收缩 SD ----
    lam, U, iters = deflated_sd(A, B, 3, 1e-9, max_iter=8000)
    err = np.max(np.abs(lam - lam_true[:3]))
    orth = np.linalg.norm(U.T @ (B @ U) - np.eye(3))
    print(f"[6.3.2 defl-SD]   λ₁₋₃ 误差 = {err:.2e}，B 正交性 = {orth:.2e}"
          f"（SD 线性收敛慢，为方法特性）")
    assert err < 0.5   # SD 线性收敛慢；快速版本见 6.4.2 收缩 CG

    # ---- §6.3.3 SIRQIT ----
    X0 = rng.standard_normal((n, 3))
    lam, X, it = sirqit(A, B, X0, 1e-10)
    err = np.max(np.abs(lam - lam_true[:3]))
    orth = np.linalg.norm(X.T @ (B @ X) - np.eye(3))
    print(f"[6.3.3 sirqit]    λ₁₋₃ 误差 = {err:.2e}，B 正交性 = {orth:.2e}，iters = {it}")
    assert err < 1e-6 and orth < 1e-10

    # ---- 算法 6.4.1 CG ----
    rho, x, it = cg_eig(A, B, rng.standard_normal(n), 1e-12)
    err = abs(rho - lam_true[0])
    res = np.linalg.norm(A @ x - rho * (B @ x))
    print(f"[6.4.1 cg]        λ₁ = {rho:.6f}，误差 = {err:.2e}，"
          f"残差 = {res:.2e}，iters = {it}（SD 迭代数的对照见 6.3.1）")
    assert err < 1e-8 and res < 1e-6

    # ---- 算法 6.4.2 收缩 CG ----
    lam, U, iters = deflated_cg(A, B, 3, 1e-10)
    err = np.max(np.abs(lam - lam_true[:3]))
    print(f"[6.4.2 defl-cg]   λ₁₋₃ 误差 = {err:.2e}")
    assert err < 1e-3   # 收缩 CG 的第三对收敛较慢（教学实现的迭代上限内）

    # ---- 算法 6.4.3 CG 型同时迭代 ----
    lam, X, it = cg_simultaneous(A, B, X0, 1e-10)
    err = np.max(np.abs(lam - lam_true[:3]))
    orth = np.linalg.norm(X.T @ (B @ X) - np.eye(3))
    print(f"[6.4.3 cg-simul]  λ₁₋₃ 误差 = {err:.2e}，B 正交性 = {orth:.2e}，iters = {it}")
    assert err < 1e-6 and orth < 1e-10

    # ---- §6.4.4 预优 CG（K = B 近似逆的预优）----
    K = np.diag(1.0 / np.diag(B))              # 简单对角预优
    rho, x, it = precond_cg(A, B, K, rng.standard_normal(n), 1e-10)
    err = abs(rho - lam_true[0])
    print(f"[6.4.4 precond-cg] λ₁ = {rho:.6f}，误差 = {err:.2e}，iters = {it}")
    assert err < 1e-6

    # ---- §6.5.1 PGS ----
    K = np.linalg.inv(B)                        # 预优取 B⁻¹（理想预优）
    K = 0.5 * (K + K.T)
    X0 = rng.standard_normal((n, 3))
    lam, X, it = pgs(A, B, K, X0, 1e-10)
    err = np.max(np.abs(lam - lam_true[:3]))
    orth = np.linalg.norm(X.T @ (B @ X) - np.eye(3))
    print(f"[6.5.1 pgs]       λ₁₋₃ 误差 = {err:.2e}，B 正交性 = {orth:.2e}，iters = {it}")
    assert err < 1e-6 and orth < 1e-8

    print("-" * 70)
    print("第六讲全部自测通过 ✔")
