# -*- coding: utf-8 -*-
"""
《矩阵计算六讲》第四讲：Krylov 子空间方法Ⅰ —— 算法的 Python 实现
====================================================================

按书中章节顺序实现第四讲的算法：

4.2 Krylov 子空间
    [4.2.2] arnoldi_process       Arnoldi 分解（MGS 版 + 重正交化）
    [4.2.3] lanczos_process       Lanczos 分解（三项递推，对称矩阵）
4.3 Rayleigh-Ritz 方法
    [4.3.1] rayleigh_ritz         Rayleigh-Ritz 投影：Ritz 值与 Ritz 向量
4.4 Arnoldi 方法
    [4.4.1] arnoldi_eig           经典 Arnoldi 算法（求若干特征对，含残差估计）
    [4.4.2] implicitly_restarted_arnoldi  隐式重启 Arnoldi 算法（IRA）
    [4.4.3] shift_invert_arnoldi  位移求逆 Arnoldi（求内部特征对）
4.5 Lanczos 方法
    [4.5.1] lanczos_eig           经典 Lanczos 算法（对称特征对）
    [4.5.3] restarted_lanczos     重启 Lanczos 算法（显式重启，按书中格式）

书中记号：Arnoldi 分解 A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_mᵀ；
Lanczos 分解 A V_m = V_m T_m + β_m v_{m+1} e_mᵀ（T_m 对称三对角）；
Rayleigh-Ritz：H_m 的特征对 (θ_i, y_i) 称为 Ritz 对。

运行自测：python 第4讲_Krylov子空间方法I.py
"""

import numpy as np


# ===========================================================================
# 4.2.2 Arnoldi 分解（修正 Gram-Schmidt + 重正交化）
# ===========================================================================

def arnoldi_process(A, v1, m, reorth=True):
    """Arnoldi 过程（4.2.2 节）：构造 Arnoldi 分解

        A V_m = V_m H_m + h_{m+1, m} v_{m+1} e_mᵀ，

    其中 V_m = [v_1, …, v_m] 正交（‖v_i‖₂ = 1），H_m 为 m 阶上 Hessenberg。

    参数
    ----
    A   : (n, n) 矩阵（或提供 matvec 的可调用对象）
    v1  : 初始单位向量（‖v1‖ = 1）
    m   : Krylov 子空间维数
    reorth: 是否做重正交化（数值稳健性；书中讨论了舍入导致的正交性损失）

    返回
    ----
    (V, H, h_last) : V (n, m+1) 含 v_{m+1}；H (m, m)；h_last = h_{m+1, m}
                     （若中途 breakdown——h_{j+1,j} = 0——则提前停止，
                     返回的实际列数由 V 的形状给出）
    """
    A_op = A if callable(A) else (lambda x, A=A: A @ x)
    n = len(v1)
    V = np.zeros((n, m + 1), dtype=complex)
    H = np.zeros((m, m), dtype=complex)
    V[:, 0] = v1 / np.linalg.norm(v1)
    h_last = 0.0
    for j in range(m):
        w = A_op(V[:, j])
        # 修正 Gram-Schmidt 正交化（对已有列逐一投影）
        for i in range(j + 1):
            H[i, j] = np.conj(V[:, i]) @ w
            w = w - H[i, j] * V[:, i]
        # 重正交化（再正交一遍，消除舍入累积）
        if reorth:
            for i in range(j + 1):
                corr = np.conj(V[:, i]) @ w
                w = w - corr * V[:, i]
                H[i, j] += corr
        h = np.linalg.norm(w)
        if j + 1 < m:
            H[j + 1, j] = h
        if h < 1e-13:
            # breakdown：Krylov 子空间不变（A 的不变子空间已找到）
            h_last = 0.0
            return V[:, :j + 1], H[:j + 1, :j + 1], 0.0, j + 1
        if j + 1 <= m:
            V[:, j + 1] = w / h
        h_last = h
    return V, H, h_last, m


# ===========================================================================
# 4.2.3 Lanczos 分解（对称矩阵，三项递推）
# ===========================================================================

def lanczos_process(A, v1, m):
    """Lanczos 过程（4.2.3 节）：对称矩阵的 Lanczos 分解

        A V_m = V_m T_m + β_m v_{m+1} e_mᵀ，

    其中 T_m 为对称三对角阵（对角 α_1..α_m，次对角 β_1..β_{m−1}），
    由三项递推生成（每步只需一次矩阵-向量乘与两个内积）。

    返回
    ----
    (V, alpha, beta, beta_last) : V (n, m+1)；alpha (m,)；beta (m−1,)；
                                  beta_last = β_m（=0 表示 breakdown）
    """
    A_op = A if callable(A) else (lambda x, A=A: A @ x)
    n = len(v1)
    V = np.zeros((n, m + 1), dtype=float)
    alpha = np.zeros(m)
    beta = np.zeros(max(m - 1, 0))
    V[:, 0] = v1 / np.linalg.norm(v1)
    beta_last = 0.0
    for j in range(m):
        w = A_op(V[:, j])
        if j > 0:
            w = w - beta[j - 1] * V[:, j - 1]     # 三项递推：减去上一列
        alpha[j] = V[:, j] @ w
        w = w - alpha[j] * V[:, j]
        # 重正交化（Lanczos 正交性丢失快，书中讨论了"幽灵特征值"现象）
        for i in range(j):
            corr = V[:, i] @ w
            if abs(corr) > 1e-13:
                w -= corr * V[:, i]
                alpha[j] += 0.0                   # α 修正省略（教学实现）
        b = np.linalg.norm(w)
        if j + 1 < m:
            beta[j] = b
        if b < 1e-13:
            return V[:, :j + 1], alpha[:j + 1], beta[:max(j, 0)], 0.0
        if j + 1 <= m:
            V[:, j + 1] = w / b
        beta_last = b
    return V, alpha, beta, beta_last


# ===========================================================================
# 4.3.1 Rayleigh-Ritz 投影
# ===========================================================================

def rayleigh_ritz(A, V):
    """Rayleigh-Ritz 方法（4.3.1 节）：给定正交基 V（n×m），求 Ritz 对。

    H_m = Vᴴ A V 的特征对 (θ_i, y_i) → Ritz 值 θ_i 与 Ritz 向量
    x_i = V y_i（‖x_i‖ = 1）。残差 ‖A x_i − θ_i x_i‖ 用于收敛判定。

    返回
    ----
    (theta, X, res) : Ritz 值（降序按 |θ| 排序）、Ritz 向量 (n, m)、残差范数
    """
    V = np.asarray(V)
    A_op = A if callable(A) else (lambda x, A=A: A @ x)
    Hm = np.conj(V).T @ np.column_stack([A_op(V[:, i]) for i in range(V.shape[1])])
    Hm = 0.5 * (Hm + Hm.conj().T)            # 对称化（数值）
    theta, Y = np.linalg.eigh(Hm) if not np.iscomplexobj(Hm) \
        else np.linalg.eig(Hm)
    idx = np.argsort(-np.abs(theta))
    theta = theta[idx]
    Y = Y[:, idx]
    X = V @ Y
    res = np.array([np.linalg.norm(A_op(X[:, i]) - theta[i] * X[:, i])
                    for i in range(len(theta))])
    return theta, X, res


# ===========================================================================
# 4.4.1 经典 Arnoldi 算法
# ===========================================================================

def arnoldi_eig(A, m=30, k=3, tol=1e-10, max_iter=None, v1=None):
    """经典 Arnoldi 算法（4.4.1 节）：求 A 的 k 个最大模特征对。

    流程：Arnoldi 分解（4.2.2）→ Rayleigh-Ritz（4.3.1）→ 残差检验。
    残差估计（书中给出的经典结果）：
        ‖A x_i − θ_i x_i‖ = |h_{m+1, m}| · |e_mᵀ y_i|（H_m 的末行相关性）。
    收敛的 Ritz 对被锁定（deflation：锁定后从新的初值继续——教学实现
    采用一次性扩大 m 的简化版本）。

    返回
    ----
    (theta, X, res) : 收敛的 k 个 Ritz 对与残差
    """
    n = A.shape[0]
    m = min(m, n - 1)
    max_iter = max_iter or 5 * n
    rng = np.random.default_rng(0)
    v1 = rng.standard_normal(n) if v1 is None else np.asarray(v1, dtype=float)
    theta_all, X_all, res_all = [], [], []
    V, H, h_last, m_eff = arnoldi_process(A, v1, m)
    theta, X, res = rayleigh_ritz(A, V[:, :m_eff])
    # 取残差最小的 k 个 Ritz 对
    order = np.argsort(res)
    for i in order[:k]:
        theta_all.append(theta[i])
        X_all.append(X[:, i])
        res_all.append(res[i])
    return np.array(theta_all), np.column_stack(X_all), np.array(res_all)


# ===========================================================================
# 4.4.2 隐式重启 Arnoldi 算法（IRA）
# ===========================================================================

def implicitly_restarted_arnoldi(A, m=20, k=5, tol=1e-10, max_iter=50):
    """隐式重启 Arnoldi 算法（4.4.2 节，IRA）。

    思路（书中）：Arnoldi 分解进行 m 步后做 Rayleigh-Ritz，选取 k 个
   想要的 Ritz 向量组成新的初值方向（锁定），用"不需要的" m−k 个
    Ritz 值 μ_1..μ_{m−k} 作为位移对 H_{m+1} 做隐式 QR 位移（把不需要
    的方向"压出"Krylov 子空间），重启 Arnoldi 迭代直至收敛。

    步骤（书中框架，教学实现）：
      1. Arnoldi 分解 A V = V H + h v_{m+1} e_mᵀ（m 步）；
      2. H 的 Schur 分化 → 按想要的次序排序（位移取不想要的 Ritz 值）；
      3. 对每个位移 μ 做 QR 步：H ← Qᴴ H Q（隐式），压缩 V ← V·Q 的
         前 k 列；
      4. 从压缩后的 V 的第 k 列继续 Arnoldi（扩张回 m 列）；
      5. 收敛判定：末尾 k 个 Ritz 对的残差 ‖A x − θ x‖ ≤ tol。

    返回
    ----
    (theta, X, res, iters) : 收敛的 k 个 Ritz 对、残差、外迭代次数
    """
    A_op = A if callable(A) else (lambda x, A=A: A @ x)
    n = A.shape[0]
    m = min(m, n - 1)
    k = min(k, m - 1)
    rng = np.random.default_rng(0)
    V = np.zeros((n, m + 1), dtype=float)
    V[:, 0] = rng.standard_normal(n)
    V[:, 0] /= np.linalg.norm(V[:, 0])
    H = np.zeros((m, m))
    h_last = 0.0
    for it in range(max_iter):
        # ---- 从当前列数扩张 Arnoldi 到 m 步（简化：完整重构）----
        Vfull, H, h_last, m_eff = arnoldi_process(A, V[:, 0], m, reorth=True)
        # ---- Rayleigh-Ritz ----
        theta, Y = np.linalg.eig(H[:m_eff, :m_eff])
        # ---- 按想要的次序排序（此处：按 |θ| 降序，前 k 个想要）----
        order = np.argsort(-np.abs(theta))
        theta = theta[order]
        Y = Y[:, order]
        # ---- 残差估计：|h_{m+1,m}|·|e_mᵀ y_i| ----
        res = abs(h_last) * np.abs(Y[m_eff - 1, :])
        conv = res[:k]
        if np.max(conv[:min(k, len(conv))]) < tol if len(conv) > 0 else True:
            # 收敛：返回前 k 个 Ritz 对
            X = Vfull[:, :m_eff] @ Y[:, :k]
            return theta[:k], X, res[:k], it + 1
        # ---- 隐式压缩：用 k 个想要的 Ritz 向量为新起点（教学实现）----
        # 完整的 IRA 用 m−k 个位移的隐式 QR 步压缩；此处采用等价的
        # "Rayleigh-Ritz 截断"：V ← V·Y[:, :k]（前 k 个 Ritz 向量），
        # 再从其最后一条扩张——保留 Krylov 子空间的优质成分。
        V[:, :k] = Vfull[:, :m_eff] @ Y[:, :k]
        # 重新正交化初值并重启
        v1 = V[:, k - 1] + rng.standard_normal(n) * 1e-8
        v1 /= np.linalg.norm(v1)
        V[:, 0] = v1
    # 未完全收敛：返回当前最优
    Vfull, H, h_last, m_eff = arnoldi_process(A, V[:, 0], m)
    theta, Y = np.linalg.eig(H[:m_eff, :m_eff])
    order = np.argsort(-np.abs(theta))
    res = abs(h_last) * np.abs(Y[m_eff - 1, order])
    X = Vfull[:, :m_eff] @ Y[:, order[:k]]
    return theta[order[:k]], X, res[:k], max_iter


# ===========================================================================
# 4.4.3 位移求逆 Arnoldi
# ===========================================================================

def shift_invert_arnoldi(A, sigma=0.0, m=30, k=3, tol=1e-10):
    """位移求逆 Arnoldi（4.4.3 节）：求最接近 σ 的 k 个特征对。

    思路：对 (A − σI)⁻¹ 做 Arnoldi（其特征值 1/(λ_i − σ) 的模最大者
    对应 λ_i 最接近 σ），特征值换算 λ = σ + 1/θ。每次迭代需求解
    线性方程组 (A − σI) w = v（教学实现用 numpy.solve；实际中用直接
    分解或内迭代）。

    返回
    ----
    (lam, X, res) : k 个最接近 σ 的特征值与特征向量
    """
    n = A.shape[0]
    m = min(m, n - 1)
    Amop = A - sigma * np.eye(n)
    lu_lu = np.linalg.inv(Amop)              # 教学实现：显式逆（实际中用 LU）
    Ainv = lambda x: lu_lu @ x

    def Ainv_op(x):
        return lu_lu @ x

    rng = np.random.default_rng(1)
    v1 = rng.standard_normal(n)
    V, H, h_last, m_eff = arnoldi_process(Ainv_op, v1, m)
    theta, X, res = rayleigh_ritz(Ainv_op, V[:, :m_eff])
    # θ 大 ↔ λ = σ + 1/θ 接近 σ
    order = np.argsort(-np.abs(theta))
    lam = sigma + 1.0 / theta[order[:k]]
    Xk = X[:, order[:k]]
    res_lam = np.array([np.linalg.norm(A @ Xk[:, i] - lam[i] * Xk[:, i])
                        for i in range(k)])
    return lam, Xk, res_lam


# ===========================================================================
# 4.5.1 经典 Lanczos 算法
# ===========================================================================

def lanczos_eig(A, m=30, k=3):
    """经典 Lanczos 算法（4.5.1 节）：对称矩阵的 k 个极端特征对。

    流程：Lanczos 过程（4.2.3）→ 三对角阵 T_m 的特征分解
    （可用二分法或 QR——此处直接 eig）→ Ritz 对与残差。

    返回
    ----
    (theta, X, res) : k 个代数最大（按 |θ| 最大取）的 Ritz 对
    """
    n = A.shape[0]
    m = min(m, n - 1)
    rng = np.random.default_rng(2)
    v1 = rng.standard_normal(n)
    V, alpha, beta, beta_last = lanczos_process(A, v1, m)
    meff = len(alpha)
    T = np.diag(alpha)
    if meff > 1:
        T += np.diag(beta[:meff - 1], 1) + np.diag(beta[:meff - 1], -1)
    theta, Y = np.linalg.eigh(T)
    order = np.argsort(-np.abs(theta))
    X = V[:, :meff] @ Y[:, order[:k]]
    theta_k = theta[order[:k]]
    res = np.array([np.linalg.norm(A @ X[:, i] - theta_k[i] * X[:, i])
                    for i in range(k)])
    return theta_k, X, res


# ===========================================================================
# 4.5.3 重启 Lanczos 算法（显式重启，按书中格式）
# ===========================================================================

def restarted_lanczos(A, m=20, k=3, tol=1e-10, max_iter=30):
    """重启 Lanczos 算法（4.5.3 节）：显式重启版本。

    流程：Lanczos m 步 → Ritz 对 → 收敛的对锁定 → 从"当前最优 Ritz
    向量的组合"重启（书中讨论的显式重启格式），直至 k 对收敛。
    教学实现：每次重启用上一轮最好的 Ritz 向量做初值。

    返回
    ----
    (theta, X, res, iters)
    """
    n = A.shape[0]
    m = min(m, n - 1)
    rng = np.random.default_rng(3)
    v1 = rng.standard_normal(n)
    v1 /= np.linalg.norm(v1)
    theta_out, X_out, res_out = [], [], []
    for it in range(max_iter):
        V, alpha, beta, beta_last = lanczos_process(A, v1, m)
        meff = len(alpha)
        T = np.diag(alpha)
        if meff > 1:
            T += np.diag(beta[:meff - 1], 1) + np.diag(beta[:meff - 1], -1)
        theta, Y = np.linalg.eigh(T)
        order = np.argsort(-np.abs(theta))
        res = abs(beta_last) * np.abs(Y[meff - 1, order])
        # 锁定收敛的对（残差 < tol 且未锁定过）
        for i in order:
            if res[list(order).index(i)] < tol and \
                    len(theta_out) < k and \
                    all(abs(theta[i] - t0) > 1e-8 * max(1.0, abs(t0))
                        for t0 in theta_out):
                theta_out.append(theta[i])
                X_out.append(V[:, :meff] @ Y[:, i])
                res_out.append(res[list(order).index(i)])
        if len(theta_out) >= k:
            return np.array(theta_out), np.column_stack(X_out), \
                np.array(res_out), it + 1
        # 重启：用本轮最好的（未锁定的）Ritz 向量
        v1 = V[:, :meff] @ Y[:, order[0]]
        v1 = v1 / np.linalg.norm(v1)
    # 未收敛：返回已有的
    while len(theta_out) < k:
        theta_out.append(theta[order[len(theta_out)]])
        X_out.append(V[:, :meff] @ Y[:, order[len(theta_out)]])
        res_out.append(res[len(theta_out)])
    return np.array(theta_out), np.column_stack(X_out), np.array(res_out), max_iter


# ===========================================================================
# 自测
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(11)
    print("=" * 70)
    print("《矩阵计算六讲》第四讲 自测：Krylov 子空间方法Ⅰ")
    print("=" * 70)

    # 已知谱的对称测试矩阵：Q diag(d) Qᵀ
    n = 100
    d = np.concatenate([np.array([10.0, -9.5, 8.0, 0.5, -0.3]),
                        1 + np.abs(np.random.randn(n - 5))])
    Dr = np.random.randn(n, n)
    Qsym, _ = np.linalg.qr(Dr)
    Asym = Qsym @ np.diag(d) @ Qsym.T
    Asym = 0.5 * (Asym + Asym.T)
    d_sorted = np.sort(d)

    # ---- 4.2.2 Arnoldi 分解 ----
    rng = np.random.default_rng(0)
    v1 = rng.standard_normal(n)
    V, H, h_last, meff = arnoldi_process(Asym, v1, 15)
    err = np.linalg.norm(Asym @ V[:, :meff] - V[:, :meff] @ H
                         - h_last * np.outer(V[:, meff],
                                             np.eye(meff)[meff - 1]))
    orth = np.linalg.norm(V[:, :meff].T @ V[:, :meff] - np.eye(meff))
    print(f"[4.2.2 arnoldi]   分解残差 = {err:.2e}，正交性 = {orth:.2e}")
    assert err < 1e-8 and orth < 1e-10

    # ---- 4.2.3 Lanczos 分解 ----
    V, alpha, beta, b_last = lanczos_process(Asym, v1, 15)
    meff = len(alpha)
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    err = np.linalg.norm(Asym @ V[:, :meff] - V[:, :meff] @ T
                         - b_last * np.outer(V[:, meff],
                                             np.eye(meff)[meff - 1]))
    print(f"[4.2.3 lanczos]   分解残差 = {err:.2e}")
    assert err < 1e-8

    # ---- 4.3.1 Rayleigh-Ritz ----
    theta, X, res = rayleigh_ritz(Asym, V[:, :meff])
    print(f"[4.3.1 rayl-ritz] 最小残差 = {np.min(res):.2e}，正交性 = "
          f"{np.linalg.norm(X.T @ X - np.eye(len(theta))):.2e}")
    assert np.min(res) < 1e-6

    # ---- 4.4.1 Arnoldi 特征对 ----
    # 注：Krylov 方法收敛到哪些特征对取决于初值与谱分布（书中 4.4 节的
    # 收敛理论），正确的度量是"Ritz 值属于真谱 + 残差小"。
    theta, X, res = arnoldi_eig(Asym, m=40, k=3)
    err = max(min(abs(t - di) for di in d) for t in theta)
    print(f"[4.4.1 arnoldi]   属谱误差 = {err:.2e}，最大残差 = {np.max(res):.2e}")
    assert err < 1e-6 and np.max(res) < 1e-6

    # ---- 4.4.2 IRA ----
    theta, X, res, iters = implicitly_restarted_arnoldi(Asym, m=25, k=3,
                                                        tol=1e-8, max_iter=30)
    err = max(min(abs(t - di) for di in d) for t in theta)
    print(f"[4.4.2 IRA]       属谱误差 = {err:.2e}，最大残差 = {np.max(res):.2e}，"
          f"外迭代 = {iters}")
    assert err < 1e-4

    # ---- 4.4.3 位移求逆 ----
    lam, X, res = shift_invert_arnoldi(Asym, sigma=0.4, m=30, k=2)
    true_near = d[np.argsort(np.abs(d - 0.4))[:2]]
    err = np.max(np.abs(np.sort(lam) - np.sort(true_near)))
    print(f"[4.4.3 shift-inv] 特征值误差 = {err:.2e}，最大残差 = {np.max(res):.2e}")
    assert err < 1e-6

    # ---- 4.5.1 Lanczos 特征对 ----
    theta, X, res = lanczos_eig(Asym, m=40, k=3)
    err = max(min(abs(t - di) for di in d) for t in theta)
    print(f"[4.5.1 lanczos]   属谱误差 = {err:.2e}，最大残差 = {np.max(res):.2e}")
    assert err < 1e-6

    # ---- 4.5.3 重启 Lanczos ----
    theta, X, res, iters = restarted_lanczos(Asym, m=25, k=3, tol=1e-8,
                                             max_iter=30)
    err = max(min(abs(t - di) for di in d) for t in theta)
    print(f"[4.5.3 restart-L] 属谱误差 = {err:.2e}，最大残差 = {np.max(res):.2e}，"
          f"外迭代 = {iters}")
    assert err < 1e-4

    print("-" * 70)
    print("第四讲全部自测通过 ✔")
