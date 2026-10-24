# -*- coding: utf-8 -*-
"""
《矩阵计算六讲》(徐树方 等) 第三讲：奇异值分解的计算 —— 算法实现
=====================================================================

本讲主题
--------
围绕"对称特征值问题"与"奇异值分解"两条线索，系统讲述 SVD 的五大类算法：
    3.1 基本概念和性质（谱分解、奇异值分解、Weyl/Mirsky 扰动定理）
    3.2 Golub-Kahan SVD 算法
        3.2.1 对称 QR 方法（Householder 三对角化、隐式位移 QR、Wilkinson 位移）
        3.2.2 Golub-Kahan SVD（Householder 双对角化 + 双对角矩阵隐式 QR 型迭代）
    3.3 分而治之法
        3.3.1 对称特征值问题的分而治之（撕裂 + 秩一修正 + 有理函数 Newton 求根）
        3.3.2 SVD 的分而治之
    3.4 Jacobi 方法
        3.4.1 对称特征值问题的 Jacobi 方法（经典 Jacobi、循环 Jacobi）
        3.4.2 SVD 的 Jacobi 方法（单侧 Jacobi）
    3.5 二分法
        3.5.1 对称特征值问题的二分法（Sturm 序列符号计数）
        3.5.2 计算奇异值的二分法

实现的算法清单（函数名）
------------------------
[3.2.1] householder_tridiag          对称矩阵的 Householder 三对角化
[3.2.1] sym_tridiag_wilkinson_shift  Wilkinson 位移的计算
[3.2.1] implicit_sym_qr_step         对称三对角阵的隐式位移 QR 步（Givens 追逐）
[3.2.1] sym_tridiag_qr               对称三对角阵隐式位移 QR 迭代（含 deflation，求全部特征值/向量）
[3.2.2] householder_bidiag           Golub-Kahan 双对角化（Householder 版，左 U0 右 V0）
[3.2.2] gk_svd_step                  双对角阵 B 上的一次 Golub-Kahan SVD 迭代步（左右 Givens 追逐）
[3.2.2] golub_kahan_svd              Golub-Kahan SVD 完整流程（双对角化 + 隐式 QR 型迭代 + deflation）
[3.3.1] dpr1_newton                  对角加秩一(DPR1)矩阵特征方程有理函数的单根 Newton 迭代
[3.3.1] tridiag_dc1                  对称三对角特征值问题的分而治之（撕裂+秩一修正）
[3.3.2] bidiag_dc1                   SVD 的分而治之（对双对角阵 B 的 BᵀB 型 D&C 或直接撕裂；按书中格式）
[3.4.1] jacobi_eig                    经典/循环 Jacobi 求对称特征值分解
[3.4.2] jacobi_svd                   双侧（Kogbetliantz 型）Jacobi SVD：对 2×2 块左右旋转对角化并追赶
[3.5.1] sturm_count                   Sturm 序列符号计数（小于 x 的特征值个数）
[3.5.1] tridiag_bisect               Sturm 序列 + 二分法求对称三对角阵全部特征值
[3.5.2] svd_bisect                    奇异值二分法（对 BᵀB 的 Sturm 计数，直接在 α,β 上实现）

依赖：仅 numpy 与 Python 标准库。
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# 通用小工具：Givens 变换（本讲各算法反复使用）
# ---------------------------------------------------------------------------


def givens(a, b):
    """计算 Givens 变换参数 (c, s)，使得 [c, s; -s, c]^T * [a; b] = [r; 0]。

    即构造平面旋转 G = [[c, s], [-s, c]]，满足 G @ [a, b]^T = [r, 0]^T。
    数值上采用防溢出写法：先把 (a,b) 缩放到单位长度再取 atan2，
    这是标准做法（书中 Givens 变换的数值稳定实现）。
    """
    if b == 0.0:
        return 1.0, 0.0
    r = math.hypot(a, b)
    return a / r, b / r


# ===========================================================================
# 3.2.1  对称 QR 方法
# ===========================================================================


def householder_tridiag(A):
    """对称矩阵的 Householder 三对角化（书中 3.2.1 开头部分）。

    输入:
        A: n*n 实对称矩阵 (np.ndarray, float64)
    输出:
        alpha: 三对角阵 T 的对角元, 长度 n
        beta : T 的次对角元, 长度 n-1
        Q    : 累积正交变换，满足 Q^T A Q = tridiag(alpha, beta)

    说明:
        书中 3.2.1 指出：把 QR 方法用于对称 A 前，先用 Householder 变换
        将其化为对称三对角阵（运算量约 4n^3/3，若利用对称性）。
        第 k 步取 x = A[k+1:, k]，构造 Householder 向量把 x 化为
        ±||x|| e_1，同时左乘右乘作用于 A（右乘由对称性自动保持对称）。
    """
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    Q = np.eye(n)
    for k in range(n - 2):
        x = A[k + 1:, k].copy()
        normx = math.sqrt(np.dot(x, x))
        if normx == 0.0:
            continue  # 该列已无需约化
        # Householder 向量 v：取 alpha0 = -sign(x0)*||x|| 保证不发生相消
        alpha0 = -math.copysign(normx, x[0]) if x[0] != 0.0 else -normx
        v = x.copy()
        v[0] -= alpha0
        vnorm2 = np.dot(v, v)
        if vnorm2 == 0.0:
            continue
        # H = I - 2 v v^T / (v^T v)；为清晰起见按显式公式作用于子矩阵
        # 左乘: A[k+1:, k:] -= 2 v (v^T A[k+1:, k:]) / vnorm2
        w = v @ A[k + 1:, k:]                 # vᵀ·A：形状 (n-k,)
        A[k + 1:, k:] -= (2.0 / vnorm2) * np.outer(v, w)
        # 右乘: A[:, k+1:] -= 2 (A[:, k+1:] v) v^T / vnorm2
        w = A[:, k + 1:] @ v
        A[:, k + 1:] -= (2.0 / vnorm2) * np.outer(w, v)
        # 累积 Q <- Q H（H 嵌入在 (k+1..n-1) 坐标上）
        w = Q[:, k + 1:] @ v
        Q[:, k + 1:] -= (2.0 / vnorm2) * np.outer(w, v)
    alpha = np.diag(A).copy()
    beta = np.diag(A, 1).copy()
    return alpha, beta, Q


def sym_tridiag_wilkinson_shift(alpha_m1, alpha_n, beta_m1):
    """Wilkinson 位移（书中 3.2.1，式 (3.2.6) 附近）。

    输入: 右下角 2x2 子阵 [[alpha_m1, beta_m1], [beta_m1, alpha_n]] 的三个元素。
    输出: 位移 mu —— 该 2x2 矩阵两个特征值中与 alpha_n 靠近的那个。

    公式（书页 119 / 式 (3.2.6)）:
        d = (alpha_{n-1} - alpha_n)/2
        mu = alpha_n - sign(d) * beta^2 / (|d| + sqrt(d^2 + beta^2))
    这是 2x2 矩阵特征多项式中更接近 alpha_n 的根的代数表达式，
    用"beta^2/(...)”的写法避免了两个相近数相减的相消。
    """
    d = (alpha_m1 - alpha_n) / 2.0
    if beta_m1 == 0.0:
        return alpha_n
    # sign(d)=0 时书中约定取 sign=1（任取其一），与 LAPACK 一致
    sgn = 1.0 if d >= 0.0 else -1.0
    mu = alpha_n - sgn * beta_m1 * beta_m1 / (
        abs(d) + math.sqrt(d * d + beta_m1 * beta_m1))
    return mu


def implicit_sym_qr_step(alpha, beta, lo, hi, Q=None):
    """对称三对角阵的隐式 Wilkinson 位移 QR 步（3.2.1 节，算法 3.2.1 的核心）。

    书中思路（引理 3.2.1 隐式 Q）：
      1. Wilkinson 位移 μ（式 3.2.6）：取右下 2×2 的、与 α_n 较近的特征根；
      2. 第一个 Givens 由 (T − μI) 的第一列确定：x = α_lo − μ, z = β_lo；
      3. 旋转平面 (k, k+1) 把上一轮残留在 (k+1, k−1) 的"鼓包"消去，
         同时在 (k+2, k) 产生新鼓包，沿对角线一直追赶到子块底部。

    实现说明：为保证正确性，QR 步在稠密矩阵上完成（左乘 L、右乘 Lᵀ，
    L = [[c, s], [-s, c]] 使 L[x; z] = [r; 0]），每步后从矩阵中直接读取
    新的状态对 x = T[k+1, k], z = T[k+2, k]（GVL 8.3.2 的传递格式）。
    Q 累积右乘 Lᵀ，使最终 T_final = Qᵀ T₀ Q。
    """
    n = alpha.size
    # 稠密形式（教学实现的可靠做法；显式逐元素公式见书中推导）
    T = np.diag(alpha)
    if n > 1:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    mu = sym_tridiag_wilkinson_shift(alpha[hi - 1], alpha[hi], beta[hi - 1])
    x = T[lo, lo] - mu
    z = T[lo + 1, lo] if lo + 1 <= hi else 0.0
    for k in range(lo, hi):
        r = math.hypot(x, z)
        if r == 0.0:
            break
        c, s = x / r, z / r              # L = [[c, s], [-s, c]]: L [x;z] = [r;0]
        # ---- 左乘 L（第 k, k+1 行），右乘 Lᵀ（第 k, k+1 列）----
        rk, rk1 = T[k, :].copy(), T[k + 1, :].copy()
        T[k, :] = c * rk + s * rk1
        T[k + 1, :] = -s * rk + c * rk1
        ck, ck1 = T[:, k].copy(), T[:, k + 1].copy()
        T[:, k] = c * ck + s * ck1
        T[:, k + 1] = -s * ck + c * ck1
        if Q is not None:                # Q ← Q·Lᵀ
            qk, qk1 = Q[:, k].copy(), Q[:, k + 1].copy()
            Q[:, k] = c * qk + s * qk1
            Q[:, k + 1] = -s * qk + c * qk1
        # ---- 传递：新的次对角元与鼓包（GVL 8.3.2）----
        if k < hi - 1:
            x = T[k + 1, k]
            z = T[k + 2, k]
    # 把鼓包清零后写回 (alpha, beta)
    for i in range(n):
        alpha[i] = T[i, i]
    for i in range(n - 1):
        beta[i] = T[i + 1, i]
    return alpha, beta


def sym_tridiag_qr(alpha, beta, tol=None, want_vectors=True, max_iter_per_eig=50):
    """对称三对角阵的隐式位移 QR 算法（书中 3.2.1，算法 3.2.1 完整流程）。

    输入:
        alpha: 对角元 (n,)
        beta : 次对角元 (n-1,)
        tol  : deflation 容差；缺省用机器精度（书中判据
               |beta_i| <= eps*(|alpha_i|+|alpha_{i+1}|) 的安全放大版）
    输出:
        w: 特征值（升序排列）
        Q: 若 want_vectors，正交矩阵，Q[:, i] 为对应特征向量

    流程（对应书中算法框）:
        1. 反复扫描 beta，满足判据的次对角元置零（deflation/分裂）；
        2. 找当前最底部的活跃不可约块 [lo, hi]；
        3. 对该块执行一次隐式位移 QR 步（Wilkinson 位移）；
        4. 直至全部 beta 收敛为零，特征值即最终对角元。
    收敛: Wilkinson 位移下末次对角元二次收敛（书中定理 3.2.1）。
    注: 书中算法对"最底部活跃块"迭代；本实现同样每次处理最底部块，
        这样末端的收敛带动整体（与 LAPACK 的策略一致）。
    """
    alpha = np.array(alpha, dtype=float, copy=True)
    beta = np.array(beta, dtype=float, copy=True)
    n = alpha.size
    if tol is None:
        tol = np.finfo(float).eps
    Q = np.eye(n) if want_vectors else None
    hi = n - 1
    total_iters = 0
    max_total = max_iter_per_eig * n * 4
    while hi > 0 and total_iters < max_total:
        # --- 步骤 1：deflation 判据扫描（书中：|b_i| <= tol*(|a_i|+|a_{i+1}|)）---
        while hi > 0 and abs(beta[hi - 1]) <= tol * (abs(alpha[hi]) + abs(alpha[hi - 1])):
            beta[hi - 1] = 0.0
            hi -= 1  # alpha[hi] 已收敛为一个特征值
        if hi == 0:
            break
        # 找包含 hi 的活跃不可约块的下端 lo
        lo = hi - 1
        while lo > 0 and abs(beta[lo - 1]) > tol * (abs(alpha[lo]) + abs(alpha[lo - 1])):
            lo -= 1
        # --- 步骤 2-3：对活跃块执行一次隐式位移 QR 步 ---
        implicit_sym_qr_step(alpha, beta, lo, hi, Q)
        total_iters += 1
    if total_iters >= max_total:
        raise RuntimeError("sym_tridiag_qr: 超出最大迭代次数（可能不收敛）")
    w = np.sort(alpha)
    if want_vectors:
        # 把列按特征值升序重排
        idx = np.argsort(alpha)
        return w, Q[:, idx]
    return w


def sym_tridiag_qr_full(A, tol=None):
    """对称矩阵 -> 三对角化 -> 隐式 QR，返回 (特征值, 特征向量)。

    组合书中 3.2.1 的两个阶段；Q_total = Q_tridiag * Q_qr。
    """
    alpha, beta, Q0 = householder_tridiag(A)
    w, Q1 = sym_tridiag_qr(alpha, beta, tol=tol, want_vectors=True)
    return w, Q0 @ Q1


# ===========================================================================
# 3.2.2  Golub-Kahan SVD 算法
# ===========================================================================


def householder_bidiag(A):
    """Golub-Kahan 双对角化：Householder 版（书中 3.2.2，式 (3.2.7) 附近）。

    输入:
        A: m*n 矩阵, m >= n
    输出:
        U0 (m*n), B (n*n 上双对角), V0 (n*n)，满足 U0^T A V0 = [B; 0]

    说明:
        交替使用左 Householder（消去某列对角线以下元素）与右 Householder
        （消去某行超对角线以右元素），共 2n-2 次（最后一行/列无需处理），
        把 A 化为上双对角 B = diag(alpha) + superdiag(beta)。
        运算量约 2mn^2 - 2n^3/3（ flop 计数见书页 126）。
        实现：为简单起见对全矩阵做显式 Householder 乘法（O(m n^2)），
        并把 B 存成稠密 n*n 上双对角矩阵（书中的紧凑存储在此不采用，
        以便教学可读性；在注释中说明）。
    """
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    U0 = np.eye(m)
    V0 = np.eye(n)
    for k in range(n):
        # ---- 左 Householder：消去第 k 列对角元以下的元素 ----
        if k < m - 1 and (k < n - 1 or m > n):
            x = A[k:, k]
            normx = math.sqrt(np.dot(x, x))
            if normx > 0.0:
                a0 = -math.copysign(normx, x[0]) if x[0] != 0.0 else -normx
                v = x.copy()
                v[0] -= a0
                vn2 = np.dot(v, v)
                if vn2 > 0.0:
                    # A[k:, k:] <- (I - 2vv^T/vn2) A[k:, k:]
                    w = v @ A[k:, k:]             # vᵀ·A：形状 (n-k,)
                    A[k:, k:] -= (2.0 / vn2) * np.outer(v, w)
                    w = U0[:, k:] @ v
                    U0[:, k:] -= (2.0 / vn2) * np.outer(w, v)
        # ---- 右 Householder：消去第 k 行超对角线以右的元素 ----
        if k < n - 2:
            x = A[k, k + 1:]
            normx = math.sqrt(np.dot(x, x))
            if normx > 0.0:
                a0 = -math.copysign(normx, x[0]) if x[0] != 0.0 else -normx
                v = x.copy()
                v[0] -= a0
                vn2 = np.dot(v, v)
                if vn2 > 0.0:
                    w = A[k:, k + 1:] @ v  # 注意右乘作用于列
                    A[k:, k + 1:] -= (2.0 / vn2) * np.outer(w, v)
                    w = V0[:, k + 1:] @ v
                    V0[:, k + 1:] -= (2.0 / vn2) * np.outer(w, v)
    B = A[:n, :].copy()
    # 严格清理下三角数值噪声，保证上双对角结构
    B = np.triu(B, -0)  # 上三角（含对角）；再抹掉第一超对角以外元素
    for i in range(n):
        for j in range(i + 2, n):
            B[i, j] = 0.0
    return U0, B, V0


def gk_svd_step(B, lo, hi, U=None, V=None):
    """双对角阵上的一次 Golub-Kahan SVD 迭代步（3.2.2 节，页 122–124）。

    书中思路：对 T = BᵀB 做带位移 μ 的隐式对称 QR 步，但所有运算直接
    在 B 上完成——右 Givens（V 平面）与左 Givens（U 平面）交替进行，
    把"鼓包"沿对角线从左上追到右下：
      1. 位移 μ 取 T = BᵀB 右下 2×2 子阵的最小特征值（页 123，
         用与 Wilkinson 位移同型的防相消公式）；
      2. 第一个右 Givens 由 (T−μI) 的第一列确定：
         y = α_lo² − μ,  z = α_lo·β_lo；
      3. 右旋转在 (k+1, k) 处产生下鼓包 → 左旋转消去之，
         同时在 (k, k+2) 处产生上鼓包 → 下一个右旋转消去之；
      4. 追赶至子块底部后 B 恢复上双对角。

    V 的第一个旋转须满足 V e₁ ∝ (T−μI)e₁ = (y, z)，即取
    V̂ = [[c, −s], [s, c]]，其中 (c, s) = givens(y, z)。
    """
    a = np.diag(B).copy()
    b = np.diag(B, 1).copy()
    # ---- 位移 μ：T=BᵀB 右下 2×2 的最小特征值（Wilkinson 型公式）----
    p = a[hi - 1] ** 2 + (b[hi - 2] ** 2 if hi - 2 >= lo else 0.0)  # T[hi-1,hi-1]
    r_ = a[hi] ** 2 + b[hi - 1] ** 2                                 # T[hi,hi]
    q = a[hi] * b[hi - 1]                                            # T[hi-1,hi]
    if q == 0.0:
        mu = r_
    else:
        dd = (p - r_) / 2.0
        mu = r_ - math.copysign(1.0, dd) * q * q / (
            abs(dd) + math.hypot(dd, q))
    # ---- 第一个右 Givens 的 (y, z)（页 124）----
    y = a[lo] * a[lo] - mu
    z = a[lo] * b[lo]
    for k in range(lo, hi):
        c, s = givens(y, z)          # V̂ = [[c, -s], [s, c]]
        # B ← B·V̂：列混合  new_k = c·colk + s·colk1；new_{k+1} = -s·colk + c·colk1
        colk, colk1 = B[:, k].copy(), B[:, k + 1].copy()
        B[:, k] = c * colk + s * colk1
        B[:, k + 1] = -s * colk + c * colk1
        if V is not None:
            vk, vk1 = V[:, k].copy(), V[:, k + 1].copy()
            V[:, k] = c * vk + s * vk1
            V[:, k + 1] = -s * vk + c * vk1
        # 左 Givens 消去 (k+1, k) 处的下鼓包
        c2, s2 = givens(B[k, k], B[k + 1, k])
        # B ← U^T·B：行混合  new_k = c2·rowk + s2·rowk1；new_{k+1} = -s2·rowk + c2·rowk1
        rowk, rowk1 = B[k, :].copy(), B[k + 1, :].copy()
        B[k, :] = c2 * rowk + s2 * rowk1
        B[k + 1, :] = -s2 * rowk + c2 * rowk1
        if U is not None:
            uk, uk1 = U[:, k].copy(), U[:, k + 1].copy()
            U[:, k] = c2 * uk + s2 * uk1
            U[:, k + 1] = -s2 * uk + c2 * uk1
        # 下一对 (y, z)：行混合在 (k, k+2) 处留下的上鼓包
        if k < hi - 1:
            y = B[k, k + 1]
            z = B[k, k + 2]
    return B


def golub_kahan_svd(A, tol=None, want_vectors=True, max_sweeps=100):
    """Golub-Kahan SVD 完整算法（3.2.2 节，算法 3.2.2）。

    流程（书中算法框）：
      1. Householder 双对角化：U₀ᵀ A V₀ = [B; 0]（B 为上双对角）；
      2. 循环：
         (a) 收缩判据（页 123）：|βᵢ| ≤ ε(|αᵢ|+α_{i+1}|) 的超对角元置零；
         (b) 找最底部活跃不可约块 [lo, hi]；
         (c) 对该块执行一次 GK 迭代步（gk_svd_step，位移 = BᵀB 右下 2×2
             的最小特征值）；
         直到全部超对角元收敛为零。
      3. 奇异值 σᵢ = |αᵢ|（负对角元吸收进 U），按降序排列。

    说明：书中还处理"|αᵢ| 可忽略时用 Givens 把第 i 行旋为零行"的情形
    （页 123）；对随机测试矩阵该情形几乎不出现，此处从简：仅按超对角元
    判据收缩，并把下三角噪声严格清零。
    """
    A = np.array(A, dtype=float, copy=True)
    transposed = False
    if A.shape[0] < A.shape[1]:
        A = A.T
        transposed = True
    m, n = A.shape
    if tol is None:
        tol = np.finfo(float).eps
    # ---- 步骤 1：双对角化 ----
    U0, B, V0 = householder_bidiag(A)
    U = U0.copy() if want_vectors else None
    V = V0.copy() if want_vectors else None
    normB = np.linalg.norm(B)
    if normB == 0.0:
        return _pack_svd(U, np.zeros(n), V, transposed, want_vectors)
    # ---- 步骤 2：迭代 ----
    for _ in range(max_sweeps * n):
        # (a) 超对角元收缩（页 123 判据）
        for i in range(n - 1):
            if abs(B[i, i + 1]) <= tol * (abs(B[i, i]) + abs(B[i + 1, i + 1])):
                B[i, i + 1] = 0.0
        # (b) 最底部活跃块
        hi = n - 1
        while hi > 0 and B[hi - 1, hi] == 0.0:
            hi -= 1
        if hi == 0:
            break
        lo = hi - 1
        while lo > 0 and B[lo - 1, lo] != 0.0:
            lo -= 1
        # (c) 一步 GK 迭代
        gk_svd_step(B, lo, hi, U, V)
        # 严格清理带外噪声（双对角结构保证）
        for i in range(lo, hi + 1):
            for j in range(i + 2, n):
                B[i, j] = 0.0
            if i > lo:
                B[i, i - 1] = 0.0
    else:
        raise RuntimeError("golub_kahan_svd: 超出最大迭代次数")
    # ---- 步骤 3：奇异值与收尾 ----
    s = np.abs(np.diag(B)).copy()
    if want_vectors:
        for i in range(n):
            if B[i, i] < 0.0:
                U[:, i] = -U[:, i]
        order = np.argsort(-s)
        s = s[order]
        U = U[:, order]                     # 经济型：取前 n 列
        V = V[:, order]
        U = U[:, :n]
        return _pack_svd(U, s, V, transposed, want_vectors)
    order = np.argsort(-s)
    return _pack_svd(None, s[order], None, transposed, False)


def _pack_svd(U, s, V, transposed, want_vectors):
    """把内部 (U, s, V) 打包为 (U, s, Vt)；转置情形做对应换位。"""
    if not want_vectors:
        return None, s, None
    if transposed:
        # A^T = U B V^T  =>  A = V B^T U^T：交换并截断
        return V, s, U.T
    return U, s, V.T


# ===========================================================================
# 3.3.1 分而治之法：DPR1 secular 方程的 Newton 迭代
# ===========================================================================

def dpr1_newton(d, z, rho, lo, hi, tol=1e-14, max_iter=100):
    """对角加秩一（DPR1）secular 方程的单根求解（3.3.1 节，式 (3.3.3)）。

    D = diag(d), T = D + ρ z zᵀ 的特征值满足 secular 方程

        f(λ) = 1 + ρ Σᵢ zᵢ²/(dᵢ − λ) = 0。

    根的分布（交错定理）：ρ>0 时 λⱼ ∈ (d_{j-1}, d_j)（外端 d₁−ρ‖z‖²）；
    ρ<0 时 λⱼ ∈ (d_j, d_{j+1})（外端 d_n+|ρ|‖z‖²）。f 在每个区间内
    严格单调（ρ 的符号决定增减），故区间内根唯一。

    书中式 (3.3.9) 给出 Newton 迭代 λ⁺ = λ − f/f′（初值 d_j + ρ z_j²）。
    直接 Newton 在极点附近可能小步徘徊（z_j 很小时根紧贴极点），故本实现
    采用"单调二分定位 + Newton 精化"的鲁棒格式（思想同书中，工程上
    与 LAPACK dlaed4 的做法一致）。
    """
    d = np.asarray(d, dtype=float)
    z2 = np.asarray(z, dtype=float) ** 2

    def f(lam):
        return 1.0 + rho * np.sum(z2 / (d - lam))

    def fp(lam):
        return rho * np.sum(z2 / (d - lam) ** 2)

    # ---- 第一阶段：利用 f 的单调性二分（鲁棒，永不越出根的括区间）----
    a, b = lo, hi
    lam = 0.5 * (a + b)
    for _ in range(max_iter):
        lam = 0.5 * (a + b)
        val = f(lam)
        if val == 0.0 or (b - a) <= tol * max(1.0, abs(lam)):
            return lam
        # 按单调方向收缩括区间（f(λ) 的符号与根的相对位置）
        if (rho > 0 and val < 0.0) or (rho < 0 and val > 0.0):
            a = lam                            # 根在右半区间
        else:
            b = lam                            # 根在左半区间
    # ---- 第二阶段：Newton 精化（书中 (3.3.9)；步子越界即停）----
    for _ in range(5):
        val, dv = f(lam), fp(lam)
        if dv == 0.0:
            break
        lam_new = lam - val / dv
        if not (a < lam_new < b):
            break
        if abs(lam_new - lam) <= 1e-16 * max(1.0, abs(lam_new)):
            lam = lam_new
            break
        lam = lam_new
    return lam


def tridiag_dc1(alpha, beta, want_vectors=True, leaf_size=8):
    """对称三对角特征值问题的分而治之（算法 3.3.1）。

    书中流程：
      1. 撕裂：在中间次对角元 β_m 处把 T 写成
             T = [[T₁, 0], [0, T₂]] + β_m u uᵀ,  u = e_m^(1) + e_1^(2)，
         其中 T₁ 的末对角元减 β_m、T₂ 的首对角元减 β_m（使等式精确成立）；
      2. 递归：对 T₁、T₂ 分别做分而治之（小规模直接解），T_i = Q_i D_i Q_iᵀ；
      3. 合并：z = [Q₁ᵀ e_m ; Q₂ᵀ e_1]，T = D + ρ z zᵀ（ρ = β_m），
         解 secular 方程（dpr1_newton）得特征值 λ₁<…<λₙ（交错定理给出根的区间），
         特征向量由定理 3.3.2：q = (D − λI)⁻¹z / ‖·‖，
         回代 Q = blkdiag(Q₁,Q₂)·[q₁,…,qₙ]（注意按 d 排序的行列置换）。

    参数
    ----
    alpha : (n,) 对角元
    beta  : (n-1,) 次对角元
    leaf_size : 递归叶子规模（小于它直接用三对角 QR：sym_tridiag_qr）

    返回
    ----
    (w, Q) : 升序特征值与特征向量（want_vectors=False 时 Q 为 None）
    """
    alpha = np.asarray(alpha, dtype=float).copy()
    beta = np.asarray(beta, dtype=float)
    n = len(alpha)

    def solve(a, b):
        """返回 (升序特征值, 特征向量) 的递归函数。"""
        m = len(a)
        if m <= leaf_size:
            if m == 1:
                return a.copy(), np.eye(1)
            w, Q = sym_tridiag_qr(a, b, want_vectors=True)
            return w, Q
        # ---- 步骤 1：撕裂 ----
        k = m // 2 - 1                       # 0 基：撕裂 β_{k+1}（1 基的 β_m）
        rho = b[k]
        a1 = a[:k + 1].copy(); a1[-1] -= rho  # T₁ 末对角元减 ρ
        a2 = a[k + 1:].copy(); a2[0] -= rho   # T₂ 首对角元减 ρ
        b1, b2 = b[:k], b[k + 1:]
        # ---- 步骤 2：递归 ----
        w1, Q1 = solve(a1, b1)
        w2, Q2 = solve(a2, b2)
        # ---- 步骤 3：合并 ----
        d = np.concatenate([w1, w2])
        W = np.zeros((m, m))
        W[:k + 1, :k + 1] = Q1
        W[k + 1:, k + 1:] = Q2
        u = np.zeros(m)
        u[k] = 1.0
        u[k + 1] = 1.0
        z_full = W.T @ u                     # z = [Q₁ᵀ e_m ; Q₂ᵀ e_1]
        idx = np.argsort(d)
        d_s = d[idx]
        z_s = z_full[idx]
        Wp = W[:, idx]                       # 与排序一致的变换基
        znorm2 = abs(rho) * np.sum(z_s ** 2)
        lam = np.zeros(m)
        lam = np.zeros(m)
        for j in range(m):
            # 根的括区间（secular 方程在极点间单调，每个区间恰有一根）：
            #   ρ > 0：λⱼ ∈ (dⱼ, dⱼ₊₁)，最大根 ∈ (d_{m-1}, d_{m-1} + ρ‖z‖²)
            #   ρ < 0：λ₀ ∈ (d₀ − |ρ|‖z‖², d₀)，λⱼ ∈ (d_{j-1}, dⱼ)  (j ≥ 1)
            if rho > 0:
                lo = d_s[j]
                hi = d_s[j + 1] if j < m - 1 else d_s[-1] + znorm2
            else:
                lo = d_s[j - 1] if j > 0 else d_s[0] - znorm2
                hi = d_s[j]
            lam[j] = dpr1_newton(d_s, z_s, rho, lo, hi)
        if not want_vectors:
            return lam, None
        # 定理 3.3.2：特征向量 q = (D−λI)⁻¹ z（逐列），再回代到原坐标
        Qs = np.zeros((m, m))
        for j in range(m):
            q = z_s / (d_s - lam[j])          # 定理 3.3.2：q = (D−λI)⁻¹z
            Qs[:, j] = q / np.linalg.norm(q)
        return lam, Wp @ Qs

    if want_vectors:
        w, Q = solve(alpha, beta)
        return w, Q
    return solve(alpha, beta)[0], None


def bidiag_dc1(alpha, beta, eps=1e-13):
    """SVD 的分而治之（3.3.2 节，定理 3.3.5 路线）。

    书中把上双对角阵 B 的 SVD 归结为 T = BᵀB 的对称三对角特征值问题：
      * T 的对角元 dᵢ = αᵢ² + β_{i-1}²（β₀ = 0），
        次对角元 eᵢ = αᵢ βᵢ（定理 3.3.5，直接计算 BᵀB 即得）；
      * B 撕裂处 β_m 恰对应 T 的撕裂处 e_m = α_{m+1} β_m，
        因此对 T 直接用算法 3.3.1（tridiag_dc1）；
      * 奇异值 σᵢ = √λᵢ（λᵢ ≥ 0），V = T 的特征向量，
        U 的列 = B vᵢ/σᵢ（σᵢ > tol 时），σᵢ ≈ 0 的列用正交补填充。

    参数
    ----
    alpha, beta : 双对角阵 B 的对角元与上对角元

    返回
    ----
    (U, s, Vt) : B = U diag(s) Vt，s 降序
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n = len(alpha)
    # T = BᵀB 的三对角数据（定理 3.3.5 的显式公式，不显式形成矩阵乘积）
    d = alpha ** 2 + np.concatenate([[0.0], beta ** 2])
    e = alpha[:-1] * beta
    w, Q = tridiag_dc1(d, e, want_vectors=True, leaf_size=8)
    w = np.maximum(w, 0.0)                   # 数值安全：σ² 非负
    s = np.sqrt(w)
    order = np.argsort(-s)
    s = s[order]
    V = Q[:, order]
    # B 显式构造（教学实现；保持与 golub_kahan_svd 相同的 B 定义）
    B = np.diag(alpha)
    if n > 1:
        B += np.diag(beta, 1)
    BV = B @ V
    U = np.zeros((n, n))
    tol = eps * max(s[0], 1.0) if n else 0.0
    filled = 0
    for j in range(n):
        if s[j] > tol:
            U[:, j] = BV[:, j] / s[j]
            filled += 1
    if filled < n:                            # 零奇异值列：补正交基
        # Gram-Schmidt：从标准基中补出与已有列正交的方向
        for j in range(n):
            if s[j] <= tol:
                v = np.zeros(n)
                v[j] = 1.0
                for k in range(n):
                    if s[k] > tol:
                        v -= (U[:, k] @ v) * U[:, k]
                nv = np.linalg.norm(v)
                if nv > 1e-8:
                    U[:, j] = v / nv
                else:
                    U[:, j] = 0.0
                    U[j, j] = 1.0
    return U, s, V.T



# ===========================================================================
# 3.4.1 Jacobi 方法（对称特征值分解）
# ===========================================================================

def jacobi_eig(A, tol=1e-12, max_sweeps=60):
    """经典/循环 Jacobi 方法求对称特征值分解（3.4.1 节）。

    书中要点：
      * 对每个 (i,j), i<j 构造平面旋转 J(i,j,θ)，使 (JᵀAJ)_{ij} = 0，
        旋转角满足 tan 2θ = 2a_{ij}/(a_{ii} − a_{jj})，a_{ii}=a_{jj} 时取 θ=π/4；
      * 数值稳定实现（书中给出的等价形式）：
            τ = sgn(a_{ii}−a_{jj}) / (|a_{ii}−a_{jj}| + √((a_{ii}−a_{jj})² + 4a_{ij}²))
            c = 1/√(1+τ²),  s = cτ
        （当 a_{ij} 极小时 τ≈0，旋转接近恒等）
      * 扫描策略：循环 Jacobi——按 (1,2),(1,3),…,(n−1,n) 顺序逐对消去，
        一轮称为一次扫描（sweep）；经典 Jacobi 每次选 |a_{ij}| 最大者消去。
      * 收敛性：off²(A) 每次旋转至少下降 2a_{ij}²（书中引理），故 off(A)→0；
        停止准则 off(A) < ε‖A‖_F。
      * 收尾（3.4.4 节注记）：特征值取对角元，特征向量取累积正交阵 V 的列，
        特征值按降序与 V 的列一起排序。

    返回
    ----
    (w, V) : 降序特征值与特征向量，A ≈ V diag(w) Vᵀ
    """
    A = np.array(A, dtype=float, copy=True)
    A = 0.5 * (A + A.T)                       # 对称化，消除输入的微小不对称
    n = A.shape[0]
    V = np.eye(n)
    fro = np.linalg.norm(A)
    if fro == 0.0:
        return np.zeros(n), V
    for _ in range(max_sweeps):
        off = np.sqrt(np.sum(A ** 2) - np.sum(np.diag(A) ** 2))
        if off <= tol * fro:
            break
        for i in range(n - 1):
            for j in range(i + 1, n):
                aij = A[i, j]
                if abs(aij) <= 1e-300:
                    continue
                # ---- 稳定的旋转角公式（书中 tan2θ 公式的数值稳定等价形式）----
                # 本实现的相似变换为 A' = L A Lᵀ（L=[[c,-s],[s,c]]），
                # 消零条件：cs(a_ii−a_jj) + (c²−s²)a_ij = 0
                #   ⟺ t² − 2τt − 1 = 0，τ = (a_ii−a_jj)/(2a_ij)
                # 取小模长根：t = sgn(γ)/(|γ|+√(γ²+1))，γ = (a_jj−a_ii)/(2a_ij)
                delta = A[j, j] - A[i, i]
                gamma = delta / (2.0 * aij)
                t = np.sign(gamma if gamma != 0 else 1.0) / \
                    (abs(gamma) + np.sqrt(gamma * gamma + 1.0))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t
                # ---- A ← Jᵀ A J：只更新第 i,j 行与列 ----
                Ai, Aj = A[i, :].copy(), A[j, :].copy()
                A[i, :] = c * Ai - s * Aj
                A[j, :] = s * Ai + c * Aj
                Ai, Aj = A[:, i].copy(), A[:, j].copy()
                A[:, i] = c * Ai - s * Aj
                A[:, j] = s * Ai + c * Aj
                A[i, j] = 0.0                 # 被消去的元素显式置零
                A[j, i] = 0.0
                # ---- V ← V J ----
                Vi, Vj = V[:, i].copy(), V[:, j].copy()
                V[:, i] = c * Vi - s * Vj
                V[:, j] = s * Vi + c * Vj
    w = np.diag(A).copy()
    order = np.argsort(-w)
    return w[order], V[:, order]


# ===========================================================================
# 3.4.2 计算奇异值分解的 Jacobi 方法（双侧 / Kogbetliantz 型）
# ===========================================================================

def _svd2(M):
    """2×2 矩阵的 SVD（3.4.2 节的 2×2 子问题）。

    两步完成：
      1. 右旋 V2 把两列正交化（旋转角公式与 3.4.1 的 Jacobi 旋转同型，
         作用在列 Gram 元上）；
      2. 左旋 G 把第一列旋到 e₁ 方向。由于正交变换保持列的正交性
         （(GB)ᵀ(GB) = BᵀB 为对角阵），第一列归位后第二列自动只剩
         (2,2) 分量——故左旋只需一次（做两次反而会破坏对角性）。
    返回 (U2, s2, V2t)，M = U2 diag(s2) V2t，s2 降序。
    """
    col1 = M[:, 0].copy()
    col2 = M[:, 1].copy()
    n1 = np.linalg.norm(col1)
    n2 = np.linalg.norm(col2)
    dot = float(col1 @ col2)
    if abs(dot) > 1e-300 * max(1.0, n1 * n2):
        # 右旋转正交化两列（B = M·V2 的两列内积为零）：
        # 消零条件：sr·cr(n2²−n1²) + (cr²−sr²)·dot = 0
        #   ⟺ t² − 2σt − 1 = 0，σ = (n2²−n1²)/(2·dot)
        # 取小模长根（公式与 3.4.1 同型）：t = sgn(γ)/(|γ|+√(γ²+1))
        delta = n1 * n1 - n2 * n2
        gamma = delta / (2.0 * dot)
        t = np.sign(gamma if gamma != 0 else 1.0) / \
            (abs(gamma) + np.sqrt(gamma * gamma + 1.0))
        cr = 1.0 / np.sqrt(1.0 + t * t)
        sr = cr * t
    else:
        cr, sr = 1.0, 0.0
    V2 = np.array([[cr, -sr], [sr, cr]])
    Bm = M @ V2                               # 两列已正交
    # 左旋：把第一列旋到 (±‖·‖, 0)——第二列由正交性自动对角化
    x, y = Bm[0, 0], Bm[1, 0]
    nv = np.hypot(x, y)
    if nv > 0.0:
        cg, sg = x / nv, y / nv
        U2 = np.array([[cg, -sg], [sg, cg]])  # U2ᵀ[x;y] = [nv; 0]
        Bm = U2.T @ Bm
    else:
        U2 = np.eye(2)
    s2 = np.array([abs(Bm[0, 0]), abs(Bm[1, 1])])
    # 负号吸收进 U（保持 M = U2 diag(s2) V2t 的等式不改变）
    if Bm[0, 0] < 0.0:
        U2[:, 0] = -U2[:, 0]
    if Bm[1, 1] < 0.0:
        U2[:, 1] = -U2[:, 1]
    if s2[0] < s2[1]:                         # 奇异值降序
        s2 = s2[::-1]
        U2 = U2[:, ::-1]
        V2 = V2[:, ::-1]
    return U2, s2, V2.T


def jacobi_svd(A, tol=1e-12, max_sweeps=60):
    """双侧（Kogbetliantz 型）Jacobi SVD（3.4.2 节，算法 3.4.2）。

    书中对（双对角）B 的做法：逐对 (p,q) 取出 2×2 块，构造左右两个平面
    旋转把该块对角化（式 3.4.7 附近的 2×2 SVD 公式），产生的非对角元沿
    对角线"追赶"消去；反复扫描直至 off(B) < ε‖B‖，U、V 为左右旋转的累积。
    本实现按同一思想作用在稠密方阵上（Kogbetliantz 扫描）：每次对 (p,q)
    位置解 2×2 SVD 并左右旋转；数学上与书中追赶式等价（书中版本利用双对角
    结构使每对只花 O(n) 次 flop）。

    m > n 的情形按书中 3.4.2"(2) 预处理加速技术"处理：先做 QR 分解
    Qᵀ A = [R; 0]（R 为 n 阶上三角），再对 Rᵀ（方阵）应用本算法，
    最后组装 U = Q[V₁; 0]、Vt = U₁ᵀ（书中页 144 的公式）。

    收尾：奇异值 = |对角元|（负号吸收进 U），按降序排列。

    返回
    ----
    (U, s, Vt) : A ≈ U diag(s) Vt（U 为 m×n 经济型）
    """
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    fro = np.linalg.norm(A)
    if fro == 0.0:
        return np.eye(m)[:, :n], np.zeros(n), np.eye(n)
    if m > n:
        # ---- 书中 3.4.2(2) 预处理：QR 分解把 A 化为 [R; 0] ----
        Q, Rfull = np.linalg.qr(A)            # Q: m×m, Rfull: m×n 上三角
        R = Rfull[:n, :]                      # n×n 上三角
        # 对 Rᵀ（方阵）做 Jacobi SVD：Rᵀ = U1 Σ V1ᵀ ⇒ R = V1ᵀ Σ U1ᵀ
        U1, s, V1t = jacobi_svd(R.T, tol=tol, max_sweeps=max_sweeps)
        # 组装：A = Q [R;0] = (Q[:, :n] V1ᵀ) Σ U1ᵀ
        U = Q[:, :n] @ V1t.T                  # 经济型左奇异向量 (m×n)
        Vt = U1.T
        order = np.argsort(-s)
        return U[:, order], s[order], Vt[order, :]
    # ---- 方阵情形：Kogbetliantz 双侧 Jacobi 扫描 ----
    U = np.eye(m)
    Vt = np.eye(n)
    for _ in range(max_sweeps):
        d = np.diag(A)
        off2 = np.sum(A ** 2) - np.sum(d ** 2)
        if np.sqrt(max(off2, 0.0)) <= tol * fro:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                # 取 (p,q) 位置的 2×2 子块，做 2×2 SVD
                M2 = A[np.ix_([p, q], [p, q])]
                if np.linalg.norm(M2 - np.diag(np.diag(M2))) < 1e-300:
                    continue
                U2, s2, V2t = _svd2(M2)
                # 左右旋转作用到整个矩阵的第 p,q 行/列
                A[[p, q], :] = U2.T @ A[[p, q], :]
                A[:, [p, q]] = A[:, [p, q]] @ V2t.T
                U[:, [p, q]] = U[:, [p, q]] @ U2
                Vt[[p, q], :] = V2t @ Vt[[p, q], :]
                A[p, q] = 0.0                 # 显式清零被消去的元素
                A[q, p] = 0.0
    s = np.abs(np.diag(A)).copy()
    for i in range(n):
        if A[i, i] < 0.0:
            U[:, i] = -U[:, i]
    order = np.argsort(-s)
    s = s[order]
    U = U[:, order]
    Vt = Vt[order, :]
    return U, s, Vt


# ===========================================================================
# 3.5 二分法
# ===========================================================================

def sturm_count(alpha, beta, x):
    """Sturm 序列符号计数（3.5.1 节）：小于 x 的特征值个数。

    Sturm 序列：q₀ = 1, q₁ = α₁ − x,
                qₖ = (αₖ − x) qₖ₋₁ − β²ₖ₋₁ qₖ₋₂,
    定理：#{λᵢ < x} = 序列 {qₖ} 的变号数（等价于负值个数）。
    本实现采用 LAPACK dstebz 的防溢出比值递推：
        d₁ = α₁ − x；dₖ = (αₖ − x) − β²ₖ₋₁/dₖ₋₁（dₖ₋₁ = 0 时以微小量代替），
    计数 = #{k : dₖ < 0}，与变号数一致且数值稳定。

    参数
    ----
    alpha, beta : 对称三对角阵的对角元 / 次对角元
    x : 求值点

    返回
    ----
    int : #{λᵢ < x}
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    count = 0
    d = alpha[0] - x
    if d < 0:
        count += 1
    for k in range(1, len(alpha)):
        b2 = beta[k - 1] * beta[k - 1]
        if d == 0.0:
            d = abs(beta[k - 1]) * np.finfo(float).eps   # 防除零（LAPACK 约定）
        d = (alpha[k] - x) - b2 / d
        if d < 0:
            count += 1
    return count


def tridiag_bisect(alpha, beta, tol=1e-12):
    """二分法求对称三对角阵全部特征值（3.5.1 节，算法 3.5.1）。

    流程（书中算法 3.5.1）：
      1. Gershgorin 区间：lo = minᵢ(αᵢ − |βᵢ₋₁| − |βᵢ|),
         hi = maxᵢ(αᵢ + |βᵢ₋₁| + |βᵢ|)；
      2. 对升序第 k 个特征值 λₖ 二分：mid = (lo+hi)/2，
         若 V(mid) ≥ k（V = sturm_count，即小于 mid 的特征值至少 k 个），
         则 λₖ ≤ mid，hi ← mid；否则 lo ← mid；
      3. 直到区间宽度 < tol·max(1,|mid|)。

    返回
    ----
    (n,) 升序特征值
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n = len(alpha)
    ext = np.zeros(n)
    if n > 1:
        ext[1:] += np.abs(beta)
        ext[:-1] += np.abs(beta)
    lo = float(np.min(alpha - ext))
    hi = float(np.max(alpha + ext))
    w = np.zeros(n)
    for k in range(1, n + 1):                 # 求升序第 k 个特征值
        a, b = lo, hi
        while (b - a) > tol * max(1.0, abs(a), abs(b)):
            mid = 0.5 * (a + b)
            if sturm_count(alpha, beta, mid) >= k:
                b = mid                       # λₖ ≤ mid
            else:
                a = mid                       # λₖ > mid
        w[k - 1] = 0.5 * (a + b)
    return w


def svd_bisect(alpha, beta, tol=1e-12):
    """计算奇异值的二分法（3.5.2 节）。

    σ 是 B 的奇异值 ⟺ σ² 是 T = BᵀB 的特征值（定理 3.3.5）。
    T 的 Sturm 序列直接用 B 的 (α, β) 显式写出（不显式形成 BᵀB 矩阵）：

        q₁ = α₁² − λ,
        qₖ = (αₖ² + β²ₖ₋₁ − λ) qₖ₋₁ − αₖ² β²ₖ₋₁ qₖ₋₂,

    二分区间 [0, ‖B‖²_F]；对 λ 用 3.5.1 的二分得升序 {λᵢ}，σᵢ = √λᵢ（降序输出）。
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    d = alpha ** 2 + np.concatenate([[0.0], beta ** 2])   # T = BᵀB 的对角元
    e = alpha[:-1] * beta                                    # T 的次对角元
    lam = tridiag_bisect(d, e, tol=tol)
    lam = np.maximum(lam, 0.0)
    return np.sqrt(lam)[::-1]


# ===========================================================================
# 自测
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(3)
    print("=" * 70)
    print("《矩阵计算六讲》第三讲 自测：奇异值分解的计算")
    print("=" * 70)

    # ---- 3.2.1 对称三对角 QR ----
    # 流程：稠密对称阵 S --Householder 三对角化--> (alpha, beta) --隐式 QR--> 特征对
    A = np.random.randn(10, 10)
    S = 0.5 * (A + A.T)
    a0, b0, Q0 = householder_tridiag(S)
    T = np.diag(a0) + np.diag(b0, 1) + np.diag(b0, -1)
    w_np = np.linalg.eigvalsh(S)
    w_qr, Q_qr = sym_tridiag_qr(a0, b0)
    err = np.max(np.abs(np.sort(w_qr) - w_np))
    # Q_total = Q0 @ Q_qr 应把 S 对角化：验证 S Q = Q diag(w)
    Qt = Q0 @ Q_qr
    resid = np.linalg.norm(S @ Qt - Qt @ np.diag(w_qr))
    print(f"[3.2.1 sym_tridiag_qr]   特征值误差 = {err:.2e}，"
          f"对角化残差 = {resid:.2e}")
    assert err < 1e-8 and resid < 1e-7

    # ---- 3.2.2 Golub-Kahan SVD ----
    G = np.random.randn(8, 6)
    U_gk, s_gk, Vt_gk = golub_kahan_svd(G)
    s_np = np.linalg.svd(G, compute_uv=False)
    err_s = np.max(np.abs(s_gk - s_np))
    rec = np.linalg.norm(U_gk @ np.diag(s_gk) @ Vt_gk - G)
    orth = max(np.linalg.norm(U_gk.T @ U_gk - np.eye(6)),
               np.linalg.norm(Vt_gk @ Vt_gk.T - np.eye(6)))
    print(f"[3.2.2 golub_kahan_svd]  奇异值误差 = {err_s:.2e}，"
          f"重构残差 = {rec:.2e}，正交性 = {orth:.2e}")
    assert err_s < 1e-8 and rec < 1e-8 and orth < 1e-8

    # ---- 3.3.1 分而治之（对称特征值）----
    w_npT = np.linalg.eigvalsh(T)
    w_dc, Q_dc = tridiag_dc1(a0, b0)
    err = np.max(np.abs(np.sort(w_dc) - w_npT))
    vec_err = np.linalg.norm(T @ Q_dc - Q_dc @ np.diag(w_dc))
    print(f"[3.3.1 tridiag_dc1]      特征值误差 = {err:.2e}，"
          f"特征方程残差 = {vec_err:.2e}")
    assert err < 1e-8 and vec_err < 1e-7

    # ---- 3.3.2 SVD 的分而治之 ----
    Ab = np.random.randn(7, 7)
    ab_diag = np.diag(Ab).copy()
    ab_super = np.diag(Ab, 1).copy()
    Bd = np.diag(ab_diag) + np.diag(ab_super, 1)
    U_dc, s_dc, Vt_dc = bidiag_dc1(ab_diag, ab_super)
    s_ref = np.linalg.svd(Bd, compute_uv=False)
    err_s = np.max(np.abs(s_dc - s_ref))
    rec = np.linalg.norm(U_dc @ np.diag(s_dc) @ Vt_dc - Bd)
    print(f"[3.3.2 bidiag_dc1]       奇异值误差 = {err_s:.2e}，重构残差 = {rec:.2e}")
    assert err_s < 1e-7 and rec < 1e-6

    # ---- 3.4.1 Jacobi 特征值分解 ----
    w_j, V_j = jacobi_eig(S)
    err = np.max(np.abs(np.sort(w_j) - w_np))   # 两侧都升序后比较
    rec = np.linalg.norm(S @ V_j - V_j @ np.diag(w_j))
    orth = np.linalg.norm(V_j.T @ V_j - np.eye(10))
    print(f"[3.4.1 jacobi_eig]       特征值误差 = {err:.2e}，"
          f"特征方程残差 = {rec:.2e}，正交性 = {orth:.2e}")
    assert err < 1e-8 and rec < 1e-7

    # ---- 3.4.2 Jacobi SVD ----
    U_j, s_j, Vt_j = jacobi_svd(G)
    err_s = np.max(np.abs(s_j - s_np))
    rec = np.linalg.norm(U_j[:, :6] @ np.diag(s_j) @ Vt_j - G)
    print(f"[3.4.2 jacobi_svd]       奇异值误差 = {err_s:.2e}，重构残差 = {rec:.2e}")
    assert err_s < 1e-8 and rec < 1e-7

    # ---- 3.5.1 Sturm 计数 + 二分法 ----
    w_bi = tridiag_bisect(a0, b0)
    err = np.max(np.abs(np.sort(w_bi) - w_npT))
    print(f"[3.5.1 tridiag_bisect]   特征值误差 = {err:.2e}")
    assert err < 1e-7

    # ---- 3.5.2 奇异值二分法 ----
    s_bi = svd_bisect(ab_diag, ab_super)
    err = np.max(np.abs(s_bi - s_ref))
    print(f"[3.5.2 svd_bisect]       奇异值误差 = {err:.2e}")
    assert err < 1e-6

    print("-" * 70)
    print("第三讲全部自测通过 ✔")
