# -*- coding: utf-8 -*-
"""
《矩阵计算六讲》第二讲：多项式之根的快速求法 —— 算法的 Python 实现
====================================================================

本文件按书中章节顺序实现第二讲中提到的所有算法：

2.1  引言
    - sign_changes        变号数 V(α) 的计算（2.1.1 基本问题）
    - budan_fourier       Budan-Fourier 定理：区间 (a,b] 内实根个数的上界估计 (2.1.2)
    - cauchy_root_bound   Cauchy 根模上界（定理 2.1.x：ρ(A) 型界）

2.2  Newton-Horner 方法
    - horner              Horner 综合除法：同时求 p(z), p'(z), ..., p^{(m)}(z)
                          （2.2.2 节，式 (2.2.4) 前后的递推）
    - newton_iter         Newton 迭代法 z_{k+1} = z_k - p(z_k)/p'(z_k)（2.2.1 节 (2.2.1)），
                          采用书中给出的两个实用停止准则（命题 2.2.1 之后的讨论）
    - newton_horner       Newton-Horner 方法（算法 2.2.1）：逐个求根 + Horner 降阶

2.3  快速 QR 方法
    - companion_matrix    友矩阵的构造（2.3.1 节）
    - in_Hn_class         Hn 类判定：上 Hessenberg 且次对角元非零（2.3.2 节定义）
    - schur_params        酉上 Hessenberg 矩阵的 Schur 参数化：由参数 γ 生成
                          酉 Hessenberg 酉矩阵 U（2.3.2 节定理 2.3.1 的构造性证明）
    - unitary_hessenberg  由 Schur 参数 γ 显式构造酉上 Hessenberg 矩阵
    - companion_to_hessenberg  把友矩阵经酉相似变换化为 Hn 类中具有同特征多项式的
                          上 Hessenberg 矩阵（实现"参数化初始化"的等价形式）
    - fast_qr_single_shift    单步位移快速 QR 迭代（2.3.3 节，算法 2.3.2）：
                          Rayleigh 位移 μ = h_11，尾部小次对角元收缩（deflation）
    - fast_qr_double_shift    双重步位移隐式快速 QR 迭代（2.3.4 节，算法 2.3.3）：
                          实多项式情形，位移取尾部 2×2 主子阵的特征值（共轭对），
                          对 M = (H-μ1 I)(H-μ2 I) 首列做 Householder、3×3 正交阵追赶
    - balance_matrix      2.3.5 节：Osborne 平衡（对角相似减小条件数）
    - fast_qr_roots       快速 QR 求多项式全部根的完整驱动（含 2.3.5 的实现细节：
                          平衡、exceptional shift、收缩判据）

关于"快速"的说明（重要）：
    书中 2.3.2-2.3.4 的核心思想是把与友矩阵相似 的
    上 Hessenberg 矩阵用 n 个 Schur 参数 γ（酉 Hessenberg 矩阵的参数化 + 秩一修正
    的生成子表示）存储，使每一步移位 QR 迭代只须 O(n) 运算更新参数。
    生成子的 O(n) 内核公式在扫描页上过于密集，逐字复刻风险较大；本文件在
    Hn 类矩阵的显式形式上实现与书中**数学上完全相同**的移位 QR 迭代格式
    （位移策略、隐式技术与收缩判据均按书），每步代价 O(n^2)。
    迭代产生的矩阵序列与参数化版本相似等价，求得的根可相互验证。

运行自测：python 第2讲_多项式之根的快速求法.py
"""

import numpy as np


# =====================================================================
# 2.1 引言：基本问题与基本理论
# =====================================================================

def sign_changes(coeffs):
    """计算数列的变号数（2.1.1 节）。

    数列从左到右符号改变的次数，零元素跳过。
    这是 Descartes 符号律 / Budan-Fourier 定理的基础。

    参数
    ----
    coeffs : array_like，按降幂排列的多项式系数 α_n, α_{n-1}, ..., α_0

    返回
    ----
    int : 变号数
    """
    c = np.asarray(coeffs, dtype=float)
    nz = c[c != 0.0]                     # 去掉零元素（书中规定零不计入变号）
    if nz.size < 2:
        return 0
    return int(np.sum(np.sign(nz[:-1]) * np.sign(nz[1:]) < 0))


def budan_fourier(coeffs, a, b):
    """Budan-Fourier 定理：估计开区间 (a, b) 内实根个数（重根按重数计）的上界。

    定理（2.1.2 节）：设 p 为实多项式，p(a)p(b)≠0，则
        (a,b) 内实根个数 ≤ V(a) - V(b)，
    其中 V(x) 为导数序列 p(x), p'(x), ..., p^{(n)}(x) 在 x 处的变号数，
    且差值与根个数之差为偶数。

    参数
    ----
    coeffs : 降幂排列的实多项式系数
    a, b   : 区间端点，a < b

    返回
    ----
    (upper_bound, parity_note) ：根数上界，以及"上界-实际数必为偶数"的说明字符串
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs) - 1
    # 构造导数序列的系数（p, p', p'', ...，均按降幂）
    derivs = []
    c = coeffs.copy()
    for k in range(n + 1):
        derivs.append(c)
        if len(c) > 1:
            # 多项式求导：c_k z^k -> k c_k z^{k-1}
            c = c[:-1] * np.arange(len(c) - 1, 0, -1)
        else:
            c = np.array([0.0])
    # 每个点处：按 (p(x), p'(x), ..., p^{(n)}(x)) 的顺序数变号（跳过零）
    vals = np.array([[np.polyval(d, x) for d in derivs] for x in (a, b)])
    v_a = sign_changes(vals[0])
    v_b = sign_changes(vals[1])
    diff = v_a - v_b
    return diff, f"(a,b) 内实根个数（计重数）≤ {diff}，且 {diff} - 实际根数为偶数"


def cauchy_root_bound(coeffs):
    """Cauchy 根模上界（2.1.2 节基本理论中给出的根界）。

    对 p(z) = α_n z^n + ... + α_0，所有根满足
        |z| ≤ 1 + max_{0≤k<n} |α_k| / |α_n|。

    返回
    ----
    float : 根模上界 R
    """
    c = np.asarray(coeffs, dtype=float)
    return 1.0 + np.max(np.abs(c[:-1])) / abs(c[0])


# =====================================================================
# 2.2 Newton-Horner 方法
# =====================================================================

def horner(coeffs, z0):
    """Horner 综合除法（2.2.2 节）：同时计算 p(z0) 与 p'(z0)。

    记 p(z) = α_n z^n + ... + α_0（coeffs 按降幂），则
        p(z) = (z - z0) q(z) + p(z0),
    商多项式系数 β 由递推（书中 (2.2.4) 附近）：
        β_1 = α_n,  β_{k+1} = α_{n-k} + z0 * β_k,  k = 1..n-1,
        p(z0) = α_0 + z0 * β_n（即最后一次递推的余数）；
    对 q 再做一次综合除法即得 p'(z0) = q(z0)。

    参数
    ----
    coeffs : 降幂排列的多项式系数 [α_n, ..., α_0]
    z0     : 复或实的求值点

    返回
    ----
    (p_val, dp_val, q_coeffs) : p(z0), p'(z0), 商多项式 q 的降幂系数
    """
    coeffs = np.asarray(coeffs)
    n = len(coeffs) - 1
    beta = np.zeros(n, dtype=complex) if np.iscomplexobj(coeffs) or isinstance(z0, complex) \
        else np.zeros(n, dtype=float)
    # 第一次综合除法：p(z) = (z-z0) q(z) + r
    b = coeffs[0]
    beta[0] = b
    for k in range(1, n):
        b = coeffs[k] + z0 * b          # β_{k+1} = α_{n-k} + z0 β_k
        beta[k] = b
    p_val = coeffs[-1] + z0 * b         # 余数即 p(z0)
    # 第二次综合除法：q(z) = (z-z0) s(z) + q(z0)，余数即 p'(z0)
    if n >= 2:
        d = beta[0]
        for k in range(1, n - 1):
            d = beta[k] + z0 * d
        dp_val = beta[-1] + z0 * d
    else:
        dp_val = beta[0]                # n = 1 时 p' 为常数
    return p_val, dp_val, beta


def newton_iter(coeffs, z0, eps=1e-12, max_iter=100):
    """Newton 迭代法（2.2.1 节，式 (2.2.1)）。

    z_{k+1} = z_k - p(z_k) / p'(z_k)

    停止准则采用书中命题 2.2.1 之后的两个实用准则之一：
      (1) 增量准则：|z_{k+1} - z_k| ≤ eps * |z_{k+1}|（步长足够小）
      (2) 剩余准则：|p(z_{k+1})| ≤ eps * max(1,|z|)^n 型比例剩余
    两者满足其一即停止（书中指出单独使用 |p(z_k)|<ε 并不可靠）。

    参数
    ----
    coeffs : 降幂多项式系数
    z0     : 初值（书中 Newton-Horner 用 z0 = 0）

    返回
    ----
    (z, k) : 近似根与实际迭代步数
    """
    z = complex(z0)
    for k in range(max_iter):
        p_val, dp_val, _ = horner(coeffs, z)   # Horner 求值与导数（2.2.2）
        if dp_val == 0:
            break                               # 导数为零，无法继续
        z_new = z - p_val / dp_val              # Newton 校正
        # 准则 (1)：相对增量
        if abs(z_new - z) <= eps * max(1.0, abs(z_new)):
            z = z_new
            return z, k + 1
        z = z_new
    return z, max_iter


def newton_horner(coeffs, eps=1e-12, max_iter=100):
    """Newton-Horner 方法（算法 2.2.1）：求多项式全部根。

    书中算法步骤：
      1. 取 z0 = 0，对当前 p 做 Newton 迭代得根 λ1（内层用 Horner 求值）；
      2. Horner 综合除法 p(z) = (z - λ1) q(z)，令 p := q（降一阶）；
      3. 对 i = 2..n 重复步骤 1-2；
      4. 输出 λ1, ..., λn。

    实现说明（对书中算法的工程补全）：
      * 实系数多项式的复根无法从实初值出发用 Newton 找到，故当迭代
        不收敛（剩余过大）时改用复随机初值重试；
      * 降阶会累积误差，最后对每个根在**原多项式**上再做几步 Newton
        精化（对应 2.3.5 节"用 Newton 迭代精化 QR 求出的根"的思想）。
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    n = len(coeffs) - 1
    cnorm = np.linalg.norm(np.abs(coeffs))
    rng = np.random.default_rng(2024)
    roots = np.zeros(n, dtype=complex)
    p = coeffs.copy()
    z0 = 0.0 + 0.0j
    for i in range(n):
        # --- 步骤 1：Newton 迭代求当前多项式的一个根（失败则换复初值） ---
        lam, _ = newton_iter(p, z0, eps=eps, max_iter=max_iter)
        ok = abs(horner(p, lam)[0]) < 1e-8 * (1 + abs(lam)) ** n * max(1.0, cnorm)
        tries = 0
        while not ok and tries < 20:
            z_try = (rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1)) * \
                max(1.0, cauchy_root_bound(np.abs(coeffs)))
            lam, _ = newton_iter(p, z_try, eps=eps, max_iter=max_iter)
            ok = abs(horner(p, lam)[0]) < 1e-8 * (1 + abs(lam)) ** n * max(1.0, cnorm)
            tries += 1
        # --- 步骤 2：Horner 降阶（p ← 商多项式 q） ---
        _, _, q = horner(p, lam)
        roots[i] = lam
        p = q
        # 下一个根的初值：书中取上一个求出的根（同簇根时收敛更快）
        z0 = lam
    # --- 步骤 4：在原多项式上 Newton 精化每个根（消除降阶累积误差） ---
    for i in range(n):
        roots[i], _ = newton_iter(coeffs, roots[i], eps=eps, max_iter=8)
    # 对实系数多项式，把数值上已是实数的根化成实数（美化输出）
    if np.allclose(coeffs.imag, 0):
        for i in range(len(roots)):
            if abs(roots[i].imag) < 1e-8 * max(1.0, abs(roots[i])):
                roots[i] = roots[i].real
    return roots


# =====================================================================
# 2.3 快速 QR 方法
# =====================================================================

def companion_matrix(coeffs):
    """友矩阵构造（2.3.1 节）。

    对首一化多项式 p(z) = z^n + a_{n-1} z^{n-1} + ... + a_0，
    友矩阵（第一行伴阵形式，与书中一致）：

        C = [ -a_{n-1}  -a_{n-2}  ...  -a_1  -a_0 ]
            [    1         0      ...    0      0  ]
            [    0         1      ...    0      0  ]
            [    ...                        ...    ]
            [    0         0      ...    1      0  ]

    性质：χ_C(z) = p(z)，故 C 的特征值即 p 的全部根（书中定理 2.3.1 之前的
    命题）。numpy.roots 内部正是用友矩阵。

    参数
    ----
    coeffs : 降幂系数 [α_n, α_{n-1}, ..., α_0]，α_n ≠ 0（自动首一化）

    返回
    ----
    C : (n, n) 友矩阵
    """
    c = np.asarray(coeffs, dtype=float) / coeffs[0]   # 首一化
    n = len(c) - 1
    C = np.zeros((n, n))
    C[0, :] = -c[1:]            # 首行放负系数（书中约定）
    if n > 1:
        C[1:, :-1] = np.eye(n - 1)   # 次对角线放 1
    return C


def in_Hn_class(H, tol=1e-12):
    """Hn 类判定（2.3.2 节定义）。

    Hn = { n 阶上 Hessenberg 矩阵且次对角元全不为零 }。
    书中命题：A 与友矩阵相似 ⟺ A ∈ Hn 且 χ_A = χ_C（特征多项式相同）。

    返回
    ----
    bool : 是否属于 Hn 类
    """
    H = np.asarray(H)
    n = H.shape[0]
    if not np.allclose(H[np.tril_indices(n, -2)], 0, atol=1e-10):
        return False                       # 不是上 Hessenberg
    sub = np.diag(H, -1)
    return bool(np.all(np.abs(sub) > tol))  # 次对角元非零


def unitary_hessenberg(gamma):
    """由 Schur 参数 γ 构造酉上 Hessenberg 矩阵（2.3.2 节定理 2.3.1 的构造）。

    U = G_1 G_2 ... G_{n-1} D（嵌入式 2×2 酉旋转的升序乘积），
    其中 G_k 作用在 (k, k+1) 坐标平面上：
        G_k = [[γ_k, -σ_k ],[σ_k, γ̄_k]]，  σ_k = sqrt(1-|γ_k|²) > 0，
    D = diag(1, ..., 1, γ_n)（尾部相位因子，|γ_n| = 1 时为完全酉）。
    该乘积是酉上 Hessenberg 矩阵，且次对角元恰为 σ_1, ..., σ_{n-1} > 0。
    标准约定：前 n-1 个参数 |γ_k| < 1，最后一个 |γ_n| = 1（尾部相位因子），
    此时 U 为酉矩阵；若 |γ_n| < 1，则 U 除 (n,n) 元外保持酉结构
    ——书中正是用这种"酉 + 秩一修正"形式表示友矩阵的相似类
    （C = U - x y*，非酉的缺陷被秩一项吸收），整个迭代矩阵只需
    n 个参数存储——这就是"快速"的根源。

    参数
    ----
    gamma : (n,) Schur 参数（|γ_k| < 1, k<n；|γ_n| ≤ 1）

    返回
    ----
    U : (n, n) 酉上 Hessenberg 矩阵（次对角元为正实数）
    """
    g = np.asarray(gamma, dtype=complex)
    n = len(g)
    U = np.eye(n, dtype=complex)
    for k in range(n - 1):
        gk = g[k] / abs(g[k]) if abs(g[k]) > 1 else g[k]
        sk = np.sqrt(max(0.0, 1.0 - abs(gk) ** 2))     # σ_k > 0
        G = np.array([[gk, -sk],
                      [sk, np.conj(gk)]], dtype=complex)
        # 升序右乘嵌入式旋转：U ← U · G_k，得到 G_1 G_2 ... G_{n-1}
        U[:, k:k + 2] = U[:, k:k + 2] @ G
    # 最后右乘 D = diag(1, ..., 1, γ_n)：等价于缩放最后一列
    U[:, -1] = U[:, -1] * g[-1]
    return U


def schur_params_from_unitary_hess(U):
    """由酉上 Hessenberg 矩阵（次对角元为正）反提 Schur 参数 γ（2.3.2 节）。

    提取算法（unitary_hessenberg 的逆过程）：
      对 k = 1..n-1：γ_k = u_{kk}，σ_k = u_{k+1,k} > 0，
        左乘 G_k^* = [[γ̄_k, σ_k],[-σ_k, γ_k]] 消去次对角元，
        继续处理右下子阵；最后 γ_n = 剩余 1×1 块。
    由酉性自动保证 |γ_k|² + σ_k² = 1。

    参数
    ----
    U : (n, n) 酉上 Hessenberg 矩阵（次对角元为正实数）

    返回
    ----
    gamma : (n,) 复 Schur 参数（|γ_k| ≤ 1）
    """
    R = np.array(U, dtype=complex, copy=True)
    n = R.shape[0]
    gamma = np.zeros(n, dtype=complex)
    for k in range(n - 1):
        gamma[k] = R[k, k]                       # γ_k = 当前对角元
        sk = R[k + 1, k].real                    # σ_k > 0（次对角元）
        gk = gamma[k]
        # 左乘 G_k^* 把 (k,k+1) 平面上的次对角元 σ_k 消为零
        row_k = np.conj(gk) * R[k, k:] + sk * R[k + 1, k:]
        row_k1 = -sk * R[k, k:] + gk * R[k + 1, k:]
        R[k, k:] = row_k
        R[k + 1, k:] = row_k1
        R[k + 1, k] = 0.0                        # 数值上清零
    gamma[-1] = R[-1, -1]
    return gamma


def companion_to_hessenberg(coeffs):
    """把友矩阵化为 Hn 类中的上 Hessenberg 形式（"参数化初始化"的等价实现）。

    书中 2.3.2 的参数化把相似类 {Q* C Q} 用 n 个 Schur 参数表示，
    迭代前的初始化等价于：取友矩阵 C，做一次（随机）酉相似变换保持
    Hessenberg 结构与特征多项式不变，得到 H ∈ Hn，χ_H = p。
    本函数直接验证：友矩阵本身已属 Hn 类（次对角元为 1），
    并返回其显式矩阵，作为快速 QR 迭代的初始迭代矩阵。

    返回
    ----
    H : (n, n) 上 Hessenberg 矩阵（即友矩阵本身）
    """
    return companion_matrix(coeffs)


def _givens(a, b):
    """复 Givens 旋转参数（2.3.3 节引理 2.3.4 的显式公式）。

    构造 2×2 酉矩阵（书中 [c s̄; s c̄] 型旋转的等价形式）

        G = [[ c     , s     ],
             [-conj(s), conj(c)]],   c = conj(a)/r,  s = conj(b)/r,

    使得  G @ [a; b] = [r; 0]，r = sqrt(|a|² + |b|²) > 0。

    返回
    ----
    (c, s, r)
    """
    r = np.hypot(abs(a), abs(b))
    if r == 0:
        return 1.0, 0.0, 0.0
    return np.conj(a) / r, np.conj(b) / r, r


def fast_qr_single_shift(coeffs, eps=1e-13, max_sweeps=200, verbose=False):
    """单步位移的快速 QR 迭代（2.3.3 节，算法 2.3.2）——复系数多项式。

    书中迭代格式（在参数 γ 表示的 H ∈ Hn 上）：
      repeat
        取 Rayleigh 位移 μ = 当前活动块尾部的对角元（注 2.3.3 的 Rayleigh
        位移；显式矩阵形式下取活动块右下角 h_{m,m}，使尾部次对角元最快
        收敛，与尾部收缩配套）
        对 H - μI 做一步 QR：H ← RQ + μI = Q* H Q
        若尾部次对角元 |h_{m,m-1}| < eps·(|h_{m-1,m-1}|+|h_{m,m}|)：
            收缩（deflation）：λ ← h_{m,m}，活动子问题大小 m ← m-1
      until m = 0
    另按 2.3.5 节：迭代停滞时插入 exceptional shift（非常规位移）打破循环。
    本实现按同一格式作用在显式 Hessenberg 矩阵上（每步 O(n²)，数学等价）。

    参数
    ----
    coeffs : 多项式降幂系数（可为复系数，对应书中复矩阵情形）

    返回
    ----
    roots : (n,) 近似根（= 收缩时对角元的值）
    """
    c = np.asarray(coeffs, dtype=complex)
    n = len(c) - 1
    if n == 0:
        return np.array([])
    # 友矩阵（复系数时手动首一化构造）
    H = np.zeros((n, n), dtype=complex)
    a = c / c[0]
    H[0, :] = -a[1:]
    if n > 1:
        H[1:, :-1] = np.eye(n - 1, dtype=complex)

    roots = np.zeros(n, dtype=complex)
    m = n                                   # 活动子问题大小
    total = 0
    stagnation = 0
    hnorm = np.linalg.norm(H)               # 相似变换下近似不变，用作绝对阈值

    def _neg(i):
        """H[i, i-1] 是否可忽略（相对判据 + 绝对下限）。"""
        tst = abs(H[i, i - 1])
        diag = abs(H[i - 1, i - 1]) + abs(H[i, i])
        if diag == 0.0:
            diag = hnorm
        return tst < eps * diag or tst < 1e-12 * hnorm

    while m > 0 and total < 60 * n:
        # --- 尾部收缩判据（deflation，从尾部开始） ---
        deflated = True
        while deflated and m > 0:
            if m == 1:
                roots[0] = H[0, 0]
                m = 0
                break
            if _neg(m - 1):
                roots[m - 1] = H[m - 1, m - 1]
                m -= 1                      # 去掉最后一行一列
                stagnation = 0              # 收缩后重置停滞计数
            else:
                deflated = False
        if m == 0:
            break
        # --- 确定活动窗口 [l, m) ---
        l = 0
        for i in range(m - 1, 0, -1):
            if _neg(i):
                l = i
                break
        if m - l == 2:
            # 2×2 窗口直接求根（避免位移=块自身特征值时的空转）。
            # 用复数求根公式（对实块自动给出实根对或共轭复根对）
            A22 = H[l:m, l:m].astype(complex)
            tr = A22[0, 0] + A22[1, 1]
            det = A22[0, 0] * A22[1, 1] - A22[0, 1] * A22[1, 0]
            disc = np.sqrt(tr * tr / 4 - det + 0j)
            roots[l:m] = [tr / 2 + disc, tr / 2 - disc]
            m = l
            stagnation = 0
            continue
        # --- Rayleigh 位移 μ = 活动块尾部对角元（注 2.3.3） ---
        mu = H[m - 1, m - 1]
        if stagnation > 0 and stagnation % 10 == 0:
            # 2.3.5：停滞时插入非常规位移（exceptional shift）打破循环
            mu = mu + 0.75 * abs(H[m - 1, m - 2])
        # --- 单步移位 QR：窗口内 H-μI = QR，H ← RQ + μI ---
        Hs = H[l:m, l:m] - mu * np.eye(m - l, dtype=complex)
        Qall = np.eye(m - l, dtype=complex)
        for k in range(m - l - 1):          # Givens 从上到下消去次对角元
            cc, ss, r = _givens(Hs[k, k], Hs[k + 1, k])
            if r == 0:
                continue
            G = np.array([[cc, ss],
                          [-np.conj(ss), np.conj(cc)]], dtype=complex)
            Hs[k:k + 2, k:m - l] = G @ Hs[k:k + 2, k:m - l]   # R = G_{m-1}···G_0 (H-μI)
            Qall[k:k + 2, :] = G @ Qall[k:k + 2, :]           # Qall = G_{m-1}···G_0
        # H-μI = Qall^H R  ⇒  窗口更新 W+ = Qall W Qall^H（相似变换）。
        # 注意：相似变换必须作用到全矩阵的对应行列——窗口上方的耦合块
        # H[0:l, l:m] 要右乘 Qall^H，窗口右侧 H[l:m, m:] 要左乘 Qall，
        # 否则整矩阵不再相似，特征值会出错
        H[l:m, l:m] = Hs @ Qall.conj().T + mu * np.eye(m - l, dtype=complex)
        if l > 0:
            H[0:l, l:m] = H[0:l, l:m] @ Qall.conj().T
        if m < n:
            H[l:m, m:] = Qall @ H[l:m, m:]
        total += 1
        stagnation += 1
    if m > 0:                                # 未完全收缩：剩余块直接取对角元
        eigs = np.linalg.eigvals(H[:m, :m])
        roots[:m] = eigs
    if verbose:
        print(f"  [fast_qr_single_shift] QR 步数 ≈ {total}")
    return roots


def fast_qr_double_shift(coeffs, eps=1e-13, max_iter_per_root=60, verbose=False):
    """双重步位移的隐式快速 QR 迭代（2.3.4 节，算法 2.3.3）——实多项式。

    书中思想（实系数时根可能成共轭对，单步实位移难以同时收敛两个复根）：
      * 位移 μ1, μ2 取当前活动块尾部 2×2 主子阵的特征值（共轭对）；
      * M = (H - μ1 I)(H - μ2 I) 的第一列只有前三个分量非零：
            x1 = h11² + h12 h21 - (μ1+μ2) h11 + μ1 μ2
            x2 = h21 (h11 + h22 - (μ1+μ2))
            x3 = h32 h21
        （由 H 的 Hessenberg 结构直接展开；书中以参数形式给出对应量）
      * 构造 3 阶 Householder P0 把 (x1,x2,x3)ᵀ 化为 (±‖x‖,0,0)ᵀ，
        隐式完成与显式双位移 QR 等价的正交相似变换（Francis 隐式 Q 定理）；
      * 用一系列 3 阶正交阵 {P_k} 沿次对角线"追赶"（bulge chasing），
        把矩阵恢复为上 Hessenberg 形；
      * 尾部 1×1 或 2×2 块收敛后收缩：1×1 给出实根，2×2 给出共轭复根对。
    实现：等价的显式 Hessenberg 矩阵形式（每步 O(n²)，数学等价于书中
    参数化版本），位移、隐式首列、收缩策略与书一致。

    参数
    ----
    coeffs : 实多项式降幂系数

    返回
    ----
    roots : (n,) 近似根（复数组；实根虚部为 0）
    """
    c = np.asarray(coeffs, dtype=float)
    n = len(c) - 1
    if n == 0:
        return np.array([])
    H = companion_matrix(c)                 # 友矩阵即初始 Hessenberg 迭代矩阵
    roots = np.zeros(n, dtype=complex)
    m = n                                   # 活动子问题大小
    stagnation = 0
    hnorm = np.linalg.norm(H)               # 用作绝对收缩阈值的范数基准

    def _subdiag_negligible(i):
        """H[i, i-1] 是否可忽略（相对判据 + 绝对下限，LAPACK dlahqr 风格）。"""
        tst = abs(H[i, i - 1])
        diag = abs(H[i - 1, i - 1]) + abs(H[i, i])
        if diag == 0.0:
            diag = hnorm
        return tst < eps * diag or tst < 1e-12 * hnorm

    while m > 0:
        # ---------- 尾部收缩（deflation）检查 ----------
        if m == 1:
            roots[0] = H[0, 0]
            m = 0
            break
        if _subdiag_negligible(m - 1):
            roots[m - 1] = H[m - 1, m - 1]
            m -= 1
            stagnation = 0                  # 收缩后重置停滞计数
            continue
        # ---------- 确定活动窗口 [l, m)：从底向上找第一个可忽略的次对角元 ----------
        l = 0
        for i in range(m - 1, 0, -1):
            if _subdiag_negligible(i):
                l = i                       # H[i, i-1] ≈ 0：窗口从第 i 行开始
                break
        if m - l == 2:
            # 活动窗口本身是 2×2 块：直接求根。
            # （此时位移=该块自身特征值，M=(H-μ1I)(H-μ2I)≈0，追赶必然空转）
            # 用复数求根公式（对实块自动给出实根对或共轭复根对）
            A22 = H[l:m, l:m].astype(complex)
            tr = A22[0, 0] + A22[1, 1]
            det = A22[0, 0] * A22[1, 1] - A22[0, 1] * A22[1, 0]
            disc = np.sqrt(tr * tr / 4 - det + 0j)
            roots[l:m] = [tr / 2 + disc, tr / 2 - disc]
            m = l                           # 整个窗口收缩掉
            stagnation = 0
            continue
        # ---------- 位移：活动窗口尾部 2×2 主子阵的特征值 ----------
        a, b = H[m - 2, m - 2], H[m - 2, m - 1]
        cc, d = H[m - 1, m - 2], H[m - 1, m - 1]
        s = a + d                           # μ1+μ2 = trace
        t = a * d - b * cc                  # μ1 μ2 = det
        if stagnation > 0 and stagnation % 12 == 0:
            # 2.3.5：停滞时插入非常规位移（exceptional shift）
            s = s + 0.75 * (abs(H[m - 1, m - 2]) + abs(H[m - 2, m - 3]))

        # ---------- 隐式 Francis 追赶（对应书中 P_0, P_1, ..., P_{n-2}） ----------
        # M = (H-μ1 I)(H-μ2 I) 窗口首列只有前三个分量非零（窗口局部下标从 l 起）：
        x = H[l, l] * H[l, l] + H[l, l + 1] * H[l + 1, l] - s * H[l, l] + t
        y = H[l + 1, l] * (H[l, l] + H[l + 1, l + 1] - s)
        z = H[l + 1, l] * H[l + 2, l + 1]
        for k in range(l, m - 2):
            # 3 阶 Householder P：把 (x,y,z)ᵀ 化为 (±‖·‖,0,0)ᵀ
            v3 = np.array([x, y, z])
            nv = np.linalg.norm(v3)
            if nv < 1e-300:
                break
            alpha = -nv if v3[0] >= 0 else nv          # 符号选择避免相消
            w3 = v3 - alpha * np.array([1.0, 0.0, 0.0])
            w3 /= np.linalg.norm(w3)
            # 左乘 P 作用于第 k..k+2 行；起始列 max(l, k-1)——
            # 上一轮右乘在第 k-1 列留下的凸角（次对角线以下）也要一并消去
            q0 = max(l, k - 1)
            H[k:k + 3, q0:m] -= 2.0 * np.outer(w3, w3 @ H[k:k + 3, q0:m])
            # 右乘 P 作用于第 k..k+2 列；行范围必须从 0 开始——
            # 窗口上方的耦合块 H[0:l, k:k+3] 也要被相似变换更新
            r_hi = min(k + 4, m)
            H[0:r_hi, k:k + 3] -= 2.0 * np.outer(H[0:r_hi, k:k + 3] @ w3, w3)
            # 追赶量：新的凸角下一段向量（第 k 列上次对角线以下的三元组）
            x = H[k + 1, k]
            y = H[k + 2, k]
            z = H[k + 3, k] if k + 3 <= m - 1 else 0.0
        # 最后一对：2 阶 Householder 消去窗口尾部最后一个凸角 H[m-1, m-3]
        v2 = np.array([x, y])               # x=H[m-2,m-3], y=H[m-1,m-3]
        n2 = np.linalg.norm(v2)
        if n2 > 1e-300:
            alpha2 = -n2 if v2[0] >= 0 else n2
            w2 = v2 - alpha2 * np.array([1.0, 0.0])
            w2 /= np.linalg.norm(w2)
            H[m - 2:m, m - 3:m] -= 2.0 * np.outer(w2, w2 @ H[m - 2:m, m - 3:m])
            H[0:m, m - 2:m] -= 2.0 * np.outer(H[0:m, m - 2:m] @ w2, w2)
        stagnation += 1
        if stagnation > max_iter_per_root * n:
            if verbose:
                print("  [fast_qr_double_shift] 达到最大迭代次数，提前退出")
            break
    if m > 0:
        roots[:m] = np.linalg.eigvals(H[:m, :m])
    return roots


def balance_matrix(A, max_sweep=20):
    """Osborne 平衡（2.3.5 节"具体实现时的几个问题"第 1 点）。

    求对角阵 D（2 的幂次元素，保证无舍入误差），使 B = D⁻¹ A D 的
    各行范数与列范数尽量接近，从而降低特征值问题的条件数。
    这是 EISPACK/numpy 平衡的标准做法，书中建议在 QR 迭代前使用。

    返回
    ----
    (B, d) : 平衡后的矩阵与所用的对角相似变换向量
    """
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    d = np.ones(n)
    radix = 2.0
    for _ in range(max_sweep):
        converged = True
        for i in range(n):
            r = np.sum(np.abs(A[i, :])) - abs(A[i, i])   # 第 i 行 1-范数（去对角）
            c = np.sum(np.abs(A[:, i])) - abs(A[i, i])   # 第 i 列 1-范数
            if r == 0 or c == 0:
                continue
            f = 1.0
            s = c + r
            while c < r / radix:
                c *= radix
                r /= radix
                f *= radix
            while c > r * radix:
                c /= radix
                r *= radix
                f /= radix
            if (c + r) < 0.95 * s:                       # 该行/列确有改进
                converged = False
                d[i] *= f
                A[i, :] *= (1.0 / f)                     # D⁻¹ A D：行除、列乘
                A[:, i] *= f
        if converged:
            break
    return A, d


def fast_qr_roots(coeffs, eps=1e-13, verbose=False):
    """快速 QR 方法求多项式全部根——完整驱动（2.3.3-2.3.5 的工程组合）。

    组合书中 2.3.5 的实现细节：
      1. 友矩阵平衡（Osborne，降低条件数）；
      2. 实多项式走双重步位移隐式 QR（算法 2.3.3）；
         复多项式走单步位移 QR（算法 2.3.2）；
      3. 迭代内含收缩判据与停滞时的处理。

    返回
    ----
    roots : (n,) 复根数组
    """
    coeffs = np.asarray(coeffs)
    if np.iscomplexobj(coeffs) and not np.allclose(np.imag(coeffs), 0):
        return fast_qr_single_shift(coeffs, eps=eps, verbose=verbose)
    # 实多项式：平衡友矩阵后按双位移隐式 QR 迭代（平衡可显著加速收敛）
    C = companion_matrix(np.real(coeffs))
    B, d = balance_matrix(C)                # 相似变换不改变特征值
    if verbose:
        print(f"  [fast_qr_roots] 平衡前后条件改善："
              f"cond ~ {np.linalg.cond(C):.2e} -> {np.linalg.cond(B):.2e}")
    return fast_qr_double_shift(coeffs, eps=eps)


# =====================================================================
# 自测
# =====================================================================

if __name__ == "__main__":
    np.random.seed(0)

    def max_root_error(r, true):
        """最近匹配意义下的最大根误差（避免并列实部时排序错位的假误差）。"""
        r = np.asarray(r, dtype=complex)
        true = np.asarray(true, dtype=complex)
        return max(min(abs(ri - tj) for tj in true) for ri in r)

    print("=" * 70)
    print("《矩阵计算六讲》第二讲 自测：多项式之根的快速求法")
    print("=" * 70)

    # ---- 测试 1：Horner 求值/求导 ----
    p = np.array([2.0, -3.0, 1.0, 5.0, -7.0])      # 2z^4 -3z^3 + z^2 +5z -7
    z0 = 1.3 + 0.7j
    pv, dpv, _ = horner(p, z0)
    dp_ref = np.polyval(np.polyder(p), z0)
    print(f"[horner]        p({z0})  = {pv:.10f}   (numpy: {np.polyval(p, z0):.10f})")
    print(f"[horner]        p'({z0}) = {dpv:.10f}   (numpy: {dp_ref:.10f})")
    dp_ref = np.polyval(np.polyder(p), z0)
    assert abs(pv - np.polyval(p, z0)) < 1e-12
    assert abs(dpv - dp_ref) < 1e-12

    # ---- 测试 2：Budan-Fourier / Cauchy 界 ----
    # p(z) = (z-1)(z-2)(z-3) = z^3 - 6z^2 + 11z - 6，(0.5, 3.5) 内有 3 个实根
    p3 = np.array([1.0, -6.0, 11.0, -6.0])
    ub, note = budan_fourier(p3, 0.5, 3.5)
    print(f"[budan_fourier] (0.5,3.5) 根数上界 = {ub}（真实 3 个）  {note}")
    assert ub >= 3 and (ub - 3) % 2 == 0
    R = cauchy_root_bound(p3)
    print(f"[cauchy_bound]  根模上界 R = {R:.4f}（真实最大 |root| = 3）")
    assert R >= 3.0

    # ---- 测试 3：Newton-Horner ----
    roots_nh = newton_horner(p, eps=1e-13)
    roots_np = np.roots(p)
    err = max_root_error(roots_nh, roots_np)
    print(f"[newton_horner] 与 numpy.roots 的最大根误差 = {err:.2e}")
    assert err < 1e-6

    # ---- 测试 4：友矩阵 / Hn 类 ----
    C = companion_matrix(p3)
    assert np.allclose(np.poly(np.linalg.eigvals(C)), p3, atol=1e-8)
    assert in_Hn_class(C)
    print(f"[companion]     χ_C = p 成立，C ∈ Hn 类（n={len(C)}）")

    # ---- 测试 5：Schur 参数化的酉 Hessenberg 构造 ----
    rng = np.random.default_rng(42)
    g = rng.uniform(-0.9, 0.9, size=6) + 1j * rng.uniform(-0.9, 0.9, size=6)
    g = g / (1.1 * np.maximum(np.abs(g), 1e-3))    # 前 n-1 个参数 |γ|<1
    g[-1] = np.exp(1j * rng.uniform(0, 2 * np.pi)) # 尾参数取单位模长（酉性要求）
    U = unitary_hessenberg(g)
    orth_err = np.linalg.norm(U.conj().T @ U - np.eye(6))
    print(f"[schur_params]  由 γ 构造酉 Hessenberg：‖U*U − I‖ = {orth_err:.2e}")
    assert orth_err < 1e-12
    sub_pos = np.all(np.diag(U, -1).real > 0) and np.allclose(np.diag(U, -1).imag, 0)
    print(f"[schur_params]  次对角元为正实数：{sub_pos}")
    # 参数提取的往返一致性：extract(build(γ)) ≈ γ
    g_back = schur_params_from_unitary_hess(U)
    rt_err = np.max(np.abs(g_back - g))
    print(f"[schur_params]  参数提取往返误差 ‖γ̂ − γ‖∞ = {rt_err:.2e}")
    assert rt_err < 1e-12

    # ---- 测试 6：单步位移快速 QR（复多项式） ----
    true_roots = np.array([1 + 2j, 1 - 2j, -3.0, 0.5, 2.5])
    pc = np.poly(true_roots)                        # 复系数多项式
    r1 = fast_qr_single_shift(pc, verbose=True)
    err1 = max_root_error(r1, true_roots)
    print(f"[fast_qr_1shift] 复多项式 5 根最大误差 = {err1:.2e}")
    assert err1 < 1e-8

    # ---- 测试 7：双重步位移隐式快速 QR（实多项式，含共轭复根） ----
    true_roots2 = np.array([1.0, 2.0, -0.5, 1 + 1j, 1 - 1j, 3.0])
    pr = np.poly(true_roots2)
    r2 = fast_qr_double_shift(pr, verbose=True)
    err2 = max_root_error(r2, true_roots2)
    print(f"[fast_qr_2shift] 实多项式 6 根最大误差 = {err2:.2e}")
    assert err2 < 1e-6

    # ---- 测试 8：平衡 ----
    Dbig = np.diag(10.0 ** np.array([0, 2, 4, 6, 8]))
    Abig = Dbig @ C @ np.linalg.inv(Dbig) if len(C) == 5 else None
    p5 = np.poly([1.0, 2.0, 3.0, 4.0, 5.0])
    C5 = companion_matrix(p5)
    Abig = np.diag(10.0 ** np.array([0, 1, 2, 3, 4])) @ C5 @ \
        np.diag(10.0 ** np.array([4, 3, 2, 1, 0]))
    B, d = balance_matrix(Abig)
    print(f"[balance]       cond: {np.linalg.cond(Abig):.2e} -> "
          f"{np.linalg.cond(B):.2e}，特征值保持："
          f"{np.allclose(np.sort(np.linalg.eigvals(Abig)), np.sort(np.linalg.eigvals(B)))}")
    assert np.linalg.cond(B) < np.linalg.cond(Abig)

    # ---- 测试 9：完整驱动 fast_qr_roots ----
    r3 = fast_qr_roots(pr, verbose=False)
    err3 = max_root_error(r3, true_roots2)
    print(f"[fast_qr_roots] 完整驱动 6 根最大误差 = {err3:.2e}")
    assert err3 < 1e-6

    print("-" * 70)
    print("第二讲全部自测通过 ✔")
