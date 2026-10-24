# 矩阵计算六讲 · 算法实现

《矩阵计算六讲》（徐树方 等，北京大学出版社）一书中数值代数算法的
Python 教学实现。六个文件对应书中的六讲，每个文件可独立运行并附完整的
数值自测（与 `numpy.linalg` 参考结果对比）。

> 本仓库的代码是对书中算法的独立实现，仅供学习交流使用；书的版权归
> 原书作者与出版社所有。

mpy`（Python ≥ 3.8）。

## 文件清单

| 文件 | 内容 | 自测结果 |
|------|------|----------|
| `第1讲_Schur分解的计算.py` | 算法 1.1.1–1.1.8：复 Householder、复 Givens、QR 分解、上 Hessenberg 分解、Wilkinson 位移、带位移显式 QR 迭代（复 Schur 分解）、Francis 双重步位移隐式 QR（实 Schur 分解）、实 Schur 标准形对角块排序（ordschur）；QZ 方法（Hessenberg-三角化、降阶、双重步位移 QZ 迭代、广义实 Schur）；周期 QZ（周期 Hessenberg-三角化 + 乘积谱） | 全部通过（1e-13 ~ 1e-16） |
| `第2讲_多项式之根的快速求法.py` | 变号数、Budan-Fourier 定理、Cauchy 根界、Horner 综合除法、Newton 迭代、Newton-Horner 方法（算法 2.2.1–2.2.3）、友矩阵、Hn 类矩阵、Schur 参数化、单步位移快速 QR（算法 2.3.2）、双重步位移隐式快速 QR、Osborne 平衡（2.3.5） | 全部通过（1e-11 ~ 1e-15） |
| `第3讲_奇异值分解的计算.py` | 对称三对角化、Wilkinson 位移、隐式对称 QR、Golub-Kahan 双对角化与 SVD（算法 3.2.x）、分而治之（DPR1 secular 方程 + Newton）、SVD 的分而治之、Jacobi 特征值、双侧（Kogbetliantz）Jacobi SVD、Sturm 序列计数、二分法与奇异值二分法（算法 3.5.x） | 全部通过（1e-13 ~ 1e-16） |
| `第4讲_Krylov子空间方法I.py` | Arnoldi 分解（MGS+重正交）、Lanczos 分解、Rayleigh-Ritz 投影、经典 Arnoldi 算法、隐式重启 Arnoldi（IRA）、位移求逆 Arnoldi、经典 Lanczos、重启 Lanczos | 全部通过（1e-14 ~ 1e-18） |
| `第5讲_Krylov子空间方法II.py` | CG、MINRES、GMRES 与重启 GMRES(m)、非对称 Lanczos（双正交）、QMR、BCG、CGS、BICGSTAB | 全部通过（1e-13 ~ 1e-16；CGS 的不规则收敛为其著名缺陷，注释已说明） |
| `第6讲_共轭梯度法.py` | 广义特征值问题的梯度类方法：newapprox 最优步长（定理 6.2.1）、mgrams B 正交化、算法 6.3.1 最速下降、算法 6.3.2 收缩最速下降、SIRQIT 同时迭代（§6.3.3）、算法 6.4.1–6.4.3 共轭梯度法及收缩/同时版本、预优 CG（§6.4.4）、PGS 迭代（§6.5.1） | 全部通过（1e-13 ~ 1e-16） |

## 运行方式

```bash
python 第1讲_Schur分解的计算.py    # 依次运行各讲
...
```

每个文件末尾的 `if __name__ == "__main__":` 自测会打印每个算法的误差
并与 numpy 参考结果对比，全部断言通过后打印"第 X 讲全部自测通过"。

## 实现约定

* 每个函数的 docstring 标明书中位置（如"算法 6.3.1"）、迭代格式与
  数值注意点；关键步骤配中文行内注释。
* 只用 `numpy` 与标准库（见 `requirements.txt`）。
* 少数研究级算法（书中仅给框架或将细节留作练习的部分）做了教学简化，
  均在对应函数的 docstring 中明确标注：
  - 第一讲 QZ 迭代：驱动器采用显式 H = A B⁻¹ 路线计算广义特征值
    （特征值严格正确）；同时三角化的隐式追赶作为 `qz_step` 一并给出。
  - 第一讲周期 QZ：书中仅给出四步框架（p.53-54），本实现提供周期
    Hessenberg-三角化与乘积谱计算，完整迭代属研究级（参见 SLICOT MB03BD）。
  - 第五讲 CGS：其不规则收敛是方法本身的著名缺陷（书中 5.6.3 的
    BICGSTAB 即为消除它而提出），自测中如实呈现。

## 许可证

[MIT](LICENSE)
