<p align="center">
  <strong><h1 align="center">RunMat</h1></strong>
</p>

<p align="center">
  <strong>开源数学运行时。MATLAB 语法。CPU + GPU。无需许可证费用。</strong>
</p>

<p align="center">
  RunMat 自动融合运算并在 CPU 和 GPU 之间智能调度。<br/>
  支持 Windows、macOS、Linux 和 WebAssembly，兼容 NVIDIA、AMD、Apple Silicon 和 Intel GPU。<br/>
  无需编写内核代码。无需重写代码。无需设备标志。无厂商锁定。
</p>

<p align="center">
  <a href="https://github.com/runmat-org/runmat/actions"><img src="https://img.shields.io/github/actions/workflow/status/runmat-org/runmat/ci.yml?branch=main" alt="Build Status"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT%20with%20Attribution-blue.svg" alt="License"></a>
  <a href="https://crates.io/crates/runmat"><img src="https://img.shields.io/crates/v/runmat.svg" alt="Crates.io"></a>
  <a href="https://crates.io/crates/runmat"><img src="https://img.shields.io/crates/d/runmat.svg" alt="Downloads"></a>
</p>

<p align="center">
  <a href="https://runmat.com/sandbox"><strong>立即试用 — 无需安装</strong></a> · <a href="https://runmat.com/docs">文档</a> · <a href="https://runmat.com/blog">博客</a> · <a href="https://runmat.com">官网</a>
</p>

<p align="center"><em>状态: 预发布 (v0.4) — 核心运行时和 GPU 引擎已通过数千项测试。可能存在一些粗糙之处。</em></p>

---

## RunMat 是什么？

使用 RunMat，你可以用清晰、易读的 MATLAB 风格语法编写数学代码。RunMat 自动将你的运算融合为优化后的内核，并在最佳可用硬件（CPU 或 GPU）上运行。在 GPU 上，它在许多密集数值工作负载中通常可以匹配或超越手写 CUDA 的性能。

无论你拥有哪种 GPU —— NVIDIA、AMD、Apple Silicon、Intel —— 都通过原生 API（Metal / DirectX 12 / Vulkan）运行。无需设备管理。无厂商锁定。无需重写。

```matlab
x  = 0:0.01:4*pi;
y0 = sin(x) .* exp(-x / 10);
y1 = y0 .* cos(x / 4) + 0.25 .* (y0 .^ 2);
y2 = tanh(y1) + 0.1 .* y1;

plot(x, y2);
```

下图中的数据点对应于上述 `x` 向量中的元素数量：

![Elementwise math speedup](https://web.runmatstatic.com/elementwise-math_speedup-b.svg)

核心理念：

- **MATLAB 输入语言兼容，而非一门新语言**
- **在 CPU 和 GPU 上都很快**，使用同一个运行时
- **无需设备标志** —— Fusion 根据数据大小和传输成本启发式自动选择 CPU 或 GPU

---

## 使用 RunMat 的方式

此仓库中的开源运行时为所有 RunMat 界面提供动力：

<div align="center">
<table>
<tr>
<td align="center" width="20%">
<h3>🌐 浏览器</h3>
无需安装<br/><br/>
通过 WebAssembly + WebGPU 运行。<br/>
你的代码永远不会离开你的机器。<br/><br/>
<a href="https://runmat.com/sandbox"><strong>立即试用 →</strong></a>
</td>
<td align="center" width="20%">
<h3>⌨️ CLI</h3>
开源（本仓库）<br/><br/>
运行 <code>.m</code> 文件，进行基准测试，<br/>
集成到 CI/CD 中。<br/><br/>
<code>cargo install runmat</code>
</td>
<td align="center" width="20%">
<h3>📦 NPM</h3>
嵌入到任何地方<br/><br/>
完整的运行时 —— 执行、GPU、<br/>
绘图 —— 适用于任何 Web 应用。<br/><br/>
<a href="https://www.npmjs.com/package/runmat"><code>npm install runmat</code></a>
</td>
<td align="center" width="20%">
<h3>🖥️ 桌面端</h3>
即将推出<br/><br/>
原生 IDE，支持本地文件<br/>
和完整的 GPU 加速。<br/><br/>
&nbsp;
</td>
<td align="center" width="20%">
<h3>☁️ 云端</h3>
提供 Hobby 免费层<br/><br/>
版本控制、协作、<br/>
团队管理。<br/><br/>
<a href="https://runmat.com/pricing"><strong>价格 →</strong></a>
</td>
</tr>
</table>
</div>

---

## ✨ 功能一览

- **MATLAB 输入语言兼容，而非一门新语言**

  - 熟悉的 `.m` 文件、数组、控制流
  - 许多 MATLAB / Octave 脚本几乎无需修改即可运行

- **Fusion：自动 CPU+GPU 选择**

  - 构建数组运算的内部图
  - 将逐元素运算和归约融合为更大的内核
  - 根据形状和传输成本为每个内核选择 CPU 或 GPU
  - 当在设备上运行更快时，保持数组驻留在设备上

- **现代 CPU 运行时**

  - VM 解释器实现快速启动
  - Turbine JIT (Cranelift) 处理热路径
  - 针对数值代码优化的分代 GC
  - 设计上保证内存安全（Rust）

- **跨平台 GPU 后端**

  - 使用 wgpu / WebGPU
  - 支持 **Metal (macOS)、DirectX 12 (Windows)、Vulkan (Linux)、WebGPU (浏览器)**
  - 当工作负载过小而无法在 GPU 上获得优势时自动回退到 CPU

- **异步运行时**

  - 评估基于 Rust futures 构建 —— 非阻塞设计，非后期添加
  - GPU 回读、交互式输入和长时运行脚本永远不会阻塞宿主
  - 语言级 `async`/`await` 与协作式任务已在路线图中
  - MATLAB 没有等效功能 —— RunMat 脚本可以在浏览器中交互运行而不冻结页面

- **WebAssembly 目标 + NPM 包**

  - 完整的运行时编译为 WASM 并作为本仓库的一部分发布 (`runmat-wasm`)
  - 可在 [NPM 上的 `runmat`](https://www.npmjs.com/package/runmat) 获取 —— 将执行、GPU 加速和绘图嵌入到任何 Web 应用中
  - 通过 WebGPU 在浏览器中实现 GPU 加速
  - 驱动 [浏览器沙盒](https://runmat.com/sandbox) —— 你的代码在本地运行，从不在服务器上运行

- **绘图**

  - 交互式 2D 和 3D 绘图，GPU 加速渲染
  - 30+ 种绘图类型：线图、散点图、条形图、曲面图、网格图、直方图、stem、errorbar、area、等高线图、饼图、plot3、imagesc 以及对数尺度变体
  - 图形句柄、子图状态、注释内置函数（`title`、`sgtitle`、`xlabel`、`legend`）和 3D 相机控制

  开源绘图引擎演示（可在 CLI 和浏览器沙盒中运行）：

  ![RunMat open-source 3D plotting demo](.github/assets/runmat-sandbox-3d-plotting.gif)

  <p align="center">
    <a href=".github/assets/runmat-sandbox-3d-plotting.gif"><strong>直接打开 GIF</strong></a> · <a href="https://runmat.com/sandbox"><strong>在浏览器沙盒中试用 →</strong></a>
  </p>

- **开源运行时**

  - 完整的运行时、GPU 引擎、JIT、GC 和绘图 —— 本仓库中的所有内容 —— 均采用 MIT 许可证
  - 小巧的二进制文件，CLI 优先设计

---

## 📊 性能

在蒙特卡罗模拟中，比 **NumPy 快 131 倍**，比 **PyTorch 快 7 倍**。硬件：Apple M2 Max，Metal。取 3 次运行的中位数。

![Monte Carlo speedup](https://web.runmatstatic.com/monte-carlo-analysis_speedup-b.svg)

<details>
<summary><strong>蒙特卡罗原始数据</strong></summary>

| 路径数 (模拟次数) | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|--------------------:|-----------:|-------------:|-----------:|---------------:|-----------------:|
| 250k   | 108.58 |   824.42 |  4,065.87 | 37.44× | 7.59× |
| 500k   | 136.10 |   900.11 |  8,206.56 | 60.30× | 6.61× |
| 1M     | 188.00 |   894.32 | 16,092.49 | 85.60× | 4.76× |
| 2M     | 297.65 | 1,108.80 | 32,304.64 |108.53× | 3.73× |
| 5M     | 607.36 | 1,697.59 | 79,894.98 |131.55× | 2.80× |

</details>

<details>
<summary><strong>4K 图像处理管线</strong> —— 比 NumPy 快 10 倍</summary>

![4K image pipeline speedup](https://web.runmatstatic.com/4k-image-processing_speedup-b.svg)

| B | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|---|---:|---:|---:|---:|---:|
| 4  | 142.97 | 801.29 | 500.34 | 3.50× | 5.60× |
| 8  | 212.77 | 808.92 | 939.27 | 4.41× | 3.80× |
| 16 | 241.56 | 907.73 | 1783.47 | 7.38× | 3.76× |
| 32 | 389.25 | 1141.92 | 3605.95 | 9.26× | 2.93× |
| 64 | 683.54 | 1203.20 | 6958.28 | 10.18× | 1.76× |

</details>

<details>
<summary><strong>逐元素数学运算</strong> —— 在 10 亿元素时比 PyTorch 快 144 倍</summary>

![Elementwise math speedup](https://web.runmatstatic.com/elementwise-math_speedup-b.svg)

| 数据点 | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|---|---:|---:|---:|---:|---:|
| 1M   | 145.15 | 856.41  |   72.39 | 0.50× | 5.90× |
| 2M   | 149.75 | 901.05  |   79.49 | 0.53× | 6.02× |
| 5M   | 145.14 | 1111.16 |  119.45 | 0.82× | 7.66× |
| 10M  | 143.39 | 1377.43 |  154.38 | 1.08× | 9.61× |
| 100M | 144.81 | 16,404.22 | 1,073.09 | 7.41× | 113.28× |
| 200M | 156.94 | 16,558.98 | 2,114.66 | 13.47× | 105.51× |
| 500M | 137.58 | 17,882.11 | 5,026.94 | 36.54× | 129.97× |
| 1B | 144.40 | 20,841.42 | 11,931.93 | 82.63× | 144.34× |

</details>

在较小数组上，Fusion 会将运算保留在 CPU 上，因此你仍然可以获得低开销和快速 JIT。

*查看 [benchmarks/](benchmarks/) 获取可复现的测试脚本、详细结果以及与 NumPy、PyTorch 和 Julia 的对比。*


---

## 🎯 快速开始

### 安装

```bash
# 快速安装 (Linux/macOS)
curl -fsSL https://runmat.com/install.sh | sh

# 快速安装 (Windows PowerShell)
iwr https://runmat.com/install.ps1 | iex

# Homebrew (macOS/Linux)
brew install runmat-org/tap/runmat

# 或从 crates.io 安装
cargo install runmat --features gui

# 或从源码构建
git clone https://github.com/runmat-org/runmat.git
cd runmat && cargo build --release --features gui
```

#### Linux 前置依赖

如需在 Linux 上使用 BLAS/LAPACK 加速，请在构建前安装系统 OpenBLAS 包：

```bash
sudo apt-get update && sudo apt-get install -y libopenblas-dev
```

### 运行你的第一个脚本

```bash
# 启动交互式 REPL
runmat

# 或运行已有的 .m 文件
runmat script.m

# 或将脚本通过管道传入 RunMat
echo "a = 10; b = 20; c = a + b" | runmat
```

### CLI 功能

```bash
# 检查 GPU 加速状态
runmat accel-info

# 对脚本进行基准测试
runmat benchmark script.m --iterations 5 --jit

# 创建快照以加速启动
runmat snapshot create -o stdlib.snapshot

# 查看系统信息
runmat info
```

查看 [CLI 文档](https://runmat.com/docs/cli) 获取完整的命令参考。

### Jupyter 集成

```bash
# 将 RunMat 注册为 Jupyter 内核
runmat --install-kernel

# 启动支持 RunMat 的 JupyterLab
jupyter lab
```

---

## 🧱 架构：CPU+GPU 性能

RunMat 使用分层 CPU 运行时加上一个融合引擎，自动为每个数学运算块选择 CPU 或 GPU。以下所有组件均为开源，并存在于本仓库中。

### 核心组件

| 组件              | 用途                                  | 技术 / 说明                                                  |
| ---------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| ⚙️ runmat-vm         | 即时启动的基线解释器 | HIR → 字节码编译器，基于栈的解释器                    |
| ⚡ runmat-turbine     | 热代码优化 JIT              | Cranelift 后端，针对数值工作负载优化                      |
| 🧠 runmat-gc         | 高性能内存管理       | 分代 GC，支持指针压缩                            |
| 🚀 runmat-accelerate | GPU 加速子系统               | 融合引擎 + 自动卸载规划器 + `wgpu` 后端               |
| 🔥 Fusion engine       | 折叠运算链，选择 CPU 或 GPU  | 构建运算图，融合运算，估算成本，保持张量驻留设备 |
| 🎨 runmat-plot       | 绘图层                           | 交互式 2D/3D 绘图；部分高级绘图类型仍在开发中 |
| 🌐 runmat-wasm       | 运行时的 WebAssembly 构建         | 在任何浏览器中运行；驱动 runmat.com 上的沙盒               |
| 📸 runmat-snapshot   | 快速启动快照                   | 二进制 blob 序列化 / 恢复                                 |
| 🧰 runmat-runtime    | 核心运行时 + 330+ 内置函数    | BLAS/LAPACK 集成和其他 CPU/GPU 加速运算    |


### 为什么重要

- **分层 CPU 执行** 提供快速启动和强大的单机性能。
- **融合引擎** 消除了大部分手动设备管理和内核调优。
- **GPU 后端** 通过 Metal / DirectX 12 / Vulkan 在 NVIDIA、AMD、Apple Silicon 和 Intel 上运行，无厂商锁定。

---

## 🚀 GPU 加速：Fusion 与自动卸载

RunMat 自动加速你的 MATLAB 代码在 GPU 上的运行，无需编写内核代码或重写代码。系统通过四个阶段工作：

### 1. 捕获数学运算
RunMat 构建一个"加速图"，捕获你运算的意图 —— 形状、运算类别、依赖关系和常量。此图提供了脚本计算内容的完整视图。

### 2. 决定在 GPU 上运行什么
融合引擎检测长链的逐元素运算和关联的归约，计划将它们作为组合的 GPU 程序执行。自动卸载规划器估算盈亏平衡点并智能地路由工作：
- **融合检测**：将多个运算组合为单次 GPU 调度
- **自动卸载启发式**：考虑元素数量、归约大小和矩阵乘法饱和度
- **驻留感知**：一旦值得，就将张量驻留在设备上

### 3. 生成 GPU 内核
RunMat 生成可移植的 WGSL (WebGPU Shading Language) 内核，跨平台工作：
- **Metal** 在 macOS 上
- **DirectX 12** 在 Windows 上
- **Vulkan** 在 Linux 上

内核只编译一次并缓存以备后续运行，消除重新编译开销。

### 4. 高效执行
运行时通过以下方式最小化主机↔设备传输：
- 一次上传张量并使其驻留
- 在 GPU 内存上直接执行融合内核
- 仅在需要时收集结果（例如用于 `fprintf` 或显示）

### 示例：自动 GPU 融合

```matlab
x = rand(1024, 1, 'single');
y = sin(x) .* x + 0.5;        % 融合: sin, multiply, add
m = mean(y, 'all');            % 归约保留在 GPU 上
fprintf('m=%.6f\n', double(m)); % 仅在输出端单次下载
```

RunMat 检测逐元素链（`sin`、`.*`、`+`），将它们融合为一次 GPU 调度，使 `y` 驻留在 GPU 上，仅在需要输出时才下载 `m`。

更多详情，请参阅 [RunMat GPU 简介](https://runmat.com/docs/accelerate/fusion-intro)。

---

## 💡 设计理念

RunMat 遵循 **默认快速运行时，开放扩展模型** 的设计理念：

- **高保真语言覆盖**：核心 MATLAB 语法、运算符、控制流、OOP 和索引 —— 不是子集，也不是新语言
- **丰富的内置函数**：330+ 函数覆盖核心 MATLAB 内置函数，持续添加中
- **分层执行**：VM 解释器实现快速启动，Turbine JIT 处理热代码
- **GPU 优先数学**：融合引擎自动将 MATLAB 代码转为快速 GPU 工作负载
- **单一可移植二进制文件**：一个静态二进制文件包含运行时、GPU 引擎和绘图 —— 快速启动、现代 CLI、Jupyter 内核支持
- **工具箱即包**：信号处理、统计、图像处理和其他领域以包的形式存在 —— 包管理器正在[积极设计中](https://runmat.com/docs/package-manager)

运行时有意将 GPU 加速、融合、JIT 和绘图作为一等子系统而非可选插件提供 —— 这正是 RunMat 默认快速的原因。领域特定的工具箱（信号处理、统计、图像处理等）以包的形式存在。使用你喜欢的任何编辑器，或内置的[浏览器 IDE](https://runmat.com/sandbox) 和即将推出的桌面应用。

查看 [设计理念](https://runmat.com/docs/design-philosophy) 获取完整的设计原理说明。

---

## 🌍 谁在使用 RunMat？

RunMat 为许多领域中的数组密集型数学而构建。

<div align="center">
<table>
<tr>
<td align="center" width="25%">
<strong>成像 / 地理空间</strong><br/>
4K+ 瓦片、归一化、辐射校正、QC 指标
</td>
<td align="center" width="25%">
<strong>量化 / 模拟</strong><br/>
蒙特卡罗风险、情景分析、协方差、因子模型
</td>
<td align="center" width="25%">
<strong>信号处理 / 控制</strong><br/>
滤波器、NLMS、大规模时序任务
</td>
<td align="center" width="25%">
<strong>研究人员和学生</strong><br/>
MATLAB 背景，需要在笔记本或集群上更快运行
</td>
</tr>
</table>
</div>

如果你用 MATLAB 编写数学代码并在 CPU 上遇到性能瓶颈，RunMat 就是为你而建的。

---

## 📚 快速链接

- **入门指南**
  - [安装](https://runmat.com/docs/getting-started)
  - [浏览器沙盒](https://runmat.com/sandbox)
  - [CLI 参考](docs/CLI.md)
  - [配置](docs/CONFIG.md)

- **语言与运行时**
  - [MATLAB 兼容性](docs/COMPATIBILITY.md)
  - [语言参考](docs/LANGUAGE.md)
  - [语言覆盖](docs/LANGUAGE_COVERAGE.md)
  - [内置函数库](docs/LIBRARY.md)
  - [设计理念](docs/DESIGN_PHILOSOPHY.md)

- **GPU 加速**
  - [RunMat GPU 简介](docs/INTRODUCTION_TO_RUNMAT_GPU.md)
  - [GPU 行为说明](docs/GPU_BEHAVIOR_NOTES.md)
  - [融合与自动卸载](https://runmat.com/docs/accelerate/fusion-intro)

- **绘图**
  - [绘图指南](docs/PLOTTING.md)

- **运行时架构**
  - [架构概述](docs/ARCHITECTURE.md)
  - [异步设计](docs/ARCH_ASYNC.md)
  - [文件系统](docs/FILESYSTEM.md)
  - [路线图](docs/ROADMAP.md)

- **嵌入与集成**
  - [NPM 包 (`runmat`)](bindings/ts/README.md)
  - [浏览器沙盒指南](docs/DESKTOP_BROWSER_GUIDE.md)

- **贡献**
  - [贡献指南](docs/CONTRIBUTING.md)
  - [开发者设置](docs/DEVELOPING.md)

- **博客**
  - [介绍 RunMat](https://runmat.com/blog/introducing-runmat)
  - [为什么选择 Rust](https://runmat.com/blog/why-rust)
  - [2026 年 MATLAB 替代品](https://runmat.com/blog/matlab-alternatives)
  - [如何在 MATLAB 中使用 GPU](https://runmat.com/blog/how-to-use-gpu-in-matlab)
  - [捍卫 MATLAB 白板式代码](https://runmat.com/blog/in-defense-of-matlab-whiteboard-style-code)

---

## 🤝 加入这项使命

RunMat 不仅仅是软件 —— 它是朝着 **开放、快速和可访问的科学计算** 的运动。我们正在构建数值编程的未来，我们需要你的帮助。

### 🛠️ 如何贡献

<table>
<tr>
<td width="33%">

**🚀 面向 Rust 开发者**
- 实现新的内置函数
- 优化 JIT 编译器
- 增强垃圾回收器
- 构建开发者工具

[**贡献代码 →**](https://github.com/runmat-org/runmat/discussions)

</td>
<td width="33%">

**🔬 面向领域专家**
- 添加数学函数
- 编写全面的测试
- 创建基准测试

[**加入讨论 →**](https://github.com/runmat-org/runmat/discussions)

</td>
<td width="33%">

**📚 面向其他所有人**
- 报告 bug 和功能请求
- 改进文档
- 创建教程和示例
- 传播信息

[**开始使用 →**](https://github.com/runmat-org/runmat/issues/labels/good-first-issue)

</td>
</tr>
</table>

### 💬 联系我们

- **GitHub Discussions**: [分享想法和获取帮助](https://github.com/runmat-org/runmat/discussions)
- **X (Twitter)**: [@runmat_com](https://x.com/runmat_com) 获取更新和公告
- **LinkedIn**: [RunMat](https://www.linkedin.com/company/runmat)

---

## 📜 许可证

RunMat 运行时是开源的，采用 **MIT 许可证（附归属要求）**。这意味着：

✅ **对所有人免费** —— 个人、学术界、大多数公司
✅ **永远开源** —— 运行时将始终保持免费和开放
✅ **允许商业使用** —— 自由嵌入到你的产品中
⚠️ **需要归属** —— 在公开发布中注明 "RunMat by Dystr"
⚠️ **特殊条款** —— 大型科学软件公司必须保持修改内容开源

RunMat Cloud 和桌面应用是构建在此开源运行时之上的独立产品。详情请参见 [runmat.com/pricing](https://runmat.com/pricing)。

查看 [LICENSE.md](LICENSE.md) 获取完整条款，或访问 [runmat.com/license](https://runmat.com/license) 查看常见问题。

---

**Built with ❤️ by [Dystr Inc.](https://dystr.com) and the RunMat community**

⭐ 如果 RunMat 对你有用，请在 **GitHub 上给我们点星**。

[**🚀 开始使用**](https://runmat.com/docs/getting-started) • [**📖 文档**](https://runmat.com/docs) • [**📝 博客**](https://runmat.com/blog) • [**💰 价格**](https://runmat.com/pricing) • [**𝕏 @runmat_com**](https://x.com/runmat_com) • [**LinkedIn**](https://www.linkedin.com/company/runmat)

---

*MATLAB® 是 The MathWorks, Inc. 的注册商标。RunMat 与 The MathWorks, Inc. 无关联、未获其认可、也未受其赞助。*
