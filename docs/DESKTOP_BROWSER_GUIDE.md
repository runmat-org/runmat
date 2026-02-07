# Desktop & Browser Guide

RunMat Desktop is a browser-based development environment for writing and running MATLAB-style code with GPU acceleration. It provides a full IDE experience—code editor, file explorer, console output, and live plotting—all running locally in your browser.

---

## What is RunMat Desktop?

RunMat Desktop lets you:

- **Write MATLAB-style code** in a full-featured editor with syntax highlighting
- **Run scripts instantly** with automatic CPU/GPU acceleration
- **See live plots** — interactive 2D and 3D surfaces rendered directly in the browser
- **Catch errors before you run** — hover to see matrix dimensions, with red underlines for dimension mismatches
- **Inspect variables** and console output in real time
- **Trace execution** with detailed diagnostic logging

All computation happens locally in your browser. There's no server-side execution—your code and data never leave your machine.

---

## Accessing RunMat Desktop

Visit **[runmat.org/sandbox](https://runmat.org/sandbox)** to launch the desktop environment.

No installation required. Works in Chrome, Edge, Firefox, and Safari.

> **Note:** For GPU acceleration, use a browser that supports WebGPU (Chrome 113+, Edge 113+, Safari 18+, Firefox 139+). Resource limits may vary by browser—some browsers are more conservative to preserve battery life.

---

## The Interface

When you open RunMat Desktop, you'll see three main areas:

```
┌─────────────┬────────────────────────┬─────────────────────┐
│   Sidebar   │        Editor          │   Runtime Panel     │
│  (Files)    │  (Code + Tabs)         │  (Run, Output, Vars)│
└─────────────┴────────────────────────┴─────────────────────┘
```

### Sidebar (Left)

- **File tree** showing your project files
- Click files to open them in the editor
- Use the **+** button to create new files or folders

### Editor (Center)

- Full code editor with syntax highlighting and language services
- **Shape tracking**: hover over any variable to see its type and matrix dimensions as they flow through your code
- **Live diagnostics**: red underlines for syntax errors and dimension mismatches (e.g., multiplying incompatible matrices) — caught before you run
- Multiple file tabs (click a file to open, double-click to pin)
- Unsaved changes shown with a dot indicator
- **Cmd/Ctrl+S** to save

### Runtime Panel (Right)

- **Run button** to execute the current file
- **Figure tabs** for viewing plots (2D and interactive 3D — rotate, zoom, and pan)
- **Console** for standard output and input prompts
- **Variables pane** showing workspace variables, their types, shapes, and whether they're on CPU or GPU
- **Trace / Logs** for execution tracing and diagnostic output — step through what each line does

All three panels are resizable—drag the borders to adjust.

---

## First Run Walkthrough

When you first open RunMat Desktop, a demo file (`demo.m`) is already loaded:

```matlab:runnable
a = 0:pi/100:2*pi;

b = sin(a);
c = cos(a);

g = b.^2 + c.^2;

sum_g = sum(g);
max_g = max(g);

disp([ ...
   'Sum of g: ', num2str(sum_g), ' | ', ...
   'Max of g: ', num2str(max_g) ...
]);
```

### Step 1: Click "Run demo.m"

In the Runtime Panel on the right, click the purple **▶ Run demo.m** button.

### Step 2: View the Output

After execution:
- The **Console** shows the printed output
- The **Variables** tab (at the bottom of the Runtime Panel) displays all workspace variables with their classes, shapes, and residency (CPU/GPU)

### Step 3: Modify and Re-run

Try editing the script. For example, add a plot:

```matlab
plot(a, b);
title("Sine Wave");
```

Click **Run** again. A new **Figure** tab appears in the Runtime Panel showing your plot.

---

## Creating Your Own Scripts

### Create a New File

1. Click the **+** button in the sidebar
2. Select **Create file (.m)**
3. Enter a filename (e.g., `my_script.m`)
4. Press Enter

### Write Your Code

The editor supports standard MATLAB syntax:

```matlab:runnable
% Element-wise operations (automatically GPU-accelerated for large arrays)
x = rand(1000000, 1);
y = sin(x) .* exp(-x);

% Built-in functions
avg = mean(y);
fprintf('Average: %.6f\n', avg);

% Plotting
plot(x(1:1000), y(1:1000));
xlabel('x');
ylabel('sin(x) * exp(-x)');
```

### Run Your Script

- Click **Run** in the Runtime Panel, or
- Use the keyboard shortcut **Ctrl+Enter** (Windows/Linux) or **Cmd+Enter** (macOS)

---

## How It Works

### Local Execution

RunMat Desktop runs entirely in your browser using WebAssembly. When you click Run:

1. Your script is compiled and executed locally
2. RunMat automatically decides whether to use CPU or GPU for each operation
3. Results stream to the console and variable inspector in real time

### GPU Acceleration

For large arrays and intensive math, RunMat automatically fuses operations and runs them on your GPU (via WebGPU). This happens transparently—you don't need to write any special GPU code. (Using MATLAB with NVIDIA GPUs? See our [MATLAB on NVIDIA GPUs](/blog/matlab-nvidia-gpu) guide.)

The Variables pane shows **Residency** for each variable:
- **cpu** — data is on the CPU
- **gpu** — data is on the GPU (faster for large operations)

### Plotting

Plots render directly in the browser using GPU-accelerated graphics. Supported plot types include:
- `plot` — 2D line plots
- `scatter` — scatter plots
- `surf`, `mesh` — interactive 3D surface plots (rotate, zoom, and pan with the mouse)
- `plot3` — 3D line plots

3D plots are fully interactive: click and drag to rotate, scroll to zoom, and right-click to pan. Plots render as crisp, high-fidelity surfaces regardless of data size.

> **Note:** Additional plot types (bar charts, histograms, subplots, figure handles) are still in development.

### Type & Shape Tracking

RunMat tracks the type and dimensions of every variable as it flows through your code:

- **Hover to see dimensions**: place your cursor over any variable to see its current type and shape (e.g., `double [3×4]`)
- **Dimension mismatch warnings**: if you try to multiply matrices with incompatible sizes, a red underline appears in the editor *before* you run the script
- **LSP-powered hovers**: built-in function documentation appears on hover, showing expected inputs and outputs

This works in both the browser sandbox and the desktop app.

### Execution Tracing

When you run a script, the Trace panel shows a step-by-step log of what each line did:

- Which functions were called and what they returned
- Diagnostic messages and warnings
- Clear, readable error messages with line numbers

Use the Trace panel (in the Runtime Panel on the right) to debug unexpected behavior without adding `disp()` statements everywhere.

---

## Sandbox Storage

The sandbox runs entirely in your browser. Your files are stored in memory within your browser tab, and the RunMat runtime executes locally via WebAssembly. In sandbox mode, your code stays on your machine—it's never sent to our servers.

This means:

- ✅ No account required
- ✅ Your code stays local—we can't see it
- ⚠️ Files are cleared when you close or refresh the tab

**Tip:** Copy your code to a local file before closing the tab. We'll be adding sign-in and download options soon to make saving easier.

---

## Common Questions

### Why don't I see GPU acceleration?

GPU acceleration requires:
- A browser that supports WebGPU (Chrome 113+, Edge 113+, Safari 18+, Firefox 139+)
- A compatible GPU with up-to-date drivers

If WebGPU isn't available, RunMat falls back to CPU execution. You can still run all the same scripts—they just won't get GPU speedups.

### Can I use my existing MATLAB files?

Yes. RunMat supports standard `.m` file syntax. Many MATLAB scripts run with few or no changes.

See the [Language Coverage](/docs/language-coverage) guide for supported features.

### How do I handle user input?

Use `input()` for interactive prompts:

```matlab:runnable
name = input('Enter your name: ', 's');
fprintf('Hello, %s!\n', name);
```

When your script calls `input()`, the console prompts you for input.

### What if I want full desktop performance?

Browsers limit how much GPU and CPU a website can use to preserve battery life. For maximum performance:

1. **Download RunMat CLI** — Run scripts from your terminal with full native GPU access
2. **Use RunMat Desktop App** — Coming soon, provides native performance in a desktop window

Visit [runmat.org/download](https://runmat.org/download) for installation options.

---

## What's Next

Now that you've run your first script:

- **Explore the built-in functions** — See the [Library Reference](/docs/library)
- **Learn about GPU acceleration** — Read [Introduction to RunMat GPU](/docs/accelerate/fusion-intro)
- **Try interactive plotting** — Add `surf(peaks)` to a script and rotate the 3D surface
- **Install the CLI** — For local file access and scripting: [CLI Guide](/docs/cli)
- **Try the benchmarks** — Compare RunMat performance against NumPy and PyTorch

---

## Getting Help

- **Documentation**: [runmat.org/docs](https://runmat.org/docs)
- **GitHub Issues**: [Report bugs or request features](https://github.com/runmat-org/runmat/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/runmat-org/runmat/discussions)

