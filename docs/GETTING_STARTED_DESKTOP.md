# Getting Started with RunMat Desktop

RunMat Desktop is a browser-based development environment for writing and running MATLAB-style code with GPU acceleration. It provides a full IDE experience—code editor, file explorer, console output, and live plotting—all running locally in your browser.

---

## What is RunMat Desktop?

RunMat Desktop lets you:

- **Write MATLAB-style code** in a full-featured editor with syntax highlighting
- **Run scripts instantly** with automatic CPU/GPU acceleration
- **See live plots** rendered directly in the browser
- **Inspect variables** and console output in real time

All computation happens locally in your browser. There's no server-side execution—your code and data never leave your machine.

---

## Accessing RunMat Desktop

Visit **[runmat.org/sandbox](https://runmat.org/sandbox)** to launch the desktop environment.

No installation required. Works in Chrome, Edge, Firefox, and Safari.

> **Note:** For maximum GPU performance, Chrome or Edge on macOS/Windows is recommended. These browsers support WebGPU, which enables full hardware acceleration.

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

- Full code editor with syntax highlighting for `.m` files
- Multiple file tabs (click a file to open, double-click to pin)
- Unsaved changes shown with a dot indicator
- **Cmd/Ctrl+S** to save

### Runtime Panel (Right)

- **Run button** to execute the current file
- **Figure tabs** for viewing plots
- **Console** for standard output and input prompts
- **Variables pane** showing workspace variables, their types, shapes, and whether they're on CPU or GPU

All three panels are resizable—drag the borders to adjust.

---

## First Run Walkthrough

When you first open RunMat Desktop, a demo file (`demo.m`) is already loaded:

```matlab
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

```matlab
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

For large arrays and intensive math, RunMat automatically fuses operations and runs them on your GPU (via WebGPU). This happens transparently—you don't need to write any special GPU code.

The Variables pane shows **Residency** for each variable:
- **cpu** — data is on the CPU
- **gpu** — data is on the GPU (faster for large operations)

### Plotting

Plots render directly in the browser using GPU-accelerated graphics. Supported plot types include:
- `plot` — line plots
- `scatter` — scatter plots
- `bar` — bar charts
- `hist` — histograms

More plot types are being added.

---

## Sandbox Storage

Your files are stored in your browser's local storage (IndexedDB). This means:

- ✅ Files persist across page refreshes
- ✅ No account required
- ⚠️ Files are specific to this browser
- ⚠️ Clearing browser data will delete your files

**Tip:** For important work, copy your code to a local file on your computer.

---

## Common Questions

### Why is my script slow on the first run?

The first execution includes compilation time. Subsequent runs are faster because compiled code is cached.

### Why don't I see GPU acceleration?

GPU acceleration requires:
- A browser that supports WebGPU (Chrome 113+, Edge 113+)
- A compatible GPU with up-to-date drivers

If WebGPU isn't available, RunMat falls back to CPU execution. You can still run all the same scripts—they just won't get GPU speedups.

Check the sidebar's runtime status indicator to see if GPU is enabled.

### Can I use my existing MATLAB files?

Yes. RunMat supports standard `.m` file syntax. Many MATLAB scripts run with few or no changes.

See the [Language Coverage](LANGUAGE_COVERAGE.md) guide for supported features.

### How do I handle user input?

Use `input()` for interactive prompts:

```matlab
name = input('Enter your name: ', 's');
fprintf('Hello, %s!\n', name);
```

When your script calls `input()`, the console prompts you for input.

### What if I want full desktop performance?

The browser-based sandbox has some GPU performance limits due to browser security sandboxing. For maximum performance:

1. **Download RunMat CLI** — Run scripts from your terminal with full native GPU access
2. **Use RunMat Desktop App** — Coming soon, provides native performance in a desktop window

Visit [runmat.org/download](https://runmat.org/download) for installation options.

---

## What's Next

Now that you've run your first script:

- **Explore the built-in functions** — See the [Library Reference](LIBRARY.md)
- **Learn about GPU acceleration** — Read [Introduction to RunMat GPU](INTRODUCTION_TO_RUNMAT_GPU.md)
- **Install the CLI** — For local file access and scripting: [CLI Guide](CLI.md)
- **Try the benchmarks** — Compare RunMat performance against NumPy and PyTorch

---

## Getting Help

- **Documentation**: [runmat.org/docs](https://runmat.org/docs)
- **GitHub Issues**: [Report bugs or request features](https://github.com/runmat-org/runmat/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/runmat-org/runmat/discussions)

