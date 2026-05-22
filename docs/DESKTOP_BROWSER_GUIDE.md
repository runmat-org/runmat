---
image: "https://web.runmatstatic.com/runmat-sandbox-dark.png"
imageAlt: "RunMat Sandbox"
jsonLd:
  "@context": "https://schema.org"
  "@type": "WebApplication"
  name: "RunMat Sandbox"
  url: "https://runmat.com/sandbox"
  applicationCategory: "DeveloperApplication"
  operatingSystem: "Any"
  offers:
    "@type": "Offer"
    price: "0"
    priceCurrency: "USD"
  featureList:
    - "GPU-accelerated MATLAB-style code execution via WebGPU"
    - "Interactive 2D and 3D plotting"
    - "Real-time variable inspector with type and shape tracking"
    - "Syntax highlighting and error diagnostics"
    - "Persistent storage and automatic file versioning (with free account)"
    - "Built-in agent with runtime execution and workspace inspection"
    - "Compatibility-guided script adaptation with reviewable diffs"
---

# Browser Guide

The RunMat sandbox is a browser-based development environment for writing and running MATLAB-style code with GPU acceleration. It provides a full IDE experience—code editor, file explorer, console output, and live plotting—all running locally in your browser.

---

## Sandbox vs Sandbox + App

| Feature | Sandbox (no account) | Sandbox + App (signed in) |
|---|---|---|
| Installation required | No | No |
| Account required | No | Yes (free) |
| GPU acceleration | WebGPU (browser-throttled) | WebGPU (browser-throttled) |
| Interactive IDE | Yes | Yes |
| Interactive plotting | Yes | Yes |
| Variable inspector | Yes | Yes |
| File storage | In-memory (cleared on tab close) | App (persists across sessions and devices) |
| File versioning | No | Automatic on every save |
| Project sharing | No | Paid plans |
| Built-in agent | Yes | Yes |

---

## What is the RunMat sandbox?

The sandbox lets you:

- **Write MATLAB-style code** in a full-featured editor with syntax highlighting
- **Run scripts instantly** with automatic CPU/GPU acceleration
- **See live plots** rendered directly in the browser
- **Inspect variables** and console output in real time
- **Ask the built-in agent** to run code, check results, and propose reviewable edits for script adaptation

All computation happens locally in your browser via WebAssembly — there's no server-side execution. If you sign in for cloud storage or use the built-in agent, data is sent to RunMat's servers and the configured LLM provider respectively.

---

## Accessing the sandbox

Visit **[runmat.com/sandbox](https://runmat.com/sandbox)** to launch the sandbox.

No installation required. Works in Chrome, Edge, Firefox, and Safari.

> **Note:** For GPU acceleration, use a browser that supports WebGPU (Chrome 113+, Edge 113+, Safari 18+, Firefox 139+). Resource limits may vary by browser—some browsers are more conservative to preserve battery life.

---

## The Interface

When you open the sandbox, you'll see four main areas:

```text
┌──────────┬──────────────────────┬──────────────────┐
│ Sidebar  │       Editor         │  Runtime Panel   │
│ (Files)  │  (Code + Tabs)       │  (Run, Vars,     │
│          │                      │   Console, Figs) │
│          ├──────────────────────┤                  │
│          │    Agent Panel       │                  │
│          │  (Chat + Diffs)      │                  │
└──────────┴──────────────────────┴──────────────────┘
```

### Sidebar (Left)

- **File tree** showing your project files
- Click files to open them in the editor
- Use the **+** button to create new files or folders

### Editor (Center)

- Full code editor with syntax highlighting and language services (e.g., red underlines for errors)
- Multiple file tabs (click a file to open, double-click to pin)
- Unsaved changes shown with a dot indicator
- **Cmd/Ctrl+S** to save

### Runtime Panel (Right)

- **Run button** to execute the current file
- **Figure tabs** for viewing plots
- **Console** for standard output and input prompts
- **Variables pane** showing workspace variables, their types, shapes, and whether they're on CPU or GPU

### Agent Panel (Below Editor)

The agent panel is a chat interface connected to your live runtime session. The agent can execute code, read workspace variables and plot output, and search project files. When it suggests changes, they appear as diffs you can accept or reject individually. Sessions are persisted as project files so you can revisit them later.

All four panels are resizable—drag the borders to adjust.

---

## First Run Walkthrough

When you first open the sandbox, a set of example files is already loaded.

### Step 1: Click "Run demo.m"

In the Runtime Panel on the right, click the purple **▶ Run demo.m** button.

### Step 2: View the Output

After execution:
- The **Console** shows the printed output
- The **Variables** tab (at the bottom of the Runtime Panel) displays all workspace variables with their classes, shapes, and residency (CPU/GPU)

### Step 3: Modify and Re-run

Try editing the script. For example, add a plot:

```matlab:runnable
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

## Notebook Editor

The sandbox includes a notebook editor for mixing markdown and code in a single document. Create a `.md` file in the sidebar and it opens in notebook mode.

- Markdown cells for documentation, headings, and LaTeX math
- Code cells with the same GPU acceleration and variable inspector as regular `.m` scripts
- Run cells individually or execute all cells in sequence
- Output (console and plots) appears inline below each code cell

---

## How It Works

### Local Execution

The sandbox runs entirely in your browser using WebAssembly. When you click Run:

1. Your script is compiled and executed locally
2. RunMat automatically decides whether to use CPU or GPU for each operation
3. Results stream to the console and variable inspector in real time

### GPU Acceleration

For large arrays and intensive math, RunMat automatically fuses operations and runs them on your GPU (via WebGPU). This happens transparently—you don't need to write any special GPU code. Browsers throttle GPU and CPU usage to preserve battery life, so sandbox benchmarks will be slower than the CLI running the same code natively. For full GPU throughput, use the [CLI](https://runmat.com/docs/cli).

The Variables pane shows **Residency** for each variable:
- **cpu** — data is on the CPU
- **gpu** — data is on the GPU (faster for large operations)

### Plotting

Plots render directly in the browser using GPU-accelerated graphics. RunMat includes 40+ plotting builtins:

- **2D:** `plot`, `scatter`, `bar`, `histogram`, `hist`, `area`, `stairs`, `stem`, `errorbar`, `pie`, `contour`, `contourf`, `image`, `imagesc`, `imshow`, `quiver`, `heatmap`
- **3D:** `plot3`, `surf`, `surfc`, `mesh`, `meshc`, `scatter3` (interactive rotate, zoom, pan)
- **Log-scale:** `semilogx`, `semilogy`, `loglog`
- **Figure management:** `figure`, `subplot`, `hold`, `clf`, `cla`, `close`, `title`, `sgtitle`, `xlabel`, `ylabel`, `zlabel`, `legend`, `colorbar`, `colormap`, `axis`, `grid`, `box`, `shading`, `view`, `drawnow`, `pause`

Advanced/specialized chart types (`polar`, `geobubble`, `wordcloud`) are not yet supported. For the full list, see the [Compatibility](/docs/compatibility) page.

---

## Storage

### Without an account (sandbox mode)

Without signing in, the sandbox runs entirely in your browser. Your files are stored in memory within your browser tab, and the RunMat runtime executes locally via WebAssembly.

- No account required
- Code execution is local — nothing is uploaded to run your script
- Using the built-in agent sends context to the configured LLM provider
- Files are cleared when you close or refresh the tab

### With a RunMat App account

Sign in to get cloud storage, automatic file versioning, and access to your projects from any device. Your files sync to RunMat App and persist across sessions.

- **Hobby tier** — 100 MB storage, unlimited projects, automatic version history
- **Pro** — 10 GB storage, version history ($30/mo per user)
- **Team** — 100 GB storage, project sharing, SSO ($100/mo per user)

#### File versioning

Every time you save a file, RunMat App records a version automatically. You can browse and restore previous versions of any file at any time. Version history is available on all App tiers (Hobby, Pro, and Team). Stored versions count toward your storage quota.

When signed in, your code is transmitted to RunMat App for storage and sync. Execution still happens locally in your browser—your code is not executed on our servers.

See [pricing](https://runmat.com/pricing) for full plan details. For local persistence without an account, use the [CLI](https://runmat.com/docs/cli).

---

## Common Questions

### Why don't I see GPU acceleration?

GPU acceleration requires:
- A browser that supports WebGPU (Chrome 113+, Edge 113+, Safari 18+, Firefox 139+)
- A compatible GPU with up-to-date drivers

If WebGPU isn't available, RunMat falls back to CPU execution. You can still run all the same scripts—they just won't get GPU speedups.

### Can I use my existing MATLAB files?

Yes. RunMat supports standard `.m` file syntax. Many MATLAB scripts run with few or no changes. If a script uses a function RunMat doesn't ship yet, the built-in agent can often help adapt it — running your code, reading the diagnostics, and proposing reviewable edits. See [Agent-assisted migration](/docs/compatibility#agent-assisted-migration) for details.

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

1. **Install the RunMat CLI** — Run scripts from your terminal with full native GPU access. See the [CLI guide](https://runmat.com/docs/cli).
2. **RunMat desktop app** — Coming soon. Will provide a native IDE with full local file system access.

---

## Teams and Enterprise

### Team features (App Team plan)

- **Project sharing** — share projects with collaborators in your organization. Each editor is billed as a seat.
- **Team workspaces** — organize projects under your team's organization.
- **SSO / SAML** — single sign-on for your identity provider.
- **Priority support** — faster response times from the RunMat team.

### Enterprise

For organizations that need on-premises or air-gapped deployment:

- **Self-hosted deployment** — run RunMat on your own infrastructure.
- **Data residency and ITAR compliance** — keep data in your environment.
- **Audit logs** — track access and changes for compliance.
- **Offline licensing** — no internet connection required.
- **SCIM provisioning** — automated user management.

See [pricing](https://runmat.com/pricing) for plan details. For Enterprise inquiries, [contact sales](https://runmat.com/contact?type=enterprise) or email [team@runmat.com](mailto:team@runmat.com).

---

## What's Next

Now that you've run your first script:

- **Explore the built-in functions** — See the [Function Reference](https://runmat.com/docs/matlab-function-reference)
- **Check MATLAB compatibility** — What works, what doesn't, and agent-assisted migration: [Compatibility](https://runmat.com/docs/compatibility)
- **Learn about GPU acceleration** — Read [GPU Acceleration](https://runmat.com/docs/accelerate/fusion-intro)
- **Install the CLI** — For local file access and scripting: [CLI Guide](https://runmat.com/docs/cli)
- **Try the benchmarks** — Compare RunMat performance: [Benchmarks](https://runmat.com/benchmarks)

---

## Getting Help

- **Documentation**: [runmat.com/docs](https://runmat.com/docs)
- **GitHub Issues**: [Report bugs or request features](https://github.com/runmat-org/runmat/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/runmat-org/runmat/discussions)

