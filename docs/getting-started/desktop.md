---
title: "Desktop"
category: "Getting Started"
section: "1.2"
last_updated: "May 29, 2026"
---

# RunMat Desktop

RunMat Desktop is the full local RunMat experience: an editor, file tree, console, plots, variable inspector, notebooks, and project settings in a high-performance native app. 

It runs MATLAB-syntax code with the same RunMat runtime used by the CLI and browser, but adds native file access and hardware GPU support for local projects.

Use Desktop when you want to work on files on your machine, inspect results interactively, and keep plots, stdout, variables, and run history visible while you iterate.

## Install

Download the latest build from [Download RunMat Desktop](/download/latest).

RunMat Desktop is available for:

| Platform | Package |
| --- | --- |
| macOS Apple Silicon | DMG |
| macOS Intel | DMG |
| Windows x86_64 | Installer |
| Linux x86_64 | AppImage |

After installing, open RunMat from your applications folder, start menu, or launcher.

## Open a Project

On first launch, you can either open a local folder or sign in to use cloud projects.

- **Local projects** use files on your machine and can run offline.
- **Cloud projects** sync through RunMat App and require sign-in.

For local work, choose or create a folder for your scripts. RunMat treats that folder as the project root.

## Run Your First Script

Create a file named `hello.m`:

```matlab
x = linspace(0, 2*pi, 200);
y = sin(x);

plot(x, y);
title("Hello from RunMat Desktop");
disp("Done");
```

Open the file in RunMat Desktop and click **Run** or use the run shortcut shown in the app.

Desktop sends the script to the RunMat runtime, then updates the runtime panel with:

- stdout from `disp`, `fprintf`, and other console output
- figures created by plotting commands
- workspace variables such as `x` and `y`
- diagnostics if parsing, compilation, or execution fails

The workspace stays available after the run, so you can inspect variables and rerun after editing.

## Work With Plots and Variables

Plots appear in the runtime panel as interactive figures. You can switch between figures, resize the panel, and export images when you need to save results.

Variables appear in the variables pane with type, shape, and residency information. Use the inspector to materialize values when you need to look at array contents without printing everything to the console.

## Use Project Settings

RunMat Desktop reads the same project manifests used by the CLI:

```toml
[package]
name = "my-project"

[sources]
roots = ["."]

[entrypoints.analysis]
path = "analysis.m"
```

Save this as `runmat.toml` in your project root. The CLI can run the same entrypoint with:

```bash
runmat run analysis
```

Desktop also stores app-specific project preferences, such as artifact location, notebook behavior, GPU preference, and run history settings. See [Projects](/docs/runtime/getting-started/projects) for project layout and [Configuration Reference](/docs/runtime/getting-started/config) for shared runtime settings.

## Desktop, Browser, and CLI

RunMat uses the same core runtime across all hosts:

| Host | Best for |
| --- | --- |
| Desktop | Local projects, native files, interactive plots, variables, and hardware GPU access. |
| Browser | Zero-install experiments and sharing small examples. |
| CLI | Scripts, automation, CI, benchmarking, and terminal workflows. |

You can move between them as your workflow changes. A script that runs in Desktop should also run from the CLI as long as it uses the same files, configuration, and runtime features.

## Next Steps

- Try the [Hello World](/docs/runtime/getting-started/hello-world) example.
- Learn the [Command Line Interface](/docs/runtime/getting-started/cli) for scripts and automation.
- Review [GPU Acceleration](/docs/runtime/gpu) if your project uses large arrays or plotting workloads.
