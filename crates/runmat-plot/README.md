# RunMat Plot

RunMat Plot is a **high performance interactive plotting library for Rust**, designed to provide comprehensive 2D/3D plotting. It is built from the ground up for performance, featuring a GPU-accelerated rendering pipeline using `wgpu`, a modern theming system, and seamless integration with Jupyter notebooks.

As a core component of the [RunMat project](../../docs/ARCHITECTURE.md), it serves as the powerful handle-graphics engine for all visualization tasks.

## Key Features

- **GPU-Accelerated Rendering**: High-performance plotting powered by `wgpu` for fast, smooth, and interactive visualizations.
- **Comprehensive 2D Plots**: Support for line plots, scatter plots, bar charts, and histograms with extensive styling options.
- **Advanced 3D Visualization**: Create stunning 3D surface plots and point clouds with configurable colormaps and shading.
- **Interactive GUI**: A feature-rich interactive window built with `winit` and `egui`, offering smooth camera controls, zooming, panning, and UI overlays.
- **Multi-Plot Figures**: Combine multiple plot types in a single figure with automatic bounds computation, legends, and grid lines.
- **Language Compatibility**: A familiar, language-agnostic API for quickly creating plots (e.g., `plot()`, `surf()`, `scatter3()`).
- **Jupyter Notebook Integration**: Display plots directly in Jupyter notebooks as static images or interactive HTML widgets.
- **Modern Theming System**: A professional and configurable styling system with beautiful presets like `ModernDark`.

## Architecture

The `runmat-plot` crate is designed with a layered architecture to separate concerns, providing both high-level simplicity and low-level control.

-   **`src/core` - The Rendering Engine**: This is the heart of the library.
    -   `WgpuRenderer` abstracts over `wgpu` to manage render pipelines, shaders, and GPU buffers.
    -   `Scene` provides a scene graph to manage renderable objects, their transformations, and visibility.
    -   `Camera` implements both orthographic (2D) and perspective (3D) cameras with interactive navigation controls.
    -   `PlotRenderer` is a unified pipeline that handles rendering for both interactive windows and static exports, ensuring consistent output.

-   **`src/plots` - High-Level Plot Types**: This module defines the user-facing API for creating plots.
    -   `Figure` is the main container for a visualization, managing multiple overlaid plots.
    -   Structs like `LinePlot`, `ScatterPlot`, `SurfacePlot`, etc., encapsulate the data and styling for a specific plot type. They are responsible for generating `RenderData` (vertices, indices) to be consumed by the core renderer.

-   **`src/gui` - Interactive Windowing**: This module provides the interactive GUI.
    -   `PlotWindow` is the main entry point, creating a `winit` window and managing the event loop.
    -   `PlotOverlay` uses `egui` to draw UI elements (axes, grids, titles, controls) on top of the `wgpu` canvas.
    -   `thread_manager` and `native_window` contain robust, cross-platform logic to handle GUI operations, especially the main-thread requirements on macOS.

-   **`src/styling` - Theming & Appearance**: This module controls the visual style.
    -   `PlotThemeConfig` allows for complete customization of colors, fonts, and layout via RunMat's central configuration system (e.g., `.runmat.yaml`).
    -   `ModernDarkTheme` provides a professional, out-of-the-box dark theme.

## Crate Layout

The crate is organized to clearly separate rendering, plot logic, and UI.

```
runmat-plot/
├── Cargo.toml          # Dependencies and feature flags (gui, jupyter)
├── README.md           # This file
├── examples/           # Runnable examples (interactive_demo.rs, etc.)
├── shaders/            # WGSL shaders for GPU rendering pipelines
└── src/
    ├── core/           # Low-level rendering engine (WGPU, scene, camera)
    ├── data/           # Data processing, LOD, buffer management (TODO)
    ├── export/         # Static export to PNG, SVG, HTML (TODO)
    ├── gui/            # Interactive GUI window, controls, and UI overlays
    ├── jupyter/        # Jupyter Notebook integration
    ├── lib.rs          # Main library entry point and public API
    ├── plots/          # High-level plot types (LinePlot, SurfacePlot, etc.)
    ├── simple_plots.rs # Legacy static plotting with `plotters`
    └── styling/        # Theming, colors, and layout configuration
```

## Usage Examples

### 1. Simple Line Plot

Create a figure, add a line plot, and prepare it for rendering.

```rust
use runmat_plot::plots::{Figure, LinePlot, LineStyle};
use glam::Vec4;

let x: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
let y: Vec<f64> = x.iter().map(|&x| x.sin()).collect();

let mut figure = Figure::new()
    .with_title("Sine Wave")
    .with_labels("X-axis", "Y-axis")
    .with_grid(true);

let line_plot = LinePlot::new(x, y)?
    .with_style(Vec4::new(0.35, 0.78, 0.48, 1.0), 2.0, LineStyle::Solid)
    .with_label("sin(x)");

figure.add_line_plot(line_plot);
```

### 2. Multi-Plot Figure

Overlay multiple plot types in a single figure for comprehensive visualizations.

```rust
use runmat_plot::plots::{Figure, LinePlot, ScatterPlot, BarChart, MarkerStyle};
use glam::Vec4;

let mut figure = Figure::new().with_title("Sales Data Analysis");

// Add a bar chart for monthly sales
let sales_bars = BarChart::new(
    vec!["Jan".to_string(), "Feb".to_string(), "Mar".to_string()],
    vec![120.0, 135.0, 155.0]
)?.with_label("Monthly Sales");

// Add a line plot for the sales trend
let trend_line = LinePlot::new(vec![0.0, 1.0, 2.0], vec![120.0, 135.0, 155.0])?
    .with_label("Sales Trend");

// Add a scatter plot for KPI targets
let kpi_points = ScatterPlot::new(vec![0.0, 1.0, 2.0], vec![125.0, 130.0, 160.0])?
    .with_style(Vec4::new(1.0, 0.3, 0.3, 1.0), 8.0, MarkerStyle::Star)
    .with_label("KPI Targets");

figure.add_bar_chart(sales_bars);
figure.add_line_plot(trend_line);
figure.add_scatter_plot(kpi_points);
```

### 3. 3D Surface Plot

Create a 3D surface plot from a mathematical function.

```rust
use runmat_plot::plots::{SurfacePlot, ColorMap, ShadingMode};

let surface = SurfacePlot::from_function(
    (-3.0, 3.0),
    (-3.0, 3.0),
    (50, 50),
    |x, y| (-(x*x + y*y)).exp() * 2.0 - 1.0 // Gaussian
).unwrap()
    .with_colormap(ColorMap::Viridis)
    .with_shading(ShadingMode::Smooth);

let mut figure = Figure::new().with_title("3D Gaussian Surface");
// Figure currently only supports 2D plots, but 3D integration is planned.
```

### 4. Interactive Plotting

Launch a GPU-accelerated interactive window to display a figure.

```rust
// This requires the "gui" feature
// Add to Cargo.toml: runmat-plot = { version = "...", features = ["gui"] }

#[cfg(feature = "gui")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... create your figure as in the examples above ...
    let mut figure = Figure::new().with_title("Interactive Demo");
    // ... add plots to figure ...

    // Launch the interactive window
    runmat_plot::show_interactive_with_figure(&figure).await?;
    Ok(())
}
```

## Current Status & Known Issues

`runmat-plot` is under active development. The core GPU-accelerated architecture is implemented and functional, but there are several important limitations to be aware of:

### ⚠️ Current Limitations

-   **Triangle Rendering Issue (macOS Metal)**: Bar charts and filled 2D shapes currently render as thin lines instead of filled areas on macOS. This is a low-level WGPU/Metal triangle rasterization issue that affects the triangle rendering pipeline. Line plots work correctly.
    
    **Status**: Under investigation. The issue has been isolated to triangle primitive assembly in the Metal backend. All high-level geometry generation, vertex data, shaders, and draw calls are correct.
    
    **Workaround**: Line plots, scatter plots, and 3D point clouds work as expected. Bar charts will display but appear as outlines only.

-   **Sequential Plotting**: Opening multiple plot windows sequentially may cause EventLoop recreation errors on macOS due to `winit` limitations. The first plot window works correctly.
    
    **Workaround**: Restart the application between different plotting sessions.

-   **Legacy Export Backend**: Static PNG exports currently use a fallback `plotters`-based renderer, which may not match the interactive GPU rendering exactly.

### ✅ What Works

-   **Interactive GUI**: Full-featured interactive plotting windows with smooth navigation, zooming, and real-time controls
-   **Line Plots**: Complete 2D line plotting with multiple series, styling, and legends  
-   **Scatter Plots**: 2D and 3D scatter plots with configurable markers and colors
-   **3D Point Clouds**: High-performance 3D visualization with interactive camera controls
-   **GPU Performance**: Excellent rendering performance for large datasets via WGPU acceleration
-   **Cross-Platform**: Works on Windows, Linux, and macOS (with triangle rendering limitation)
-   **Language Compatibility**: Familiar `plot()`, `scatter()`, `scatter3()` function interface

## Roadmap & TODOs

Active areas of development, in priority order:

-   **🔥 PRIORITY: Fix Triangle Rendering (macOS Metal)**: Resolve the triangle rasterization issue preventing filled shapes from rendering correctly. Investigation points to WGPU primitive topology or Metal driver interaction.

-   **EventLoop Management**: Implement robust sequential plotting support to eliminate EventLoop recreation errors.

-   **Unified Static Export**: Replace the legacy `plotters`-based backend in `simple_plots.rs` with a unified headless rendering mode using the `wgpu` engine. This will ensure that exported PNGs and SVGs are pixel-perfect matches of the interactive plots.

-   **Complete Export Modules**: Fully implement the `src/export` modules for high-quality vector (SVG, PDF) and web (HTML, interactive widgets) outputs.

-   **Advanced Data Handling**: Implement the `src/data` modules for optimized GPU buffer management, level-of-detail (LOD) for large datasets, and advanced geometry processing.

-   **Volume Rendering**: Implement the `VolumePlot` type for 3D volumetric data visualization.

-   **Jupyter WebGL Widget**: Complete the WebGL-based interactive widget for Jupyter to provide a fully interactive experience within notebooks, matching the native GUI.

-   **Expanded Theming**: Add more built-in themes and expand the customizability of the styling system.

-   **3D in Figures**: Fully integrate 3D plots like `SurfacePlot` into the `Figure` system for multi-plot 3D scenes.

---

**For Developers**: If you're contributing to the triangle rendering fix, see the detailed technical investigation in the Git history. Key files: `crates/runmat-plot/src/plots/bar.rs`, `crates/runmat-plot/src/core/plot_renderer.rs`, and `crates/runmat-plot/src/gpu/shaders/vertex/triangle.rs`. The issue has been isolated to direct vertex drawing with `PrimitiveTopology::TriangleList` on Metal backend.
