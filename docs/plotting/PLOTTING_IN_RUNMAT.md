# Plotting in RunMat

Plotting in RunMat is a graphics workflow for turning arrays, sampled fields, distributions, and derived data into interpretable figures.

RunMat supports a MATLAB compatible plotting input syntax while treating plotting as a first-class part of the system. 

Figures, axes, legends, labels, plot objects, and handle-based updates all belong to a graphics model. This model is the foundation of the plotting workflow which can begin with a quick exploratory command and then grow into a structured, stateful figure.

## The basic plotting workflow

Most plotting sessions follow the same broad pattern.

1. Choose a plot type that matches the structure of the data.
2. Create or select the axes in which the plot should appear.
3. Draw the plot.
4. Label and style the result.
5. Inspect or refine the figure through handles if needed.

A figure is often assembled in layers: data first, then interpretation, then annotation, then refinement.

## Figures, axes, and plot objects

In RunMat, a figure is the top-level graphics container. It may hold one axes or several axes arranged through `subplot`. Each axes then owns most of the visible plotting state for its panel: plotted objects, labels, legends, limits, scales, color settings, and view.

Plot-producing builtins create plot objects inside an axes. Annotation builtins such as `title`, `xlabel`, `ylabel`, and `zlabel` create or update axes-owned text objects. `legend` creates or updates an axes-owned legend object. These objects participate in the same handle-based graphics system, so they can be inspected and restyled after creation.

This object structure makes plotting compositional. A figure can contain several axes. Each axes can contain several plots and annotations. Later commands can modify those existing objects rather than starting over from scratch.

## Current axes and subplot-local state

Many plotting commands operate on the current axes by default. This makes short scripts convenient, but it also means that axes selection matters. `subplot` both selects a panel and returns an axes handle for that panel. Once multiple axes exist, later commands apply to the current or explicitly targeted axes rather than to the figure as a whole.

This matters because titles, labels, legends, limits, scales, colorbars, colormaps, and views are axes-local state. A two-panel figure does not have one shared title state, one shared legend state, or one shared view state. Each subplot behaves as its own plotting context.

For simple figures, current-axes behavior is often enough. For multi-panel figures, explicit axes handles make the code easier to reason about.

## Choosing a plot family

The first analytical choice in plotting is usually the plot family. Different plot types preserve different structure in the data.

- `plot`, `stairs`, `stem`, and `errorbar` emphasize ordered samples.
- `scatter` and `scatter3` emphasize isolated observations.
- `bar` emphasizes categorical comparison.
- `histogram` emphasizes distributional structure after binning.
- `image`, `imagesc`, `contour`, `contourf`, `surf`, and `mesh` emphasize different aspects of sampled scalar fields.
- `quiver` emphasizes vector fields.
- `plot3` emphasizes connected trajectories in 3-D.
- `pie` emphasizes part-to-whole structure.

The right choice depends on what the reader should notice. If continuity matters, use a plot that makes continuity visible. If values live on a grid, decide whether the reader should see cells, level sets, or height. If direction matters, use a representation that preserves direction.

## Common plot families at a glance

The following table is a practical starting point.

| Plot type | Best first interpretation |
| --- | --- |
| `plot` | ordered samples over a continuous domain |
| `scatter` | isolated observations without connecting order |
| `bar` | category or group comparison |
| `histogram` | sample distribution after binning |
| `imagesc` | dense scalar field shown through color |
| `contour` | level sets of a scalar field |
| `contourf` | level regions of a scalar field |
| `surf` | scalar field shown as height |
| `mesh` | scalar field shown as height with visible grid structure |
| `quiver` | vector field with direction and magnitude |
| `plot3` | connected trajectory in 3-D |
| `scatter3` | point cloud in 3-D |
| `pie` | part-to-whole proportions |

For a fuller decision guide, see `CHOOSING_THE_RIGHT_PLOT_TYPE.md`.

## Styling and annotation

Initial drawing is only part of plotting. Labels, legends, limits, view, grid lines, and color settings are part of the interpretation of the figure.

Common annotation and styling builtins include:

- `title`, `xlabel`, `ylabel`, `zlabel`
- `legend`
- `grid`, `box`
- `axis`, `view`
- `colormap`, `colorbar`, `shading`
- `get`, `set`

These commands are not just decorative. A title establishes context. Limits determine visible domain. Color limits and colormap determine scalar interpretation. View determines how a 3-D object is observed. Styling is therefore part of the analytical workflow, not only presentation polish.

## Plotting with handles

Many plotting and annotation calls return handles. A handle is the reference to a graphics object after it has been created. Handles make it possible to inspect an object, change its properties, and build more structured plotting code.

This matters especially when a script becomes more than a one-liner. In a multi-panel figure, axes handles make targeting explicit. In a reusable plotting function, line or surface handles make later refinement straightforward. In annotation-heavy figures, text and legend handles keep the figure editable after creation.

The full object model is described in `GRAPHICS_HANDLES.md`, but the basic practical idea is simple: capture handles for objects you expect to revisit.

## Scalar fields, views, and 3-D reasoning

Scalar fields can be visualized in several different ways. `imagesc` treats the field as values over a sampled grid and uses color as the main carrier of structure. `contour` and `contourf` emphasize level sets and level regions. `surf` and `mesh` embed the field as height in 3-D. These are different interpretations of the same underlying data, not interchangeable styles.

View also matters. A 3-D surface is not simply a 2-D image with a different style. It is a geometric object observed from a chosen orientation. This means that color, height, and viewpoint together determine what the reader perceives. For the mathematical side of that story, see `GPU_PLOTTING_AND_RENDERING_MATH.md`.

## GPU plotting in context

Plotting may begin from GPU-resident arrays, but plotting is not only a numerical tensor operation. A rendered figure requires scene construction, axes-local state, geometry or image interpretation, and frame rendering. GPU plotting is therefore about more than whether the source data lives on the device.

This distinction becomes important when reasoning about performance and residency. Source tensors, derived plot representations, and render buffers are related but not identical. For the computational side of that story, see `GPU_PLOTTING_AND_RESIDENCY.md`.

## A small example workflow

The following example illustrates several core ideas at once: subplot-local state, plot-family choice, annotation, and handle-based refinement.

```matlab
ax1 = subplot(1, 2, 1);
h1 = plot(0:0.1:1, sin(2*pi*(0:0.1:1)));
title("Signal");
xlabel("Time");
ylabel("Amplitude");
set(h1, "DisplayName", "sine");
legend();

ax2 = subplot(1, 2, 2);
imagesc(peaks(40));
title("Scalar field");
colorbar;
set(ax2, "XLim", [1 40], "YLim", [1 40]);
```

The left axes shows an ordered signal, so `plot` is the natural plot family. The right axes shows a dense scalar field, so `imagesc` is a natural starting point. Each subplot has its own title and styling state. The line handle on the left is refined with `set`, while the right subplot is configured through its axes handle. This is a typical RunMat plotting workflow: choose a representation, draw it, then refine the figure through graphics state.

## Where to go next

This overview is the entry point to the broader plotting documentation.

- Read `CHOOSING_THE_RIGHT_PLOT_TYPE.md` for a decision-oriented guide to plot families.
- Read `GRAPHICS_HANDLES.md` for the object model behind figures, axes, legends, labels, and plot handles.
- Read `GPU_PLOTTING_AND_RENDERING_MATH.md` for the mathematical story of plotting as geometry, projection, and sampled observation.
- Read `GPU_PLOTTING_AND_RESIDENCY.md` for the computational story of device residency, scene construction, and render execution.

Together, these pages describe plotting in RunMat as both a practical workflow and a coherent graphics system.
