# Styling Plots and Axes

Good plotting style clarifies which structure the reader should see. A readable plot makes important comparisons easy to notice, keeps secondary structure available without making it dominant, and reduces visual choices that distract from the analytical task.

## The two layers of styling

In RunMat, styling decisions belong to one of two layers: plot-object styling and axes styling.

### Plot-object styling

Plot-object styling controls the appearance of the plotted data itself. This includes properties such as color, line width, marker appearance, filled faces, edge emphasis, transparency where supported, and display-name metadata used by legends.

### Axes styling

Axes styling controls the context in which the data is read. This includes titles, axis labels, limits, scaling, grid state, box state, colormap, colorbar, and 3-D view. These choices shape how the data is interpreted.

This distinction is useful in practice. If the question is how one series should stand out, the change belongs on the plot object. If the question is how the reader should interpret the domain, scale, or field values, the change belongs on the axes.

## Start with the analytical purpose

Styling works best when it begins with the analytical purpose of the figure. Before changing colors or line widths, it helps to ask a few basic questions.

- What should the reader notice first?
- What should remain visible but secondary?
- Is the figure about trend, uncertainty, distribution, field structure, or comparison across groups?
- Should the reader compare values, levels, directions, or shape?
- Is the plot meant for close inspection, quick recognition, or panel-to-panel comparison?

These questions often determine style more reliably than aesthetic preference alone. A plot with one primary series should not usually style every series equally. A scalar field shown through color should not usually hide its quantitative meaning by omitting the colorbar. A 3-D plot should not choose a view that hides the very structure the plot was meant to reveal.

## Styling plot objects

Plot objects carry the visual weight of the data. Their styling should make the intended structure clear without introducing unnecessary clutter.

### Lines and markers

Line plots usually work best when the visual emphasis matches the analytical emphasis. If a line is the main object of interest, line width and color should make it easy to follow. If the line is only context, it should usually appear lighter or less visually dominant than the primary series.

Markers are useful when individual sample locations matter, but they can quickly become distracting if every point is emphasized equally in a dense series. A practical rule is to let the line carry continuity and let markers carry discreteness only when the samples themselves are important.

```matlab
h1 = plot(0:0.1:1, sin(2*pi*(0:0.1:1)));
set(h1, "LineWidth", 2, "DisplayName", "signal");
hold on;
h2 = plot(0:0.1:1, 0.5*cos(2*pi*(0:0.1:1)));
set(h2, "DisplayName", "reference");
legend();
grid on;
title("Primary signal with reference");
```

### Scatter and point emphasis

Scatter plots place most of their meaning in point position, so size and color should support spatial reading rather than overwhelm it. Larger or darker markers can help emphasize sparse or especially important observations, but dense point clouds often benefit from restraint. When many points overlap, excessive marker size can obscure the actual distribution.

The main goal is to preserve the structure of the point set. If styling turns a cloud into a blot of ink, the plot has lost part of its analytical value.

### Bars, surfaces, and filled objects

Filled objects require a balance between face visibility and boundary clarity. Bar plots often benefit from faces that remain easy to compare without overemphasizing edges. Surface-like plots often benefit from color and shading choices that clarify field structure rather than only making the figure more dramatic.

Edges are useful when boundaries matter. They are less useful when they dominate the reading of a dense field. Transparency can help in overlapping scenes, but it should be used to reveal structure rather than to add visual complexity for its own sake.

### Display names and legends

When a legend is needed, `DisplayName` is usually the cleanest way to connect plot-object identity to legend entries. This keeps labels attached to the plotted objects themselves rather than managing them as separate, disconnected strings.

If a plot contains several related objects, styling and naming should work together. A legend should clarify distinctions already present in the figure rather than rescue a figure whose styling never made those distinctions visible.

## Styling axes

Axes styling defines how the reader enters the plot. It determines domain, scale, context, and often the interpretation of visual encodings such as color.

### Labels and titles

Titles and axis labels should frame the figure in analytical terms. A good title usually gives the reader the point of the figure, not just the name of the builtin. Axis labels should identify variables and units whenever the units matter.

In practice, labels should remove ambiguity rather than repeat obvious information. If the x-axis is time in seconds, that should be stated clearly. If the plotted quantity is dimensionless, that can often be stated more lightly.

### Limits and scaling

Limits are part of interpretation, not only layout. They determine what portion of the data domain is visible and how tightly the figure focuses on that domain. Scale settings matter just as much. A logarithmic axis changes the meaning of distance and comparison along that dimension.

Use limits and scaling to support the comparison the figure is meant to invite. Wide limits can provide context. Tight limits can reveal local structure. Log scaling is appropriate when multiplicative change or order-of-magnitude comparison matters more than absolute linear difference.

### Grid and box

Grid lines help the reader estimate position and compare values, but they should support the data rather than compete with it. A light grid can make reading easier. A heavy grid can become the most visually dominant object in the axes.

The same is true of the axes box. It can help frame the plotting area, but it should not usually become one of the strongest shapes in the figure unless that framing is truly needed.

### Legends

Legends are most useful when the plot contains distinctions that cannot be read directly from position alone. They are less useful when a figure contains only one obvious series or when direct labeling would be clearer.

When a legend is present, it should explain real differences in the plot rather than simply restate what is already obvious. A legend is explanatory context, not a substitute for thoughtful object styling.

### Colormap and colorbar

When color carries quantitative meaning, colormap choice and colorbar visibility become part of the interpretation of the figure. A scalar field shown through color should usually make that mapping legible. The reader should not have to guess whether color is decorative or data-bearing.

Different colormaps emphasize different structure. Some make level changes easy to see. Others make gradual variation easier to follow. In all cases, the colormap should support reading of the field rather than calling attention only to itself.

```matlab
imagesc(peaks(40));
colormap("turbo");
colorbar;
title("Scalar field");
```

### View

For 3-D plots, view is both a styling choice and an interpretive choice. The same surface can appear smooth, steep, cluttered, or revealing depending on the orientation from which it is observed. A good view exposes the structure the plot was meant to communicate.

If the chosen view hides important geometry, the figure may be technically correct but analytically weak.

```matlab
surf(peaks(40));
shading interp;
colormap("parula");
colorbar;
view(45, 30);
zlabel("Height");
title("Surface structure");
```

## Plot styling versus axes styling

Many styling questions become easier once the boundary between plot-object state and axes state is clear.

- line color, line width, marker choice, and display name belong to the plot object
- title, axis labels, limits, scales, grid, box, colormap, colorbar, and view belong to the axes
- legend entries depend on plot objects, but the legend itself is an axes-owned object
- shading affects how surface-like plots are read through axes-local interpretation state

In practice, this means a styling change should be applied where its meaning lives. If the change belongs to one series, target the plot object. If the change belongs to the reading of the whole panel, target the axes.

## Common workflows

Styling is easiest to learn through small, repeatable workflows.

### Emphasize one series among several

When several series appear together, usually only one or two should dominate visually. The main series may use a stronger line width, a clearer color, or a more visible marker choice. Context series can be lighter or thinner.

### Make a scalar field readable

A scalar field often becomes much easier to interpret when colormap, colorbar, limits, and plot family are chosen together. `imagesc`, `contourf`, and `surf` may all show the same data, but each asks for different styling decisions.

### Clean up a multi-panel figure

In a subplot figure, each axes should remain readable on its own while still fitting the figure as a whole. That usually means consistent labels and scales when comparison across panels matters, and intentional differences only when the panels serve different analytical roles.

### Style a 3-D plot deliberately

For a surface or 3-D trajectory, view, labels, shading, and color interpretation should work together. A visually dramatic 3-D figure is not necessarily an analytically useful one.

## Multi-panel figures

Multi-panel figures deserve special care because each subplot carries its own axes-local state. Titles, labels, legends, limits, colorbars, colormaps, and views do not automatically belong to the figure as a whole. They belong to specific axes.

This means consistency across panels should be intentional. If the goal is direct comparison, similar axes should usually share comparable scaling and visual language. If the goal is contrast, then different styling choices should be clearly motivated by different analytical roles.

Subplot figures become much easier to manage when each axes is treated as its own readable unit within a larger composition.

```matlab
ax1 = subplot(1, 2, 1);
plot(1:5, [1 4 2 5 3]);
title("Left panel");
grid on;

ax2 = subplot(1, 2, 2);
plot(1:5, [5 4 3 2 1]);
title("Right panel");
set(ax2, "XLim", [1 5], "YLim", [0 6]);
box on;
```

## Common styling mistakes

Several mistakes appear repeatedly in plotting workflows.

- styling every series equally when one should be primary
- choosing a 3-D view that hides the structure of interest
- adding a legend when the plot is already self-explanatory
- using strong grids or boxes that dominate the data
- omitting the colorbar when color carries quantitative meaning
- overusing markers, edge lines, or transparency until the plot becomes noisy
- treating styling as a final cosmetic step instead of part of the interpretive process

Most of these mistakes arise when visual choices are made independently of the analytical task.

## Styling with builtins and with `set`

Common axes-level styling tasks are often easiest through builtins such as `title`, `xlabel`, `ylabel`, `legend`, `grid`, `box`, `axis`, `view`, `colormap`, `colorbar`, and `shading`. These are convenient when the change affects the reading of the axes as a whole.

`set` becomes especially useful when styling should be explicit and object-specific. A handle-based workflow is often the clearest way to adjust one line, one legend, one text object, or one subplot without relying too heavily on implicit current-axes behavior.

The deeper object model behind these operations is described in `GRAPHICS_HANDLES.md`, but the practical rule is simple: use builtins for common panel-level actions and use handles plus `set` for targeted control.

## See also

Read `PLOTTING_IN_RUNMAT.md` for the broader plotting overview, `CHOOSING_THE_RIGHT_PLOT_TYPE.md` for guidance on plot-family selection, and `GRAPHICS_HANDLES.md` for the underlying handle and object model. The rendering-math and GPU-residency documents provide deeper conceptual background for view, color, geometry, and computational behavior.
