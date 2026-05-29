# Plotting

Plotting in RunMat is a graphics system for turning arrays, sampled fields, distributions, and derived results into interpretable figures. 

These documents guide you through the plotting workflow, from choosing a representation, organizing axes, annotating and styling the figure, understanding the graphics state when refinement is needed, and then reasoning about rendering, persistence, and output when the figure becomes part of a larger workflow.

## Start here

### Plotting in RunMat

Plotting in RunMat is about building figures, organizing axes, choosing plot families that match the structure of the data, and refining the result into a readable figure. The overview in `PLOTTING_IN_RUNMAT.md` introduces figures, axes, plot objects, subplot-local state, and the general shape of plotting from first command to finished figure.

### Choosing the Right Plot Type

Different plot families preserve different structure in the data: continuity, discreteness, levels, field values, direction, category, or part-to-whole composition. The guide in `CHOOSING_THE_RIGHT_PLOT_TYPE.md` helps decide when to use `plot` rather than `scatter`, `imagesc` rather than `contourf`, `surf` rather than `mesh`, and similar neighboring choices.

### Styling Plots and Axes

After choosing a plot family, the next task is making the figure readable and intentional. Styling in RunMat lives partly on plot objects and partly on axes state, so good figures usually come from coordinating both layers. The guide in `STYLING_PLOTS_AND_AXES.md` explains how labels, limits, legends, colormaps, grids, and view work together, and treats styling as part of interpretation rather than decoration.

## Understand the graphics system

### Graphics Handles and Plot Objects

RunMat plotting is built on a graphics object model. Figures contain axes, axes contain plot objects and annotations, and handles make it possible to inspect and update those objects after they are created. The guide in `GRAPHICS_HANDLES.md` explains how figures, axes, legends, labels, and plotted objects fit together, and how `get` and `set` interact with the resulting stateful graphics system.

### Plot Replay and Export

Once a figure exists, it can be treated as more than the one-time output of code. The guide in `PLOT_REPLAY_AND_EXPORT.md` explains how a figure persists as scene state, how replay differs from recomputation, and how export differs from preserving a live plot scene. It is the right page to read when plotting becomes part of a broader workflow involving persistence, restoration, or fixed visual output.

## Understand rendering and GPU behavior

### GPU Plotting and Rendering Math

Plotting also has a mathematical side. Sampled values become geometry, axes and view state act as transformations, and the visible figure emerges through projection and finite sampling. The guide in `GPU_PLOTTING_AND_RENDERING_MATH.md` explains how plotting moves from arrays to geometry to observed image, and why a rendered plot is a sampled visual observation rather than a direct display of raw arrays.

### GPU Plotting and Residency

Plotting also has a computational side. GPU-resident arrays are not yet plots; they must become scene data, derived geometry, and renderable state. The guide in `GPU_PLOTTING_AND_RESIDENCY.md` explains why plotting terminates ordinary numeric fusion, how residency changes across source tensors and render buffers, and how to think realistically about plotting cost.

## After reading

After reading these documents, you should have a good understanding of the following topics:

- workflow: how a figure is built and refined
- representation: which plot family best matches the data
- styling: how the figure becomes readable
- state: how figures, axes, and plot objects are organized
- rendering: how a scene becomes a visible observation
- computation: how plotting interacts with GPU residency and render execution
- persistence: how a figure can be replayed or exported