# Choosing the Right Plot Type

Choosing a plot type is choosing which structure in the data should be visually primary. A good plot makes the important mathematical or analytical structure easy to see. A poor plot may still be technically correct, but it can suggest continuity where none exists, hide variation that matters, or emphasize geometry that is irrelevant to the question being asked.

Different plot families represent different kinds of objects. Some plots emphasize ordered samples, some emphasize isolated observations, some emphasize categorical comparison, and some emphasize scalar or vector fields over a domain. Choosing well therefore begins with the structure of the data rather than with the appearance of a plotting command.

## Start with the structure of the data

Before choosing a plotting function, it helps to ask what kind of mathematical object the data represents.

- Is it an ordered sequence of samples along a line or in time?
- Is it a set of observations with no meaningful connecting order?
- Is it a scalar field sampled over a grid?
- Is it a vector field with direction and magnitude at each location?
- Is it a distribution built from many repeated samples?
- Is it a set of category totals or group comparisons?
- Is the third dimension structurally meaningful, or only available?

These questions are often more important than the number of dimensions alone. A matrix can represent an image, a scalar field, a table of categories, or a sampled surface. The right plot type depends on what the entries mean and what the reader should notice.

## A quick decision map

The following guide is deliberately simple.

| If your data is mainly... | Usually start with... |
| --- | --- |
| ordered samples along a continuous domain | `plot` |
| unordered point observations | `scatter` |
| piecewise-constant sampled values | `stairs` |
| discrete samples tied to a baseline | `stem` |
| values with uncertainty intervals | `errorbar` |
| category totals | `bar` |
| a sample distribution | `histogram` |
| a scalar field on a grid | `imagesc`, `contour`, `contourf`, `surf`, or `mesh` |
| a vector field on a grid | `quiver` |
| a connected 3-D trajectory | `plot3` |
| a 3-D point cloud | `scatter3` |
| part-to-whole proportions | `pie` |

This map is only a first pass. The more important question is what the plot should help the reader perceive.

## Ordered samples: `plot`, `stairs`, `stem`, and `errorbar`

Ordered samples are among the most common plotting inputs, but even within this category the right representation depends on what order means.

### `plot`

Use `plot` when the samples belong to an ordered domain and the visual connection between neighboring values is meaningful. Time series, sampled trajectories, and smoothly varying measurements usually belong here. A line plot tells the reader that adjacency matters.

`plot` is often the right choice when the main structure is trend, smooth variation, or relative change across an ordered domain.

### `stairs`

Use `stairs` when the data is piecewise constant between sample locations. This is the right visual language when each sample represents a value that persists until the next boundary. A staircase plot communicates interval structure more honestly than a smooth or straight-line interpolation.

If the value should be understood as holding over an interval, `stairs` is usually better than `plot`.

### `stem`

Use `stem` when each sample should remain visibly discrete but still tied to a baseline. Stem plots are useful when the positions of the samples matter, but the reader should not infer a continuous curve between them. Discrete-time signals and impulse-like sequences often fit this form.

If the key fact is that the samples are separate rather than continuously connected, `stem` is often more faithful than `plot`.

### `errorbar`

Use `errorbar` when uncertainty, tolerance, or interval width is part of the data meaning. Error bars do more than decorate central values. They communicate that each point should be read together with a range or estimate of variability.

If uncertainty matters analytically, it should usually appear directly in the plot rather than only in surrounding text.

### Common comparisons

- Use `plot` rather than `scatter` when order and continuity should be emphasized.
- Use `stairs` rather than `plot` when values are constant over intervals.
- Use `stem` rather than `plot` when samples are discrete and should remain visually separate.
- Use `errorbar` rather than plain markers when interval information is part of the result.

## Point sets: `scatter` and `scatter3`

Use scatter plots when the data consists of observations, not a connected path. A scatter plot tells the reader that point location matters, but the order of the samples is either absent or not analytically important. This is often the correct choice for measurement clouds, experimental observations, and point distributions.

`scatter3` is the three-dimensional extension of the same idea. It is useful when the data is genuinely a point cloud in 3-D and when the third coordinate carries structural meaning rather than merely offering another way to decorate the display.

The central distinction from line plots is simple: a scatter plot does not imply continuity between neighboring points. If connecting the points would create a false suggestion of trajectory or progression, use `scatter` or `scatter3` instead of `plot` or `plot3`.

## Categorical magnitudes: `bar`

Use `bar` when the horizontal organization is categorical or group-based rather than continuous. A bar chart is usually the right choice when the main task is comparison across a finite set of labels, groups, or named categories.

The most common mistake in this area is using `plot` for values that are not actually sampled over a continuous domain. A line between category values suggests intermediate structure that does not exist. If adjacency should not imply continuity, `bar` is usually the better representation.

## Distributions: `hist` and `histogram`

A histogram is not an ordered-sample plot. It is a plot of distributional structure after binning. The visual object represents how values accumulate into intervals, not how they vary along a sequence.

Use `histogram` when you want a persistent plotted histogram object with plotting-handle and property workflows. RunMat supports `hist` for backwards compatibility when the older count-oriented MATLAB-style workflow is the better fit for the task. In practice, `histogram` is often the better default when the histogram itself is meant to be styled, inspected, or treated as a first-class plot object.

It is also useful to distinguish histograms from bar charts. A bar chart usually compares categories. A histogram usually summarizes the distribution of a continuous or repeated sample set. The geometry may look similar, but the semantics are different.

## Scalar fields on a grid

Scalar fields are one of the richest areas of plot choice because several plot families can represent the same sampled field in very different ways. The most important question is not simply whether the data is on a grid, but what the reader should see in that grid.

### `image` and `imagesc`

Use `image` or `imagesc` when cell-wise values or raster structure are primary. These plots are often the clearest way to show a dense matrix or scalar field when the reader should focus on value variation across the domain rather than on geometric height.

`imagesc` is especially appropriate when color scaling is part of the interpretation. It is often the best choice when the scalar field should be understood as a colored map rather than as a raised surface.

### `contour`

Use `contour` when level sets are more important than every cell or every sample. Contour lines are useful when equal-value structure matters: ridges, valleys, thresholds, and level boundaries become visually explicit.

If the reader should compare where the field reaches the same value rather than how every local cell is colored, `contour` is often the right representation.

### `contourf`

Use `contourf` when the main structure lies in level regions rather than only in contour lines. Filled contours often provide a strong middle ground between line-only level information and dense image-like scalar maps. They are especially good when the field is best understood in bands or zones.

### `surf`

Use `surf` when height itself is meaningful and a geometric embedding in 3-D helps interpretation. A surface plot makes the scalar field visible as shape. This can be powerful when slope, curvature, or relative elevation is part of the point.

At the same time, `surf` introduces view dependence and possible occlusion. It is excellent when shape is the message, but often less clear than a 2-D field representation when exact regional comparison matters more than geometric impression.

### `mesh`

Use `mesh` when the grid structure itself should remain visually legible. A mesh shows the same surface-like data as `surf`, but preserves stronger awareness of the sampled lattice. This can be useful when the reader should see both height and the underlying discretization.

### `surfc` and `meshc`

Use `surfc` or `meshc` when both height and level structure should be visible together. These are useful when the reader should understand the 3-D shape and its level organization at the same time.

### Common comparisons for scalar fields

- Use `image` or `imagesc` when cell values are primary.
- Use `contour` when level sets are primary.
- Use `contourf` when level regions are primary.
- Use `surf` when height and geometric shape are primary.
- Use `mesh` when height matters but the sampled grid should remain more visible.
- Use `surfc` or `meshc` when both height and level structure should be seen together.

## Vector fields: `quiver`

Use `quiver` when direction and magnitude over position are the main structure. A vector field is not merely a scalar field with extra decoration. It represents orientation as well as size, and that directional information is often the most important part of the data.

If the task is to show flow, force, gradient direction, or local motion, `quiver` is usually the correct visual language. A scalar-field plot of magnitudes may still be useful as a companion view, but it cannot replace vector geometry when direction matters analytically.

## Three-dimensional trajectories and point clouds

Three-dimensional plots should be used when the third coordinate is structurally meaningful rather than merely available.

- Use `plot3` for connected trajectories in 3-D.
- Use `scatter3` for unconnected observations in 3-D.
- Use `surf` or `mesh` for scalar fields embedded as gridded surfaces in 3-D.

The main caution is that 3-D adds view dependence and occlusion. It can reveal true geometry, but it can also make comparison harder. If the third dimension is not conceptually essential, a 2-D representation is often clearer.

## Part-to-whole data: `pie`

Use `pie` only when angular part-to-whole structure is genuinely the point. A pie chart can be useful for a small number of categories when the main message is composition of a whole. It is usually weak when precise quantitative comparison matters.

If the reader needs to compare magnitudes accurately, a bar chart is often clearer than a pie chart. `pie` is best reserved for a narrow class of part-to-whole comparisons where exact measurement is less important than broad proportional structure.

## When not to use a plot type

Negative guidance is often more useful than positive guidance alone.

- Do not use `plot` for unordered observations.
- Do not use `bar` when the horizontal axis is a true continuous domain.
- Do not use `pie` when precise comparison is important.
- Do not use `surf` only because 3-D looks impressive if `imagesc` or `contourf` shows the field more clearly.
- Do not use 3-D when the extra dimension adds clutter but no real structure.
- Do not use a scalar-field view when direction is the real quantity of interest and `quiver` is the honest representation.

## Choosing among scalar-field plots

Scalar fields are worth revisiting because they present one of the most common high-level decisions in plotting. The right plot type depends on what the reader should perceive first.

| If the reader should see... | Prefer... |
| --- | --- |
| exact cell values or dense raster variation | `image` / `imagesc` |
| level sets | `contour` |
| level regions | `contourf` |
| height and shape | `surf` |
| height together with visible grid structure | `mesh` |
| both height and level structure | `surfc` / `meshc` |

This choice is not only aesthetic. It changes the visual metaphor through which the field is interpreted. The same sampled values can be read as an image, a set of bands, a set of level curves, or a raised surface. Each emphasizes different structure.

## Choosing between 2-D and 3-D

Three-dimensional plots are most useful when geometry in three dimensions is itself the message. They are less useful when the same structure can be seen more clearly in 2-D. A 3-D surface may communicate height well, but it may also hide values behind perspective and occlusion. A contour map or image can often make regional variation easier to compare.

As a practical rule, use 3-D when the reader truly needs 3-D geometry. Use 2-D when the third dimension adds spectacle more than understanding.

## Plot type and computational shape

Different plot types also ask the plotting system to construct different kinds of representations.

- `plot` and `scatter` are relatively direct sample-to-geometry mappings.
- `surf` and `mesh` use gridded geometry.
- `image` and `imagesc` align naturally with dense sampled fields.
- `contour` and `contourf` require derived level geometry.
- `quiver` requires vector-to-arrow expansion.
- `histogram` requires binning before visible bars exist.

This should not usually dominate plot choice, but it is still useful to think about what representation the plotting system is being asked to build. A plot type is not only a visual choice. It is also a choice of geometric and computational model.

## Practical decision examples

The most useful way to choose a plot is often to begin with the analysis task.

- If the data is a time series or ordered signal, start with `plot`.
- If the data is a set of independent observations, start with `scatter`.
- If the data is a matrix of temperatures, elevations, or intensities over a grid, choose among `imagesc`, `contourf`, and `surf` based on whether values, levels, or height should be primary.
- If the data gives wind or flow direction over a domain, start with `quiver`.
- If the data is a 3-D path through space, use `plot3`.
- If the task is to compare counts across bins of repeated samples, use `histogram`.
- If the task is to compare a small number of category totals, use `bar`.

These examples are simple, but the deeper rule is the same in each case: begin with the structure that should remain visible in the final figure.

## Common mistakes

Several mistakes appear repeatedly across plotting workflows.

- Using `plot` for categories and accidentally implying continuity.
- Using 3-D because it looks richer even when it obscures the actual comparison.
- Using `pie` for too many categories.
- Using `surf` when a scalar field is better understood as an image or contour map.
- Using `hist` when a persistent histogram object would be more useful.
- Using scalar-only displays when direction is the most important part of the data.

Most of these mistakes come from choosing a plot by appearance rather than by structure.

## See also

The builtin references for `plot`, `scatter`, `stairs`, `stem`, `errorbar`, `bar`, `hist`, `histogram`, `image`, `imagesc`, `contour`, `contourf`, `surf`, `mesh`, `surfc`, `meshc`, `quiver`, `plot3`, `scatter3`, and `pie` provide syntax-level details. The conceptual documents on graphics handles, rendering math, and GPU plotting explain how these plot families behave once chosen.
