# Graphics Handles and Plot Objects

Each plotting command in RunMat creates or updates graphics objects such as figures, axes, labels, legends, and plotted series. Those objects have identity, state, and relationships to one another, and RunMat exposes that model through graphics handles.

This document explains how the graphics object model works in practice. It shows what kinds of objects exist, how handles are created, how `get` and `set` interact with those objects, and why axes-local state matters so much in subplot workflows. If you want to understand what a plotting command returns, what object a later command will modify, or how to inspect and restyle an existing plot, this is the right place to start.

## What graphics handles are

A graphics handle is a value that refers to a graphics object after that object has been created. Instead of redrawing a plot from scratch every time you want to inspect or restyle it, you can keep the handle, ask the object for its current properties, and update those properties directly. This is what makes plotting stateful rather than write-only.

In practice, a handle is how plotting code stays connected to the figure it built. A line handle refers to a line object, an axes handle refers to a specific axes, and a legend handle refers to the legend attached to an axes. The handle is not the plotted data itself; it is the reference you use to work with the resulting graphics object.

The smallest handle workflow is: create an object, keep the handle, inspect it, and then change it. This is the foundation of the plotting stateful graphics system.

```matlab
h = plot(1:5, [1 4 2 5 3]);
get(h, "Type")
set(h, "LineWidth", 2)
```

Here `plot` creates a line object and returns its handle. `get` asks the object what kind of graphics object it is, and `set` updates one of its style properties without recreating the line.

## What kinds of graphics objects exist

RunMat's graphics object model is built around a small number of object families. At the top level are figure objects, which hold one or more axes. Each axes can then own plotted series, axis labels, titles, legends, view state, limits, and related plotting configuration.

The most important object families are figures, axes, text objects such as `title` and axis labels, legend objects, and plot objects such as lines, scatter plots, bars, surfaces, contours, images, pie charts, quiver fields, and areas. Different object kinds expose different property sets, but they all participate in the same handle-based model.

The important categories of graphics objects are:

- figure objects
- axes objects
- text objects created by `title`, `xlabel`, `ylabel`, and `zlabel`
- legend objects created by `legend`
- plot objects created by plotting commands such as `plot`, `scatter`, `bar`, `stairs`, `stem`, `errorbar`, `surf`, `mesh`, `contour`, `histogram`, `image`, `pie`, `quiver`, `area`, `plot3`, and `scatter3`

## How the object model is organized

The graphics object model is hierarchical. Figures contain axes, and axes contain most of the user-visible plotting state. That includes plotted children, axis labels, legend state, limits, scales, colormap settings, and view settings. A plotted line belongs to an axes, not directly to the figure. A title also belongs to an axes, not to the figure as a whole.

This hierarchy matters because commands like `get`, `set`, `legend`, `title`, `axis`, and `view` operate on specific objects. Once you understand which object owns which state, plotting behavior becomes much easier to predict. Properties such as `Type`, `Parent`, and `Children` are the user-facing way to inspect that structure, with `Parent` being the clearest way to connect a plotted object back to its owning axes.

Conceptually, the hierarchy looks like this:

- a `figure` contains one or more `axes`
- an `axes` owns its title, labels, legend, limits, scales, and plotted children
- each plotted child belongs to exactly one axes at a time

That structure is visible through ordinary handle operations.

```matlab
ax = subplot(1, 1, 1);
h = plot(1:4, [1 3 2 4]);

get(h, "Type")
get(h, "Parent")
get(ax, "Title")
```

In this example, the line handle reports its own type and its parent axes, while the axes handle exposes one of the axes-owned objects attached to that plotting context.

## How handles are created

Handles are created as part of normal plotting workflows. `figure` returns a figure handle, `subplot` returns an axes handle, plot-producing builtins return plot-object handles, label functions such as `title` and `xlabel` return text handles, and `legend` returns a legend handle. These return values are how plotting code can continue to work with objects after they appear.

Not every plotting-related command creates a new object. Commands such as `grid`, `box`, `axis`, `colormap`, `shading`, and `colorbar` usually modify the state of the current axes instead. That distinction is important: some commands create objects you can keep, while others primarily reconfigure the current plotting context.

As a rule of thumb:

- use returned figure handles when figure selection matters later
- use returned plot handles when you want to restyle or inspect a series later
- use returned axes handles when you want to configure a specific subplot later
- use returned text and legend handles when you want to restyle annotations after creating them

The main handle-returning families are easy to summarize:

| Builtin family | Returned handle kind | What it refers to | Typical next step |
| --- | --- | --- | --- |
| `figure` | figure handle | one figure | inspect `CurrentAxes` or switch figure context |
| `subplot` | axes handle | one subplot axes | configure limits, scales, labels, grid, view |
| `plot`, `scatter`, `bar`, `stem`, `errorbar`, `surf`, `contour`, `image`, `histogram`, `pie`, `quiver`, `area`, `plot3`, `scatter3` | plot handle | one plotted object | restyle the series, set `DisplayName`, inspect `Parent` |
| `title`, `xlabel`, `ylabel`, `zlabel` | text handle | one axes-owned text object | restyle annotation text |
| `legend` | legend handle | one axes-owned legend object | inspect or restyle legend state |
| `grid`, `box`, `axis`, `colormap`, `shading`, `colorbar` | usually no new graphics object | current axes state | mutate plotting state rather than keep a new handle |

## Current axes and explicit handle targeting

Many plotting commands act on the current axes by default. This makes short scripts convenient, because you can create an axes and then issue a sequence of plotting and styling commands without repeating the target each time. In single-axes code, that default is often exactly what you want.

In multi-axes code, explicit handles are more reliable. Once a figure contains multiple subplot panels, it becomes important to know which axes a command will affect. Capturing axes handles and using them intentionally makes plotting code easier to read, easier to maintain, and less sensitive to ordering mistakes.

```matlab
ax1 = subplot(1, 2, 1);
plot(1:5, 1:5);
title("Left");

ax2 = subplot(1, 2, 2);
plot(1:5, [1 4 2 5 3]);
title("Right");

set(ax1, "XLim", [0 6]);
set(ax2, "Grid", true);
```

This example works because each call to `subplot` makes a specific axes current and returns its handle. The later `set(ax1, ...)` and `set(ax2, ...)` calls then target each axes explicitly instead of relying on whatever happens to be current at that point in the script.

## How property access works

The graphics object model becomes most visible when you use `get` and `set`. `get(h)` returns a snapshot of the supported properties for the object referenced by `h`, while `get(h, "Property")` reads one property. `set(h, "Property", value)` changes object state directly, and multiple property-value pairs can be applied in one call.

Property access is handle-specific rather than universal. A line handle exposes line-oriented properties such as color, line width, marker settings, and display name. An axes handle exposes limits, scaling, grid state, box state, colormap, colorbar, and view configuration. A text handle exposes text content and styling. The property model is consistent, but the surface depends on the object type.

Some properties mainly describe the object hierarchy:

- `Type` identifies what kind of object a handle refers to
- `Parent` identifies the owning figure or axes
- `Children` exposes attached child objects for handle kinds that report them

Other properties are object-specific. For example, a line may expose `Color` and `LineWidth`, while an axes may expose `XLim`, `YLim`, `Grid`, `Box`, or view-related properties. When learning a new handle type, `get(h)` is often the fastest way to see its property surface.

## What state lives on each object type

One of the best ways to understand the graphics object model is to ask which object owns which kind of state. Figures own top-level figure identity and current-axes selection. Axes own most of the plotting environment: limits, scales, grid and box behavior, legends, labels, color settings, and view state. Plot objects own data and object-specific styling such as line width, marker appearance, baseline configuration, or face transparency.

This split is the reason many plotting workflows feel natural once the model clicks. If you want to change how an axes behaves, work with the axes. If you want to restyle one plotted series, work with that plot object. If you want to inspect an annotation, work with the text or legend handle returned by the relevant command.

In practical terms:

- figure handles are about top-level plotting context
- axes handles are about plotting environment and layout-local state
- text and legend handles are about annotation objects attached to one axes
- plot handles are about one plotted object and its own data or styling

## Subplot-local state

Each subplot panel is its own axes object, with its own independent state. That means titles, axis labels, legends, limits, log scaling, grid state, box state, colorbar state, colormap state, and 3-D view settings are all local to the axes you are working with. They do not automatically apply across the whole figure.

This is one of the most important ideas in the entire plotting system. Once a figure has multiple axes, later commands affect the current or explicitly targeted axes, not some figure-wide pool of shared plotting state. Understanding this explains why storing subplot handles is so useful, and why otherwise similar commands can change different parts of a figure depending on context.

If a script creates two subplot panels, each panel can have a different title, a different x-axis range, a different legend, a different colormap, or a different 3-D view. That independence is not a special case; it is the normal consequence of axes being separate graphics objects.

## Text and legend objects

Text and legend objects are first-class parts of the graphics object model. Commands such as `title`, `xlabel`, `ylabel`, and `zlabel` create or update axes-owned text objects, and `legend` creates or updates an axes-owned legend object. These objects are not just side effects; they have handles, properties, and inspectable state.

That means annotation workflows can use the same handle-based model as plotted data. You can capture a title handle, inspect its type, and restyle it with `set`. You can create a legend, inspect its properties, and understand it as an object attached to one axes rather than a figure-global overlay.

```matlab
plot(1:5, [1 4 2 5 3]);
t = title("Signal");
x = xlabel("Sample");
l = legend("Series A");

set(t, "FontWeight", "bold");
set(x, "FontAngle", "italic");
get(l, "Type")
```

This example uses the same pattern as plot-handle workflows: create an object, keep the handle, and update it later. The difference is that the objects are annotations owned by the current axes rather than plotted data series.

## Plot objects and styling workflows

Plot objects are the part of the graphics object model most users interact with first. Calls such as `plot`, `scatter`, `bar`, `stem`, `errorbar`, `surf`, `contour`, `image`, `histogram`, `pie`, `quiver`, and `area` produce plot objects that can later be queried and restyled through their handles. This lets plotting code separate creation from refinement.

This workflow is especially useful when styling becomes more deliberate. A script can create several series first, then assign display names, adjust line widths, change marker settings, and build a legend from those object-level properties. In that sense, plot handles are the bridge between "draw this" and "now shape the result into the figure I want."

```matlab
h1 = plot(1:5, [1 2 3 4 5]);
set(h1, "DisplayName", "Linear");

hold on;
h2 = plot(1:5, [1 4 9 16 25]);
set(h2, "DisplayName", "Quadratic");

legend();
```

Here each plot object carries its own display name, and `legend()` uses that object-level metadata to build the legend. This is often more maintainable than treating legend labels as a completely separate concern.

## Special and composite cases

Most plotting builtins map cleanly to one obvious object kind, but some deserve special explanation. Composite plotting workflows such as `surfc` and `meshc` combine more than one visual component while still returning a single surface-oriented handle. Pie charts behave differently from line-style series because legends are naturally slice-based. `histogram` follows a handle-based object model, while older `hist` workflows are better understood as count-producing plotting commands.

There are also cases where user-facing semantics should stay simple even if the deeper rendering path is shared with another plot family. `image` and `imagesc`, for example, should be understood as image-oriented plotting objects from the user's perspective, even when they share lower-level machinery with surface rendering. The important question for this document is not how those objects are rendered internally, but how they behave as graphics objects you can inspect and modify.

When documenting or using these cases, prefer the user-facing mental model:

- composite plotting functions may coordinate more than one visual component
- some legends are object-based, while pie legends are naturally slice-based
- `histogram` should be treated as its own plot object model
- `image` and `imagesc` should be understood as image plotting workflows

## Practical handle workflows

The graphics object model is most useful when it changes how you write plotting code. In simple scripts, that might mean capturing a line handle so you can restyle it after plotting. In larger figures, it often means capturing axes handles so each subplot can be configured independently. In reusable plotting code, it often means setting `DisplayName` on plot objects and letting `legend` build from those names.

A good practical rule is to capture the handle for any object you expect to inspect, restyle, or revisit later. Handles make code more explicit, and they reduce the need to rely on implicit state. Even when current-axes behavior is convenient, explicit object references usually make nontrivial plotting scripts easier to reason about.

Useful habits include:

- capture plot handles when you plan to restyle an individual series
- capture axes handles when you build figures with more than one panel
- capture text or legend handles when annotation styling matters
- use `get(h)` when you need to learn what a handle supports

## Common misconceptions

A few misconceptions come up repeatedly in handle-based plotting systems. One is assuming that labels or legends are figure-wide by default, when in practice they belong to a specific axes. Another is assuming that all plotting state lives globally on the current figure, when much of the important state is actually axes-local. A third is expecting every object to expose the same property surface, even though different handle kinds are intentionally different.

The easiest way to stay oriented is to ask three questions. What object do I have a handle for? What object owns the state I want to change? Is this command creating a new object, or mutating an existing one? Once those answers are clear, the rest of the plotting workflow usually follows naturally.

## See also

For syntax-level details, see the reference pages for `subplot`, `get`, `set`, `legend`, `title`, `xlabel`, `ylabel`, `zlabel`, `axis`, and `view`. For a broader conceptual overview of plotting workflows, pair this page with the general plotting guide.

---

## Related

- [Plotting in RunMat](/docs/plotting/plotting-in-runmat): the plotting workflow from first command to finished figure.
- [Styling Plots and Axes](/docs/plotting/styling-plots-and-axes): labels, legends, colormaps, and coordinated styling.
- [Plot Replay and Export](/docs/plotting/plot-replay-and-export): persist, replay, and export figures.
- [Builtin Function Reference](/docs/matlab-function-reference): full list of supported MATLAB functions.
