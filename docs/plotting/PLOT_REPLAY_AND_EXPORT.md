# Plot Replay and Export

Once a plot exists, it can be treated as a graphics scene with its own persistent state. A figure contains plot objects, axes-local interpretation state, labels, legends, limits, color settings, and view configuration. In RunMat, replay and export operate on that scene state.

This distinction matters because computation, scene state, and rendered output are not the same thing. Recomputing data is not the same as replaying a figure. Replaying a figure is not the same as exporting an image. A clear plotting model therefore needs to distinguish source computation, persistent scene structure, and fixed visual observation.

## What replay means

Replay means reconstructing or re-presenting an existing figure as a graphics scene. The important point is that replay operates on interpreted plotting state rather than only on the original source code path. A replayed figure is a scene whose plot objects, axes state, annotations, and view can be rendered again.

This makes replay different from rerunning a script in the simplest possible sense. A script may generate a figure, but once the figure exists, it has a structure of its own. Replay preserves that structure, and can re-open and re-render it as a scene.

In this way, replay is a way to persist the figure as a scene, and then re-open and re-render it as a scene - not re-execute the original script.

## Replay versus recomputation

Recomputation begins again from source data and plotting code. Replay begins from an existing scene description. The two are related, but they are not conceptually identical.

When a script is rerun, the plotting system rebuilds the scene from data, interpretation, and plotting commands. When a figure is replayed, the system starts from the interpreted graphics state already associated with that figure. In other words, recomputation rebuilds the scene from the analytical pipeline, while replay rebuilds the visible figure from scene state held in either persisted or volatile memory.

This distinction becomes especially clear in interactive plotting. Rotating a 3-D surface after it has already been created does not require redefining the sampled field from scratch. The scene already exists. What changes is how it is observed and re-rendered.

## What parts of a figure are preserved

Replay is meaningful because a figure preserves more than raw plotted arrays. The preserved state includes the interpreted graphics objects and the axes-local context in which they are read.

That preserved state includes, among other things:

- plot objects and their relevant properties
- subplot layout and axes membership
- axis limits and scales
- titles, axis labels, and legend state
- colormap, color limits, colorbar state, and shading
- 3-D view settings
- visibility and style choices that affect the rendered result

The important idea is that replay preserves interpreted graphics state, not just undifferentiated numerical input. The figure remains a structured scene.

## Scene state as the unit of replay

The most useful way to think about replay is to treat the scene as the persistent unit. A scene contains the objects to be drawn and the context in which they are drawn. That includes geometry or field-oriented plot state, annotation state, axes-local interpretation state, and the visual parameters required to produce the current figure.

This is why replay makes sense independently of the original code that constructed the plot. The code may have created the scene, but once created, the scene has its own identity as graphics state. Replay operates on that identity.

## Replay and the rendering pipeline

Replay does not bypass rendering. A replayed figure still has to pass through the rendering pipeline in order to become visible. Geometry or field data must still be interpreted through current scene state, projected into a visible frame, and drawn as an image on the display.

This means replay preserves the scene, not a single immutable pixel arrangement. If the same scene is rendered under a different viewing configuration, output size, or display context, the visible frame may change even though the underlying scene state is the same. Replay therefore preserves renderable meaning rather than only one fixed raster.

## Export as fixed visual output

Export turns the current figure into an external visual artifact. An exported image is a fixed observation of the scene in its present state. It captures the current layout, current view, current styling, and current visible arrangement of the figure.

This is different from preserving a live scene. Export does not primarily preserve the full editable graphics structure in the same way replay does. Instead, it preserves the rendered result of the scene at a particular moment of observation.

That is why export is best understood as output rather than replay. It records what the figure looks like now.

## Export versus replay

Replay preserves or reconstructs a figure as a scene. Export preserves a rendered observation of that scene. Replay keeps the figure conceptually alive as a plotting object that can still be rendered again. Export freezes one visible rendering of it.

This is the most important distinction in the page. A replayable figure still participates in the plotting system as scene state. An exported image participates as a visual artifact. Both are valuable, but they preserve different things.

## Why this distinction matters

The difference between replay and export affects both interpretation and workflow. An exported image is ideal for reporting, sharing, publication, and stable visual reference. A replayable figure is ideal when the scene itself must remain available for continued interaction, restoration, or re-rendering.

An exported image cannot substitute for a live scene when later edits, alternate views, or updated renderings are required. Conversely, a replayable scene is not automatically the best format for a fixed visual deliverable. The right choice depends on whether the goal is persistence of structure or persistence of appearance.

## Replay in multi-panel figures

Multi-panel figures make the value of replay especially clear. Each subplot carries its own axes-local state: titles, labels, legends, limits, color settings, and view. A replayed multi-panel figure preserves that composition as a structured whole rather than as a single undifferentiated picture.

This is important because subplot layout is not only a matter of placement. It is part of the analytical organization of the figure. Replay preserves that organization at the scene level. Export preserves the visible result of that organization in one fixed frame.

## Replay and 3-D plots

Three-dimensional plots further sharpen the distinction between scene and image. In a 3-D plot, the current view is part of the scene state. Replay preserves the ability to render that 3-D structure again as a scene. Export preserves one current observation of that structure.

This means a 3-D surface or trajectory should not be thought of only as a single image. It is a geometric object together with view-dependent interpretation state. Replay preserves that object as something still renderable. Export preserves one currently chosen observation of it.

## Practical workflows

Several practical workflows follow naturally from this model.

### Create, refine, then replay

A figure can be created, styled, annotated, and treated as persistent scene state. Replay later restores or re-renders that scene without collapsing it into only a static image.

### Adjust a scene, then export the current result

This is a common workflow for polished figures. The plot is developed as a live scene, refined through handles and axes-local state, and then exported when the current observation is the one worth sharing.

### Preserve a multi-panel analytical composition

When a figure consists of several coordinated panels, replay preserves the subplot structure and axes-local interpretation of each panel. Export preserves the resulting composed page of graphics as one visible outcome.

### Distinguish scene updates from image output

When a figure changes through new limits, a new view, new annotations, or new plotted objects, the scene changes. When the current scene is exported, the output changes because the observed state has changed. This is easier to reason about when scene and observation are kept conceptually separate.

## Common misconceptions

Several misunderstandings appear often when replay and export are not distinguished clearly.

- replay is not simply rerunning the original script in the same way
- export does not preserve a live scene in the same sense that replay does
- identical source data does not eliminate the conceptual difference between recomputation and replay
- subplot layout is not figure-wide decoration; it is part of the scene structure
- view is not merely presentation in 3-D plots; it is part of the interpreted figure state

Most of these misunderstandings disappear once the figure is understood as a scene and the export is understood as one rendered observation of that scene.

## Relationship to other plotting docs

This page fits naturally beside the other conceptual plotting documents. `GRAPHICS_HANDLES.md` explains what the figure’s objects are. `GPU_PLOTTING_AND_RENDERING_MATH.md` explains how a scene becomes a visible observation. `GPU_PLOTTING_AND_RESIDENCY.md` explains the computational and residency side of that rendering process. This page explains how the resulting figure persists as scene state and how that state differs from exported output.

Together, these documents separate three important layers: source computation, scene structure, and rendered observation.

## See also

Read `PLOTTING_IN_RUNMAT.md` for the broader plotting overview, `GRAPHICS_HANDLES.md` for the object model behind figures and axes, `GPU_PLOTTING_AND_RENDERING_MATH.md` for the geometric and observational model of rendering, and `GPU_PLOTTING_AND_RESIDENCY.md` for the computational model of scene construction and render execution. The reference pages for `subplot`, `view`, `legend`, `colormap`, `colorbar`, `get`, and `set` describe the builtin-level controls that shape replayable figure state.

---

## Related

- [Plotting in RunMat](/docs/plotting/plotting-in-runmat): the plotting workflow from first command to finished figure.
- [Graphics Handles](/docs/plotting/graphics-handles): inspect and update plot objects with handles.
- [Large Dataset Persistence](/docs/large-dataset-persistence): chunked, content-addressed storage for large arrays.
- [GPU Residency and Precision](/docs/accelerate/gpu-behavior): when data moves to and from the GPU.
