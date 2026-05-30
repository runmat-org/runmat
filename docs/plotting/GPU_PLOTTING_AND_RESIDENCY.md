# GPU Plotting and Residency

Plotting sits at the boundary between computation and observation. The numerical side of the system operates on arrays. The rendering side operates on scene data derived from those arrays.

This distinction is the foundation of GPU plotting in RunMat. A tensor on the GPU is not yet a plot. A plot is a scene representation built from numerical data and axes-local state. The central questions are therefore: whether data is on the GPU, what form that data is in, when it changes representation, and what work must be repeated as the scene changes.

## Numeric arrays versus renderable scene data

Numerical tensors and renderable plots are different kinds of objects. A tensor stores values in a form suited to array algebra, elementwise kernels, reductions, and fused numerical computation. A plot stores a visual interpretation of values: points, lines, surfaces, image cells, contour segments, colors, limits, and view-dependent state.

A tensor answers questions such as value, shape, and dtype. A plot answers questions such as geometry, visibility, color mapping, and viewpoint. The same source samples may therefore lead to very different plot representations depending on how they are interpreted.

## What residency means in plotting

Residency in plotting refers to where the relevant data lives at each stage of the pipeline. Source samples may remain in device memory. Derived geometry may also remain in device-oriented form. Render buffers may then be prepared for drawing without first becoming a host-side numerical structure. These are related but distinct kinds of residency.

For that reason, "stays on the GPU" is not a complete description. The important question is: what remains on the GPU, in what representation, and for what stage of the pipeline? Source residency concerns the original numerical tensors. Derived residency concerns plot-specific geometry or image data. Render-buffer residency concerns the packed representation consumed by the renderer.

## The plotting boundary: how plotting terminates fusion graphs

Fusion is designed to take chains of numerical tensor operations, and combine them into a single GPU kernel to save memory and reduce launch overhead. As long as a computation remains an array computation, materialization can be delayed and operations can be combined. Plotting crosses a different boundary. A plotting call does not produce another tensor. It produces a scene.

That boundary is fundamental rather than accidental. A rendered plot requires axes state, geometry generation, style resolution, view configuration, and a render target. A fused numerical graph can remain abstract while it still represents algebra on arrays. A plotting call cannot remain only an array computation, because its output is no longer an array in the same semantic sense. Plotting therefore terminates ordinary numeric fusion and begins scene construction.

This marks a point in the fusion graph where a tensor is required to represent the scene. This tensor is a scene representation built from numerical data and axes-local state.

## The stages of GPU plotting

The GPU-side story of plotting is easiest to understand when broken into stages.

### 1. Device-resident numeric input

The process begins with one or more tensors, possibly already resident on the GPU. Numerical preprocessing may still occur here in ordinary array form. At this stage, the data remains value-oriented.

### 2. Plot interpretation

The plotting system interprets those values according to plot semantics. The same tensor may be read as a sequence of samples, a structured grid, an image field, a vector field, or a source for a histogram. This changes the semantic type of the data even when the underlying numeric storage is unchanged.

### 3. Geometry or image-data construction

The interpreted values are converted into plot-specific structure. A line plot builds a polyline. A surface plot builds a sampled surface. A contour plot builds derived level geometry. A quiver plot builds arrow-oriented geometry. An image plot builds image-oriented cells or color payloads. This is the stage at which the plot becomes something renderable in principle.

### 4. Scene-state attachment

Axes-local state is attached to the plot representation. Limits, color limits, colormap, shading, and view all affect how the derived structure is interpreted at render time. The source values alone are not yet enough to define the final visible result.

### 5. Render-oriented packing

The plot representation is packed into a form suitable for drawing. This representation is no longer simply the source tensor layout. It is specialized for rendering.

### 6. Frame rendering

The current scene state is projected, sampled, and drawn into a visible frame. The output is not a tensor result in the ordinary numerical sense. It is a displayed observation of the scene.

## Plot-family differences in residency behavior

Different plot families cross the plotting boundary in different ways.

### Lines and scatter

Line and scatter families are relatively direct. Source samples map naturally to point or polyline structure. The representation change is still real, but the geometric expansion is comparatively simple.

### Surfaces and meshes

Surface-style plots begin from structured sampled domains. They often support strong GPU-side handling because the input already has regular geometric meaning. Once interpreted as a grid, the data can be carried forward into surface-oriented render structure with relatively direct correspondence between source samples and derived geometry.

### Images and imagesc

Image-style plots are especially natural for GPU residency because the data is already image-like. Scalar-valued images still require color mapping, and truecolor images still require rendering-specific handling, but the underlying representation is already close to what the renderer needs: values attached to a sampled rectangular domain.

### Contours and contourf

Contour families are more derived. The source field does not directly define the rendered geometry. Level geometry must first be extracted from the sampled field. That means contour plots often require more preprocessing than direct sample-to-geometry families.

### Quiver

Quiver plots begin with vector components, but the renderer does not draw abstract vectors. It draws arrow geometry. Source data must therefore expand into shafts, heads, and oriented marker-like structures. The cost is influenced not only by source size, but by geometric expansion.

### Histogram and related aggregations

Histogram-style plots introduce a different kind of boundary. Before a visible plot exists, the source data must be reduced into bins or related summary structure. The plot is built from derived aggregate data rather than from direct sample positions.

## Source data, derived geometry, and render buffers

Residency questions are easiest to reason about when three layers are kept separate.

### Source data

This is the original tensor or gpuArray input. It is the value-oriented numerical state from which the plot begins.

### Derived plot representation

This is the plot-specific structure produced from the source data. Examples include vertices, contour segments, image color grids, quiver arrows, histogram bars, and surface faces. This layer is semantically visual rather than purely numerical.

### Render buffers

This is the packed representation specialized for drawing. It is optimized for render execution rather than for general array algebra.

Many conversations about GPU plotting become confused because these layers are merged together conceptually. A source tensor may remain on device while the derived geometry changes representation. A render buffer may be device-resident while no longer resembling the source tensor at all. Performance and residency both depend on which layer is under discussion.

## Axes-local state as control data

Axes-local state is usually small in memory footprint but large in semantic effect. Limits determine coordinate normalization. Log modes determine coordinate transforms. Color limits determine scalar normalization for color mapping. Colormap selection determines how normalized values become visible colors. View settings determine how the geometry is observed.

This state is best understood as control data rather than bulk data. It does not dominate memory use, but it strongly affects the meaning of the rendered scene. A change in axes state can therefore alter the visible result substantially even when the source tensors remain unchanged.

## What changes when the user changes the plot

Different kinds of plot updates have different computational meanings.

### Changing source data

Changing the source tensors may require rebuilding the derived plot representation. For many families, this is one of the most expensive kinds of update because it changes the numerical foundation of the scene.

### Changing axes limits or color limits

Changing limits may alter coordinate normalization or scalar-to-color mapping without requiring the same degree of source reinterpretation. The scene meaning changes even if the source arrays do not.

### Changing view

Changing view usually alters observation rather than source semantics. The underlying geometric object may remain the same while projection and visible arrangement change. This distinction is especially important for interactive 3-D plots.

### Changing style

Changing line style, marker appearance, shading mode, or colormap may affect visible appearance more than source interpretation. The visual result can change substantially without a full rebuild of the originating numeric data path.

## Frame production and the render loop

Once a plot has become scene state, rendering produces frames from that scene. A frame is not the tensor itself. It is the current visible image implied by derived plot data, axes state, view, and style. The render loop repeatedly samples and draws that scene.

This makes it useful to distinguish scene updates from frame redraws. A redraw may occur even when the scene is unchanged. A scene update changes the state from which frames are generated. The computational meaning of these two events is different, even though both are part of the user’s experience of plotting.

## The GPU cost model of plotting

The cost of GPU plotting is not determined by source size alone. It depends on several distinct factors.

### Input size

Larger source tensors may increase interpretation and packing work simply because more samples exist.

### Geometry expansion

Some families produce render structures whose size is close to the source sample count. Others, such as contour or quiver families, may produce substantially more derived structure than the raw source shape suggests.

### Data movement

Host-device transfer costs matter when data cannot remain entirely in device-oriented form. Even when large transfers are avoided, repacking into render-oriented structures still has computational cost.

### Scene-state changes

Some updates are small control changes. Others force reinterpretation or rebuilding of derived plot state. Limits, view, and colormap do not all carry the same cost as new source data.

### View and sampling cost

Projection, visibility, and final image production depend on the rendered scene and the display resolution. Final rendering cost is influenced by what is actually observed, not only by the abstract domain from which the plot began.

### Repeated frames

A static scene and a continuously changing scene have different computational character. Repeated rendering of unchanged state is not the same as repeatedly rebuilding the scene from new input data.

Taken together, these factors show that plotting cost is a function of representation changes and rendered observation, not merely of original tensor size.

## Why plotted world size is not the final performance scale

A plot may arise from a large domain in world coordinates, but the rendered figure is governed by the representation actually being observed. Once a scene is projected into image space and sampled at display resolution, the relevant output is finite. The total size of the original numerical domain does not, by itself, determine the final scale of the visual observation.

This does not mean source size is irrelevant. It means that source size is only one part of the cost model. What ultimately matters is the chain from source data to derived geometry to projected observation. The plotted figure is a finite sampled image of a scene, not an unbounded display of the original domain.

## Practical implications for users

Several practical rules follow from this model.

- GPU-resident input is useful when the plot family supports meaningful device-side handling, but residency must be understood stage by stage.
- Plot families differ substantially in how much derived work they require. Direct sample-to-geometry paths behave differently from contour extraction, quiver expansion, or histogram aggregation.
- View changes and limit changes are not the same kind of update as changing source data.
- Style and color changes can be semantically important even when they are cheaper than rebuilding geometry.
- Performance questions should be asked in terms of source size, derived structure, transfer boundaries, and rendered observation together.

## Worked examples

### GPU-resident surface

A surface plot from device-resident sampled data begins with source tensors on the GPU, interprets them as a structured domain, constructs surface-oriented plot state, attaches limits and color mapping, and renders the result as a viewed scene. The key question is not only whether the input began on the GPU, but how much of the surface-oriented representation can remain in device-oriented form through rendering.

### Truecolor image

A truecolor image begins from data that is already close to image structure. The representation change is therefore more direct than for many geometric families. The source data still becomes a plot-specific render representation, but the semantic gap between input and rendered object is smaller than for contour extraction or vector expansion.

### Contour field

A contour plot begins from a sampled scalar field, but the renderer draws extracted level geometry rather than the field itself. The cost therefore includes the derivation of contour structure before rendering begins. The relevant representation is the extracted geometry, not only the original field.

### View-only update versus data update

Rotating a surface and recomputing a surface are different computational events. A view-only update changes observation of an existing scene. A data update may require rebuilding the plot representation itself. This distinction is often more useful than a simple binary contrast between "GPU" and "CPU."

## Relationship to the rendering-math model

The rendering-math model explains what transformations plotting performs. The residency model explains where the relevant data lives and what computational work is required to carry those transformations out. The two perspectives describe the same pipeline from different sides: one geometric, one computational.

## See also

For the math behind plotting, see `GPU_PLOTTING_AND_RENDERING_MATH.md`. 

For the graphics handles that control the plot objects, see `GRAPHICS_HANDLES.md`. 

For the plot-family-specific rendering explainers, see the builtin references for `surf`, `mesh`, `image`, `imagesc`, `contour`, `contourf`, `quiver`, `histogram`, `view`, `colormap`, and `shading`.