# GPU Plotting and Rendering Math

Plotting begins with numerical data, but a plotted figure is not a direct display of raw arrays. Between data and image lies a sequence of mathematical constructions. Values are interpreted as samples over a domain, converted into geometric structure, normalized by axes state, assigned visible color, transformed by the current view, and finally rendered. The resulting figure is therefore a derived object: a visual representation built from data through geometry and transformation.

A sampled scalar field may be written as

$$
z_{ij} = f(x_j, y_i), \qquad i = 1,\dots,m,\quad j = 1,\dots,n,
$$

over the discrete domain

$$
\mathcal{X} \times \mathcal{Y}
= \{x_1,\dots,x_n\} \times \{y_1,\dots,y_m\}.
$$

In a surface-style plot, these samples are embedded into three-dimensional space by associating each sampled value with a point

$$
\mathbf{p}_{ij} = (x_j, y_i, z_{ij}).
$$

The matrix of sampled values is thereby reinterpreted as geometry over a domain. Subsequent transformations act on that geometry to determine visibility, color, and view. The visible plot is the final result of those transformations.

## Plotting as a sequence of transformations

The mathematical structure of plotting is easiest to see when it is separated into stages. Numerical input is first interpreted according to plotting semantics. That interpretation produces geometric objects. Those objects are then mapped into an axes-dependent coordinate system, assigned visual attributes such as color, transformed into a viewing configuration, and expressed in a renderable form.

This decomposition separates questions that are easy to conflate when plotting is treated as a single operation. Input interpretation determines what the data means. Geometry generation determines what object is being drawn. Coordinate normalization determines how that object is placed within an axes. Color mapping determines how scalar variation becomes visible. View transformation determines how three-dimensional structure becomes a two-dimensional figure.

## Sampled fields and domains

A surface plot begins with samples of a scalar field. When written as `surf(Z)`, the matrix `Z` is interpreted as field values sampled over a rectangular grid. If `Z` has `m` rows and `n` columns, the default domain is the discrete index set

$$
x_j = j, \qquad j = 1,\dots,n,
$$

$$
y_i = i, \qquad i = 1,\dots,m.
$$

When written as `surf(X, Y, Z)`, the domain is given more explicitly. The horizontal coordinates may be provided either as axis vectors or as full coordinate matrices. In either case, plotting begins by establishing where the sampled values live. The array alone is not yet a visible object. It is a sampled function attached to a domain.

## From samples to surface geometry

Once the sampled field has been placed on a domain, each sample becomes a point in space. If the sample at row `i` and column `j` has horizontal coordinates `(x_j, y_i)` and scalar value `z_{ij}`, then the corresponding surface vertex is

$$
\mathbf{p}_{ij} = (x_j, y_i, z_{ij}).
$$

Neighboring samples are then connected across the grid to form faces. At this stage, the data is no longer merely a collection of sampled values. It has become a geometric object embedded in three-dimensional space.

This passage from sampled values to geometry appears across many plot families. A line plot turns samples into a polyline. A scatter plot turns samples into positioned markers. A bar plot turns values into rectangles. A contour plot turns a scalar field into level geometry. The details differ, but the common pattern is the same: plotting constructs geometry from interpreted data.

## Axes limits as coordinate transformations

The surface produced so far still lives in data coordinates. To place it inside a figure, the plotting system maps those coordinates into the visible range of the active axes. For linear axes, this is an affine normalization. If the visible limits are `x_min` to `x_max`, `y_min` to `y_max`, and `z_min` to `z_max`, then the normalized coordinates are

$$
x_n = \frac{x - x_{\min}}{x_{\max} - x_{\min}}, \qquad
y_n = \frac{y - y_{\min}}{y_{\max} - y_{\min}}, \qquad
z_n = \frac{z - z_{\min}}{z_{\max} - z_{\min}}.
$$

This is the point at which axes state becomes mathematically active. Changing axis limits changes the mapping from data coordinates to visible coordinates. The same surface can therefore appear differently under different axis choices even when the underlying data is unchanged.

## Logarithmic coordinates

A logarithmic axis changes the coordinate system before normalization occurs. The plotted object is no longer interpreted in a linear domain along that axis. Instead, the data is transformed through a logarithm and then normalized in the transformed space. For a base-10 logarithmic x-axis,

$$
x_n = \frac{\log_{10}(x) - \log_{10}(x_{\min})}{\log_{10}(x_{\max}) - \log_{10}(x_{\min})}.
$$

This formula makes the domain restriction transparent. A logarithmic axis requires positive values in the transformed dimension because the logarithm itself requires a positive argument. A logarithmic axis is therefore not a relabeling convention. It is a different coordinate map.

## Scalar values and visible color

Color often carries quantitative meaning. In surface-style plots, scalar values may determine visible color through normalization against color limits followed by sampling from a colormap. If `c_min` and `c_max` denote the active color limits, then a scalar value `z` is normalized as

$$
t = \frac{z - c_{\min}}{c_{\max} - c_{\min}}, \qquad t \in [0,1].
$$

The normalized value `t` is then mapped into the active colormap. In this way, scalar magnitude becomes visible variation in hue or brightness. Color is therefore not merely decorative. It is another transformation from numerical structure into visual structure.

The same geometry can communicate different scalar structure under different color settings. Narrow color limits emphasize local variation. Broader color limits emphasize larger-scale variation. Changing the colormap changes how normalized scalar values are encoded visually. Both operations alter the visible interpretation of the same data.

## Shading as interpolation of appearance

Once geometry and color values exist, the plotting system must determine how visual attributes vary across each face of the surface. This is the role of shading. A surface is not only a set of vertices; it also carries a rule for how appearance behaves between those vertices.

Flat shading emphasizes per-face uniformity. Interpolated shading blends values across faces. Faceted rendering preserves stronger visual separation between neighboring grid elements. These are not merely stylistic variations. They correspond to different interpolation assumptions for appearance over geometry.

## View as geometric transformation

A three-dimensional surface must still be expressed relative to an observer. This is the role of the view transformation. A point on the normalized surface is mapped into a view-dependent coordinate system determined by the active orientation. The view is therefore part of the mathematical transformation chain rather than an external presentation layer.

If the normalized point is written as

$$
\mathbf{p}_n = (x_n, y_n, z_n),
$$

then the viewed point may be written abstractly as

$$
\mathbf{p}_v = T_{\mathrm{view}}(\mathbf{p}_n),
$$

where $T_{\mathrm{view}}$ is the transformation induced by the current viewing configuration. The point is now being described relative to an observer rather than only relative to the axes in which it was normalized.

## Projection onto the image plane

A viewed three-dimensional object is not observed directly as a three-dimensional object. It is observed through projection onto a two-dimensional image plane. Projection removes one geometric degree of freedom and produces image-plane coordinates for visible points.

If

$$
\mathbf{p}_v = (x_v, y_v, z_v),
$$

then projection produces image-plane coordinates

$$
(u, v) = \pi(\mathbf{p}_v).
$$

For a simple perspective-style model, one may write

$$
u = \frac{x_v}{z_v}, \qquad v = \frac{y_v}{z_v},
$$

up to scale and translation factors. The exact projection model may vary, but the mathematical role is the same: three-dimensional geometry is converted into two-dimensional observable position.

## Sampling on a discrete image lattice

The image plane is still a continuous mathematical object. A rendered figure, however, is observed on a discrete grid. The projected image is therefore sampled on a finite lattice of screen locations. If the display has width $W$ and height $H$, then the visible image is represented on indices

$$
i \in \{1,\dots,W\}, \qquad j \in \{1,\dots,H\}.
$$

Conceptually, image-plane coordinates are mapped into pixel indices by a sampling map

$$
S : (u, v) \mapsto (i, j).
$$

This is the point at which continuous geometry becomes discrete observation. The surface itself may be smooth or finely sampled, but the displayed result is limited by the finite resolution of the image lattice. The observer does not see the geometric object directly. The observer sees a sampled projection of that object.

## Rendering as repeated finite observation

Once geometry, color, and view have been resolved, rendering becomes the repeated production of a finite sampled image. Each frame is an observation of the current scene state on a discrete image lattice. If the scene state does not change, the rendered frame is a repeated sampling of the same projected object. If the scene state changes, the sampled image changes accordingly.

This is the point at which plotting becomes a display process rather than only a geometric construction. The earlier stages determine what object exists and how it is viewed. The rendering stage repeatedly converts that viewed object into a finite image at the resolution of the display.

## Why world-space size is not the final scale of observation

A plotted object may occupy a large or small domain in its original coordinates, but the rendered figure is governed by its projected and sampled form. Once the object has been transformed into image-plane coordinates, the relevant observation is finite-resolution sampling on the display grid. The total size of the original domain is therefore not, by itself, the final determinant of what is observed.

This is why plotting can be understood as the composition of two different kinds of mathematics. First, there is the geometry of the plotted object in data, normalized, and viewed coordinates. Second, there is the discrete sampling of that geometry into an image. The final figure belongs to the second category: it is a finite observation of the first.

## Axes state as mathematical input

Axes state enters plotting as mathematical input. Limits define normalization maps. Log settings define transformed coordinate systems. Color limits define scalar normalization. Colormap selection defines how normalized values become visible color. View settings define how geometry is observed.

This perspective helps unify plotting semantics. The figure is shaped jointly by the data and by the state through which the data is interpreted. Plotting is therefore not only about what values are given, but also about what transformations are active when those values are converted into geometry and appearance.

## Edge cases and singular situations

The structure of the plotting pipeline also explains its edge cases. If an axis range collapses so that a minimum equals a maximum, the corresponding normalization becomes undefined. If a logarithmic axis receives nonpositive data, the transformed coordinate cannot be formed. If surface coordinates and sampled values do not define a compatible grid, valid geometry cannot be constructed.

These are consequences of the mathematical maps involved. In the same way, collapsed color limits affect scalar normalization, and malformed domains prevent consistent geometric interpretation. The edge cases are part of the model rather than exceptions to it.

## A surface plot through the full chain

A surface plot can now be read as a complete chain of maps. A sampled field

$$
z_{ij} = f(x_j, y_i)
$$

is embedded into geometry through points

$$
\mathbf{p}_{ij} = (x_j, y_i, z_{ij}).
$$

That geometry is normalized by the active axes, transformed by the view map, projected into the image plane, and sampled on a finite lattice. The visible figure is the resulting sampled image together with the color and shading values attached to that image.

Seen this way, plotting is neither a direct display of raw arrays nor an abstract geometric construction alone. It is a transformation from sampled data to geometry, and from geometry to discrete visual observation.

## See also

The same ideas connect naturally to graphics handles and plot objects, axis and view controls, colormap and shading semantics, and the plot-family-specific rendering mathematics for surfaces, contours, images, vector fields, pie charts, and histograms.
