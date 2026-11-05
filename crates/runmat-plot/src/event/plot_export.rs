//! A scene plot from a figure's plot element, for export, and the checks it passes first.

use super::*;

impl ScenePlot {
    pub(super) async fn from_plot_for_export(
        plot: &PlotElement,
        axes_index: u32,
    ) -> Result<Self, SceneExportFailure> {
        let scene_plot = match plot {
            PlotElement::Line(line) => {
                let (x_data, y_data) = line
                    .export_numeric_xy_data_typed()
                    .await
                    .map_err(SceneExportFailure::readback_typed)?
                    .ok_or_else(|| {
                        SceneExportError::unexportable("line plot has no exportable scene data")
                    })?;
                let x = x_data.materialize_f64();
                let y = y_data.materialize_f64();
                if x.is_empty() && y.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "line plot has no exportable scene data",
                    )
                    .into());
                }
                Self::Line {
                    x,
                    y,
                    x_source: Some((&x_data).into()),
                    y_source: Some((&y_data).into()),
                    color_rgba: vec4_to_rgba(line.color),
                    line_width: line.line_width,
                    line_style: format!("{:?}", line.line_style),
                    marker_style: line.marker.as_ref().map(|m| format!("{:?}", m.kind)),
                    marker_size: line.marker.as_ref().map(|m| m.size),
                    marker_face_color: line.marker.as_ref().map(|m| vec4_to_rgba(m.face_color)),
                    marker_edge_color: line.marker.as_ref().map(|m| vec4_to_rgba(m.edge_color)),
                    marker_filled: line.marker.as_ref().map(|m| m.filled),
                    handle_visibility: (!line.handle_visibility.eq_ignore_ascii_case("on"))
                        .then(|| line.handle_visibility.clone()),
                    axes_index,
                    label: line.label.clone(),
                    visible: line.visible,
                }
            }
            PlotElement::ReferenceLine(_) => Self::from_plot(plot, axes_index),
            PlotElement::Scatter(scatter) => {
                let (x_data, y_data) = scatter
                    .export_numeric_xy_data_typed()
                    .await
                    .map_err(SceneExportFailure::readback_typed)?
                    .ok_or_else(|| {
                        SceneExportError::unexportable("scatter plot has no exportable scene data")
                    })?;
                let x = x_data.materialize_f64();
                let y = y_data.materialize_f64();
                if x.is_empty() && y.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "scatter plot has no exportable scene data",
                    )
                    .into());
                }
                Self::Scatter {
                    x,
                    y,
                    x_source: Some((&x_data).into()),
                    y_source: Some((&y_data).into()),
                    color_rgba: vec4_to_rgba(scatter.color),
                    marker_size: scatter.marker_size,
                    marker_style: format!("{:?}", scatter.marker_style),
                    axes_index,
                    label: scatter.label.clone(),
                    visible: scatter.visible,
                }
            }
            PlotElement::Bar(bar) => {
                let values = bar
                    .export_scene_values()
                    .await
                    .map_err(SceneExportError::readback)?;
                if values.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "bar chart has no exportable scene data",
                    )
                    .into());
                }
                Self::Bar {
                    labels: bar.labels.clone(),
                    values,
                    histogram_bin_edges: bar.histogram_bin_edges().map(|edges| edges.to_vec()),
                    polar_histogram: bar.is_polar_histogram(),
                    polar_histogram_display_style: Some(format!(
                        "{:?}",
                        bar.polar_histogram_display_style()
                    )),
                    color_rgba: vec4_to_rgba(bar.color),
                    outline_color_rgba: bar.outline_color.map(vec4_to_rgba),
                    bar_width: bar.bar_width,
                    outline_width: bar.outline_width,
                    orientation: format!("{:?}", bar.orientation),
                    group_index: bar.group_index as u32,
                    group_count: bar.group_count as u32,
                    stack_offsets: bar.stack_offsets().map(|offsets| offsets.to_vec()),
                    axes_index,
                    label: bar.label.clone(),
                    visible: bar.visible,
                }
            }
            PlotElement::ErrorBar(error) => {
                let (x, y, y_neg, y_pos, x_neg, x_pos) = error
                    .export_scene_data()
                    .await
                    .map_err(SceneExportError::readback)?;
                if x.is_empty() && y.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "errorbar plot has no exportable scene data",
                    )
                    .into());
                }
                Self::ErrorBar {
                    x,
                    y,
                    err_low: y_neg,
                    err_high: y_pos,
                    x_err_low: x_neg,
                    x_err_high: x_pos,
                    orientation: format!("{:?}", error.orientation),
                    color_rgba: vec4_to_rgba(error.color),
                    line_width: error.line_width,
                    line_style: format!("{:?}", error.line_style),
                    cap_width: error.cap_size,
                    marker_style: error.marker.as_ref().map(|m| format!("{:?}", m.kind)),
                    marker_size: error.marker.as_ref().map(|m| m.size),
                    marker_face_color: error.marker.as_ref().map(|m| vec4_to_rgba(m.face_color)),
                    marker_edge_color: error.marker.as_ref().map(|m| vec4_to_rgba(m.edge_color)),
                    marker_filled: error.marker.as_ref().map(|m| m.filled),
                    axes_index,
                    label: error.label.clone(),
                    visible: error.visible,
                }
            }
            PlotElement::Stairs(stairs) => {
                let (x, y) = stairs
                    .export_scene_xy_data()
                    .await
                    .map_err(SceneExportError::readback)?;
                if x.is_empty() && y.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "stairs plot has no exportable scene data",
                    )
                    .into());
                }
                Self::Stairs {
                    x,
                    y,
                    x_source: stairs.x_source.as_ref().map(SceneNumericData::from),
                    y_source: stairs.y_source.as_ref().map(SceneNumericData::from),
                    color_rgba: vec4_to_rgba(stairs.color),
                    line_width: stairs.line_width,
                    axes_index,
                    label: stairs.label.clone(),
                    visible: stairs.visible,
                }
            }
            PlotElement::Stem(stem) => {
                let (x, y) = stem
                    .export_scene_xy_data()
                    .await
                    .map_err(SceneExportError::readback)?;
                if x.is_empty() && y.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "stem plot has no exportable scene data",
                    )
                    .into());
                }
                Self::Stem {
                    x,
                    y,
                    x_source: stem.x_source.as_ref().map(SceneNumericData::from),
                    y_source: stem.y_source.as_ref().map(SceneNumericData::from),
                    baseline: stem.baseline,
                    color_rgba: vec4_to_rgba(stem.color),
                    line_width: stem.line_width,
                    line_style: format!("{:?}", stem.line_style),
                    baseline_color_rgba: vec4_to_rgba(stem.baseline_color),
                    baseline_visible: stem.baseline_visible,
                    marker_color_rgba: vec4_to_rgba(
                        stem.marker
                            .as_ref()
                            .map(|m| m.face_color)
                            .unwrap_or(stem.color),
                    ),
                    marker_size: stem.marker.as_ref().map(|m| m.size).unwrap_or(0.0),
                    marker_filled: stem.marker.as_ref().map(|m| m.filled).unwrap_or(false),
                    axes_index,
                    label: stem.label.clone(),
                    visible: stem.visible,
                }
            }
            PlotElement::Area(area) => {
                let (x, y) = area
                    .export_scene_xy_data()
                    .await
                    .map_err(SceneExportError::readback)?;
                if x.is_empty() && y.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "area plot has no exportable scene data",
                    )
                    .into());
                }
                Self::Area {
                    x,
                    y,
                    lower_y: area.lower_y.clone(),
                    baseline: area.baseline,
                    color_rgba: vec4_to_rgba(area.color),
                    axes_index,
                    label: area.label.clone(),
                    visible: area.visible,
                }
            }
            PlotElement::Quiver(quiver) => {
                let (x, y, z, u, v, w) = quiver
                    .export_scene_vector_data()
                    .await
                    .map_err(SceneExportError::readback)?;
                if x.is_empty() && y.is_empty() && u.is_empty() && v.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "quiver plot has no exportable scene data",
                    )
                    .into());
                }
                let source = quiver.source_data();
                Self::Quiver {
                    x,
                    y,
                    z,
                    u,
                    v,
                    w,
                    x_source: source.map(|source| source.0.into()),
                    y_source: source.map(|source| source.1.into()),
                    z_source: source.and_then(|source| source.2.map(Into::into)),
                    u_source: source.map(|source| source.3.into()),
                    v_source: source.map(|source| source.4.into()),
                    w_source: source.and_then(|source| source.5.map(Into::into)),
                    color_rgba: vec4_to_rgba(quiver.color),
                    line_width: quiver.line_width,
                    scale: quiver.scale,
                    head_size: quiver.head_size,
                    axes_index,
                    label: quiver.label.clone(),
                    visible: quiver.visible,
                }
            }
            PlotElement::Surface(surface) => {
                let (x, y, z) = surface
                    .export_scene_grid_data()
                    .await
                    .map_err(SceneExportError::readback)?;
                if x.is_empty() && y.is_empty() && z.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "surface plot has no exportable scene data",
                    )
                    .into());
                }
                let color_grid = surface
                    .export_scene_color_grid()
                    .await
                    .map_err(SceneExportError::readback)?;
                let (source_x, source_y, source_z) = surface.source_data();
                Self::Surface {
                    x,
                    y,
                    z,
                    x_source: source_x.map(Into::into),
                    y_source: source_y.map(Into::into),
                    z_source: source_z.map(Into::into),
                    c_source: surface.color_source().map(Into::into),
                    x_grid: surface.x_grid.clone(),
                    y_grid: surface.y_grid.clone(),
                    colormap: surface.colormap.to_serialized_token(),
                    shading_mode: format!("{:?}", surface.shading_mode),
                    wireframe: surface.wireframe,
                    alpha: surface.alpha,
                    flatten_z: surface.flatten_z,
                    color_mapping: surface.color_mapping,
                    image_mode: surface.image_mode,
                    cell_labels: surface.cell_labels.clone(),
                    color_grid_rgba: color_grid.as_ref().map(|grid| {
                        grid.iter()
                            .map(|row| row.iter().map(|color| vec4_to_rgba(*color)).collect())
                            .collect()
                    }),
                    color_limits: surface.color_limits.map(|(lo, hi)| [lo, hi]),
                    axes_index,
                    label: surface.label.clone(),
                    visible: surface.visible,
                }
            }
            PlotElement::Patch(_) | PlotElement::Mesh(_) | PlotElement::Pie(_) => {
                Self::from_plot(plot, axes_index)
            }
            PlotElement::Line3(line) => {
                let (x_data, y_data, z_data) = line
                    .export_numeric_xyz_data()
                    .await
                    .map_err(SceneExportError::readback)?
                    .ok_or_else(|| {
                        SceneExportError::unexportable("plot3 line has no exportable scene data")
                    })?;
                let x = x_data.materialize_f64();
                let y = y_data.materialize_f64();
                let z = z_data.materialize_f64();
                if x.is_empty() && y.is_empty() && z.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "plot3 line has no exportable scene data",
                    )
                    .into());
                }
                Self::Line3 {
                    x,
                    y,
                    z,
                    x_source: Some((&x_data).into()),
                    y_source: Some((&y_data).into()),
                    z_source: Some((&z_data).into()),
                    color_rgba: vec4_to_rgba(line.color),
                    line_width: line.line_width,
                    line_style: format!("{:?}", line.line_style),
                    axes_index,
                    label: line.label.clone(),
                    visible: line.visible,
                }
            }
            PlotElement::Scatter3(scatter3) => {
                let (x_data, y_data, z_data) = scatter3
                    .export_numeric_xyz_data()
                    .await
                    .map_err(SceneExportError::readback)?
                    .ok_or_else(|| {
                        SceneExportError::unexportable("scatter3 plot has no exportable scene data")
                    })?;
                let points = x_data
                    .materialize_f64()
                    .into_iter()
                    .zip(y_data.materialize_f64())
                    .zip(z_data.materialize_f64())
                    .map(|((x, y), z)| [x as f32, y as f32, z as f32])
                    .collect::<Vec<_>>();
                if points.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "scatter3 plot has no exportable scene data",
                    )
                    .into());
                }
                let colors = scatter3
                    .export_scene_colors(points.len())
                    .await
                    .map_err(SceneExportError::readback)?;
                Self::Scatter3 {
                    points,
                    x_source: Some((&x_data).into()),
                    y_source: Some((&y_data).into()),
                    z_source: Some((&z_data).into()),
                    colors_rgba: colors.into_iter().map(vec4_to_rgba).collect(),
                    point_size: scatter3.point_size,
                    point_sizes: scatter3.point_sizes.clone(),
                    axes_index,
                    label: scatter3.label.clone(),
                    visible: scatter3.visible,
                }
            }
            PlotElement::Contour(contour) => {
                let vertices = contour
                    .export_scene_vertices()
                    .await
                    .map_err(SceneExportError::readback)?;
                if vertices.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "contour plot has no exportable scene data",
                    )
                    .into());
                }
                Self::Contour {
                    vertices: vertices.into_iter().map(Into::into).collect(),
                    bounds_min: vec3_to_xyz(contour.bounds().min),
                    bounds_max: vec3_to_xyz(contour.bounds().max),
                    base_z: contour.base_z,
                    line_width: contour.line_width,
                    axes_index,
                    label: contour.label.clone(),
                    visible: contour.visible,
                    force_3d: contour.force_3d,
                    value_range: contour.value_range.map(|(a, b)| [a, b]),
                }
            }
            PlotElement::ContourFill(fill) => {
                let vertices = fill
                    .export_scene_vertices()
                    .await
                    .map_err(SceneExportError::readback)?;
                if vertices.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "filled contour plot has no exportable scene data",
                    )
                    .into());
                }
                Self::ContourFill {
                    vertices: vertices.into_iter().map(Into::into).collect(),
                    bounds_min: vec3_to_xyz(fill.bounds().min),
                    bounds_max: vec3_to_xyz(fill.bounds().max),
                    axes_index,
                    label: fill.label.clone(),
                    visible: fill.visible,
                    value_range: fill.value_range.map(|(a, b)| [a, b]),
                }
            }
        };
        scene_plot.validate_exportable()?;
        Ok(scene_plot)
    }

    pub(super) fn validate_exportable(&self) -> Result<(), SceneExportError> {
        match self {
            ScenePlot::Line { x, y, .. }
            | ScenePlot::Scatter { x, y, .. }
            | ScenePlot::Stairs { x, y, .. }
            | ScenePlot::Stem { x, y, .. }
            | ScenePlot::Area { x, y, .. } => {
                validate_required_equal_lengths(
                    "plot X/Y scene data",
                    &[("x", x.len()), ("y", y.len())],
                )?;
            }
            ScenePlot::ErrorBar {
                x,
                y,
                err_low,
                err_high,
                x_err_low,
                x_err_high,
                ..
            } => {
                validate_required_equal_lengths(
                    "errorbar scene data",
                    &[
                        ("x", x.len()),
                        ("y", y.len()),
                        ("err_low", err_low.len()),
                        ("err_high", err_high.len()),
                        ("x_err_low", x_err_low.len()),
                        ("x_err_high", x_err_high.len()),
                    ],
                )?;
            }
            ScenePlot::Quiver {
                x, y, z, u, v, w, ..
            } => {
                validate_required_equal_lengths(
                    "quiver vector field scene data",
                    &[
                        ("x", x.len()),
                        ("y", y.len()),
                        ("u", u.len()),
                        ("v", v.len()),
                    ],
                )?;
                if let Some(z) = z {
                    validate_required_equal_lengths(
                        "quiver3 position scene data",
                        &[("x", x.len()), ("z", z.len())],
                    )?;
                }
                if let Some(w) = w {
                    validate_required_equal_lengths(
                        "quiver3 vector scene data",
                        &[("u", u.len()), ("w", w.len())],
                    )?;
                }
                if z.is_some() != w.is_some() {
                    return Err(SceneExportError::unexportable(
                        "quiver3 scene data requires both z and w arrays",
                    ));
                }
            }
            ScenePlot::Bar {
                labels,
                values,
                histogram_bin_edges,
                stack_offsets,
                ..
            } => {
                validate_required_equal_lengths(
                    "bar value scene data",
                    &[("labels", labels.len()), ("values", values.len())],
                )?;
                if let Some(edges) = histogram_bin_edges {
                    if edges.len() != values.len() + 1 {
                        return Err(SceneExportError::unexportable(format!(
                            "bar histogram bin edge count ({}) must be values length + 1 ({})",
                            edges.len(),
                            values.len() + 1
                        )));
                    }
                }
                if let Some(offsets) = stack_offsets {
                    if offsets.len() != values.len() {
                        return Err(SceneExportError::unexportable(format!(
                            "bar stack offset count ({}) must match value count ({})",
                            offsets.len(),
                            values.len()
                        )));
                    }
                }
            }
            ScenePlot::Surface {
                x,
                y,
                z,
                x_grid,
                y_grid,
                color_grid_rgba,
                ..
            } => {
                match (x_grid, y_grid) {
                    (Some(x_grid), Some(y_grid)) => validate_surface_coordinate_grids(
                        "surface coordinate grid scene data",
                        x_grid,
                        y_grid,
                        z,
                    )?,
                    (None, None) => validate_surface_grid("surface grid scene data", x, y, z)?,
                    _ => {
                        return Err(SceneExportError::unexportable(
                            "surface coordinate grid scene data must include both xGrid and yGrid",
                        ));
                    }
                }
                if let Some(color_grid) = color_grid_rgba {
                    validate_matching_grid_shape("surface color grid scene data", z, color_grid)?;
                }
            }
            ScenePlot::Patch {
                vertices, faces, ..
            } => {
                if vertices.is_empty() || faces.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "patch plot has no exportable mesh scene data",
                    ));
                }
            }
            ScenePlot::Mesh {
                vertices,
                triangles,
                ..
            } => {
                if vertices.is_empty() || triangles.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "mesh plot has no exportable mesh scene data",
                    ));
                }
            }
            ScenePlot::Line3 { x, y, z, .. } => {
                validate_required_equal_lengths(
                    "plot3 line scene data",
                    &[("x", x.len()), ("y", y.len()), ("z", z.len())],
                )?;
            }
            ScenePlot::Scatter3 {
                points,
                x_source,
                y_source,
                z_source,
                colors_rgba,
                point_sizes,
                ..
            } => {
                if points.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "scatter3 plot has no exportable point scene data",
                    ));
                }
                match (x_source, y_source, z_source) {
                    (Some(x_source), Some(y_source), Some(z_source)) => {
                        let x_data: NumericPlotData = x_source
                            .clone()
                            .try_into()
                            .map_err(SceneExportError::unexportable)?;
                        let y_data: NumericPlotData = y_source
                            .clone()
                            .try_into()
                            .map_err(SceneExportError::unexportable)?;
                        let z_data: NumericPlotData = z_source
                            .clone()
                            .try_into()
                            .map_err(SceneExportError::unexportable)?;
                        validate_required_equal_lengths(
                            "scatter3 typed source scene data",
                            &[
                                ("points", points.len()),
                                ("xSource", x_data.len()),
                                ("ySource", y_data.len()),
                                ("zSource", z_data.len()),
                            ],
                        )?;
                    }
                    (None, None, None) => {}
                    _ => {
                        return Err(SceneExportError::unexportable(
                            "scatter3 scene contains partial typed source data",
                        ))
                    }
                }
                if !colors_rgba.is_empty() && colors_rgba.len() != points.len() {
                    return Err(SceneExportError::unexportable(format!(
                        "scatter3 color count ({}) must match point count ({})",
                        colors_rgba.len(),
                        points.len()
                    )));
                }
                if let Some(sizes) = point_sizes {
                    if sizes.len() != points.len() {
                        return Err(SceneExportError::unexportable(format!(
                            "scatter3 point size count ({}) must match point count ({})",
                            sizes.len(),
                            points.len()
                        )));
                    }
                }
            }
            ScenePlot::Contour { vertices, .. } | ScenePlot::ContourFill { vertices, .. } => {
                if vertices.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "contour plot has no exportable vertex scene data",
                    ));
                }
            }
            ScenePlot::Pie { values, .. } => {
                if values.is_empty() {
                    return Err(SceneExportError::unexportable(
                        "pie plot has no exportable value scene data",
                    ));
                }
            }
            ScenePlot::ReferenceLine { .. } => {}
            ScenePlot::Unsupported { plot_kind, .. } => {
                return Err(SceneExportError::unexportable(format!(
                    "unsupported plot kind cannot be exported: {plot_kind:?}"
                )));
            }
        }
        Ok(())
    }
}
