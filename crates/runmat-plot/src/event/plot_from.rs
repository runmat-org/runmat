//! A scene plot from a figure's plot element (the snapshot form).

use super::*;

impl ScenePlot {
    pub(super) fn from_plot(plot: &PlotElement, axes_index: u32) -> Self {
        match plot {
            PlotElement::Line(line) => {
                let (x, y) = line.host_xy_f64().ok().flatten().unwrap_or_default();
                let (source_x, source_y) = line.source_data();
                Self::Line {
                    x,
                    y,
                    x_source: source_x.map(Into::into),
                    y_source: source_y.map(Into::into),
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
            PlotElement::ReferenceLine(line) => Self::ReferenceLine {
                orientation: match line.orientation {
                    ReferenceLineOrientation::Vertical => "vertical",
                    ReferenceLineOrientation::Horizontal => "horizontal",
                }
                .into(),
                value: line.value,
                color_rgba: vec4_to_rgba(line.color),
                line_width: line.line_width,
                line_style: format!("{:?}", line.line_style),
                label: line.label.clone(),
                display_name: line.display_name.clone(),
                label_orientation: line.label_orientation.clone(),
                axes_index,
                visible: line.visible,
                color_auto: line.color_auto,
            },
            PlotElement::Scatter(scatter) => {
                let (x, y) = scatter.host_xy_f64().ok().flatten().unwrap_or_default();
                let (source_x, source_y) = scatter.source_data();
                Self::Scatter {
                    x,
                    y,
                    x_source: source_x.map(Into::into),
                    y_source: source_y.map(Into::into),
                    color_rgba: vec4_to_rgba(scatter.color),
                    marker_size: scatter.marker_size,
                    marker_style: format!("{:?}", scatter.marker_style),
                    axes_index,
                    label: scatter.label.clone(),
                    visible: scatter.visible,
                }
            }
            PlotElement::Bar(bar) => Self::Bar {
                labels: bar.labels.clone(),
                values: bar.values().unwrap_or(&[]).to_vec(),
                histogram_bin_edges: bar.histogram_bin_edges().map(|edges| edges.to_vec()),
                color_rgba: vec4_to_rgba(bar.color),
                outline_color_rgba: bar.outline_color.map(vec4_to_rgba),
                bar_width: bar.bar_width,
                outline_width: bar.outline_width,
                orientation: format!("{:?}", bar.orientation),
                group_index: bar.group_index as u32,
                group_count: bar.group_count as u32,
                stack_offsets: bar.stack_offsets().map(|offsets| offsets.to_vec()),
                polar_histogram: bar.is_polar_histogram(),
                polar_histogram_display_style: Some(format!(
                    "{:?}",
                    bar.polar_histogram_display_style()
                )),
                axes_index,
                label: bar.label.clone(),
                visible: bar.visible,
            },
            PlotElement::ErrorBar(error) => Self::ErrorBar {
                x: error.x.clone(),
                y: error.y.clone(),
                err_low: error.y_neg.clone(),
                err_high: error.y_pos.clone(),
                x_err_low: error.x_neg.clone(),
                x_err_high: error.x_pos.clone(),
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
            },
            PlotElement::Stairs(stairs) => Self::Stairs {
                x: stairs.x.clone(),
                y: stairs.y.clone(),
                x_source: stairs.x_source.as_ref().map(SceneNumericData::from),
                y_source: stairs.y_source.as_ref().map(SceneNumericData::from),
                color_rgba: vec4_to_rgba(stairs.color),
                line_width: stairs.line_width,
                axes_index,
                label: stairs.label.clone(),
                visible: stairs.visible,
            },
            PlotElement::Stem(stem) => Self::Stem {
                x: stem.x.clone(),
                y: stem.y.clone(),
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
            },
            PlotElement::Area(area) => Self::Area {
                x: area.x.clone(),
                y: area.y.clone(),
                lower_y: area.lower_y.clone(),
                baseline: area.baseline,
                color_rgba: vec4_to_rgba(area.color),
                axes_index,
                label: area.label.clone(),
                visible: area.visible,
            },
            PlotElement::Quiver(quiver) => {
                let source = quiver.source_data();
                Self::Quiver {
                    x: quiver.x.clone(),
                    y: quiver.y.clone(),
                    z: quiver.z.clone(),
                    u: quiver.u.clone(),
                    v: quiver.v.clone(),
                    w: quiver.w.clone(),
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
            PlotElement::Surface(surface) => Self::Surface {
                x: surface.x_data.clone(),
                y: surface.y_data.clone(),
                z: surface.z_data.clone().unwrap_or_default(),
                x_source: surface.source_data().0.map(Into::into),
                y_source: surface.source_data().1.map(Into::into),
                z_source: surface.source_data().2.map(Into::into),
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
                color_grid_rgba: surface.color_grid.as_ref().map(|grid| {
                    grid.iter()
                        .map(|row| row.iter().map(|color| vec4_to_rgba(*color)).collect())
                        .collect()
                }),
                color_limits: surface.color_limits.map(|(lo, hi)| [lo, hi]),
                axes_index,
                label: surface.label.clone(),
                visible: surface.visible,
            },
            PlotElement::Patch(patch) => {
                let (source_x, source_y, source_z, source_c, source_faces, source_vertices) =
                    patch.source_data();
                Self::Patch {
                    vertices: patch
                        .vertices()
                        .iter()
                        .map(|point| vec3_to_xyz(*point))
                        .collect(),
                    faces: patch
                        .faces()
                        .iter()
                        .map(|face| face.iter().map(|idx| *idx as u32).collect())
                        .collect(),
                    face_color_rgba: vec4_to_rgba(patch.face_color()),
                    edge_color_rgba: vec4_to_rgba(patch.edge_color()),
                    face_color_mode: format!("{:?}", patch.face_color_mode()),
                    edge_color_mode: format!("{:?}", patch.edge_color_mode()),
                    face_alpha: patch.face_alpha(),
                    edge_alpha: patch.edge_alpha(),
                    line_width: patch.line_width(),
                    axes_index,
                    label: patch.label().map(str::to_string),
                    visible: patch.is_visible(),
                    force_3d: patch.force_3d(),
                    color_mapping: patch.color_mapping(),
                    color_source_kind: patch.color_source_kind(),
                    x_source: source_x.map(Into::into),
                    y_source: source_y.map(Into::into),
                    z_source: source_z.map(Into::into),
                    c_source: source_c.map(Into::into),
                    faces_source: source_faces.map(Into::into),
                    vertices_source: source_vertices.map(Into::into),
                }
            }
            PlotElement::Mesh(mesh) => Self::Mesh {
                vertices: mesh
                    .vertices()
                    .iter()
                    .map(|point| vec3_to_xyz(*point))
                    .collect(),
                triangles: mesh.triangles().to_vec(),
                mesh_id: mesh.mesh_id().map(str::to_string),
                face_color_rgba: vec4_to_rgba(mesh.face_color()),
                edge_color_rgba: vec4_to_rgba(mesh.edge_color()),
                face_alpha: mesh.face_alpha(),
                edge_alpha: mesh.edge_alpha(),
                edge_width: mesh.edge_width(),
                edge_mode: mesh.edge_mode().as_str().to_string(),
                feature_edge_groups: mesh
                    .feature_edge_groups()
                    .map(|groups| groups.to_vec())
                    .unwrap_or_default(),
                vertex_normals: mesh
                    .vertex_normals_source()
                    .map(|normals| normals.iter().map(|n| n.to_array()).collect())
                    .unwrap_or_default(),
                vertex_colors_rgba: mesh
                    .vertex_colors()
                    .map(|colors| colors.iter().copied().map(vec4_to_rgba).collect())
                    .unwrap_or_default(),
                triangle_colors_rgba: mesh
                    .triangle_colors()
                    .map(|colors| colors.iter().copied().map(vec4_to_rgba).collect())
                    .unwrap_or_default(),
                axes_index,
                label: mesh.label().map(str::to_string),
                regions: mesh.regions().iter().map(Into::into).collect(),
                highlighted_region_id: mesh.highlighted_region_id().map(str::to_string),
                highlight_color_rgba: Some(vec4_to_rgba(mesh.highlight_color())),
                scalar_field: mesh.scalar_field().map(|field| Box::new(field.into())),
                vector_field: mesh.vector_field().map(|field| Box::new(field.into())),
                deformation: mesh.deformation().map(|field| Box::new(field.into())),
                instances: Vec::new(),
                lod_pixels: None,
                packed: None,
                visible: mesh.is_visible(),
            },
            PlotElement::Line3(line) => {
                let (x, y, z) = line.host_xyz_f64().ok().flatten().unwrap_or_default();
                let (source_x, source_y, source_z) = line.source_data();
                Self::Line3 {
                    x,
                    y,
                    z,
                    x_source: source_x.map(Into::into),
                    y_source: source_y.map(Into::into),
                    z_source: source_z.map(Into::into),
                    color_rgba: vec4_to_rgba(line.color),
                    line_width: line.line_width,
                    line_style: format!("{:?}", line.line_style),
                    axes_index,
                    label: line.label.clone(),
                    visible: line.visible,
                }
            }
            PlotElement::Scatter3(scatter3) => {
                let source = scatter3.source_data();
                Self::Scatter3 {
                    points: scatter3
                        .points
                        .iter()
                        .map(|point| vec3_to_xyz(*point))
                        .collect(),
                    x_source: source.map(|source| source.0.into()),
                    y_source: source.map(|source| source.1.into()),
                    z_source: source.map(|source| source.2.into()),
                    colors_rgba: scatter3
                        .colors
                        .iter()
                        .map(|color| vec4_to_rgba(*color))
                        .collect(),
                    point_size: scatter3.point_size,
                    point_sizes: scatter3.point_sizes.clone(),
                    axes_index,
                    label: scatter3.label.clone(),
                    visible: scatter3.visible,
                }
            }
            PlotElement::Contour(contour) => Self::Contour {
                vertices: contour
                    .cpu_vertices()
                    .unwrap_or(&[])
                    .iter()
                    .cloned()
                    .map(Into::into)
                    .collect(),
                bounds_min: vec3_to_xyz(contour.bounds().min),
                bounds_max: vec3_to_xyz(contour.bounds().max),
                base_z: contour.base_z,
                line_width: contour.line_width,
                axes_index,
                label: contour.label.clone(),
                visible: contour.visible,
                force_3d: contour.force_3d,
                value_range: contour.value_range.map(|(a, b)| [a, b]),
            },
            PlotElement::ContourFill(fill) => Self::ContourFill {
                vertices: fill
                    .cpu_vertices()
                    .unwrap_or(&[])
                    .iter()
                    .cloned()
                    .map(Into::into)
                    .collect(),
                bounds_min: vec3_to_xyz(fill.bounds().min),
                bounds_max: vec3_to_xyz(fill.bounds().max),
                axes_index,
                label: fill.label.clone(),
                visible: fill.visible,
                value_range: fill.value_range.map(|(a, b)| [a, b]),
            },
            PlotElement::Pie(pie) => Self::Pie {
                values: pie.values.clone(),
                colors_rgba: pie.colors.iter().map(|c| vec4_to_rgba(*c)).collect(),
                slice_labels: pie.slice_labels.clone(),
                legend_labels: pie.legend_labels.clone(),
                label_format: pie.label_format.clone(),
                explode: pie.explode.clone(),
                axes_index,
                label: pie.label.clone(),
                visible: pie.visible,
            },
        }
    }
}
