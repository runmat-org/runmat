//! Restoring a scene plot into a figure.

use super::*;

impl ScenePlot {
    pub(super) fn apply_to_figure(self, figure: &mut Figure) -> Result<(), String> {
        match self {
            ScenePlot::Line {
                x,
                y,
                x_source,
                y_source,
                color_rgba,
                line_width,
                line_style,
                marker_style,
                marker_size,
                marker_face_color,
                marker_edge_color,
                marker_filled,
                handle_visibility,
                axes_index,
                label,
                visible,
            } => {
                let mut line = match (x_source, y_source) {
                    (Some(x_source), Some(y_source)) => {
                        LinePlot::from_numeric_data(x_source.try_into()?, y_source.try_into()?)?
                    }
                    (None, None) => LinePlot::new(x, y)?,
                    _ => return Err("line scene contains partial typed source data".to_string()),
                };
                line.set_color(rgba_to_vec4(color_rgba));
                line.set_line_width(line_width);
                line.set_line_style(parse_line_style(&line_style));
                if let Some(visibility) = handle_visibility {
                    line.handle_visibility = visibility;
                }
                if let Some(size) = marker_size {
                    line.set_marker(Some(crate::plots::line::LineMarkerAppearance {
                        kind: parse_marker_style(marker_style.as_deref().unwrap_or("Circle")),
                        size,
                        edge_color: marker_edge_color
                            .map(rgba_to_vec4)
                            .unwrap_or(rgba_to_vec4(color_rgba)),
                        face_color: marker_face_color
                            .map(rgba_to_vec4)
                            .unwrap_or(rgba_to_vec4(color_rgba)),
                        filled: marker_filled.unwrap_or(false),
                    }));
                }
                line.label = label;
                line.set_visible(visible);
                figure.add_line_plot_on_axes(line, axes_index as usize);
            }
            ScenePlot::ReferenceLine {
                orientation,
                value,
                color_rgba,
                line_width,
                line_style,
                label,
                display_name,
                label_orientation,
                axes_index,
                visible,
                color_auto,
            } => {
                let orientation = parse_reference_line_orientation(&orientation)?;
                let mut line = ReferenceLine::new(orientation, value)?.with_style(
                    rgba_to_vec4(color_rgba),
                    line_width,
                    parse_line_style(&line_style),
                );
                line.label = label;
                line.display_name = display_name;
                line.label_orientation = label_orientation;
                line.visible = visible;
                line.color_auto = color_auto;
                figure.add_reference_line_on_axes(line, axes_index as usize);
            }
            ScenePlot::Scatter {
                x,
                y,
                x_source,
                y_source,
                color_rgba,
                marker_size,
                marker_style,
                axes_index,
                label,
                visible,
            } => {
                let mut scatter = match (x_source, y_source) {
                    (Some(x_source), Some(y_source)) => {
                        ScatterPlot::from_numeric_data(x_source.try_into()?, y_source.try_into()?)?
                    }
                    (None, None) => ScatterPlot::new(x, y)?,
                    _ => return Err("scatter scene contains partial typed source data".to_string()),
                };
                scatter.set_color(rgba_to_vec4(color_rgba));
                scatter.set_marker_size(marker_size);
                scatter.set_marker_style(parse_marker_style(&marker_style));
                scatter.label = label;
                scatter.set_visible(visible);
                figure.add_scatter_plot_on_axes(scatter, axes_index as usize);
            }
            ScenePlot::Bar {
                labels,
                values,
                histogram_bin_edges,
                polar_histogram,
                polar_histogram_display_style,
                color_rgba,
                outline_color_rgba,
                bar_width,
                outline_width,
                orientation,
                group_index,
                group_count,
                stack_offsets,
                axes_index,
                label,
                visible,
            } => {
                let mut bar = BarChart::new(labels, values)?
                    .with_style(rgba_to_vec4(color_rgba), bar_width)
                    .with_orientation(parse_bar_orientation(&orientation))
                    .with_group(group_index as usize, group_count as usize);
                if let Some(edges) = histogram_bin_edges {
                    bar.set_histogram_bin_edges(edges);
                }
                if polar_histogram {
                    bar.set_polar_histogram(true);
                    if let Some(style) = polar_histogram_display_style {
                        if style.eq_ignore_ascii_case("stairs") {
                            bar.set_polar_histogram_display_style(
                                crate::plots::PolarHistogramDisplayStyle::Stairs,
                            );
                        }
                    }
                }
                if let Some(offsets) = stack_offsets {
                    bar = bar.with_stack_offsets(offsets);
                }
                if let Some(outline) = outline_color_rgba {
                    bar = bar.with_outline(rgba_to_vec4(outline), outline_width);
                }
                bar.label = label;
                bar.set_visible(visible);
                figure.add_bar_chart_on_axes(bar, axes_index as usize);
            }
            ScenePlot::ErrorBar {
                x,
                y,
                err_low,
                err_high,
                x_err_low,
                x_err_high,
                orientation,
                color_rgba,
                line_width,
                line_style,
                cap_width,
                marker_style,
                marker_size,
                marker_face_color,
                marker_edge_color,
                marker_filled,
                axes_index,
                label,
                visible,
            } => {
                let mut error = if orientation.eq_ignore_ascii_case("Both") {
                    ErrorBar::new_both(x, y, x_err_low, x_err_high, err_low, err_high)?
                } else {
                    ErrorBar::new_vertical(x, y, err_low, err_high)?
                }
                .with_style(
                    rgba_to_vec4(color_rgba),
                    line_width,
                    parse_line_style_name(&line_style),
                    cap_width,
                );
                if let Some(size) = marker_size {
                    error.set_marker(Some(crate::plots::line::LineMarkerAppearance {
                        kind: parse_marker_style(marker_style.as_deref().unwrap_or("Circle")),
                        size,
                        edge_color: marker_edge_color
                            .map(rgba_to_vec4)
                            .unwrap_or(rgba_to_vec4(color_rgba)),
                        face_color: marker_face_color
                            .map(rgba_to_vec4)
                            .unwrap_or(rgba_to_vec4(color_rgba)),
                        filled: marker_filled.unwrap_or(false),
                    }));
                }
                error.label = label;
                error.set_visible(visible);
                figure.add_errorbar_on_axes(error, axes_index as usize);
            }
            ScenePlot::Stairs {
                x,
                y,
                x_source,
                y_source,
                color_rgba,
                line_width,
                axes_index,
                label,
                visible,
            } => {
                let mut stairs = match (x_source, y_source) {
                    (Some(x_source), Some(y_source)) => StairsPlot::new_with_source(
                        NumericPlotData::try_from(x_source)?,
                        NumericPlotData::try_from(y_source)?,
                    )?,
                    _ => StairsPlot::new(x, y)?,
                };
                stairs.color = rgba_to_vec4(color_rgba);
                stairs.line_width = line_width;
                stairs.label = label;
                stairs.set_visible(visible);
                figure.add_stairs_plot_on_axes(stairs, axes_index as usize);
            }
            ScenePlot::Stem {
                x,
                y,
                x_source,
                y_source,
                baseline,
                color_rgba,
                line_width,
                line_style,
                baseline_color_rgba,
                baseline_visible,
                marker_color_rgba,
                marker_size,
                marker_filled,
                axes_index,
                label,
                visible,
            } => {
                let mut stem = match (x_source, y_source) {
                    (Some(x_source), Some(y_source)) => StemPlot::new_with_source(
                        NumericPlotData::try_from(x_source)?,
                        NumericPlotData::try_from(y_source)?,
                    )?,
                    _ => StemPlot::new(x, y)?,
                };
                stem = stem
                    .with_style(
                        rgba_to_vec4(color_rgba),
                        line_width,
                        parse_line_style_name(&line_style),
                        baseline,
                    )
                    .with_baseline_style(rgba_to_vec4(baseline_color_rgba), baseline_visible);
                if marker_size > 0.0 {
                    stem.set_marker(Some(crate::plots::line::LineMarkerAppearance {
                        kind: crate::plots::scatter::MarkerStyle::Circle,
                        size: marker_size,
                        edge_color: rgba_to_vec4(marker_color_rgba),
                        face_color: rgba_to_vec4(marker_color_rgba),
                        filled: marker_filled,
                    }));
                }
                stem.label = label;
                stem.set_visible(visible);
                figure.add_stem_plot_on_axes(stem, axes_index as usize);
            }
            ScenePlot::Area {
                x,
                y,
                lower_y,
                baseline,
                color_rgba,
                axes_index,
                label,
                visible,
            } => {
                let mut area = AreaPlot::new(x, y)?;
                if let Some(lower_y) = lower_y {
                    area = area.with_lower_curve(lower_y);
                }
                area.baseline = baseline;
                area.color = rgba_to_vec4(color_rgba);
                area.label = label;
                area.set_visible(visible);
                figure.add_area_plot_on_axes(area, axes_index as usize);
            }
            ScenePlot::Quiver {
                x,
                y,
                z,
                u,
                v,
                w,
                x_source,
                y_source,
                z_source,
                u_source,
                v_source,
                w_source,
                color_rgba,
                line_width,
                scale,
                head_size,
                axes_index,
                label,
                visible,
            } => {
                let mut quiver = match (x_source, y_source, z_source, u_source, v_source, w_source)
                {
                    (Some(x), Some(y), Some(z), Some(u), Some(v), Some(w)) => {
                        QuiverPlot::from_numeric_data3d(
                            x.try_into()?,
                            y.try_into()?,
                            z.try_into()?,
                            u.try_into()?,
                            v.try_into()?,
                            w.try_into()?,
                        )?
                    }
                    (Some(x), Some(y), None, Some(u), Some(v), None) => {
                        QuiverPlot::from_numeric_data(
                            x.try_into()?,
                            y.try_into()?,
                            u.try_into()?,
                            v.try_into()?,
                        )?
                    }
                    (None, None, None, None, None, None) => {
                        if let (Some(z), Some(w)) = (z, w) {
                            QuiverPlot::new3d(x, y, z, u, v, w)?
                        } else {
                            QuiverPlot::new(x, y, u, v)?
                        }
                    }
                    _ => return Err("quiver scene contains partial typed source data".to_string()),
                }
                .with_style(rgba_to_vec4(color_rgba), line_width, scale, head_size)
                .with_label(label.unwrap_or_else(|| "Data".to_string()));
                quiver.set_visible(visible);
                figure.add_quiver_plot_on_axes(quiver, axes_index as usize);
            }
            ScenePlot::Surface {
                x,
                y,
                z,
                x_source,
                y_source,
                z_source,
                c_source,
                x_grid,
                y_grid,
                colormap,
                shading_mode,
                wireframe,
                alpha,
                flatten_z,
                image_mode,
                cell_labels,
                color_mapping,
                color_grid_rgba,
                color_limits,
                axes_index,
                label,
                visible,
            } => {
                let mut surface = match (x_grid, y_grid) {
                    (Some(x_grid), Some(y_grid)) => {
                        SurfacePlot::from_coordinate_grids(x_grid, y_grid, z)?
                    }
                    (None, None) => SurfacePlot::new(x, y, z)?,
                    _ => {
                        return Err(
                            "surface scene must include both xGrid and yGrid for coordinate grids"
                                .to_string(),
                        );
                    }
                };
                if x_source.is_some() || y_source.is_some() || z_source.is_some() {
                    let (fallback_x, fallback_y, fallback_z) = surface.source_data();
                    let fallback_x = fallback_x.cloned();
                    let fallback_y = fallback_y.cloned();
                    let fallback_z = fallback_z.cloned();
                    surface.set_source_data(
                        x_source.map(TryInto::try_into).transpose()?.or(fallback_x),
                        y_source.map(TryInto::try_into).transpose()?.or(fallback_y),
                        z_source.map(TryInto::try_into).transpose()?.or(fallback_z),
                    );
                }
                if let Some(source) = c_source {
                    surface.set_color_source(source.try_into()?)?;
                }
                surface.colormap = parse_colormap(&colormap);
                surface.shading_mode = parse_shading_mode(&shading_mode);
                surface.wireframe = wireframe;
                surface.alpha = alpha.clamp(0.0, 1.0);
                surface.flatten_z = flatten_z;
                surface.image_mode = image_mode;
                surface.cell_labels = cell_labels;
                surface.color_mapping = color_mapping;
                surface.color_grid = color_grid_rgba.map(|grid| {
                    grid.into_iter()
                        .map(|row| row.into_iter().map(rgba_to_vec4).collect())
                        .collect()
                });
                surface.color_limits = color_limits.map(|[lo, hi]| (lo, hi));
                surface.label = label;
                surface.visible = visible;
                figure.add_surface_plot_on_axes(surface, axes_index as usize);
            }
            ScenePlot::Patch {
                vertices,
                faces,
                face_color_rgba,
                edge_color_rgba,
                face_color_mode,
                edge_color_mode,
                face_alpha,
                edge_alpha,
                line_width,
                axes_index,
                label,
                visible,
                force_3d,
                color_mapping,
                color_source_kind,
                x_source,
                y_source,
                z_source,
                c_source,
                faces_source,
                vertices_source,
            } => {
                let vertices: Vec<Vec3> = vertices.into_iter().map(xyz_to_vec3).collect();
                let faces: Vec<Vec<usize>> = faces
                    .into_iter()
                    .map(|face| face.into_iter().map(|idx| idx as usize).collect())
                    .collect();
                let mut patch = PatchPlot::new(vertices, faces)?;
                patch.set_face_color(rgba_to_vec4(face_color_rgba));
                patch.set_edge_color(rgba_to_vec4(edge_color_rgba));
                patch.set_face_color_mode(parse_patch_face_color_mode(&face_color_mode));
                patch.set_edge_color_mode(parse_patch_edge_color_mode(&edge_color_mode));
                patch.set_face_alpha(face_alpha);
                patch.set_edge_alpha(edge_alpha);
                patch.set_line_width(line_width);
                patch.set_label(label);
                patch.set_visible(visible);
                patch.set_force_3d(force_3d);
                patch.set_color_mapping(color_mapping);
                patch.set_source_data(crate::plots::PatchSourceDataOwned {
                    x: x_source.map(TryInto::try_into).transpose()?,
                    y: y_source.map(TryInto::try_into).transpose()?,
                    z: z_source.map(TryInto::try_into).transpose()?,
                    colors: c_source.map(TryInto::try_into).transpose()?,
                    color_kind: color_source_kind,
                    faces: faces_source.map(TryInto::try_into).transpose()?,
                    vertices: vertices_source.map(TryInto::try_into).transpose()?,
                })?;
                patch.validate_color_mode()?;
                figure.add_patch_plot_on_axes(patch, axes_index as usize);
            }
            ScenePlot::Mesh {
                vertices,
                triangles,
                mesh_id,
                face_color_rgba,
                edge_color_rgba,
                face_alpha,
                edge_alpha,
                edge_width,
                edge_mode,
                feature_edge_groups,
                vertex_colors_rgba,
                triangle_colors_rgba,
                vertex_normals,
                axes_index,
                label,
                regions,
                highlighted_region_id,
                highlight_color_rgba,
                scalar_field,
                vector_field,
                deformation,
                instances: _,
                lod_pixels: _,
                packed,
                visible,
            } => {
                let (vertices, triangles, vertex_normals) = match packed {
                    Some(packed) => packed.decode()?,
                    None => (vertices, triangles, vertex_normals),
                };
                let vertices: Vec<Vec3> = vertices.into_iter().map(xyz_to_vec3).collect();
                let mut mesh = MeshPlot::new(vertices, triangles)?;
                mesh.set_mesh_id(mesh_id);
                mesh.set_face_color(rgba_to_vec4(face_color_rgba));
                mesh.set_edge_color(rgba_to_vec4(edge_color_rgba));
                mesh.set_face_alpha(face_alpha);
                mesh.set_edge_alpha(edge_alpha);
                mesh.set_edge_width(edge_width);
                mesh.set_edge_mode(parse_mesh_edge_mode(&edge_mode));
                if !feature_edge_groups.is_empty() {
                    mesh.set_feature_edge_groups(Some(feature_edge_groups))?;
                }
                if !vertex_colors_rgba.is_empty() {
                    mesh.set_vertex_colors(Some(
                        vertex_colors_rgba.into_iter().map(rgba_to_vec4).collect(),
                    ))?;
                }
                if !vertex_normals.is_empty() {
                    mesh.set_vertex_normals(Some(
                        vertex_normals.into_iter().map(Vec3::from_array).collect(),
                    ))?;
                }
                if !triangle_colors_rgba.is_empty() {
                    mesh.set_triangle_colors(Some(
                        triangle_colors_rgba.into_iter().map(rgba_to_vec4).collect(),
                    ))?;
                }
                mesh.set_label(label);
                mesh.set_regions(regions.into_iter().map(Into::into).collect());
                mesh.set_highlighted_region_id(highlighted_region_id);
                if let Some(color) = highlight_color_rgba {
                    mesh.set_highlight_color(rgba_to_vec4(color));
                }
                if let Some(field) = scalar_field {
                    mesh.set_scalar_field(Some((*field).try_into()?))?;
                }
                if let Some(field) = vector_field {
                    mesh.set_vector_field(Some((*field).try_into()?))?;
                }
                if let Some(field) = deformation {
                    mesh.set_deformation(Some((*field).into()))?;
                }
                mesh.set_visible(visible);
                figure.add_mesh_plot_on_axes(mesh, axes_index as usize);
            }
            ScenePlot::Line3 {
                x,
                y,
                z,
                x_source,
                y_source,
                z_source,
                color_rgba,
                line_width,
                line_style,
                axes_index,
                label,
                visible,
            } => {
                let line3 = match (x_source, y_source, z_source) {
                    (Some(x_source), Some(y_source), Some(z_source)) => {
                        Line3Plot::from_numeric_data(
                            x_source.try_into()?,
                            y_source.try_into()?,
                            z_source.try_into()?,
                        )?
                    }
                    (None, None, None) => Line3Plot::new(x, y, z)?,
                    _ => return Err("line3 scene contains partial typed source data".to_string()),
                };
                let mut plot = line3
                    .with_style(
                        rgba_to_vec4(color_rgba),
                        line_width,
                        parse_line_style_name(&line_style),
                    )
                    .with_label(label.unwrap_or_else(|| "Data".to_string()));
                plot.set_visible(visible);
                figure.add_line3_plot_on_axes(plot, axes_index as usize);
            }
            ScenePlot::Scatter3 {
                points,
                x_source,
                y_source,
                z_source,
                colors_rgba,
                point_size,
                point_sizes,
                axes_index,
                label,
                visible,
            } => {
                let colors: Vec<Vec4> = colors_rgba.into_iter().map(rgba_to_vec4).collect();
                let mut scatter3 = match (x_source, y_source, z_source) {
                    (Some(x_source), Some(y_source), Some(z_source)) => {
                        Scatter3Plot::from_numeric_data(
                            x_source.try_into()?,
                            y_source.try_into()?,
                            z_source.try_into()?,
                        )?
                    }
                    (None, None, None) => {
                        let points: Vec<Vec3> = points.into_iter().map(xyz_to_vec3).collect();
                        Scatter3Plot::new(points)?
                    }
                    _ => {
                        return Err("scatter3 scene contains partial typed source data".to_string())
                    }
                };
                if !colors.is_empty() {
                    scatter3 = scatter3.with_colors(colors)?;
                }
                scatter3.point_size = point_size.max(1.0);
                scatter3.point_sizes = point_sizes;
                scatter3.label = label;
                scatter3.visible = visible;
                figure.add_scatter3_plot_on_axes(scatter3, axes_index as usize);
            }
            ScenePlot::Contour {
                vertices,
                bounds_min,
                bounds_max,
                base_z,
                line_width,
                axes_index,
                label,
                visible,
                force_3d,
                value_range,
            } => {
                let mut contour = ContourPlot::from_vertices(
                    vertices.into_iter().map(Into::into).collect(),
                    base_z,
                    serialized_bounds(bounds_min, bounds_max),
                )
                .with_line_width(line_width)
                .with_force_3d(force_3d)
                .with_value_range(value_range.map(|[a, b]| (a, b)));
                contour.label = label;
                contour.set_visible(visible);
                figure.add_contour_plot_on_axes(contour, axes_index as usize);
            }
            ScenePlot::ContourFill {
                vertices,
                bounds_min,
                bounds_max,
                axes_index,
                label,
                visible,
                value_range,
            } => {
                let mut fill = ContourFillPlot::from_vertices(
                    vertices.into_iter().map(Into::into).collect(),
                    serialized_bounds(bounds_min, bounds_max),
                )
                .with_value_range(value_range.map(|[a, b]| (a, b)));
                fill.label = label;
                fill.set_visible(visible);
                figure.add_contour_fill_plot_on_axes(fill, axes_index as usize);
            }
            ScenePlot::Pie {
                values,
                colors_rgba,
                slice_labels,
                legend_labels,
                label_format,
                explode,
                axes_index,
                label,
                visible,
            } => {
                let mut pie = crate::plots::PieChart::new(
                    values,
                    Some(colors_rgba.into_iter().map(rgba_to_vec4).collect()),
                )?
                .with_slice_labels(slice_labels)
                .with_explode(explode);
                if let Some(fmt) = label_format {
                    pie = pie.with_label_format(fmt);
                }
                pie.label = label;
                pie.set_legend_labels(legend_labels);
                pie.set_visible(visible);
                figure.add_pie_chart_on_axes(pie, axes_index as usize);
            }
            ScenePlot::Unsupported { .. } => {}
        }
        Ok(())
    }
}
