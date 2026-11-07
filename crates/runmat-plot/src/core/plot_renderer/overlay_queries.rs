//! What the overlay draws for each axes: titles, labels, legends, ticks, colormaps and layout.

use super::*;

impl PlotRenderer {
    pub fn overlay_sg_title(&self) -> Option<&String> {
        self.figure_sg_title.as_ref()
    }
    pub fn overlay_sg_title_style(&self) -> &TextStyle {
        &self.figure_sg_title_style
    }
    /// The title of one axes. A geometry scene that draws its title (see
    /// [`Self::shows_geometry_title`]) has it on its one axes, so the overlay lays out a title
    /// band for it as it does for a figure's axes title.
    pub fn overlay_title_for_axes(&self, axes_index: usize) -> Option<&String> {
        if self.last_figure.is_none() && self.has_geometry_scene() {
            return self
                .figure_title
                .as_ref()
                .filter(|_| axes_index == 0 && self.shows_geometry_title());
        }
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .and_then(|m| m.title.as_ref())
    }
    pub fn overlay_title_style_for_axes(&self, axes_index: usize) -> Option<&TextStyle> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| &m.title_style)
    }
    pub fn overlay_subtitle_for_axes(&self, axes_index: usize) -> Option<&String> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .and_then(|m| m.subtitle.as_ref())
    }
    pub fn overlay_subtitle_style_for_axes(&self, axes_index: usize) -> Option<&TextStyle> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| &m.subtitle_style)
    }
    pub fn overlay_x_label(&self) -> Option<&String> {
        self.figure_x_label.as_ref()
    }
    pub fn overlay_x_label_for_axes(&self, axes_index: usize) -> Option<&String> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .and_then(|m| m.x_label.as_ref())
    }
    pub fn overlay_x_label_style_for_axes(&self, axes_index: usize) -> Option<&TextStyle> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| &m.x_label_style)
    }
    pub fn overlay_axes_style_for_axes(&self, axes_index: usize) -> Option<&TextStyle> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| &m.axes_style)
    }
    pub fn overlay_y_label(&self) -> Option<&String> {
        self.figure_y_label.as_ref()
    }
    pub fn overlay_y_label_for_axes(&self, axes_index: usize) -> Option<&String> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .and_then(|m| m.y_label.as_ref())
    }
    pub fn overlay_y_label_style_for_axes(&self, axes_index: usize) -> Option<&TextStyle> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| &m.y_label_style)
    }
    pub fn overlay_z_label(&self) -> Option<&String> {
        self.figure_z_label.as_ref()
    }
    pub fn overlay_z_label_for_axes(&self, axes_index: usize) -> Option<&String> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .and_then(|m| m.z_label.as_ref())
    }
    pub fn overlay_z_label_style_for_axes(&self, axes_index: usize) -> Option<&TextStyle> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| &m.z_label_style)
    }
    pub fn active_axes_pie_labels(&self) -> Vec<(String, glam::Vec2)> {
        let Some(fig) = self.last_figure.as_ref() else {
            return Vec::new();
        };
        fig.pie_labels_for_axes(fig.active_axes_index)
            .into_iter()
            .map(|entry| (entry.label, entry.position))
            .collect()
    }
    pub fn pie_labels_for_axes(&self, axes_index: usize) -> Vec<(String, glam::Vec2)> {
        let Some(fig) = self.last_figure.as_ref() else {
            return Vec::new();
        };
        fig.pie_labels_for_axes(axes_index)
            .into_iter()
            .map(|entry| (entry.label, entry.position))
            .collect()
    }

    pub fn reference_line_labels_for_axes(
        &self,
        axes_index: usize,
    ) -> Vec<(
        crate::plots::ReferenceLineOrientation,
        f64,
        String,
        Option<glam::Vec4>,
    )> {
        self.last_figure
            .as_ref()
            .map(|f| f.reference_line_labels_for_axes(axes_index))
            .unwrap_or_default()
    }

    pub fn cell_label_marks_for_axes(
        &self,
        axes_index: usize,
    ) -> Vec<(glam::Vec3, String, glam::Vec4)> {
        self.last_figure
            .as_ref()
            .map(|f| f.cell_label_marks_for_axes(axes_index))
            .unwrap_or_default()
    }

    pub fn world_text_annotations_for_axes(
        &self,
        axes_index: usize,
    ) -> Vec<(glam::Vec3, String, TextStyle)> {
        self.last_figure
            .as_ref()
            .map(|f| {
                f.axes_text_annotations(axes_index)
                    .iter()
                    .map(|annotation| {
                        (
                            annotation.position,
                            annotation.text.clone(),
                            annotation.style.clone(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn world_axis_label_annotations_for_axes(
        &self,
        axes_index: usize,
    ) -> Vec<(glam::Vec3, String, TextStyle)> {
        let Some(fig) = self.last_figure.as_ref() else {
            return Vec::new();
        };
        let Some(meta) = fig.axes_metadata(axes_index) else {
            return Vec::new();
        };
        let Some(bounds) = self.display_bounds_3d_for_axes(axes_index) else {
            return Vec::new();
        };
        let dx = (bounds.max.x - bounds.min.x).abs().max(1.0e-3);
        let dy = (bounds.max.y - bounds.min.y).abs().max(1.0e-3);
        let dz = (bounds.max.z - bounds.min.z).abs().max(1.0e-3);
        let camera = self
            .axes_camera(axes_index)
            .or_else(|| Some(self.camera()))
            .expect("plot renderer must always have a camera");
        let center = (bounds.min + bounds.max) * 0.5;
        let cam_delta = camera.position - center;
        let sx = if cam_delta.x >= 0.0 { 1.0 } else { -1.0 };
        let sy = if cam_delta.y >= 0.0 { 1.0 } else { -1.0 };
        let sz = if cam_delta.z >= 0.0 { 1.0 } else { -1.0 };
        let x_anchor = glam::Vec3::new(bounds.min.x + dx * 0.82, bounds.min.y, bounds.min.z)
            + glam::Vec3::new(0.0, -sy * dy * 0.10, -sz * dz * 0.08);
        let y_anchor = glam::Vec3::new(bounds.min.x, bounds.min.y + dy * 0.82, bounds.min.z)
            + glam::Vec3::new(-sx * dx * 0.10, 0.0, -sz * dz * 0.08);
        let z_anchor = glam::Vec3::new(bounds.min.x, bounds.min.y, bounds.min.z + dz * 0.82)
            + glam::Vec3::new(-sx * dx * 0.08, -sy * dy * 0.08, 0.0);
        let mut out = Vec::new();
        if let Some(label) = meta.x_label.clone().filter(|s| !s.is_empty()) {
            out.push((x_anchor, label, meta.x_label_style.clone()));
        }
        if let Some(label) = meta.y_label.clone().filter(|s| !s.is_empty()) {
            out.push((y_anchor, label, meta.y_label_style.clone()));
        }
        if let Some(label) = meta.z_label.clone().filter(|s| !s.is_empty()) {
            out.push((z_anchor, label, meta.z_label_style.clone()));
        }
        out
    }
    pub fn overlay_show_legend(&self) -> bool {
        self.figure_show_legend
    }
    pub fn overlay_show_legend_for_axes(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.legend_enabled)
            .unwrap_or(self.figure_show_legend)
    }
    pub fn overlay_legend_entries(&self) -> &Vec<LegendEntry> {
        &self.legend_entries
    }
    pub fn overlay_legend_entries_for_axes(&self, axes_index: usize) -> Vec<LegendEntry> {
        self.last_figure
            .as_ref()
            .map(|f| f.legend_entries_for_axes(axes_index))
            .unwrap_or_default()
    }
    pub fn overlay_x_log(&self) -> bool {
        self.figure_x_log
    }
    /// Width over height of the 2-D plot box when the axes fix their data aspect
    /// (`axis equal`, `daspect`); `None` when the box stretches to fill.
    pub fn overlay_plot_box_aspect_for_axes(&self, axes_index: usize) -> Option<f64> {
        let meta = self.last_figure.as_ref()?.axes_metadata(axes_index)?;
        if !(meta.axis_equal || meta.data_aspect_ratio_mode == "manual") || meta.x_log || meta.y_log
        {
            return None;
        }
        let (x_min, x_max, y_min, y_max) = self.display_bounds_for_axes(axes_index)?;
        let x_units = (x_max - x_min).abs() / meta.data_aspect_ratio[0].abs().max(1.0e-12);
        let y_units = (y_max - y_min).abs() / meta.data_aspect_ratio[1].abs().max(1.0e-12);
        let aspect = x_units / y_units;
        (aspect.is_finite() && aspect > 0.0).then_some(aspect)
    }
    pub fn overlay_x_log_for_axes(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.x_log)
            .unwrap_or(self.figure_x_log)
    }
    pub fn overlay_y_log(&self) -> bool {
        self.figure_y_log
    }
    pub fn overlay_y_log_for_axes(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.y_log)
            .unwrap_or(self.figure_y_log)
    }
    pub fn overlay_colormap(&self) -> ColorMap {
        self.figure_colormap.clone()
    }
    pub fn overlay_colorbar_enabled_for_axes(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.colorbar_enabled)
            .unwrap_or(self.figure_colorbar_enabled)
    }
    pub fn overlay_colormap_for_axes(&self, axes_index: usize) -> ColorMap {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.colormap.clone())
            .unwrap_or_else(|| self.figure_colormap.clone())
    }
    pub fn overlay_color_range_for_axes(&self, axes_index: usize) -> Option<(f64, f64)> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_color_range(axes_index))
    }
    pub fn overlay_colorbar_enabled(&self) -> bool {
        self.figure_colorbar_enabled
    }
    /// Subplot grid
    pub fn figure_axes_grid(&self) -> (usize, usize) {
        self.last_figure
            .as_ref()
            .map(|f| f.axes_grid())
            .unwrap_or((1, 1))
    }
    pub fn figure_axes_count(&self) -> usize {
        self.last_figure
            .as_ref()
            .map(|f| f.axes_count())
            .unwrap_or(1)
    }
    pub fn overlay_parent_for_axes(&self, axes_index: usize) -> Option<usize> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_overlay_parent(axes_index))
    }
    pub fn axes_position_explicit_for_axes(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.position_explicit)
            .unwrap_or(false)
    }
    pub(super) fn axes_grid_cell_for_axes(
        &self,
        axes_index: usize,
    ) -> Option<crate::plots::figure::AxesGridCell> {
        self.last_figure
            .as_ref()
            .and_then(|figure| figure.axes_grid_cell(axes_index))
    }
    pub fn overlay_position_for_axes(&self, axes_index: usize) -> Option<([f64; 4], String)> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .filter(|m| m.position_explicit)
            .map(|m| (m.position, m.units.clone()))
    }
    pub fn overlay_y_axis_location_for_axes(&self, axes_index: usize) -> &str {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.y_axis_location.as_str())
            .unwrap_or("left")
    }
    /// Return categorical labels if any (is_x_axis, &labels)
    pub fn overlay_categorical_labels(&self) -> Option<(bool, &Vec<String>)> {
        if let (Some(is_x), Some(labels)) = (
            &self.figure_categorical_is_x,
            &self.figure_categorical_labels,
        ) {
            Some((*is_x, labels))
        } else {
            None
        }
    }

    pub fn overlay_categorical_labels_for_axes(
        &self,
        axes_index: usize,
    ) -> Option<(bool, Vec<String>)> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.categorical_axis_labels_for_axes(axes_index))
    }

    pub fn overlay_x_tick_labels_for_axes(&self, axes_index: usize) -> Option<Vec<String>> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.x_axis_tick_labels_for_axes(axes_index))
    }

    pub fn overlay_y_tick_labels_for_axes(&self, axes_index: usize) -> Option<Vec<String>> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.y_axis_tick_labels_for_axes(axes_index))
    }

    pub fn overlay_x_tick_format_for_axes(&self, axes_index: usize) -> Option<String> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.x_axis_tick_format_for_axes(axes_index))
    }

    pub fn overlay_y_tick_format_for_axes(&self, axes_index: usize) -> Option<String> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.y_axis_tick_format_for_axes(axes_index))
    }

    pub fn overlay_x_tick_label_rotation_for_axes(&self, axes_index: usize) -> f64 {
        self.last_figure
            .as_ref()
            .and_then(|f| f.x_axis_tick_label_rotation_for_axes(axes_index))
            .unwrap_or(0.0)
    }

    pub fn overlay_y_tick_label_rotation_for_axes(&self, axes_index: usize) -> f64 {
        self.last_figure
            .as_ref()
            .and_then(|f| f.y_axis_tick_label_rotation_for_axes(axes_index))
            .unwrap_or(0.0)
    }

    pub fn overlay_x_ticks_for_axes(&self, axes_index: usize) -> Option<Vec<f64>> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.x_axis_ticks_for_axes(axes_index))
    }

    pub fn overlay_y_ticks_for_axes(&self, axes_index: usize) -> Option<Vec<f64>> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.y_axis_ticks_for_axes(axes_index))
    }

    pub fn overlay_histogram_edges_for_axes(&self, axes_index: usize) -> Option<(bool, Vec<f64>)> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.histogram_axis_edges_for_axes(axes_index))
    }

    /// Get stable display bounds for an axes (data/explicit limits), excluding transient pan/zoom.
    pub fn overlay_display_bounds_for_axes(
        &self,
        axes_index: usize,
    ) -> Option<(f64, f64, f64, f64)> {
        self.display_bounds_for_axes(axes_index)
    }

    /// Get bounds used for display (manual axis limits override data bounds when provided)
    pub fn data_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        let base = self.data_bounds;
        base.map(|(bx_min, bx_max, by_min, by_max)| {
            let (mut x_min, mut x_max) = (bx_min, bx_max);
            let (mut y_min, mut y_max) = (by_min, by_max);
            if let Some((xl, xr)) = self.figure_x_limits {
                x_min = xl;
                x_max = xr;
            }
            if let Some((yl, yr)) = self.figure_y_limits {
                y_min = yl;
                y_max = yr;
            }
            (x_min, x_max, y_min, y_max)
        })
    }

    /// Get mutable reference to a specific axes camera when using subplots
    pub fn axes_camera_mut(&mut self, idx: usize) -> Option<&mut Camera> {
        self.axes_cameras.get_mut(idx)
    }

    /// Get view bounds for a specific axes camera (l, r, b, t)
    pub fn view_bounds_for_axes(&self, idx: usize) -> Option<(f64, f64, f64, f64)> {
        if let Some(cam) = self.axes_cameras.get(idx) {
            if let crate::core::camera::ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                ..
            } = cam.projection
            {
                return Some((left as f64, right as f64, bottom as f64, top as f64));
            }
        }
        None
    }

    pub fn axes_bounds(&self, axes_index: usize) -> Option<crate::core::BoundingBox> {
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut saw_any = false;

        for node in self.scene.get_visible_nodes() {
            if node.axes_index != axes_index {
                continue;
            }
            let Some(render_data) = &node.render_data else {
                continue;
            };
            if let Some(bounds) = render_data.bounds {
                min = min.min(bounds.min);
                max = max.max(bounds.max);
                saw_any = true;
                continue;
            }
            for v in &render_data.vertices {
                let p = Vec3::new(v.position[0], v.position[1], v.position[2]);
                min = min.min(p);
                max = max.max(p);
                saw_any = true;
            }
        }

        if !saw_any {
            return None;
        }
        Some(crate::core::BoundingBox { min, max })
    }

    /// Every visible reference line on an axes (orientation and value).
    pub fn overlay_reference_lines_for_axes(&self, axes_index: usize) -> Vec<(crate::plots::ReferenceLineOrientation, f64)> {
        self.last_figure
            .as_ref()
            .map(|f| f.reference_lines_for_axes(axes_index))
            .unwrap_or_default()
    }

    /// The legend style of an axes (its `Location` and box), when the figure set one.
    pub fn overlay_legend_style_for_axes(&self, axes_index: usize) -> Option<&crate::plots::figure::LegendStyle> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| &m.legend_style)
    }

    /// Prefer exporting the original figure if available
    pub fn export_figure_clone(&self) -> crate::plots::Figure {
        if let Some(f) = &self.last_figure {
            return f.clone();
        }
        // As a strict fallback, produce an empty figure with current metadata only
        let mut fig = crate::plots::Figure::new();
        fig.title = self.figure_title.clone();
        fig.sg_title = self.figure_sg_title.clone();
        fig.sg_title_style = self.figure_sg_title_style.clone();
        fig.x_label = self.figure_x_label.clone();
        fig.y_label = self.figure_y_label.clone();
        fig.legend_enabled = self.figure_show_legend;
        fig.grid_enabled = self.figure_show_grid;
        fig.minor_grid_enabled = self.figure_show_minor_grid;
        fig.box_enabled = self.figure_show_box;
        fig.x_limits = self.figure_x_limits;
        fig.y_limits = self.figure_y_limits;
        fig.x_log = self.figure_x_log;
        fig.y_log = self.figure_y_log;
        fig.axis_equal = self.figure_axis_equal;
        fig.colormap = self.figure_colormap.clone();
        fig.colorbar_enabled = self.figure_colorbar_enabled;
        let (rows, cols) = self.figure_axes_grid();
        fig.set_subplot_grid(rows, cols);
        fig
    }
}
