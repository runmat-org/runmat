//! Loading a figure or geometry scene into the renderer, and the axes bounds and views it keeps.

use super::*;

impl PlotRenderer {
    /// Set the figure to render
    pub fn set_figure(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        figure: Figure,
    ) {
        self.last_geometry_scene_key = None;
        self.last_geometry_scene = None;
        self.geometry_overlay = None;
        self.geometry_xray_enabled = false;
        self.geometry_node_owner_ids.clear();
        self.geometry_hidden_owner_node_ids.clear();
        // Clear existing scene
        self.scene.clear();
        self.scene.set_translucent_last(false);
        self.scene_buffer_cache.borrow_mut().clear();

        // Convert figure to scene nodes
        self.cache_figure_meta(&figure);
        self.last_figure = Some(figure.clone());
        self.last_axes_plot_sizes_px = None;
        self.last_axes_view_bounds = None;
        // Initialize axes cameras for subplot grid
        let (rows, cols) = figure.axes_grid();
        let num_axes = figure.axes_count();
        let axes_view_contract = Self::axes_view_contract_for_figure(&figure);
        let axes_view_contract_changed =
            self.last_axes_view_contract.as_ref() != Some(&axes_view_contract);
        if axes_view_contract_changed {
            log::debug!(
                target: "runmat_plot.camera_refit",
                "figure axes view contract changed; resetting script-owned camera fit rows={} cols={} axes_count={}",
                rows,
                cols,
                num_axes
            );
            self.clear_all_axes_camera_interaction();
            self.camera_auto_fit = true;
        }
        self.last_axes_view_contract = Some(axes_view_contract);

        if self.axes_cameras.len() != num_axes {
            self.axes_cameras
                .resize_with(num_axes, Self::create_default_camera);
            self.axes_2d_camera_user_controlled.resize(num_axes, false);
            self.axes_applied_view_revisions.resize(num_axes, None);
            self.camera_auto_fit = true;
        }

        for axes_index in 0..num_axes {
            let wants_3d = self.axes_has_3d_content(axes_index);
            let has_3d_camera = self
                .axes_cameras
                .get(axes_index)
                .map(|cam| {
                    matches!(
                        cam.projection,
                        crate::core::camera::ProjectionType::Perspective { .. }
                    )
                })
                .unwrap_or(false);
            if wants_3d != has_3d_camera {
                self.axes_cameras[axes_index] = if wants_3d {
                    Camera::new()
                } else {
                    Self::create_default_camera()
                };
                self.clear_axes_camera_interaction(axes_index);
                if let Some(revision) = self.axes_applied_view_revisions.get_mut(axes_index) {
                    *revision = None;
                }
                self.camera_auto_fit = true;
            }
        }

        self.add_figure_to_scene(emission, figure);

        // Mark for update
        self.needs_update = true;

        // Recompute bounds and fit camera immediately (only once per initial dataset).
        let fit_applied = if self.camera_auto_fit {
            if num_axes > 1 {
                self.fit_cameras_to_axes_data()
            } else {
                self.fit_camera_to_data()
            }
        } else {
            false
        };
        if self.camera_auto_fit && fit_applied {
            // Freeze the initial fit (CAD-like): don't re-fit as data updates (e.g. animations)
            // unless the user explicitly asks (Fit Extents / Reset View) or we change plot mode.
            self.camera_auto_fit = false;
        }
        self.apply_stored_axes_views();
        self.figure_camera_fit
            .resize(num_axes, FigureCameraFitState::Inactive);
        if fit_applied {
            self.arm_figure_camera_fits();
        }
    }

    /// Set a chunked geometry scene to render.
    ///
    /// Unlike figures, geometry scenes are expected to represent large, stable CAD/FEA
    /// datasets. Re-applying the same cache key preserves scene nodes and GPU buffers so
    /// camera motion only updates uniforms and redraws the existing buffers.
    pub fn set_geometry_scene(&mut self, geometry_scene: GeometryScene) {
        self.set_geometry_scene_with_presentation(
            geometry_scene,
            self.geometry_presentation.clone(),
        );
    }

    pub fn set_geometry_scene_with_presentation(
        &mut self,
        geometry_scene: GeometryScene,
        mut presentation: GeometryScenePresentation,
    ) {
        self.figure_camera_fit.fill(FigureCameraFitState::Inactive);
        let cache_key = geometry_scene.cache_key();
        let same_scene_identity = self
            .last_geometry_scene_key
            .as_ref()
            .map(|key| key.scene_id.as_str())
            == Some(cache_key.scene_id.as_str());
        if same_scene_identity {
            self.preserve_geometry_scene_state_when_unspecified(&mut presentation);
        }
        let preserved_camera = same_scene_identity
            .then(|| self.axes_cameras.first().cloned())
            .flatten();
        let preserved_xray_enabled = same_scene_identity.then_some(self.geometry_xray_enabled);
        self.geometry_overlay = geometry_scene.overlay.clone();
        let geometry_xray_enabled = scene_authored_as_xray(&geometry_scene);
        if self.last_geometry_scene_key.as_ref() == Some(&cache_key) {
            self.last_geometry_scene = Some(geometry_scene.clone());
            let show_grid = presentation.show_grid.unwrap_or(geometry_scene.show_grid);
            if self.figure_show_grid != show_grid {
                self.figure_show_grid = show_grid;
                self.needs_update = true;
            }
            if self.geometry_presentation != presentation {
                let mut previous = self.geometry_presentation.clone();
                previous.show_grid = presentation.show_grid;
                self.geometry_presentation = presentation;
                if previous != self.geometry_presentation {
                    self.refresh_geometry_scene_render_data(&geometry_scene);
                }
            }
            return;
        }

        self.scene.clear();
        self.scene.set_translucent_last(true);
        self.scene_buffer_cache.borrow_mut().clear();
        self.geometry_node_owner_ids.clear();
        if !same_scene_identity {
            self.geometry_hidden_owner_node_ids.clear();
        }
        self.last_geometry_scene_key = Some(cache_key);
        self.last_geometry_scene = Some(geometry_scene.clone());
        self.geometry_presentation = presentation;
        self.geometry_xray_enabled = preserved_xray_enabled.unwrap_or(geometry_xray_enabled);
        self.last_figure = None;
        self.last_scene_viewport_px = Some((
            self.wgpu_renderer.surface_config.width.max(1),
            self.wgpu_renderer.surface_config.height.max(1),
        ));
        self.last_axes_plot_sizes_px = None;
        self.last_axes_view_bounds = None;
        self.last_axes_view_contract = None;

        self.figure_title = geometry_scene.title.clone();
        self.figure_sg_title = None;
        self.figure_sg_title_style = TextStyle::default();
        self.figure_x_label = Some("X".to_string());
        self.figure_y_label = Some("Y".to_string());
        self.figure_z_label = Some("Z".to_string());
        self.figure_show_grid = self
            .geometry_presentation
            .show_grid
            .unwrap_or(geometry_scene.show_grid);
        self.figure_show_legend = false;
        self.figure_show_box = true;
        self.figure_x_limits = None;
        self.figure_y_limits = None;
        self.legend_entries.clear();
        self.figure_x_log = false;
        self.figure_y_log = false;
        self.figure_axis_equal = geometry_scene.axis_equal;
        self.figure_colorbar_enabled = false;
        self.figure_categorical_is_x = None;
        self.figure_categorical_labels = None;

        self.axes_cameras.clear();
        self.axes_cameras
            .push(preserved_camera.clone().unwrap_or_else(Camera::new));
        self.axes_2d_camera_user_controlled.clear();
        self.axes_2d_camera_user_controlled.push(false);
        self.axes_applied_view_revisions.clear();
        self.axes_applied_view_revisions.push(None);

        self.rebuild_geometry_scene_nodes(&geometry_scene);

        if preserved_camera.is_none() {
            let mut camera = Camera::new();
            if Self::bounds_are_finite(geometry_scene.bounds) {
                camera.target = geometry_scene.bounds.center();
                camera.up = Vec3::Z;
                camera.position = camera.target + Vec3::new(1.0, -1.0, 1.0);
                camera.fit_bounds(geometry_scene.bounds.min, geometry_scene.bounds.max);
            }
            self.axes_cameras[0] = camera;
        }
        self.camera_auto_fit = false;
        self.needs_update = true;
    }

    /// Add a figure to the current scene
    pub(super) fn add_figure_to_scene(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        figure: Figure,
    ) {
        self.add_figure_to_scene_with_axes_plot_sizes(emission, figure, None);
    }

    pub(super) fn add_figure_to_scene_with_axes_plot_sizes(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        mut figure: Figure,
        axes_plot_sizes_px: Option<&[(u32, u32)]>,
    ) {
        use crate::core::SceneNode;

        // Lines with an automatic color take the theme's axes color.
        figure.resolve_auto_reference_line_colors(self.theme.build_theme().get_axis_color());
        let (rows, cols) = figure.axes_grid();

        // Convert figure to render data first, then create scene nodes.
        // For subplot figures, avoid baking full-surface line stroke geometry before overlay
        // layout has provided per-axes plot viewports.
        let viewport_px = (
            self.wgpu_renderer.surface_config.width.max(1),
            self.wgpu_renderer.surface_config.height.max(1),
        );
        self.last_scene_viewport_px = Some(viewport_px);
        let gpu = crate::core::GpuPackContext { emission };
        let axes_count = figure.axes_count();
        let view_bounds = self.axes_view_bounds_for_count(axes_count);
        let viewport_hint = if axes_plot_sizes_px.is_some() || rows.max(1) * cols.max(1) <= 1 {
            Some(viewport_px)
        } else {
            None
        };
        let render_data_list = figure.render_data_with_axes_with_viewport_and_gpu(
            viewport_hint,
            axes_plot_sizes_px,
            Some(&view_bounds),
            Some(&gpu),
        );

        for (node_id_counter, (axes_index, render_data)) in render_data_list.into_iter().enumerate()
        {
            let axes_index = axes_index.min(axes_count.saturating_sub(1));
            // Create scene node for this plot element
            let node = SceneNode {
                id: node_id_counter as u64,
                name: format!("Plot {node_id_counter} @axes {axes_index}"),
                transform: Mat4::IDENTITY,
                visible: true,
                cast_shadows: false,
                receive_shadows: false,
                axes_index,
                parent: None,
                children: Vec::new(),
                render_data: Some(render_data),
                bounds: crate::core::BoundingBox::default(),
                lod_levels: Vec::new(),
                current_lod: 0,
                instances: None,
            };

            let nid = self.scene.add_node(node);
            // Tag node with axes index via a no-op mechanism for now (could extend SceneNode in future)
            let _ = nid;
            let _ = axes_index;
            let _ = rows;
            let _ = cols;
        }
    }

    pub fn ensure_scene_viewport_dependent_geometry_for_axes(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        axes_plot_sizes_px: &[(u32, u32)],
    ) {
        let mut normalized: Vec<(u32, u32)> = axes_plot_sizes_px
            .iter()
            .map(|&(w, h)| (w.max(1), h.max(1)))
            .collect();
        let Some(figure) = self.last_figure.clone() else {
            let view_bounds = self.axes_view_bounds_for_count(normalized.len().max(1));
            self.last_axes_plot_sizes_px = Some(normalized);
            self.last_axes_view_bounds = Some(view_bounds);
            return;
        };
        let axes_count = figure.axes_count();
        if normalized.len() < axes_count {
            let grid_len = normalized.len().max(1);
            for axes_index in normalized.len()..axes_count {
                let parent = figure
                    .axes_overlay_parent(axes_index)
                    .unwrap_or(axes_index)
                    .min(grid_len - 1);
                let size = normalized.get(parent).copied().unwrap_or((
                    self.wgpu_renderer.surface_config.width.max(1),
                    self.wgpu_renderer.surface_config.height.max(1),
                ));
                normalized.push(size);
            }
        }
        if normalized.iter().any(|&(w, h)| w < 2 || h < 2) {
            log::debug!(
                target: "runmat_plot.viewport_rebuild",
                "skipped viewport-dependent scene geometry rebuild for unstable viewport_sizes={:?}",
                normalized
            );
            return;
        }
        let view_bounds = self.axes_view_bounds_for_count(axes_count);
        if self.last_axes_plot_sizes_px.as_ref() == Some(&normalized)
            && self.last_axes_view_bounds.as_ref() == Some(&view_bounds)
        {
            return;
        }
        self.scene.clear();
        self.scene_buffer_cache.borrow_mut().clear();
        self.add_figure_to_scene_with_axes_plot_sizes(emission, figure, Some(&normalized));
        log::debug!(
            target: "runmat_plot.viewport_rebuild",
            "rebuilt viewport-dependent scene geometry axes_count={} viewport_sizes={:?}",
            normalized.len(),
            normalized
        );
        self.refit_2d_cameras_to_scene_bounds();
        self.last_axes_plot_sizes_px = Some(normalized);
        self.last_axes_view_bounds = Some(view_bounds);
        self.needs_update = true;
    }

    pub(super) fn axes_view_bounds_for_count(&self, axes_count: usize) -> PerAxesViewBounds {
        (0..axes_count)
            .map(|idx| self.view_bounds_for_axes(idx))
            .collect()
    }

    pub(super) fn direct_log_flags_for_axes(&self, axes_index: usize) -> [u32; 2] {
        [
            u32::from(self.overlay_x_log_for_axes(axes_index)),
            u32::from(self.overlay_y_log_for_axes(axes_index)),
        ]
    }

    pub(super) fn refit_2d_cameras_to_scene_bounds(&mut self) {
        for idx in 0..self.axes_cameras.len() {
            if self.axes_has_3d_content(idx) {
                continue;
            }
            if self
                .axes_2d_camera_user_controlled
                .get(idx)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let Some((x_min, x_max, y_min, y_max)) = self.display_bounds_for_axes(idx) else {
                continue;
            };
            let geometry_bounds = self.axes_bounds(idx);
            let Some(cam) = self.axes_cameras.get_mut(idx) else {
                continue;
            };
            if let crate::core::camera::ProjectionType::Orthographic {
                ref mut left,
                ref mut right,
                ref mut bottom,
                ref mut top,
                ..
            } = cam.projection
            {
                *left = x_min as f32;
                *right = x_max as f32;
                *bottom = y_min as f32;
                *top = y_max as f32;
                let camera_left = *left;
                let camera_right = *right;
                let camera_bottom = *bottom;
                let camera_top = *top;
                cam.position.z = 1.0;
                cam.target.z = 0.0;
                cam.mark_dirty();
                if let Some(bounds) = geometry_bounds {
                    log::debug!(
                        target: "runmat_plot.camera_refit",
                        "refit 2d camera to rebuilt scene bounds axes_index={} geometry=({}, {})..({}, {}) camera=({}, {})..({}, {}) margins=top:{} bottom:{} left:{} right:{}",
                        idx,
                        bounds.min.x,
                        bounds.min.y,
                        bounds.max.x,
                        bounds.max.y,
                        camera_left,
                        camera_bottom,
                        camera_right,
                        camera_top,
                        camera_top - bounds.max.y,
                        bounds.min.y - camera_bottom,
                        bounds.min.x - camera_left,
                        camera_right - bounds.max.x
                    );
                } else {
                    log::debug!(
                        target: "runmat_plot.camera_refit",
                        "refit 2d camera without geometry bounds axes_index={} camera=({}, {})..({}, {})",
                        idx,
                        camera_left,
                        camera_bottom,
                        camera_right,
                        camera_top
                    );
                }
                if let Some(display_bounds) = self.display_bounds_for_axes(idx) {
                    log::debug!(
                        target: "runmat_plot.bounds_chain",
                        "bounds chain axes_index={} axes_bounds=({}, {})..({}, {}) display_bounds=({}, {})..({}, {}) camera_bounds=({}, {})..({}, {})",
                        idx,
                        geometry_bounds.map(|b| b.min.x as f64).unwrap_or(f64::NAN),
                        geometry_bounds.map(|b| b.min.y as f64).unwrap_or(f64::NAN),
                        geometry_bounds.map(|b| b.max.x as f64).unwrap_or(f64::NAN),
                        geometry_bounds.map(|b| b.max.y as f64).unwrap_or(f64::NAN),
                        display_bounds.0,
                        display_bounds.2,
                        display_bounds.1,
                        display_bounds.3,
                        camera_left,
                        camera_bottom,
                        camera_right,
                        camera_top
                    );
                }
            }
        }
    }

    /// Cache figure metadata for overlay consumption
    pub(super) fn cache_figure_meta(&mut self, figure: &Figure) {
        self.figure_title = figure.title.clone();
        self.figure_sg_title = figure.sg_title.clone();
        self.figure_sg_title_style = figure.sg_title_style.clone();
        self.figure_x_label = figure.x_label.clone();
        self.figure_y_label = figure.y_label.clone();
        self.figure_z_label = figure.z_label.clone();
        self.figure_show_grid = figure.grid_enabled;
        self.figure_show_minor_grid = figure.minor_grid_enabled;
        self.figure_show_legend = figure.legend_enabled;
        self.figure_show_box = figure.box_enabled;
        self.figure_x_limits = figure.x_limits;
        self.figure_y_limits = figure.y_limits;
        self.legend_entries = figure.legend_entries();
        self.figure_x_log = figure.x_log;
        self.figure_y_log = figure.y_log;
        self.figure_axis_equal = figure.axis_equal;
        self.figure_colormap = figure.colormap.clone();
        self.figure_colorbar_enabled = figure.colorbar_enabled;
        // Cache categorical labels for overlay
        if let Some((is_x, labels)) = figure.categorical_axis_labels() {
            self.figure_categorical_is_x = Some(is_x);
            self.figure_categorical_labels = Some(labels);
        } else {
            self.figure_categorical_is_x = None;
            self.figure_categorical_labels = None;
        }
    }

    pub(super) fn apply_stored_axes_views(&mut self) {
        let Some(fig) = self.last_figure.as_ref() else {
            return;
        };
        for (idx, cam) in self.axes_cameras.iter_mut().enumerate() {
            if !matches!(
                cam.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            ) {
                continue;
            }
            if let Some(meta) = fig.axes_metadata(idx) {
                if let (Some(az), Some(el)) = (meta.view_azimuth_deg, meta.view_elevation_deg) {
                    if self.axes_applied_view_revisions.get(idx).copied().flatten()
                        == Some(meta.view_revision)
                    {
                        continue;
                    }
                    cam.set_view_angles_deg(az, el);
                    if let Some(revision) = self.axes_applied_view_revisions.get_mut(idx) {
                        *revision = Some(meta.view_revision);
                    }
                } else if let Some(revision) = self.axes_applied_view_revisions.get_mut(idx) {
                    *revision = None;
                }
            }
        }
    }

    pub(super) fn display_bounds_for_axes(
        &self,
        axes_index: usize,
    ) -> Option<(f64, f64, f64, f64)> {
        let base = self.axes_bounds(axes_index)?;
        let mut x_min = base.min.x as f64;
        let mut x_max = base.max.x as f64;
        let mut y_min = base.min.y as f64;
        let mut y_max = base.max.y as f64;

        if let Some(fig) = self.last_figure.as_ref() {
            ((x_min, x_max), (y_min, y_max)) =
                fig.auto_axes_bounds_2d(axes_index, (x_min, x_max), (y_min, y_max));
            if let Some(meta) = fig.axes_metadata(axes_index) {
                if let Some((xl, xr)) = meta.x_limits {
                    x_min = xl;
                    x_max = xr;
                }
                if let Some((yl, yr)) = meta.y_limits {
                    y_min = yl;
                    y_max = yr;
                }
                // A fixed data aspect (`axis equal`) keeps these limits; the overlay layout
                // shrinks the plot box to the aspect instead (see `overlay_plot_box_aspect_for_axes`).
            }
        }
        if let Some(meta) = self
            .last_figure
            .as_ref()
            .and_then(|fig| fig.axes_metadata(axes_index))
        {
            if meta.x_log {
                (x_min, x_max) = crate::plots::figure::positive_log_bounds(x_min, x_max);
            }
            if meta.y_log {
                (y_min, y_max) = crate::plots::figure::positive_log_bounds(y_min, y_max);
            }
        }

        Some((x_min, x_max, y_min, y_max))
    }

    pub(super) fn apply_3d_display_limits_to_bounds(
        bounds: BoundingBox,
        figure: Option<&Figure>,
        axes_index: usize,
    ) -> BoundingBox {
        crate::plots::figure::display_bounds_3d(
            bounds,
            figure.and_then(|fig| fig.axes_metadata(axes_index)),
        )
    }

    pub(super) fn bounds_are_finite(bounds: BoundingBox) -> bool {
        bounds.min.x.is_finite()
            && bounds.min.y.is_finite()
            && bounds.min.z.is_finite()
            && bounds.max.x.is_finite()
            && bounds.max.y.is_finite()
            && bounds.max.z.is_finite()
    }

    pub(super) fn current_3d_display_bounds_for_axes(
        &self,
        axes_index: usize,
    ) -> Option<BoundingBox> {
        let bounds = self.axes_bounds(axes_index)?;
        let bounds =
            Self::apply_3d_display_limits_to_bounds(bounds, self.last_figure.as_ref(), axes_index);
        Self::bounds_are_finite(bounds).then_some(bounds)
    }

    pub(super) fn display_bounds_3d_for_axes(&self, axes_index: usize) -> Option<BoundingBox> {
        self.current_3d_display_bounds_for_axes(axes_index)
    }

    /// The initial 3-D view before any `view` call: MATLAB's default `view(-37.5, 30)` for
    /// plots, and the CAD isometric view from (+x, -y, +z) for geometry.
    pub(super) fn default_3d_view_direction(&self, axes_index: usize) -> Vec3 {
        if self.axes_shows_geometry(axes_index) {
            Vec3::new(1.0, -1.0, 1.0)
        } else {
            Camera::view_direction_deg(-37.5, 30.0)
        }
    }

    /// Whether this axes shows CAD geometry (a geometry scene or a geometry mesh), whose axes
    /// are anchored at the world origin like a CAD view.
    pub(super) fn axes_shows_geometry(&self, axes_index: usize) -> bool {
        self.last_geometry_scene.is_some()
            || self.last_figure.as_ref().is_some_and(|f| {
                f.plots().enumerate().any(|(idx, plot)| {
                    f.plot_axes_indices().get(idx).copied().unwrap_or(0) == axes_index
                        && matches!(plot, crate::plots::PlotElement::Mesh(_))
                })
            })
    }

    /// The 3-D axis frame of one axes: where its three axis lines start, how long they are, and
    /// the tick values along them. The renderer draws the lines and tick marks from it and the
    /// overlay labels the same ticks, so both read this one function.
    ///
    /// Plots (`surf`, `mesh`, `plot3`, ...) anchor the axes at the lower corner of the axis
    /// limits and tick the limits on each axis, as MATLAB's plot box does: a surface over spot
    /// 60..140 shows ticks from 60 to 140 along its own edge. CAD geometry keeps a world-origin
    /// triad that reaches the far edge of the model.
    pub fn axis_frame_3d(&self, axes_index: usize) -> Option<AxisFrame3d> {
        let bounds = if self.last_geometry_scene.is_some() {
            self.last_geometry_scene.as_ref().map(|scene| scene.bounds)
        } else {
            self.display_bounds_3d_for_axes(axes_index)
        }?;
        let mut frame = AxisFrame3d::from_bounds(
            bounds.min,
            bounds.max,
            self.axes_shows_geometry(axes_index),
        )?;
        if !frame.world_origin {
            // Stand the z-axis on the bottom corner that sits leftmost on screen, as MATLAB's
            // plot box does, so it frames the data instead of crossing in front of it.
            if let Some(camera) = self.axes_cameras.get(axes_index) {
                let forward = camera.target - camera.position;
                let right = forward.cross(camera.up);
                let scale = camera.axis_scale;
                let leftmost = [
                    Vec3::new(bounds.min.x, bounds.min.y, bounds.min.z),
                    Vec3::new(bounds.max.x, bounds.min.y, bounds.min.z),
                    Vec3::new(bounds.min.x, bounds.max.y, bounds.min.z),
                    Vec3::new(bounds.max.x, bounds.max.y, bounds.min.z),
                ]
                .into_iter()
                .filter(|corner| right.is_finite() && right.length_squared() > 0.0 && corner.is_finite())
                .min_by(|a, b| (*a * scale).dot(right).total_cmp(&(*b * scale).dot(right)));
                if let Some(corner) = leftmost {
                    frame.z_base = corner;
                }
            }
        }
        Some(frame)
    }

    /// A tick step per data axis for the 3-D origin triad: a nice value near `target_px` on
    /// screen along that axis.
    pub(crate) fn world_axis_steps(
        project_to_px: &dyn Fn(Vec3) -> Option<(f32, f32)>,
        center: Vec3,
        nice_step: &dyn Fn(f64) -> f64,
    ) -> [f64; 3] {
        let target_px = 96.0_f64;
        [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| {
            let px = match (project_to_px(center), project_to_px(center + axis)) {
                (Some((ax, ay)), Some((bx, by))) => ((bx - ax).hypot(by - ay) as f64).max(1e-6),
                _ => 1.0,
            };
            nice_step((target_px / px).max(1e-9))
        })
    }

    /// The 3-D axis scale: each axis stretched to fill a cube (MATLAB's default plot box),
    /// or one scale for all axes under `axis equal`, a set data aspect, or a geometry mesh.
    pub(super) fn axes_3d_axis_scale(&self, axes_index: usize, bounds: BoundingBox) -> Vec3 {
        let size = (bounds.max - bounds.min).abs();
        let largest = size.x.max(size.y).max(size.z);
        if !(largest.is_finite() && largest > 0.0) {
            return Vec3::ONE;
        }
        // Geometry scenes are always true-scale; importing one clears `last_figure`, so its
        // axis-equal flag lives in `figure_axis_equal`.
        if self.last_geometry_scene.is_some() || self.figure_axis_equal {
            return Vec3::splat(1.0 / largest);
        }
        let figure = self.last_figure.as_ref();
        let meta = figure.and_then(|f| f.axes_metadata(axes_index));
        let has_geometry_mesh = figure.is_some_and(|f| {
            f.plots().enumerate().any(|(idx, plot)| {
                f.plot_axes_indices().get(idx).copied().unwrap_or(0) == axes_index
                    && matches!(plot, crate::plots::PlotElement::Mesh(_))
            })
        });
        if has_geometry_mesh {
            return Vec3::splat(1.0 / largest);
        }
        if let Some(meta) =
            meta.filter(|m| m.axis_equal || m.data_aspect_ratio_mode == "manual")
        {
            // One data unit along axis k is `ratio[k]` long relative to the others; the largest
            // relative extent fills the unit box and the limits stay as they are.
            let ratio = if meta.data_aspect_ratio_mode == "manual" {
                meta.data_aspect_ratio.map(|r| (r.abs() as f32).max(1.0e-12))
            } else {
                [1.0; 3]
            };
            return data_aspect_axis_scale(size, ratio);
        }
        let axis = |extent: f32| {
            if extent > largest * 1.0e-6 {
                1.0 / extent
            } else {
                1.0 / largest
            }
        };
        Vec3::new(axis(size.x), axis(size.y), axis(size.z))
    }

    pub(super) fn fit_cameras_to_axes_data(&mut self) -> bool {
        let mut applied = false;
        for idx in 0..self.axes_cameras.len() {
            if self.axes_has_3d_content(idx) {
                let Some(bounds) = self.display_bounds_3d_for_axes(idx) else {
                    continue;
                };
                let center = (bounds.min + bounds.max) * 0.5;
                let mut cam = Camera::new();
                cam.axis_scale = self.axes_3d_axis_scale(idx, bounds);
                cam.target = center * cam.axis_scale;
                cam.up = Vec3::Z;
                cam.position = cam.target + self.default_3d_view_direction(idx);
                cam.fit_bounds(bounds.min, bounds.max);
                self.axes_cameras[idx] = cam;
                applied = true;
                continue;
            }

            let Some((x_min, x_max, y_min, y_max)) = self.display_bounds_for_axes(idx) else {
                continue;
            };
            let mut cam = Self::create_default_camera();
            if let crate::core::camera::ProjectionType::Orthographic {
                ref mut left,
                ref mut right,
                ref mut bottom,
                ref mut top,
                ..
            } = cam.projection
            {
                *left = x_min as f32;
                *right = x_max as f32;
                *bottom = y_min as f32;
                *top = y_max as f32;
            }
            cam.position.z = 1.0;
            cam.target.z = 0.0;
            cam.mark_dirty();
            self.axes_cameras[idx] = cam;
            applied = true;
        }
        applied
    }
}
