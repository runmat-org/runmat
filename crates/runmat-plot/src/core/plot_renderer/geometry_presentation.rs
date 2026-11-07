//! The geometry scene's overlay, presentation, visibility, x-ray and theme.

use super::*;

impl PlotRenderer {
    pub fn overlay_show_box(&self) -> bool {
        self.figure_show_box
    }
    pub fn overlay_show_box_for_axes(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.box_enabled)
            .unwrap_or(self.figure_show_box)
    }
    pub fn overlay_title(&self) -> Option<&String> {
        self.figure_title.as_ref()
    }
    pub fn geometry_overlay(&self) -> Option<&GeometrySceneOverlay> {
        self.geometry_overlay.as_ref()
    }
    pub fn has_geometry_scene(&self) -> bool {
        self.last_geometry_scene.is_some()
    }
    /// Whether the overlay draws the figure title. A CAD preview's title is its source name,
    /// which the host shows elsewhere; a result scene (one with a scalar legend) draws its
    /// title, such as a caption stating what was solved.
    pub fn shows_geometry_title(&self) -> bool {
        match &self.geometry_overlay {
            None => true,
            Some(overlay) => overlay.scalar_legend.is_some() && self.figure_title.is_some(),
        }
    }

    /// Whether a geometry scene is a result: one colored by a field, with a scalar legend.
    pub fn geometry_scene_is_result(&self) -> bool {
        self.geometry_overlay
            .as_ref()
            .is_some_and(|overlay| overlay.scalar_legend.is_some())
    }

    /// Whether one 3-D axes draws its world-axis lines, tick marks and tick labels. A result
    /// scene never does: a CAD world-origin triad drawn through a colored field reads as data.
    pub fn shows_world_triad(&self, axes_index: usize) -> bool {
        !(self.axes_shows_geometry(axes_index) && self.geometry_scene_is_result())
    }

    /// Whether the corner orientation triad is drawn: a geometry scene's own setting; figures
    /// follow `axis on`/`axis off` alone.
    pub fn shows_orientation_triad(&self) -> bool {
        self.last_geometry_scene
            .as_ref()
            .is_none_or(|scene| scene.show_orientation_triad)
    }

    pub fn geometry_xray_enabled(&self) -> bool {
        self.geometry_xray_enabled
    }
    pub fn geometry_scene_presentation(&self) -> &GeometryScenePresentation {
        &self.geometry_presentation
    }
    #[cfg(feature = "egui-overlay")]
    pub(crate) fn geometry_screen_annotations(
        &self,
        viewport_size: [f32; 2],
    ) -> Vec<GeometryScreenAnnotation> {
        let Some(scene) = self.last_geometry_scene.as_ref() else {
            return Vec::new();
        };
        if viewport_size[0] <= 0.0 || viewport_size[1] <= 0.0 {
            return Vec::new();
        }
        let mut camera = self.camera().clone();
        let view_projection = camera.view_proj_matrix();
        self.geometry_presentation
            .region_annotations
            .iter()
            .filter_map(|annotation| {
                let label = annotation.label.as_deref().unwrap_or("").trim();
                if label.is_empty() && annotation.direction.is_none() {
                    return None;
                }
                let anchor = scene.region_anchor(&annotation.region_id)?;
                let clip = view_projection * anchor.extend(1.0);
                if !clip.w.is_finite() || clip.w <= 0.0 {
                    return None;
                }
                let ndc = clip.truncate() / clip.w;
                if !ndc.is_finite()
                    || !(-1.05..=1.05).contains(&ndc.x)
                    || !(-1.05..=1.05).contains(&ndc.y)
                    || !(-0.05..=1.05).contains(&ndc.z)
                {
                    return None;
                }
                Some(GeometryScreenAnnotation {
                    position: [
                        (ndc.x + 1.0) * 0.5 * viewport_size[0],
                        (1.0 - ndc.y) * 0.5 * viewport_size[1],
                    ],
                    color: annotation.color.unwrap_or([0.9, 0.9, 0.9, 1.0]),
                    kind: annotation.kind,
                    label: label.to_string(),
                    vector_extent: annotation.direction.and_then(|direction| {
                        let points =
                            scene.annotation_vector_points(&annotation.region_id, direction)?;
                        let mut minimum = [f32::INFINITY; 2];
                        let mut maximum = [f32::NEG_INFINITY; 2];
                        for point in points {
                            let clip = view_projection * point.extend(1.0);
                            if !clip.is_finite() || clip.w <= 0.0 {
                                return None;
                            }
                            let ndc = clip.truncate() / clip.w;
                            let screen = [
                                (ndc.x + 1.0) * 0.5 * viewport_size[0],
                                (1.0 - ndc.y) * 0.5 * viewport_size[1],
                            ];
                            for axis in 0..2 {
                                minimum[axis] = minimum[axis].min(screen[axis]);
                                maximum[axis] = maximum[axis].max(screen[axis]);
                            }
                        }
                        Some([minimum, maximum])
                    }),
                })
            })
            .collect()
    }
    /// Draws the 3-D views with perspective or as a parallel projection. Framing is kept: the
    /// parallel view's height is the perspective view's height at the target.
    pub fn set_geometry_projection(
        &mut self,
        projection: crate::geometry_scene::GeometrySceneProjection,
    ) {
        let parallel = projection == crate::geometry_scene::GeometrySceneProjection::Orthographic;
        for camera in &mut self.axes_cameras {
            if matches!(
                camera.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            ) && camera.parallel != parallel
            {
                camera.parallel = parallel;
                camera.mark_dirty();
                self.needs_update = true;
            }
        }
    }

    pub fn set_geometry_scene_presentation(&mut self, mut presentation: GeometryScenePresentation) {
        let requested_view_preset = presentation.view_preset.take();
        if let Some(projection) = presentation.projection.take() {
            self.set_geometry_projection(projection);
        }
        self.preserve_geometry_scene_state_when_unspecified(&mut presentation);
        if let Some(view_preset) = requested_view_preset {
            self.set_camera_view_preset(geometry_scene_view_preset_to_camera(view_preset));
        }
        log::info!(
            target: "runmat_plot",
            "geometry_scene.presentation_renderer selected_region_id={} selected_region_count={} has_scene={} unchanged={}",
            presentation.selected_region_id.as_deref().unwrap_or("none"),
            presentation.selected_region_ids.len(),
            self.last_geometry_scene.is_some(),
            self.geometry_presentation == presentation,
        );
        if self.geometry_presentation == presentation {
            return;
        }
        let mut previous = self.geometry_presentation.clone();
        previous.show_grid = presentation.show_grid;
        self.figure_show_grid = presentation
            .show_grid
            .or_else(|| {
                self.last_geometry_scene
                    .as_ref()
                    .map(|scene| scene.show_grid)
            })
            .unwrap_or(false);
        self.geometry_presentation = presentation;
        if previous == self.geometry_presentation {
            self.needs_update = true;
            return;
        }
        self.apply_geometry_presentation_visibility();
        if let Some(scene) = self.last_geometry_scene.clone() {
            self.refresh_geometry_scene_render_data(&scene);
        }
    }
    pub fn geometry_owner_visible(&self, owner_id: &str) -> bool {
        !self.geometry_hidden_owner_node_ids.contains(owner_id)
    }
    pub fn set_geometry_owner_visible(&mut self, owner_id: impl Into<String>, visible: bool) {
        let owner_id = owner_id.into();
        if owner_id.trim().is_empty() {
            return;
        }
        if visible {
            if !self.geometry_hidden_owner_node_ids.remove(&owner_id) {
                return;
            }
        } else if !self.geometry_hidden_owner_node_ids.insert(owner_id) {
            return;
        }
        self.geometry_presentation.hidden_owner_node_ids = Some(
            self.geometry_hidden_owner_node_ids
                .iter()
                .cloned()
                .collect(),
        );
        self.geometry_presentation.isolated_owner_node_ids = None;
        self.apply_geometry_visibility_filter();
        self.needs_update = true;
    }
    pub(super) fn refresh_geometry_scene_render_data(&mut self, geometry_scene: &GeometryScene) {
        log::debug!(
            target: "runmat_plot",
            "geometry_scene.refresh_render_data chunks={} selected_region_id={} selected_region_count={}",
            geometry_scene.chunks.len(),
            self.geometry_presentation
                .selected_region_id
                .as_deref()
                .unwrap_or("none"),
            self.geometry_presentation.selected_region_ids.len(),
        );
        self.rebuild_geometry_scene_nodes(geometry_scene);
    }

    /// Sets the plot theme. Scene line colors depend on whether the background is light, so a
    /// loaded geometry scene is rebuilt when that changes.
    pub fn set_theme(&mut self, theme: crate::styling::PlotThemeConfig) {
        let was_light = self.theme_background_is_light();
        self.theme = theme;
        if was_light != self.theme_background_is_light() {
            if let Some(scene) = self.last_geometry_scene.clone() {
                self.rebuild_geometry_scene_nodes(&scene);
            }
        }
    }

    pub(super) fn theme_background_is_light(&self) -> bool {
        relative_luminance(self.theme.build_theme().get_background_color()) > 0.5
    }

    pub(super) fn rebuild_geometry_scene_nodes(&mut self, geometry_scene: &GeometryScene) {
        self.scene.clear();
        self.scene.set_translucent_last(true);
        self.scene_buffer_cache.borrow_mut().clear();
        self.geometry_node_owner_ids.clear();
        for (index, chunk) in geometry_scene.chunks.iter().enumerate() {
            let node_id = geometry_scene.chunk_node_id(index, &chunk.chunk_id);
            if !chunk.owner_node_ids.is_empty() {
                self.geometry_node_owner_ids
                    .insert(node_id, chunk.owner_node_ids.clone());
            }
        }
        self.apply_geometry_presentation_visibility();
        let light_background = self.theme_background_is_light();
        for mut node in geometry_scene.nodes_with_presentation(&self.geometry_presentation) {
            if light_background {
                if let Some(render_data) = node.render_data.as_mut() {
                    darken_pale_lines_for_light_background(render_data);
                }
            }
            self.scene.add_node_preserving_id(node);
        }
        self.apply_geometry_xray_to_nodes();
        self.apply_geometry_visibility_filter();
        self.needs_update = true;
    }
    pub(super) fn preserve_geometry_scene_state_when_unspecified(
        &self,
        presentation: &mut GeometryScenePresentation,
    ) {
        if !presentation.resolves_owner_visibility() {
            presentation.hidden_owner_node_ids =
                self.geometry_presentation.hidden_owner_node_ids.clone();
            presentation.isolated_owner_node_ids =
                self.geometry_presentation.isolated_owner_node_ids.clone();
        }
        if !presentation.resolves_section() {
            presentation.section = self.geometry_presentation.section.clone();
        }
    }
    pub(super) fn apply_geometry_presentation_visibility(&mut self) {
        if !self.geometry_presentation.resolves_owner_visibility() {
            return;
        }
        let all_owner_node_ids = self
            .geometry_node_owner_ids
            .values()
            .flat_map(|owner_ids| owner_ids.iter().map(String::as_str));
        self.geometry_hidden_owner_node_ids =
            self.geometry_presentation.resolved_hidden_owner_node_ids(
                all_owner_node_ids,
                &self.geometry_hidden_owner_node_ids,
            );
    }
    pub(super) fn apply_geometry_visibility_filter(&mut self) {
        if self.geometry_node_owner_ids.is_empty() {
            return;
        }
        let hidden_owner_ids = &self.geometry_hidden_owner_node_ids;
        let node_owner_ids = &self.geometry_node_owner_ids;
        self.scene.for_each_node_mut(|node| {
            let Some(owner_ids) = node_owner_ids.get(&node.id) else {
                return;
            };
            node.visible = !owner_ids
                .iter()
                .any(|owner_id| hidden_owner_ids.contains(owner_id));
        });
    }
    pub fn set_geometry_xray_enabled(&mut self, enabled: bool) {
        if self.geometry_overlay.is_none() || self.geometry_xray_enabled == enabled {
            return;
        }
        self.geometry_xray_enabled = enabled;
        if let Some(scene) = self.last_geometry_scene.clone() {
            self.refresh_geometry_scene_render_data(&scene);
        } else {
            self.apply_geometry_xray_to_nodes();
            self.scene_buffer_cache.borrow_mut().clear();
            self.needs_update = true;
        }
    }
    pub(super) fn apply_geometry_xray_to_nodes(&mut self) {
        if !self.geometry_xray_enabled {
            return;
        }
        let alpha = 0.38;
        self.scene.for_each_node_mut(|node| {
            if let Some(render_data) = node.render_data.as_mut() {
                if matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Triangles
                ) {
                    render_data.material.albedo.w = if self.geometry_xray_enabled {
                        render_data.material.albedo.w.min(alpha)
                    } else {
                        render_data.material.albedo.w.max(alpha)
                    };
                    render_data.material.alpha_mode = if render_data.material.albedo.w < 0.98 {
                        crate::core::AlphaMode::Blend
                    } else {
                        crate::core::AlphaMode::Opaque
                    };
                    for vertex in &mut render_data.vertices {
                        vertex.color[3] = if self.geometry_xray_enabled {
                            vertex.color[3].min(alpha)
                        } else {
                            vertex.color[3].max(alpha)
                        };
                    }
                }
            }
        });
    }
}
