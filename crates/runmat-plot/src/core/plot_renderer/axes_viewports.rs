//! Rendering every axes to its viewport, and the renderer's camera, scene and bounds accessors.

use super::*;

impl PlotRenderer {
    /// Render all axes of a subplot grid into their respective viewport rectangles.
    /// `axes_viewports` is a vector of (x, y, w, h) in physical pixels, length equals rows*cols.
    pub fn render_axes_to_viewports(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        axes_viewports: &[(u32, u32, u32, u32)],
        msaa_samples: u32,
        base_config: &PlotRenderConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        log::debug!(
            "runmat-plot: renderer.axes_to_viewports.start viewport_count={} msaa_samples={} width={} height={}",
            axes_viewports.len(),
            msaa_samples,
            base_config.width,
            base_config.height
        );
        // Build map axes_index -> node ids
        let mut axes_to_nodes: std::collections::HashMap<usize, Vec<crate::core::scene::NodeId>> =
            std::collections::HashMap::new();
        for node in self.scene.get_visible_nodes() {
            axes_to_nodes
                .entry(node.axes_index)
                .or_default()
                .push(node.id);
        }

        if self.axes_cameras.is_empty() {
            self.axes_cameras.push(Self::create_default_camera());
        }
        self.wgpu_renderer
            .ensure_axes_uniform_capacity(emission, axes_viewports.len().max(1));

        // Pre-collect all node ids
        let all_ids: Vec<crate::core::scene::NodeId> = self
            .scene
            .get_visible_nodes()
            .into_iter()
            .map(|n| n.id)
            .collect();
        let active_axes: Vec<usize> = axes_viewports
            .iter()
            .enumerate()
            .filter_map(|(ax_idx, _)| {
                axes_to_nodes
                    .get(&ax_idx)
                    .filter(|ids| !ids.is_empty())
                    .map(|_| ax_idx)
            })
            .collect();
        if active_axes.is_empty() {
            log::debug!("runmat-plot: renderer.axes_to_viewports.no_active_axes");
            return Ok(());
        }

        self.wgpu_renderer.ensure_msaa(msaa_samples.max(1));
        let shared_msaa_view = if self.wgpu_renderer.msaa_sample_count > 1 {
            Some(self.wgpu_renderer.ensure_msaa_color_view(emission))
        } else {
            None
        };

        for (ax_idx, viewport) in axes_viewports.iter().enumerate() {
            log::debug!(
                "runmat-plot: renderer.axes_to_viewports.viewport axes_index={} viewport=({}, {}, {}, {})",
                ax_idx,
                viewport.0,
                viewport.1,
                viewport.2,
                viewport.3
            );
            let ids_for_axes = axes_to_nodes.get(&ax_idx).cloned().unwrap_or_default();
            if ids_for_axes.is_empty() {
                log::debug!(
                    "runmat-plot: renderer.axes_to_viewports.skip_empty_axes axes_index={}",
                    ax_idx
                );
                continue;
            }

            // Hide nodes not belonging to this axes
            let mut hidden_ids: Vec<crate::core::scene::NodeId> = Vec::new();
            for id in &all_ids {
                if !ids_for_axes.contains(id) {
                    if let Some(node) = self.scene.get_node_mut(*id) {
                        if node.visible {
                            node.visible = false;
                            hidden_ids.push(*id);
                        }
                    }
                }
            }
            // Update camera and bounds
            let cam = self
                .axes_cameras
                .get(ax_idx)
                .cloned()
                .unwrap_or_else(Self::create_default_camera);
            let _ = self.calculate_data_bounds();

            // Render this axes into its viewport
            let mut cfg = base_config.clone();
            cfg.width = viewport.2;
            cfg.height = viewport.3;
            cfg.msaa_samples = msaa_samples.max(1);
            let is_first_axes = Some(&ax_idx) == active_axes.first();
            let is_last_axes = Some(&ax_idx) == active_axes.last();
            log::debug!(
                "runmat-plot: renderer.axes_to_viewports.axes_ready axes_index={} node_count={} first_axes={} last_axes={}",
                ax_idx,
                ids_for_axes.len(),
                is_first_axes,
                is_last_axes
            );
            let render_target = if let Some(ref msaa_view) = shared_msaa_view {
                RenderTarget {
                    view: msaa_view.as_ref(),
                    resolve_target: if is_last_axes {
                        Some(target_view)
                    } else {
                        None
                    },
                }
            } else {
                RenderTarget {
                    view: target_view,
                    resolve_target: None,
                }
            };
            let _ = self.render_camera_to_target_viewport(
                emission,
                encoder,
                render_target,
                *viewport,
                &cfg,
                &cam,
                ax_idx,
                is_first_axes,
            )?;
            log::debug!(
                "runmat-plot: renderer.axes_to_viewports.axes_render_ok axes_index={}",
                ax_idx
            );

            // Restore hidden nodes visibility
            for id in hidden_ids {
                if let Some(node) = self.scene.get_node_mut(id) {
                    node.visible = true;
                }
            }
        }
        log::debug!("runmat-plot: renderer.axes_to_viewports.ok");
        Ok(())
    }

    /// Create default 2D camera for plotting
    pub(super) fn create_default_camera() -> Camera {
        let mut camera = Camera::new();
        camera.projection = crate::core::camera::ProjectionType::Orthographic {
            left: -5.0,
            right: 5.0,
            bottom: -5.0,
            top: 5.0,
            // Use a deeper z range so the default camera position doesn't clip z=0 geometry.
            near: -10.0,
            far: 10.0,
        };
        camera.depth_mode = DepthMode::default();
        // For 2D plotting we keep the camera close to the z=0 plane.
        camera.position = Vec3::new(0.0, 0.0, 1.0);
        camera.target = Vec3::new(0.0, 0.0, 0.0);
        camera.up = Vec3::new(0.0, 1.0, 0.0);
        camera
    }

    // Removed simple data_bounds getter in favor of overlay-aware bounds below

    /// Get the primary (axes 0) camera.
    pub fn set_presence(
        &mut self,
        frame: crate::collaboration::PresenceFrame,
    ) -> Result<(), String> {
        if !frame.validate() {
            return Err("invalid presence frame".into());
        }
        self.presence = frame;
        Ok(())
    }

    pub fn presence(&self) -> &crate::collaboration::PresenceFrame {
        &self.presence
    }

    pub fn camera(&self) -> &Camera {
        self.axes_cameras
            .first()
            .expect("axes_cameras must contain at least one camera")
    }

    /// Get mutable reference to the primary (axes 0) camera.
    pub fn camera_mut(&mut self) -> &mut Camera {
        self.axes_cameras
            .first_mut()
            .expect("axes_cameras must contain at least one camera")
    }

    pub fn axes_camera(&self, axes_index: usize) -> Option<&Camera> {
        self.axes_cameras.get(axes_index)
    }

    /// Get scene reference
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Get scene statistics
    pub fn scene_statistics(&self) -> crate::core::SceneStatistics {
        self.scene.statistics()
    }

    /// Get current view bounds (camera frustum) in world/data space for 2D
    pub fn view_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        match self.camera().projection {
            crate::core::camera::ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                ..
            } => Some((left as f64, right as f64, bottom as f64, top as f64)),
            _ => None,
        }
    }
}
