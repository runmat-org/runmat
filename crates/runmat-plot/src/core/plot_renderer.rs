//! Unified plot rendering pipeline for both interactive GUI and static export
//!
//! This module provides the core rendering logic that is shared between
//! interactive plotting windows and static file exports, ensuring consistent
//! high-quality output across all use cases.

use crate::core::renderer::Vertex;
use crate::core::{
    BoundingBox, Camera, CameraViewPreset, ClipPolicy, DepthMode, Scene, WgpuRenderer,
};
use crate::geometry_scene::{
    GeometryScene, GeometrySceneCacheKey, GeometrySceneOverlay, GeometryScenePresentation,
};
use crate::plots::figure::{LegendEntry, TextStyle};
use crate::plots::surface::ColorMap;
use crate::plots::Figure;
use glam::{Mat4, Vec3, Vec4};
use runmat_time::Instant;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

type ViewBounds2D = (f64, f64, f64, f64);
type PerAxesViewBounds = Vec<Option<ViewBounds2D>>;

#[derive(Clone, Debug)]
struct CachedSceneBuffers {
    vertex_signature: (usize, usize),
    vertex_buffer: Arc<wgpu::Buffer>,
    index_signature: Option<(usize, usize)>,
    index_buffer: Option<Arc<wgpu::Buffer>>,
}

#[derive(Clone, Debug, PartialEq)]
struct AxesViewContract {
    rows: usize,
    cols: usize,
    axes: Vec<AxesViewContractEntry>,
}

#[derive(Clone, Debug, PartialEq)]
struct AxesViewContractEntry {
    has_3d_content: bool,
    x_limits: Option<(f64, f64)>,
    y_limits: Option<(f64, f64)>,
    z_limits: Option<(f64, f64)>,
    axis_equal: bool,
    x_log: bool,
    y_log: bool,
    view_azimuth_deg: Option<f32>,
    view_elevation_deg: Option<f32>,
    view_revision: u64,
}

const PATCH_3D_ABS_EPSILON: f32 = 1e-9;
const PATCH_3D_REL_EPSILON: f32 = 1e-6;
const MAX_2D_GRID_LINES_PER_AXIS: usize = 4096;

fn render_item_diagnostics_enabled() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        false
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| match std::env::var("RUNMAT_PLOT_RENDER_ITEM_LOGS") {
            Ok(value) => matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
            Err(_) => false,
        })
    }
}

/// Unified plot renderer that handles both interactive and static rendering
pub struct PlotRenderer {
    /// WGPU renderer for GPU-accelerated rendering
    pub wgpu_renderer: WgpuRenderer,

    /// Current scene being rendered
    pub scene: Scene,

    /// Current theme configuration  
    pub theme: crate::styling::PlotThemeConfig,

    /// Cached rendering state
    data_bounds: Option<(f64, f64, f64, f64)>,
    needs_update: bool,

    // Cached figure metadata for overlay
    figure_title: Option<String>,
    figure_sg_title: Option<String>,
    figure_sg_title_style: TextStyle,
    figure_x_label: Option<String>,
    figure_y_label: Option<String>,
    figure_z_label: Option<String>,
    figure_show_grid: bool,
    figure_show_minor_grid: bool,
    figure_show_legend: bool,
    figure_show_box: bool,
    figure_x_limits: Option<(f64, f64)>,
    figure_y_limits: Option<(f64, f64)>,
    legend_entries: Vec<LegendEntry>,
    figure_x_log: bool,
    figure_y_log: bool,
    figure_axis_equal: bool,
    figure_colormap: ColorMap,
    figure_colorbar_enabled: bool,
    // Categorical axis cache
    figure_categorical_is_x: Option<bool>,
    figure_categorical_labels: Option<Vec<String>>,
    /// Per-axes cameras (for subplots and single-axes figures).
    axes_cameras: Vec<Camera>,
    /// Keep a clone of the last figure set for export/UX operations
    pub(crate) last_figure: Option<crate::plots::Figure>,
    /// Current chunked geometry scene key, when the renderer is driven by CAD/FEA scene data.
    last_geometry_scene_key: Option<GeometrySceneCacheKey>,
    last_geometry_scene: Option<GeometryScene>,
    geometry_overlay: Option<GeometrySceneOverlay>,
    geometry_presentation: GeometryScenePresentation,
    geometry_xray_enabled: bool,
    geometry_node_owner_ids: HashMap<u64, Vec<String>>,
    geometry_hidden_owner_node_ids: BTreeSet<String>,

    /// Last surface extent (in pixels) that was used to build viewport-dependent geometry.
    /// Used so we can rebuild the scene after the canvas is resized (common on wasm).
    last_scene_viewport_px: Option<(u32, u32)>,
    /// Last per-axes plot viewport sizes used to build viewport-dependent geometry.
    last_axes_plot_sizes_px: Option<Vec<(u32, u32)>>,
    /// Last per-axes orthographic view bounds used for viewport-dependent 2D stroke geometry.
    last_axes_view_bounds: Option<PerAxesViewBounds>,
    /// Last figure view contract used to decide whether script-owned axes state changed.
    last_axes_view_contract: Option<AxesViewContract>,

    /// If false, do not auto-fit camera when the figure updates (user has interacted).
    camera_auto_fit: bool,
    /// Per-axes 2D camera ownership. True means the user has interacted and automatic 2D refits
    /// should not overwrite the camera during ordinary figure updates.
    axes_2d_camera_user_controlled: Vec<bool>,
    /// Last script-owned `view(...)` revision applied to each axes camera.
    axes_applied_view_revisions: Vec<Option<u64>>,
    /// Per-node GPU buffer cache for stable interactive redraws.
    scene_buffer_cache: RefCell<HashMap<u64, CachedSceneBuffers>>,
}

/// Configuration for plot rendering
#[derive(Debug, Clone)]
pub struct PlotRenderConfig {
    /// Output dimensions
    pub width: u32,
    pub height: u32,

    /// Background color
    pub background_color: Vec4,

    /// Whether to draw grid
    pub show_grid: bool,

    /// Whether to draw axes
    pub show_axes: bool,

    /// Whether to draw title
    pub show_title: bool,

    /// Anti-aliasing samples
    pub msaa_samples: u32,

    /// Depth mode for 3D rendering (standard vs reversed-Z).
    pub depth_mode: DepthMode,

    /// Clip plane policy for 3D rendering.
    pub clip_policy: ClipPolicy,

    /// Theme to use
    pub theme: crate::styling::PlotThemeConfig,
}

impl Default for PlotRenderConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            background_color: Vec4::new(0.08, 0.09, 0.11, 1.0), // Dark theme background
            show_grid: true,
            show_axes: true,
            show_title: true,
            msaa_samples: 4,
            depth_mode: DepthMode::default(),
            clip_policy: ClipPolicy::default(),
            theme: crate::styling::PlotThemeConfig::default(),
        }
    }
}

/// Target surface information for rendering with optional MSAA resolve.
pub struct RenderTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub resolve_target: Option<&'a wgpu::TextureView>,
}

/// Result of rendering operation
#[derive(Debug)]
pub struct RenderResult {
    /// Whether rendering was successful
    pub success: bool,

    /// Rendered data bounds
    pub data_bounds: Option<(f64, f64, f64, f64)>,

    /// Axes viewports in physical pixels as `(x, y, width, height)`.
    ///
    /// Hosts use these rectangles to route pointer events and picking through the
    /// same viewport that was used for rendering.
    pub axes_viewports_px: Vec<(u32, u32, u32, u32)>,

    /// Performance metrics
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub render_time_ms: f64,
}

impl PlotRenderer {
    /// Notify the renderer that the underlying surface configuration has changed (e.g. resize).
    /// On wasm the canvas is often created at a tiny size and resized shortly after; some
    /// CPU-generated geometry (like thick 2D lines) depends on viewport pixels, so we rebuild
    /// the scene when the surface extent changes.
    pub fn on_surface_config_updated(&mut self) {
        let current = (
            self.wgpu_renderer.surface_config.width.max(1),
            self.wgpu_renderer.surface_config.height.max(1),
        );
        if self.last_scene_viewport_px == Some(current) {
            return;
        }
        let Some(figure) = self.last_figure.clone() else {
            self.last_scene_viewport_px = Some(current);
            return;
        };
        // Rebuild scene using the updated surface extent.
        self.set_figure(figure);
    }

    fn prepare_buffers_for_render_data(
        &self,
        node_id: u64,
        render_data: &crate::core::RenderData,
    ) -> Option<(Arc<wgpu::Buffer>, Option<Arc<wgpu::Buffer>>)> {
        let mut cache = self.scene_buffer_cache.borrow_mut();
        let vertex_signature = (
            render_data.vertices.as_ptr() as usize,
            render_data.vertices.len(),
        );
        let index_signature = render_data
            .indices
            .as_ref()
            .map(|indices| (indices.as_ptr() as usize, indices.len()));

        if let Some(cached) = cache.get(&node_id) {
            if cached.vertex_signature == vertex_signature
                && cached.index_signature == index_signature
            {
                return Some((cached.vertex_buffer.clone(), cached.index_buffer.clone()));
            }
        }

        let vertex_buffer = self
            .wgpu_renderer
            .vertex_buffer_from_sources(render_data.gpu_vertices.as_ref(), &render_data.vertices)?;
        let index_buffer = render_data
            .indices
            .as_ref()
            .map(|indices| Arc::new(self.wgpu_renderer.create_index_buffer(indices)));

        cache.insert(
            node_id,
            CachedSceneBuffers {
                vertex_signature,
                vertex_buffer: vertex_buffer.clone(),
                index_signature,
                index_buffer: index_buffer.clone(),
            },
        );

        Some((vertex_buffer, index_buffer))
    }

    fn gpu_indirect_args(render_data: &crate::core::RenderData) -> Option<(&wgpu::Buffer, u64)> {
        render_data
            .gpu_vertices
            .as_ref()
            .and_then(|buf| buf.indirect.as_ref())
            .map(|indirect| (indirect.args.as_ref(), indirect.offset))
    }

    /// Create a new plot renderer
    pub async fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_config: wgpu::SurfaceConfiguration,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let wgpu_renderer = WgpuRenderer::new(device, queue, surface_config).await;
        let scene = Scene::new();
        let theme = crate::styling::PlotThemeConfig::default();

        Ok(Self {
            wgpu_renderer,
            scene,
            theme,
            data_bounds: None,
            needs_update: true,
            figure_title: None,
            figure_sg_title: None,
            figure_sg_title_style: TextStyle::default(),
            figure_x_label: None,
            figure_y_label: None,
            figure_z_label: None,
            figure_show_grid: true,
            figure_show_minor_grid: false,
            figure_show_legend: true,
            figure_show_box: true,
            figure_x_limits: None,
            figure_y_limits: None,
            legend_entries: Vec::new(),
            figure_x_log: false,
            figure_y_log: false,
            figure_axis_equal: false,
            figure_colormap: ColorMap::Parula,
            figure_colorbar_enabled: false,
            figure_categorical_is_x: None,
            figure_categorical_labels: None,
            axes_cameras: vec![Self::create_default_camera()],
            last_figure: None,
            last_geometry_scene_key: None,
            last_geometry_scene: None,
            geometry_overlay: None,
            geometry_presentation: GeometryScenePresentation::default(),
            geometry_xray_enabled: false,
            geometry_node_owner_ids: HashMap::new(),
            geometry_hidden_owner_node_ids: BTreeSet::new(),
            last_scene_viewport_px: None,
            last_axes_plot_sizes_px: None,
            last_axes_view_bounds: None,
            last_axes_view_contract: None,
            camera_auto_fit: true,
            axes_2d_camera_user_controlled: vec![false],
            axes_applied_view_revisions: vec![None],
            scene_buffer_cache: RefCell::new(HashMap::new()),
        })
    }

    fn plot_element_is_3d(plot: &crate::plots::figure::PlotElement) -> bool {
        match plot {
            crate::plots::figure::PlotElement::Surface(surface) => !surface.image_mode,
            crate::plots::figure::PlotElement::Mesh(_) => true,
            crate::plots::figure::PlotElement::Patch(patch) => {
                if patch.force_3d() {
                    return true;
                }

                let mut max_xy = 0.0_f32;
                let mut max_z = 0.0_f32;
                for point in patch.vertices() {
                    max_xy = max_xy.max(point.x.abs().max(point.y.abs()));
                    max_z = max_z.max(point.z.abs());
                }

                max_z > PATCH_3D_ABS_EPSILON.max(max_xy * PATCH_3D_REL_EPSILON)
            }
            crate::plots::figure::PlotElement::Line3(_) => true,
            crate::plots::figure::PlotElement::Scatter3(_) => true,
            crate::plots::figure::PlotElement::Contour(contour) => contour.is_3d(),
            _ => false,
        }
    }

    pub fn axes_has_3d_content(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .map(|figure| {
                figure
                    .plots()
                    .zip(figure.plot_axes_indices().iter().copied())
                    .any(|(plot, plot_axes_index)| {
                        plot_axes_index == axes_index && Self::plot_element_is_3d(plot)
                    })
            })
            .unwrap_or(false)
    }

    fn axes_view_contract_for_figure(figure: &Figure) -> AxesViewContract {
        let (rows, cols) = figure.axes_grid();
        let axes_count = rows.max(1) * cols.max(1);
        let mut has_3d_content = vec![false; axes_count];
        for (plot, axes_index) in figure
            .plots()
            .zip(figure.plot_axes_indices().iter().copied())
        {
            if axes_index < axes_count && Self::plot_element_is_3d(plot) {
                has_3d_content[axes_index] = true;
            }
        }
        let axes = (0..axes_count)
            .map(|axes_index| {
                let meta = figure.axes_metadata(axes_index);
                AxesViewContractEntry {
                    has_3d_content: has_3d_content[axes_index],
                    x_limits: meta.and_then(|m| m.x_limits),
                    y_limits: meta.and_then(|m| m.y_limits),
                    z_limits: meta.and_then(|m| m.z_limits),
                    axis_equal: meta.map(|m| m.axis_equal).unwrap_or(false),
                    x_log: meta.map(|m| m.x_log).unwrap_or(false),
                    y_log: meta.map(|m| m.y_log).unwrap_or(false),
                    view_azimuth_deg: meta.and_then(|m| m.view_azimuth_deg),
                    view_elevation_deg: meta.and_then(|m| m.view_elevation_deg),
                    view_revision: meta.map(|m| m.view_revision).unwrap_or(0),
                }
            })
            .collect();
        AxesViewContract { rows, cols, axes }
    }

    /// Mark that the user has interacted with the camera (disable auto-fit-on-update).
    pub fn note_camera_interaction(&mut self) {
        if self.camera_auto_fit {
            log::debug!(target: "runmat_plot", "camera_auto_fit disabled (user interaction)");
        }
        self.camera_auto_fit = false;
    }

    pub fn note_axes_camera_interaction(&mut self, axes_index: usize) {
        self.note_camera_interaction();
        if self.axes_has_3d_content(axes_index) {
            return;
        }
        if let Some(flag) = self.axes_2d_camera_user_controlled.get_mut(axes_index) {
            *flag = true;
        }
    }

    pub fn set_axes_camera_interaction_flags(&mut self, flags: &[bool]) {
        self.axes_2d_camera_user_controlled
            .resize(self.axes_cameras.len(), false);
        let mut any_user_controlled = false;
        for idx in 0..self.axes_cameras.len() {
            let controlled = flags.get(idx).copied().unwrap_or(false);
            if let Some(flag) = self.axes_2d_camera_user_controlled.get_mut(idx) {
                *flag = controlled;
            }
            any_user_controlled |= controlled;
        }
        if any_user_controlled {
            self.note_camera_interaction();
        }
    }

    pub fn axes_camera_interaction_flags(&self) -> &[bool] {
        &self.axes_2d_camera_user_controlled
    }

    fn clear_axes_camera_interaction(&mut self, axes_index: usize) {
        if let Some(flag) = self.axes_2d_camera_user_controlled.get_mut(axes_index) {
            *flag = false;
        }
    }

    fn clear_all_axes_camera_interaction(&mut self) {
        for flag in &mut self.axes_2d_camera_user_controlled {
            *flag = false;
        }
    }

    /// Set the figure to render
    pub fn set_figure(&mut self, figure: Figure) {
        self.last_geometry_scene_key = None;
        self.geometry_overlay = None;
        self.geometry_xray_enabled = false;
        self.geometry_node_owner_ids.clear();
        self.geometry_hidden_owner_node_ids.clear();
        // Clear existing scene
        self.scene.clear();
        self.scene_buffer_cache.borrow_mut().clear();

        // Convert figure to scene nodes
        self.cache_figure_meta(&figure);
        self.last_figure = Some(figure.clone());
        self.last_axes_plot_sizes_px = None;
        self.last_axes_view_bounds = None;
        // Initialize axes cameras for subplot grid
        let (rows, cols) = figure.axes_grid();
        let num_axes = rows.max(1) * cols.max(1);
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

        self.add_figure_to_scene(figure);

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
        presentation: GeometryScenePresentation,
    ) {
        let cache_key = geometry_scene.cache_key();
        let same_scene_identity = self
            .last_geometry_scene_key
            .as_ref()
            .map(|key| key.scene_id.as_str())
            == Some(cache_key.scene_id.as_str());
        let preserved_camera = same_scene_identity
            .then(|| self.axes_cameras.first().cloned())
            .flatten();
        let preserved_xray_enabled = same_scene_identity.then_some(self.geometry_xray_enabled);
        self.geometry_overlay = geometry_scene.overlay.clone();
        let geometry_xray_enabled = geometry_scene.chunks.iter().any(|chunk| {
            chunk.material.albedo.w < 0.95
                || matches!(chunk.material.alpha_mode, crate::core::AlphaMode::Blend)
        });
        if self.last_geometry_scene_key.as_ref() == Some(&cache_key) {
            self.last_geometry_scene = Some(geometry_scene.clone());
            if self.geometry_presentation != presentation {
                self.geometry_presentation = presentation;
                self.refresh_geometry_scene_render_data(&geometry_scene);
            }
            return;
        }

        self.scene.clear();
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
        self.figure_show_grid = geometry_scene.show_grid;
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

        for (index, chunk) in geometry_scene.chunks.iter().enumerate() {
            let node_id = geometry_scene.chunk_node_id(index, &chunk.chunk_id);
            if !chunk.owner_node_ids.is_empty() {
                self.geometry_node_owner_ids
                    .insert(node_id, chunk.owner_node_ids.clone());
            }
        }

        for node in geometry_scene.nodes_with_presentation(&self.geometry_presentation) {
            self.scene.add_node(node);
        }
        self.apply_geometry_xray_to_nodes();
        self.apply_geometry_visibility_filter();

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
    fn add_figure_to_scene(&mut self, figure: Figure) {
        self.add_figure_to_scene_with_axes_plot_sizes(figure, None);
    }

    fn add_figure_to_scene_with_axes_plot_sizes(
        &mut self,
        mut figure: Figure,
        axes_plot_sizes_px: Option<&[(u32, u32)]>,
    ) {
        use crate::core::SceneNode;

        let (rows, cols) = figure.axes_grid();

        // Convert figure to render data first, then create scene nodes.
        // For subplot figures, avoid baking full-surface line stroke geometry before overlay
        // layout has provided per-axes plot viewports.
        let viewport_px = (
            self.wgpu_renderer.surface_config.width.max(1),
            self.wgpu_renderer.surface_config.height.max(1),
        );
        self.last_scene_viewport_px = Some(viewport_px);
        let gpu = crate::core::GpuPackContext {
            device: &self.wgpu_renderer.device,
            queue: &self.wgpu_renderer.queue,
        };
        let view_bounds = self.axes_view_bounds_for_count(rows.max(1) * cols.max(1));
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
            let axes_index = axes_index.min(rows * cols - 1);
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
        axes_plot_sizes_px: &[(u32, u32)],
    ) {
        let normalized: Vec<(u32, u32)> = axes_plot_sizes_px
            .iter()
            .map(|&(w, h)| (w.max(1), h.max(1)))
            .collect();
        if normalized.iter().any(|&(w, h)| w < 2 || h < 2) {
            log::debug!(
                target: "runmat_plot.viewport_rebuild",
                "skipped viewport-dependent scene geometry rebuild for unstable viewport_sizes={:?}",
                normalized
            );
            return;
        }
        let view_bounds = self.axes_view_bounds_for_count(normalized.len().max(1));
        if self.last_axes_plot_sizes_px.as_ref() == Some(&normalized)
            && self.last_axes_view_bounds.as_ref() == Some(&view_bounds)
        {
            return;
        }
        let Some(figure) = self.last_figure.clone() else {
            self.last_axes_plot_sizes_px = Some(normalized);
            self.last_axes_view_bounds = Some(view_bounds);
            return;
        };
        self.scene.clear();
        self.scene_buffer_cache.borrow_mut().clear();
        self.add_figure_to_scene_with_axes_plot_sizes(figure, Some(&normalized));
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

    fn axes_view_bounds_for_count(&self, axes_count: usize) -> PerAxesViewBounds {
        (0..axes_count)
            .map(|idx| self.view_bounds_for_axes(idx))
            .collect()
    }

    fn refit_2d_cameras_to_scene_bounds(&mut self) {
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
    fn cache_figure_meta(&mut self, figure: &Figure) {
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
        self.figure_colormap = figure.colormap;
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

    fn apply_stored_axes_views(&mut self) {
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

    fn display_bounds_for_axes(&self, axes_index: usize) -> Option<(f64, f64, f64, f64)> {
        let base = self.axes_bounds(axes_index)?;
        let mut x_min = base.min.x as f64;
        let mut x_max = base.max.x as f64;
        let mut y_min = base.min.y as f64;
        let mut y_max = base.max.y as f64;

        if let Some(fig) = self.last_figure.as_ref() {
            if let Some(meta) = fig.axes_metadata(axes_index) {
                if let Some((xl, xr)) = meta.x_limits {
                    x_min = xl;
                    x_max = xr;
                }
                if let Some((yl, yr)) = meta.y_limits {
                    y_min = yl;
                    y_max = yr;
                }
                if meta.axis_equal {
                    let cx = (x_min + x_max) * 0.5;
                    let cy = (y_min + y_max) * 0.5;
                    let size = (x_max - x_min).abs().max((y_max - y_min).abs()).max(0.1);
                    x_min = cx - size * 0.5;
                    x_max = cx + size * 0.5;
                    y_min = cy - size * 0.5;
                    y_max = cy + size * 0.5;
                }
            }
        }

        Some((x_min, x_max, y_min, y_max))
    }

    fn apply_3d_display_limits_to_bounds(
        bounds: BoundingBox,
        figure: Option<&Figure>,
        axes_index: usize,
    ) -> BoundingBox {
        let Some(meta) = figure.and_then(|fig| fig.axes_metadata(axes_index)) else {
            return bounds;
        };
        let mut min = bounds.min;
        let mut max = bounds.max;
        if let Some((lo, hi)) = meta.x_limits {
            min.x = lo as f32;
            max.x = hi as f32;
        }
        if let Some((lo, hi)) = meta.y_limits {
            min.y = lo as f32;
            max.y = hi as f32;
        }
        if let Some((lo, hi)) = meta.z_limits {
            min.z = lo as f32;
            max.z = hi as f32;
        }
        BoundingBox { min, max }
    }

    fn bounds_are_finite(bounds: BoundingBox) -> bool {
        bounds.min.x.is_finite()
            && bounds.min.y.is_finite()
            && bounds.min.z.is_finite()
            && bounds.max.x.is_finite()
            && bounds.max.y.is_finite()
            && bounds.max.z.is_finite()
    }

    fn current_3d_display_bounds_for_axes(&self, axes_index: usize) -> Option<BoundingBox> {
        let bounds = self.axes_bounds(axes_index)?;
        let bounds =
            Self::apply_3d_display_limits_to_bounds(bounds, self.last_figure.as_ref(), axes_index);
        Self::bounds_are_finite(bounds).then_some(bounds)
    }

    fn display_bounds_3d_for_axes(&self, axes_index: usize) -> Option<BoundingBox> {
        self.current_3d_display_bounds_for_axes(axes_index)
    }

    fn axes_model_matrix(&self, _axes_index: usize) -> Mat4 {
        Mat4::IDENTITY
    }

    fn fit_cameras_to_axes_data(&mut self) -> bool {
        let mut applied = false;
        for idx in 0..self.axes_cameras.len() {
            if self.axes_has_3d_content(idx) {
                let Some(bounds) = self.display_bounds_3d_for_axes(idx) else {
                    continue;
                };
                let center = (bounds.min + bounds.max) * 0.5;
                let mut cam = Camera::new();
                cam.target = center;
                cam.up = Vec3::Z;
                cam.position = center + Vec3::new(1.0, -1.0, 1.0);
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

    /// Calculate data bounds from scene
    pub fn calculate_data_bounds(&mut self) -> Option<(f64, f64, f64, f64)> {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some(bounds) = render_data.bounds {
                    min_x = min_x.min(bounds.min.x as f64);
                    max_x = max_x.max(bounds.max.x as f64);
                    min_y = min_y.min(bounds.min.y as f64);
                    max_y = max_y.max(bounds.max.y as f64);
                    continue;
                }
                for vertex in &render_data.vertices {
                    let x = vertex.position[0] as f64;
                    let y = vertex.position[1] as f64;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        if min_x != f64::INFINITY && max_x != f64::NEG_INFINITY {
            // Add a small margin around data for readability without making
            // the plotted curve appear to exceed the labeled axis range.
            let x_range = (max_x - min_x).max(0.1);
            let y_range = (max_y - min_y).max(0.1);
            let x_margin = x_range * 0.04;
            let y_margin = y_range * 0.04;

            let bounds = (
                min_x - x_margin,
                max_x + x_margin,
                min_y - y_margin,
                max_y + y_margin,
            );

            // println!("Calculated data bounds: {:?}", bounds); // Too noisy
            self.data_bounds = Some(bounds);
            Some(bounds)
        } else {
            self.data_bounds = None;
            None
        }
    }

    /// Fit camera to show all data.
    ///
    /// Returns `true` if a fit was applied (i.e. bounds existed).
    pub fn fit_camera_to_data(&mut self) -> bool {
        if self.axes_cameras.len() > 1 {
            return self.fit_cameras_to_axes_data();
        }

        if self.axes_has_3d_content(0) {
            let Some(bounds) = self.display_bounds_3d_for_axes(0) else {
                return false;
            };
            let center = (bounds.min + bounds.max) * 0.5;
            let mut cam = Camera::new();
            cam.target = center;
            cam.up = Vec3::Z;
            cam.position = center + Vec3::new(1.0, -1.0, 1.0);
            cam.fit_bounds(bounds.min, bounds.max);
            if let Some(axis_cam) = self.axes_cameras.get_mut(0) {
                *axis_cam = cam;
            }
            return true;
        }

        if let Some((x_min, x_max, y_min, y_max)) = self.display_bounds_for_axes(0) {
            // Match the camera bounds exactly to data bounds to align with overlay grid
            let mut cam = Self::create_default_camera();
            let l = x_min as f32;
            let r = x_max as f32;
            let b = y_min as f32;
            let t = y_max as f32;
            if let crate::core::camera::ProjectionType::Orthographic {
                ref mut left,
                ref mut right,
                ref mut bottom,
                ref mut top,
                ..
            } = cam.projection
            {
                *left = l;
                *right = r;
                *bottom = b;
                *top = t;
            }
            cam.position.z = 1.0;
            cam.target.z = 0.0;
            cam.mark_dirty();

            if let Some(axis_cam) = self.axes_cameras.get_mut(0) {
                *axis_cam = cam;
            }
            return true;
        }
        false
    }

    /// Explicit "Fit Extents" action (CAD-like). Fits the camera to current data once.
    pub fn fit_extents(&mut self) {
        let _ = if self.figure_axes_grid().0 * self.figure_axes_grid().1 > 1 {
            self.fit_cameras_to_axes_data()
        } else {
            self.fit_camera_to_data()
        };
        self.clear_all_axes_camera_interaction();
        self.camera_auto_fit = false;
        self.needs_update = true;
    }

    pub fn reset_geometry_view(&mut self) {
        self.set_camera_view_preset(CameraViewPreset::Perspective);
    }

    pub fn set_camera_view_preset(&mut self, preset: CameraViewPreset) {
        let bounds_by_axes: Vec<Option<BoundingBox>> = (0..self.axes_cameras.len())
            .map(|idx| {
                if self.axes_has_3d_content(idx) {
                    self.display_bounds_3d_for_axes(idx)
                } else {
                    self.axes_bounds(idx)
                }
            })
            .collect();
        let display_bounds: PerAxesViewBounds = (0..self.axes_cameras.len())
            .map(|idx| self.display_bounds_for_axes(idx))
            .collect();

        for (idx, camera) in self.axes_cameras.iter_mut().enumerate() {
            if matches!(
                camera.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            ) {
                Self::apply_perspective_view_preset(camera, preset, bounds_by_axes[idx]);
            } else if let Some((x_min, x_max, y_min, y_max)) = display_bounds[idx] {
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
                *camera = cam;
            }
        }

        self.clear_all_axes_camera_interaction();
        self.camera_auto_fit = false;
        self.needs_update = true;
    }

    fn apply_perspective_view_preset(
        camera: &mut Camera,
        preset: CameraViewPreset,
        bounds: Option<BoundingBox>,
    ) {
        let (direction, up) = Self::view_preset_orientation(preset);
        camera.up = up;

        if let Some(bounds) = bounds {
            let center = (bounds.min + bounds.max) * 0.5;
            camera.target = center;
            camera.position = center + direction;
            camera.fit_bounds(bounds.min, bounds.max);
        } else {
            let distance = (camera.position - camera.target).length().max(1.0);
            camera.position = camera.target + direction * distance;
            camera.mark_dirty();
        }
    }

    fn view_preset_orientation(preset: CameraViewPreset) -> (Vec3, Vec3) {
        match preset {
            CameraViewPreset::Perspective => (Vec3::new(1.0, -1.0, 1.0).normalize(), Vec3::Z),
            CameraViewPreset::Top => (Vec3::Z, Vec3::Y),
            CameraViewPreset::Bottom => (-Vec3::Z, Vec3::Y),
            CameraViewPreset::Front => (-Vec3::Y, Vec3::Z),
            CameraViewPreset::Back => (Vec3::Y, Vec3::Z),
            CameraViewPreset::Left => (-Vec3::X, Vec3::Z),
            CameraViewPreset::Right => (Vec3::X, Vec3::Z),
        }
    }

    /// Explicit "Reset Camera" action. Restores the default orientation without re-framing.
    ///
    /// For 3D, this resets the view direction around the current data center (or current target)
    /// while preserving the current zoom distance.
    /// For 2D, this is equivalent to Fit Extents (since "home" without data bounds is rarely useful).
    pub fn reset_camera_position(&mut self) {
        let dir = Vec3::new(1.0, -1.0, 1.0).normalize_or_zero();
        let data_centers: Vec<Vec3> = (0..self.axes_cameras.len())
            .map(|idx| {
                if self.axes_has_3d_content(idx) {
                    self.display_bounds_3d_for_axes(idx)
                } else {
                    self.axes_bounds(idx)
                }
                .map(|b| (b.min + b.max) * 0.5)
                .unwrap_or_else(|| self.axes_cameras[idx].target)
            })
            .collect();
        let display_bounds: PerAxesViewBounds = (0..self.axes_cameras.len())
            .map(|idx| self.display_bounds_for_axes(idx))
            .collect();
        for (idx, c) in self.axes_cameras.iter_mut().enumerate() {
            if matches!(
                c.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            ) {
                let data_center = data_centers.get(idx).copied().unwrap_or(c.target);
                let dist = (c.position - c.target).length().max(0.1);
                c.target = data_center;
                c.up = Vec3::Z;
                c.position = data_center + dir * dist;
                c.mark_dirty();
            } else if let Some((x_min, x_max, y_min, y_max)) = display_bounds[idx] {
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
                *c = cam;
            }
        }
        self.clear_all_axes_camera_interaction();
        self.camera_auto_fit = false;
        self.needs_update = true;
    }

    /// Render the current scene to a specific viewport within a texture/surface
    pub fn render_to_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        _viewport: (f32, f32, f32, f32), // (x, y, width, height) in framebuffer coordinates
        clear_background: bool,
        background_color: Option<glam::Vec4>,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        // Collect render data and create buffers first
        let mut render_items = Vec::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some(vertex_buffer) = self.wgpu_renderer.vertex_buffer_from_sources(
                    render_data.gpu_vertices.as_ref(),
                    &render_data.vertices,
                ) {
                    self.wgpu_renderer
                        .ensure_pipeline(render_data.pipeline_type);

                    log::trace!(
                        target: "runmat_plot",
                        "upload vertices={}, draw_calls={}",
                        render_data.vertex_count(),
                        render_data.draw_calls.len()
                    );

                    render_items.push((render_data, vertex_buffer));
                    total_vertices += render_data.vertex_count();

                    if render_data.pipeline_type == crate::core::PipelineType::Triangles {
                        total_triangles += render_data.vertex_count() / 3;
                    }
                }
            }
        }

        // Update uniforms
        let mut cam = self.camera().clone();
        let view_proj_matrix = cam.view_proj_matrix();

        self.wgpu_renderer
            .update_uniforms(view_proj_matrix, Mat4::IDENTITY);

        // Create render pass (respect MSAA)
        let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
        let msaa_view_opt = if use_msaa {
            let tex = self
                .wgpu_renderer
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("runmat_msaa_color_camera"),
                    size: wgpu::Extent3d {
                        width: self.wgpu_renderer.surface_config.width,
                        height: self.wgpu_renderer.surface_config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: self.wgpu_renderer.msaa_sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.wgpu_renderer.surface_config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
            Some(tex.create_view(&wgpu::TextureViewDescriptor::default()))
        } else {
            None
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Plot Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view_opt.as_ref().unwrap_or(target_view),
                resolve_target: if use_msaa { Some(target_view) } else { None },
                ops: wgpu::Operations {
                    load: if clear_background {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color.map_or(0.08, |c| c.x as f64),
                            g: background_color.map_or(0.09, |c| c.y as f64),
                            b: background_color.map_or(0.11, |c| c.z as f64),
                            a: background_color.map_or(1.0, |c| c.w as f64),
                        })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Apply viewport scissor to match overlay plot rect
        let (vx, vy, vw, vh) = _viewport;
        render_pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);

        // Configure direct-uniforms for precise data-to-NDC mapping within this viewport
        let sw = self.wgpu_renderer.surface_config.width as f32;
        let sh = self.wgpu_renderer.surface_config.height as f32;
        let ndc_left = (vx / sw) * 2.0 - 1.0;
        let ndc_right = ((vx + vw) / sw) * 2.0 - 1.0;
        let ndc_top = 1.0 - (vy / sh) * 2.0;
        let ndc_bottom = 1.0 - ((vy + vh) / sh) * 2.0;

        // data_bounds passed in from caller: (x_min, y_min, x_max, y_max)
        let (x_min, y_min, x_max, y_max) = (0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64);
        self.wgpu_renderer.update_direct_uniforms(
            [x_min as f32, y_min as f32],
            [x_max as f32, y_max as f32],
            [ndc_left, ndc_bottom],
            [ndc_right, ndc_top],
            [sw, sh],
        );

        // Continue with specific pipelines below (implementation omitted here)
        drop(render_pass);

        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            axes_viewports_px: vec![(
                vx.round().max(0.0) as u32,
                vy.round().max(0.0) as u32,
                vw.round().max(1.0) as u32,
                vh.round().max(1.0) as u32,
            )],
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: render_time,
        })
    }

    /// Render the current scene to a texture/surface
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: RenderTarget<'_>,
        config: &PlotRenderConfig,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        self.wgpu_renderer.ensure_msaa(config.msaa_samples);

        // Update WGPU uniforms from primary axes camera
        let aspect_ratio = config.width as f32 / config.height as f32;
        let mut cam = self.camera().clone();
        cam.update_aspect_ratio(aspect_ratio);
        let view_proj_matrix = cam.view_proj_matrix();
        let model_matrix = self.axes_model_matrix(0);
        self.wgpu_renderer
            .update_uniforms(view_proj_matrix, model_matrix);

        // Collect all render data and create vertex buffers first (outside render pass)
        let mut render_items = Vec::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some((vertex_buffer, index_buffer)) =
                    self.prepare_buffers_for_render_data(node.id, render_data)
                {
                    self.wgpu_renderer
                        .ensure_pipeline(render_data.pipeline_type);
                    render_items.push((render_data, vertex_buffer, index_buffer));

                    total_vertices += render_data.vertex_count();
                    if let Some(indices) = &render_data.indices {
                        total_triangles += indices.len() / 3;
                    }
                }
            }
        }

        // Pre-create image bind groups and set direct uniforms once (for textured items)
        let mut image_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        let has_textured_items = render_items.iter().any(|(render_data, _, _)| {
            render_data.pipeline_type == crate::core::PipelineType::Textured
        });
        if has_textured_items {
            // Ensure image pipeline once to avoid mutable borrow during pass.
            self.wgpu_renderer.ensure_image_pipeline();
            let mut inferred_bounds: Option<(f64, f64, f64, f64)> = None;
            for (render_data, _, _) in &render_items {
                let Some(bounds) = render_data.bounds.as_ref() else {
                    continue;
                };
                let min_x = bounds.min.x as f64;
                let max_x = bounds.max.x as f64;
                let min_y = bounds.min.y as f64;
                let max_y = bounds.max.y as f64;
                inferred_bounds = Some(match inferred_bounds {
                    Some((x0, x1, y0, y1)) => {
                        (x0.min(min_x), x1.max(max_x), y0.min(min_y), y1.max(max_y))
                    }
                    None => (min_x, max_x, min_y, max_y),
                });
            }

            let (mut x_min, mut x_max, mut y_min, mut y_max) = self
                .data_bounds
                .or(inferred_bounds)
                .unwrap_or((-1.0, 1.0, -1.0, 1.0));
            // Avoid zero ranges in the direct image shader (division by data_range).
            if (x_max - x_min).abs() < f64::EPSILON {
                x_min -= 0.5;
                x_max += 0.5;
            }
            if (y_max - y_min).abs() < f64::EPSILON {
                y_min -= 0.5;
                y_max += 0.5;
            }
            log::trace!(
                target: "runmat_plot",
                "direct uniforms bounds x=({}, {}) y=({}, {}) size=({}, {})",
                x_min,
                x_max,
                y_min,
                y_max,
                config.width,
                config.height
            );
            self.wgpu_renderer.update_direct_uniforms(
                [x_min as f32, y_min as f32],
                [x_max as f32, y_max as f32],
                [-1.0, -1.0],
                [1.0, 1.0],
                [config.width as f32, config.height as f32],
            );
        }
        for (render_data, _vb, _ib) in &render_items {
            if render_data.pipeline_type == crate::core::PipelineType::Textured {
                if let Some(crate::core::scene::ImageData::Rgba8 {
                    width,
                    height,
                    data,
                }) = &render_data.image
                {
                    let (_tex, _view, img_bg) = self
                        .wgpu_renderer
                        .create_image_texture_and_bind_group(*width, *height, data);
                    image_bind_groups.push(Some(img_bg));
                } else {
                    image_bind_groups.push(None);
                }
            } else {
                image_bind_groups.push(None);
            }
        }
        let mut point_style_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in &render_items {
            if matches!(
                render_data.pipeline_type,
                crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
            ) {
                let style = crate::core::renderer::PointStyleUniforms {
                    face_color: render_data.material.albedo.to_array(),
                    edge_color: render_data.material.emissive.to_array(),
                    edge_thickness_px: render_data.material.roughness,
                    marker_shape: render_data.material.metallic as u32,
                    _pad: [0.0, 0.0],
                };
                let (_buf, bg) = self.wgpu_renderer.create_point_style_bind_group(style);
                point_style_bind_groups.push(Some(bg));
            } else {
                point_style_bind_groups.push(None);
            }
        }
        // Expand CPU marker vertices to billboard quads.
        let mut point_buffers: Vec<Option<(wgpu::Buffer, usize)>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in &render_items {
            if matches!(
                render_data.pipeline_type,
                crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
            ) && !render_data.vertices.is_empty()
            {
                let expanded = self
                    .wgpu_renderer
                    .create_direct_point_vertices(&render_data.vertices, 0.0);
                let buffer = self.wgpu_renderer.create_vertex_buffer(&expanded);
                point_buffers.push(Some((buffer, expanded.len())));
            } else {
                point_buffers.push(None);
            }
        }
        self.wgpu_renderer.update_marker_screen_uniforms([
            config.width.max(1) as f32,
            config.height.max(1) as f32,
        ]);

        // Create render pass
        {
            let depth_view = self.wgpu_renderer.ensure_depth_view();
            let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
            let mut cached_msaa_view: Option<Arc<wgpu::TextureView>> = None;

            let (color_view, resolve_target) = if use_msaa {
                if let Some(explicit_resolve_target) = target.resolve_target {
                    (target.view, Some(explicit_resolve_target))
                } else {
                    cached_msaa_view = Some(self.wgpu_renderer.ensure_msaa_color_view());
                    (
                        cached_msaa_view
                            .as_ref()
                            .expect("msaa color view should exist")
                            .as_ref(),
                        Some(target.view),
                    )
                }
            } else {
                (target.view, target.resolve_target)
            };

            let depth_clear = match self.wgpu_renderer.depth_mode {
                crate::core::DepthMode::Standard => 1.0,
                crate::core::DepthMode::ReversedZ => 0.0,
            };
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Plot Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: config.background_color.x as f64,
                            g: config.background_color.y as f64,
                            b: config.background_color.z as f64,
                            a: config.background_color.w as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(depth_clear),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let _keep_msaa_view_alive = &cached_msaa_view;

            // Now render all items with proper bind group setup
            for (i, (render_data, vertex_buffer, index_buffer)) in render_items.iter().enumerate() {
                #[cfg(target_arch = "wasm32")]
                {
                    // On wasm, "blank but drawing" is often caused by bad vertex data (NaNs/alpha=0)
                    // or using the wrong pipeline. Emit a single summary per item.
                    if log::log_enabled!(log::Level::Debug) {
                        if let Some(v0) = render_data.vertices.first() {
                            log::debug!(
                                target: "runmat_plot",
                                "wasm draw item: pipeline={:?} verts={} v0.pos=({:.3},{:.3},{:.3}) v0.color=({:.3},{:.3},{:.3},{:.3})",
                                render_data.pipeline_type,
                                render_data.vertices.len(),
                                v0.position[0],
                                v0.position[1],
                                v0.position[2],
                                v0.color[0],
                                v0.color[1],
                                v0.color[2],
                                v0.color[3],
                            );
                        } else if render_data.gpu_vertices.is_some() {
                            log::debug!(
                                target: "runmat_plot",
                                "wasm draw item: pipeline={:?} using gpu_vertices vertex_count={}",
                                render_data.pipeline_type,
                                render_data.vertex_count(),
                            );
                        } else {
                            log::debug!(
                                target: "runmat_plot",
                                "wasm draw item: pipeline={:?} has no vertices",
                                render_data.pipeline_type
                            );
                        }
                    }
                }

                // Get the appropriate pipeline for this render data (pipeline ensured above)
                if render_data.pipeline_type == crate::core::PipelineType::Textured {
                    // Ensure image pipeline
                    let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
                    render_pass.set_pipeline(pipeline);
                    // Bind direct uniforms at set(0)
                    // Use data bounds for image mapping
                    render_pass.set_bind_group(
                        0,
                        &self.wgpu_renderer.direct_uniform_bind_group,
                        &[],
                    );
                    if let Some(ref img_bg) = image_bind_groups[i] {
                        render_pass.set_bind_group(1, img_bg, &[]);
                    }
                } else {
                    let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
                    render_pass.set_pipeline(pipeline);
                    // Set the uniform bind group (required by shaders)
                    render_pass.set_bind_group(0, self.wgpu_renderer.get_uniform_bind_group(), &[]);
                    if matches!(
                        render_data.pipeline_type,
                        crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
                    ) {
                        if let Some(ref bg) = point_style_bind_groups[i] {
                            render_pass.set_bind_group(1, bg, &[]);
                        }
                        render_pass.set_bind_group(
                            2,
                            self.wgpu_renderer.get_marker_screen_bind_group(),
                            &[],
                        );
                    }
                }

                let is_markers = matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
                );
                if is_markers {
                    if let Some((ref expanded, _len)) = point_buffers[i] {
                        render_pass.set_vertex_buffer(0, expanded.slice(..));
                    } else {
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    }
                } else {
                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                }

                if let Some(index_buffer) = index_buffer {
                    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    if let Some(indices) = &render_data.indices {
                        log::trace!(target: "runmat_plot", "draw indexed count={}", indices.len());
                        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                    }
                } else {
                    log::trace!(target: "runmat_plot", "draw direct vertices");
                    if let Some((args, offset)) = Self::gpu_indirect_args(render_data) {
                        render_pass.draw_indirect(args, offset);
                        continue;
                    }
                    if is_markers {
                        if let Some((_, len)) = point_buffers[i] {
                            render_pass.draw(0..len as u32, 0..1);
                            continue;
                        }
                    }
                    // Use draw_calls from render_data for proper vertex range handling
                    for draw_call in &render_data.draw_calls {
                        log::trace!(
                            target: "runmat_plot",
                            "draw vertices offset={} count={} instances={}",
                            draw_call.vertex_offset,
                            draw_call.vertex_count,
                            draw_call.instance_count
                        );
                        render_pass.draw(
                            draw_call.vertex_offset as u32
                                ..(draw_call.vertex_offset + draw_call.vertex_count) as u32,
                            0..draw_call.instance_count as u32,
                        );
                    }
                }
            }
            // drop render_pass at end of scope
        }

        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            axes_viewports_px: vec![(0, 0, config.width.max(1), config.height.max(1))],
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: render_time,
        })
    }

    /// Shared scene orchestration for non-overlay render targets.
    ///
    /// For single-axes figures this follows the direct full-target render path.
    /// For subplot grids, it renders each axes into a deterministic tiled viewport layout.
    pub fn render_scene_to_target(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        config: &PlotRenderConfig,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let (rows, cols) = self.figure_axes_grid();
        let axes_count = rows.saturating_mul(cols);
        log::debug!(
            "runmat-plot: renderer.scene_to_target.start rows={} cols={} axes_count={} width={} height={}",
            rows,
            cols,
            axes_count,
            config.width,
            config.height
        );
        if axes_count <= 1 {
            log::debug!("runmat-plot: renderer.scene_to_target.branch_single_axes");
            return self.render(
                encoder,
                RenderTarget {
                    view: target_view,
                    resolve_target: None,
                },
                config,
            );
        }

        let viewports =
            Self::compute_tiled_viewports(config.width.max(1), config.height.max(1), rows, cols);
        log::debug!(
            "runmat-plot: renderer.scene_to_target.branch_subplot_axes viewports={}",
            viewports.len()
        );
        self.render_axes_to_viewports(
            encoder,
            target_view,
            &viewports,
            config.msaa_samples.max(1),
            config,
        )?;
        let stats = self.scene.statistics();
        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            axes_viewports_px: viewports,
            vertex_count: stats.total_vertices,
            triangle_count: stats.total_triangles,
            render_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn compute_tiled_viewports(
        total_width: u32,
        total_height: u32,
        rows: usize,
        cols: usize,
    ) -> Vec<(u32, u32, u32, u32)> {
        if rows == 0 || cols == 0 {
            return vec![(0, 0, total_width.max(1), total_height.max(1))];
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let cell_w = (total_width / cols_u32).max(1);
        let cell_h = (total_height / rows_u32).max(1);
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows_u32 {
            for c in 0..cols_u32 {
                let x = c * cell_w;
                let y = r * cell_h;
                let mut w = cell_w;
                let mut h = cell_h;
                if c + 1 == cols_u32 {
                    w = total_width.saturating_sub(x).max(1);
                }
                if r + 1 == rows_u32 {
                    h = total_height.saturating_sub(y).max(1);
                }
                out.push((x, y, w, h));
            }
        }
        out
    }

    /// Render using the camera-based pipeline into a viewport region with a scissor rectangle.
    /// This preserves existing contents (Load) and draws only inside the viewport rectangle.
    #[allow(clippy::too_many_arguments)]
    pub fn render_camera_to_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        viewport_scissor: (u32, u32, u32, u32),
        config: &PlotRenderConfig,
        camera: &Camera,
        axes_index: usize,
        clear_background: bool,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        log::debug!(
            "runmat-plot: renderer.camera_to_viewport.start axes_index={} viewport=({}, {}, {}, {}) clear_background={}",
            axes_index,
            viewport_scissor.0,
            viewport_scissor.1,
            viewport_scissor.2,
            viewport_scissor.3,
            clear_background
        );
        let use_msaa = config.msaa_samples.max(1) > 1;
        self.wgpu_renderer.ensure_msaa(config.msaa_samples);
        let msaa_view_keepalive = if use_msaa {
            Some(self.wgpu_renderer.ensure_msaa_color_view())
        } else {
            None
        };
        let render_target = if let Some(msaa_view) = msaa_view_keepalive.as_ref() {
            RenderTarget {
                view: msaa_view.as_ref(),
                resolve_target: Some(target_view),
            }
        } else {
            RenderTarget {
                view: target_view,
                resolve_target: None,
            }
        };
        self.render_camera_to_target_viewport(
            encoder,
            render_target,
            viewport_scissor,
            config,
            camera,
            axes_index,
            clear_background,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_camera_to_target_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: RenderTarget<'_>,
        viewport_scissor: (u32, u32, u32, u32),
        config: &PlotRenderConfig,
        camera: &Camera,
        axes_index: usize,
        clear_background: bool,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        // Apply MSAA preference into pipelines
        self.wgpu_renderer.ensure_msaa(config.msaa_samples);
        self.wgpu_renderer.set_depth_mode(config.depth_mode);

        // Ensure a depth attachment exists for camera-based 3D rendering.
        // This is a no-op for pure 2D direct-mapped pipelines, but is required for correct
        // occlusion in 3D plots (surf/mesh/scatter3).
        let depth_view = self.wgpu_renderer.ensure_depth_view();

        // Update standard uniforms from the provided camera
        let aspect_ratio = (config.width.max(1)) as f32 / (config.height.max(1)) as f32;
        let mut cam = camera.clone();
        cam.update_aspect_ratio(aspect_ratio);
        cam.depth_mode = config.depth_mode;
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.camera_ready axes_index={} aspect_ratio={} msaa_samples={}",
            axes_index,
            aspect_ratio,
            config.msaa_samples
        );

        // Dynamic clip planes (CAD-like): keep near/far tight to visible bounds to avoid
        // clipping surprises and depth precision collapse on huge datasets.
        if config.clip_policy.dynamic {
            let mut bounds: Option<crate::core::scene::BoundingBox> = None;
            for node in self.scene.get_visible_nodes() {
                if let Some(rd) = &node.render_data {
                    if let Some(b) = rd.bounds {
                        bounds = Some(bounds.map_or(b, |acc| acc.union(&b)));
                    }
                }
            }
            if let Some(b) = bounds {
                cam.update_clip_planes_from_world_aabb(b.min, b.max, &config.clip_policy);
            }
        }
        let view_proj_matrix = cam.view_proj_matrix();
        let model_matrix = self.axes_model_matrix(axes_index);
        self.wgpu_renderer
            .update_uniforms_for_axes(axes_index, view_proj_matrix, model_matrix);
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.uniforms_updated axes_index={}",
            axes_index
        );

        let (mut sx, mut sy, mut sw, mut sh) = viewport_scissor;
        let target_w = self.wgpu_renderer.surface_config.width.max(1);
        let target_h = self.wgpu_renderer.surface_config.height.max(1);
        if sx >= target_w || sy >= target_h {
            return Ok(RenderResult {
                success: true,
                data_bounds: self.data_bounds,
                axes_viewports_px: vec![(0, 0, target_w, target_h)],
                vertex_count: 0,
                triangle_count: 0,
                render_time_ms: 0.0,
            });
        }
        sx = sx.min(target_w.saturating_sub(1));
        sy = sy.min(target_h.saturating_sub(1));
        sw = sw.max(1).min(target_w.saturating_sub(sx).max(1));
        sh = sh.max(1).min(target_h.saturating_sub(sy).max(1));
        let is_2d = matches!(
            cam.projection,
            crate::core::camera::ProjectionType::Orthographic { .. }
        );
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.viewport_normalized axes_index={} viewport=({}, {}, {}, {}) is_2d={}",
            axes_index,
            sx,
            sy,
            sw,
            sh,
            is_2d
        );
        match cam.projection {
            crate::core::camera::ProjectionType::Orthographic {
                left,
                right,
                bottom,
                top,
                ..
            } => {
                log::debug!(
                    target: "runmat_plot.draw_camera",
                    "draw camera axes_index={} is_2d=true viewport=({}, {}, {}, {}) bounds=({}, {})..({}, {}) cfg_wh=({}, {})",
                    axes_index,
                    sx,
                    sy,
                    sw,
                    sh,
                    left,
                    bottom,
                    right,
                    top,
                    config.width,
                    config.height
                );
            }
            crate::core::camera::ProjectionType::Perspective { .. } => {
                log::debug!(
                    target: "runmat_plot.draw_camera",
                    "draw camera axes_index={} is_2d=false viewport=({}, {}, {}, {}) cfg_wh=({}, {})",
                    axes_index,
                    sx,
                    sy,
                    sw,
                    sh,
                    config.width,
                    config.height
                );
            }
        }

        // Prepare render items outside the pass
        let mut owned_render_data: Vec<Box<crate::core::RenderData>> = Vec::new();
        let mut render_items = Vec::new();
        let mut grid_plane_buffers: Option<(wgpu::Buffer, wgpu::Buffer)> = None;
        let mut total_vertices = 0usize;
        let mut total_triangles = 0usize;
        let render_item_logs = render_item_diagnostics_enabled();
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.collect_render_items.start axes_index={}",
            axes_index
        );
        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if render_item_logs && node.axes_index == axes_index {
                    log::debug!(
                        target: "runmat_plot.draw_item",
                        "draw item axes_index={} node_axes_index={} pipeline={:?} vertex_count={} has_indices={} has_bounds={} gpu_vertices={}",
                        axes_index,
                        node.axes_index,
                        render_data.pipeline_type,
                        render_data.vertex_count(),
                        render_data.indices.is_some(),
                        render_data.bounds.is_some(),
                        render_data.gpu_vertices.is_some()
                    );
                }
                if let Some((vb, ib)) = self.prepare_buffers_for_render_data(node.id, render_data) {
                    self.wgpu_renderer
                        .ensure_pipeline(render_data.pipeline_type);
                    total_vertices += render_data.vertex_count();
                    if let Some(indices) = &render_data.indices {
                        total_triangles += indices.len() / 3;
                    }
                    render_items.push((render_data, vb, ib));
                }
            }
        }
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.collect_render_items.ok axes_index={} items={} total_vertices={} total_triangles={}",
            axes_index,
            render_items.len(),
            total_vertices,
            total_triangles
        );

        // 3D helpers: CAD-style XY grid and world-axis ticks. The compact corner
        // triad is drawn separately and remains available as navigation context.
        // These are generated per-frame so they can adapt to zoom level.
        if !is_2d && self.overlay_show_grid_for_axes(axes_index) {
            let view_proj = view_proj_matrix;
            let inv_view_proj = view_proj.inverse();

            let unproject = |ndc_x: f32, ndc_y: f32, ndc_z: f32| -> Option<Vec3> {
                let clip = Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
                let world = inv_view_proj * clip;
                if !world.w.is_finite() || world.w.abs() < 1e-6 {
                    return None;
                }
                let p = world.truncate() / world.w;
                if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
                    Some(p)
                } else {
                    None
                }
            };

            let ray_intersect_z0 = |ndc_x: f32, ndc_y: f32| -> Option<Vec3> {
                // Use a near/far pair in clip space to form a ray.
                let p0 = unproject(ndc_x, ndc_y, -1.0)?;
                let p1 = unproject(ndc_x, ndc_y, 1.0)?;
                let dir = p1 - p0;
                if !dir.z.is_finite() || dir.z.abs() < 1e-8 {
                    return None;
                }
                let t = (-p0.z) / dir.z;
                if !t.is_finite() || t <= 0.0 {
                    return None;
                }
                Some(p0 + dir * t)
            };

            let mut plane_pts: Vec<Vec3> = Vec::new();
            for (nx, ny) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                if let Some(p) = ray_intersect_z0(nx, ny) {
                    plane_pts.push(p);
                }
            }

            // Fallback region if we couldn't intersect enough rays (camera nearly parallel to plane).
            let mut min_x = 0.0_f32;
            let mut max_x = 1.0_f32;
            let mut min_y = 0.0_f32;
            let mut max_y = 1.0_f32;

            if plane_pts.len() >= 2 {
                min_x = plane_pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
                max_x = plane_pts
                    .iter()
                    .map(|p| p.x)
                    .fold(f32::NEG_INFINITY, f32::max);
                min_y = plane_pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
                max_y = plane_pts
                    .iter()
                    .map(|p| p.y)
                    .fold(f32::NEG_INFINITY, f32::max);
            } else if let crate::core::camera::ProjectionType::Perspective { fov, .. } =
                cam.projection
            {
                let dist = (cam.position - cam.target).length().max(1e-3);
                let extent = (dist * (0.5 * fov).tan() * 1.25).max(0.5);
                let center = Vec3::new(cam.target.x, cam.target.y, 0.0);
                min_x = center.x - extent;
                max_x = center.x + extent;
                min_y = center.y - extent;
                max_y = center.y + extent;
            }

            // Expand a bit so grid lines don't pop at edges.
            let dx = (max_x - min_x).abs().max(1e-3);
            let dy = (max_y - min_y).abs().max(1e-3);
            let margin_x = dx * 0.04;
            let margin_y = dy * 0.04;
            min_x -= margin_x;
            max_x += margin_x;
            min_y -= margin_y;
            max_y += margin_y;

            let project_to_px = |p: Vec3| -> Option<(f32, f32)> {
                let clip = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
                if !clip.w.is_finite() || clip.w.abs() < 1e-6 {
                    return None;
                }
                let ndc = clip.truncate() / clip.w;
                if !(ndc.x.is_finite() && ndc.y.is_finite()) {
                    return None;
                }
                let px = ((ndc.x + 1.0) * 0.5) * (sw.max(1) as f32);
                let py = ((1.0 - ndc.y) * 0.5) * (sh.max(1) as f32);
                Some((px, py))
            };

            let nice_step = |raw: f64| -> f64 {
                if !raw.is_finite() || raw <= 0.0 {
                    return 1.0;
                }
                let pow10 = 10.0_f64.powf(raw.log10().floor());
                let norm = raw / pow10;
                let mult = if norm <= 1.0 {
                    1.0
                } else if norm <= 2.0 {
                    2.0
                } else if norm <= 5.0 {
                    5.0
                } else {
                    10.0
                };
                mult * pow10
            };

            // Determine grid scale from projection at the plane center.
            let cx = (min_x + max_x) * 0.5;
            let cy = (min_y + max_y) * 0.5;
            let center = Vec3::new(cx, cy, 0.0);
            let px_per_world = {
                let a = project_to_px(center);
                let b = project_to_px(center + Vec3::new(1.0, 0.0, 0.0));
                match (a, b) {
                    (Some((ax, ay)), Some((bx, by))) => ((bx - ax).hypot(by - ay)).max(1e-3),
                    _ => 1.0,
                }
            };
            let desired_major_px = 120.0_f64;
            let major_step = nice_step((desired_major_px / (px_per_world as f64)).max(1e-6));
            let mut minor_step = major_step / 10.0;
            if !minor_step.is_finite() || minor_step <= 0.0 {
                minor_step = major_step.max(1.0);
            }

            // Cap minor line density to avoid noisy/perf-heavy grids.
            let max_minor_lines = 180.0;
            let minor_count_x = (dx as f64 / minor_step).abs();
            let minor_count_y = (dy as f64 / minor_step).abs();
            if minor_count_x > max_minor_lines || minor_count_y > max_minor_lines {
                minor_step = (major_step / 5.0).max(major_step); // effectively disable minors
            }

            let mut helper_vertices: Vec<Vertex> = Vec::new();
            let mut push_line = |a: Vec3, b: Vec3, color: Vec4| {
                helper_vertices.push(Vertex::new(a, color));
                helper_vertices.push(Vertex::new(b, color));
            };

            // Slightly offset the grid plane to reduce z-fighting with geometry on z=0.
            let z_grid = -1e-4_f32;

            // Procedural XY grid plane (depth-tested, no depth writes). This avoids far-plane
            // popping and keeps line density stable via shader derivatives.
            let show_major_grid = self.overlay_show_grid_for_axes(axes_index);
            let show_minor_grid = self.overlay_show_minor_grid_for_axes(axes_index);
            if show_major_grid || show_minor_grid {
                let theme = self.theme.build_theme();
                let bg = theme.get_background_color();
                let grid = theme.get_grid_color();
                let bg_luma = 0.2126 * bg.x + 0.7152 * bg.y + 0.0722 * bg.z;
                let mut major_rgb = [grid.x, grid.y, grid.z];
                let mut minor_rgb = [grid.x, grid.y, grid.z];
                let mut major_alpha = grid.w.clamp(0.08, 0.22);
                let mut minor_alpha = (grid.w * 0.45).clamp(0.04, 0.14);
                if bg_luma <= 0.62 {
                    major_rgb = [grid.x * 0.80, grid.y * 0.80, grid.z * 0.80];
                    minor_rgb = [grid.x * 0.68, grid.y * 0.68, grid.z * 0.68];
                }
                if bg_luma > 0.62 {
                    major_rgb = [grid.x * 0.45, grid.y * 0.45, grid.z * 0.45];
                    minor_rgb = [grid.x * 0.33, grid.y * 0.33, grid.z * 0.33];
                    major_alpha = major_alpha.max(0.24);
                    minor_alpha = minor_alpha.max(0.12);
                }
                if !show_major_grid {
                    major_alpha = 0.0;
                }
                if !show_minor_grid {
                    minor_alpha = 0.0;
                }
                self.wgpu_renderer.ensure_grid_plane_pipeline();
                self.wgpu_renderer.update_grid_uniforms_for_axes(
                    axes_index,
                    crate::core::renderer::GridUniforms {
                        major_step: major_step as f32,
                        minor_step: minor_step as f32,
                        fade_start: (0.60 * dx.max(dy)).max(major_step as f32),
                        fade_end: (0.95 * dx.max(dy)).max((major_step as f32) * 2.0),
                        camera_pos: cam.position.to_array(),
                        _pad0: 0.0,
                        target_pos: Vec3::new(cam.target.x, cam.target.y, 0.0).to_array(),
                        _pad1: 0.0,
                        major_color: [major_rgb[0], major_rgb[1], major_rgb[2], major_alpha],
                        minor_color: [minor_rgb[0], minor_rgb[1], minor_rgb[2], minor_alpha],
                    },
                );

                let quad_vertices = [
                    Vertex::new(Vec3::new(min_x, min_y, z_grid), Vec4::ONE),
                    Vertex::new(Vec3::new(max_x, min_y, z_grid), Vec4::ONE),
                    Vertex::new(Vec3::new(max_x, max_y, z_grid), Vec4::ONE),
                    Vertex::new(Vec3::new(min_x, max_y, z_grid), Vec4::ONE),
                ];
                let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
                let vb = self.wgpu_renderer.create_vertex_buffer(&quad_vertices);
                let ib = self.wgpu_renderer.create_index_buffer(&quad_indices);
                grid_plane_buffers = Some((vb, ib));
            }

            // World-axis helper for spatial awareness at model scale.
            let axis_len = (major_step as f32 * 5.0).clamp(0.5, (dx.max(dy) * 0.6).max(0.5));
            let origin = Vec3::new(0.0, 0.0, 0.0);
            let col_x = Vec4::new(0.92, 0.25, 0.25, 0.85);
            let col_y = Vec4::new(0.35, 0.90, 0.45, 0.85);
            let col_z = Vec4::new(0.35, 0.62, 0.98, 0.85);
            push_line(origin, origin + Vec3::new(axis_len, 0.0, 0.0), col_x);
            push_line(origin, origin + Vec3::new(0.0, axis_len, 0.0), col_y);
            push_line(origin, origin + Vec3::new(0.0, 0.0, axis_len), col_z);

            // Dynamic tick marks on the origin triad (major step only). Labels are drawn in the
            // overlay so they stay crisp; these marks provide a depth-correct anchor in the scene.
            // NOTE: `f32::clamp` panics if min > max. When zoomed very far in, `major_step` can
            // be tiny, making `major_step * 0.25` smaller than a fixed minimum like 0.01.
            // Keep the min <= max by adapting the minimum to the current step size.
            let tick_max = (major_step as f32 * 0.25).max(1.0e-6);
            let tick_min = 0.01_f32.min(tick_max);
            let tick_len = (axis_len * 0.04).clamp(tick_min, tick_max);
            let max_ticks = 6usize;
            let mut add_ticks = |axis: Vec3, perp: Vec3, col: Vec4| {
                if major_step <= 0.0 {
                    return;
                }
                for i in 1..=max_ticks {
                    let t = (i as f32) * (major_step as f32);
                    if t >= axis_len * 0.999 {
                        break;
                    }
                    let p = origin + axis * t;
                    push_line(
                        p - perp * tick_len,
                        p + perp * tick_len,
                        Vec4::new(col.x, col.y, col.z, col.w * 0.85),
                    );
                }
            };
            add_ticks(Vec3::X, Vec3::Y, col_x);
            add_ticks(Vec3::Y, Vec3::X, col_y);
            add_ticks(Vec3::Z, Vec3::X, col_z);

            if !helper_vertices.is_empty() {
                let rd = Box::new(crate::core::RenderData {
                    pipeline_type: crate::core::PipelineType::Lines,
                    vertices: helper_vertices,
                    indices: None,
                    gpu_vertices: None,
                    bounds: None,
                    material: crate::core::Material::default(),
                    draw_calls: vec![crate::core::DrawCall {
                        vertex_offset: 0,
                        vertex_count: 0, // filled below
                        index_offset: None,
                        index_count: None,
                        instance_count: 1,
                    }],
                    image: None,
                });
                owned_render_data.push(rd);
                let idx = owned_render_data.len() - 1;
                // Fill vertex_count now that vertices are owned.
                let vcount = owned_render_data[idx].vertices.len();
                if let Some(dc) = owned_render_data[idx].draw_calls.get_mut(0) {
                    dc.vertex_count = vcount;
                }
                let vb = Arc::new(
                    self.wgpu_renderer
                        .create_vertex_buffer(&owned_render_data[idx].vertices),
                );
                // Draw helpers first (under data, depth-tested).
                let rd_ref: &crate::core::RenderData = &owned_render_data[idx];
                render_items.insert(0, (rd_ref, vb, None));
                total_vertices += vcount;
            }
        }

        // Precompute expanded point buffers to keep them alive across the render pass
        let mut point_buffers: Vec<Option<(wgpu::Buffer, usize)>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in render_items.iter() {
            if matches!(
                render_data.pipeline_type,
                crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
            ) && !render_data.vertices.is_empty()
            {
                let expanded = self
                    .wgpu_renderer
                    // size_px=0.0 => use per-vertex normal.z sizes
                    .create_direct_point_vertices(&render_data.vertices, 0.0);
                let buf = self.wgpu_renderer.create_vertex_buffer(&expanded);
                point_buffers.push(Some((buf, expanded.len())));
            } else {
                point_buffers.push(None);
            }
        }
        // Precreate image bind groups for textured items to avoid lifetime issues
        let has_textured_items = render_items.iter().any(|(render_data, _vb, _ib)| {
            render_data.pipeline_type == crate::core::PipelineType::Textured
        });
        if has_textured_items {
            self.wgpu_renderer.ensure_image_pipeline();
        }
        let mut image_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());

        for (render_data, _vb, _ib) in render_items.iter() {
            if render_data.pipeline_type == crate::core::PipelineType::Textured {
                if let Some(crate::core::scene::ImageData::Rgba8 {
                    width,
                    height,
                    data,
                }) = &render_data.image
                {
                    let (_t, _v, bg) = self
                        .wgpu_renderer
                        .create_image_texture_and_bind_group(*width, *height, data);
                    image_bind_groups.push(Some(bg));
                } else {
                    image_bind_groups.push(None);
                }
            } else {
                image_bind_groups.push(None);
            }
        }
        // Precreate point style bind groups for points to match pipeline layout [direct uniforms, point style]
        let mut point_style_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in render_items.iter() {
            if matches!(
                render_data.pipeline_type,
                crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
            ) {
                let style = crate::core::renderer::PointStyleUniforms {
                    face_color: render_data.material.albedo.to_array(),
                    edge_color: render_data.material.emissive.to_array(),
                    edge_thickness_px: render_data.material.roughness,
                    marker_shape: render_data.material.metallic as u32,
                    _pad: [0.0, 0.0],
                };
                let (_buf, bg) = self.wgpu_renderer.create_point_style_bind_group(style);
                point_style_bind_groups.push(Some(bg));
            } else {
                point_style_bind_groups.push(None);
            }
        }

        // Precompute optional grid geometry and uniforms so we can draw it under data
        // Grid is drawn only when enabled and in 2D orthographic
        let mut grid_vb_opt: Option<wgpu::Buffer> = None;
        let show_major_grid = self.overlay_show_grid_for_axes(axes_index);
        let show_minor_grid = self.overlay_show_minor_grid_for_axes(axes_index);
        if is_2d && (show_major_grid || show_minor_grid) {
            if let Some((l, r, b, t)) = self.view_bounds_for_axes(axes_index) {
                // Update direct uniforms mapping for viewport
                self.wgpu_renderer.update_direct_uniforms_for_axes(
                    axes_index,
                    [l as f32, b as f32],
                    [r as f32, t as f32],
                    [-1.0, -1.0],
                    [1.0, 1.0],
                    [sw.max(1) as f32, sh.max(1) as f32],
                );
                self.wgpu_renderer.ensure_direct_line_pipeline();

                let x_range = (r - l).max(1e-6);
                let y_range = (t - b).max(1e-6);
                let x_step = plot_utils::calculate_tick_interval(x_range);
                let y_step = plot_utils::calculate_tick_interval(y_range);
                let mut grid_vertices: Vec<Vertex> = Vec::new();
                let g = 80.0_f32 / 255.0_f32;
                let major_col = Vec4::new(g, g, g, 1.0);
                let minor_col = Vec4::new(g, g, g, 0.42);
                if show_minor_grid && x_step.is_finite() && x_step > 0.0 {
                    let minor_step = (x_step / 5.0).max(f64::EPSILON);
                    Self::push_vertical_grid_lines(
                        &mut grid_vertices,
                        (l, r),
                        (b, t),
                        minor_step,
                        minor_col,
                        Some((x_step, minor_step * 0.25)),
                    );
                }
                if show_minor_grid && y_step.is_finite() && y_step > 0.0 {
                    let minor_step = (y_step / 5.0).max(f64::EPSILON);
                    Self::push_horizontal_grid_lines(
                        &mut grid_vertices,
                        (b, t),
                        (l, r),
                        minor_step,
                        minor_col,
                        Some((y_step, minor_step * 0.25)),
                    );
                }
                if show_major_grid && x_step.is_finite() && x_step > 0.0 {
                    Self::push_vertical_grid_lines(
                        &mut grid_vertices,
                        (l, r),
                        (b, t),
                        x_step,
                        major_col,
                        None,
                    );
                }
                if show_major_grid && y_step.is_finite() && y_step > 0.0 {
                    Self::push_horizontal_grid_lines(
                        &mut grid_vertices,
                        (b, t),
                        (l, r),
                        y_step,
                        major_col,
                        None,
                    );
                }
                if !grid_vertices.is_empty() {
                    grid_vb_opt = Some(self.wgpu_renderer.create_vertex_buffer(&grid_vertices));
                }
            }
        }

        // Before the pass: configure direct uniforms and ensure pipelines
        let bounds_opt = if is_2d {
            match cam.projection {
                crate::core::camera::ProjectionType::Orthographic {
                    left,
                    right,
                    bottom,
                    top,
                    ..
                } => Some((left as f64, right as f64, bottom as f64, top as f64)),
                _ => self.data_bounds,
            }
        } else {
            None
        };
        if is_2d {
            if let Some((l, r, b, t)) = bounds_opt {
                self.wgpu_renderer.update_direct_uniforms_for_axes(
                    axes_index,
                    [l as f32, b as f32],
                    [r as f32, t as f32],
                    [-1.0, -1.0],
                    [1.0, 1.0],
                    [sw.max(1) as f32, sh.max(1) as f32],
                );
            }
            self.wgpu_renderer.ensure_direct_triangle_pipeline();
            self.wgpu_renderer.ensure_direct_line_pipeline();
            self.wgpu_renderer.ensure_direct_point_pipeline();
        } else {
            // 3D: ensure camera-based pipelines exist so surfaces rotate with the camera.
            self.wgpu_renderer
                .ensure_pipeline(crate::core::PipelineType::Triangles);
            self.wgpu_renderer
                .ensure_pipeline(crate::core::PipelineType::Lines);
            self.wgpu_renderer
                .ensure_pipeline(crate::core::PipelineType::Points);
        }
        self.wgpu_renderer.update_marker_screen_uniforms_for_axes(
            axes_index,
            [sw.max(1) as f32, sh.max(1) as f32],
        );

        // Begin pass with Load (preserve egui)
        {
            // Prepare MSAA render target if enabled
            let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
            log::debug!(
                "runmat-plot: renderer.camera_to_target_viewport.render_pass_start axes_index={} use_msaa={} clear_background={}",
                axes_index,
                use_msaa,
                clear_background
            );

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Plot Camera Viewport Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view,
                    resolve_target: if use_msaa {
                        target.resolve_target
                    } else {
                        None
                    },
                    ops: wgpu::Operations {
                        load: if clear_background {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: config.background_color.x as f64,
                                g: config.background_color.y as f64,
                                b: config.background_color.z as f64,
                                a: config.background_color.w as f64,
                            })
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view.as_ref(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(match config.depth_mode {
                            DepthMode::Standard => 1.0,
                            DepthMode::ReversedZ => 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Apply viewport and scissor rectangle to draw into the plot rect
            render_pass.set_viewport(
                sx as f32,
                sy as f32,
                sw.max(1) as f32,
                sh.max(1) as f32,
                0.0,
                1.0,
            );
            render_pass.set_scissor_rect(sx, sy, sw.max(1), sh.max(1));
            log::debug!(
                "runmat-plot: renderer.camera_to_target_viewport.render_pass_ready axes_index={} viewport=({}, {}, {}, {})",
                axes_index,
                sx,
                sy,
                sw.max(1),
                sh.max(1)
            );
            if let Some(ref vb_grid) = grid_vb_opt {
                if let Some(ref pipeline) = self.wgpu_renderer.direct_line_pipeline {
                    log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_grid_start axes_index={} vertex_buffer_size={}",
                        axes_index,
                        vb_grid.size()
                    );
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(
                        0,
                        self.wgpu_renderer
                            .get_direct_uniform_bind_group_for_axes(axes_index),
                        &[],
                    );
                    render_pass.set_vertex_buffer(0, vb_grid.slice(..));
                    // Each grid line is two vertices (LineList)
                    // Draw full buffer
                    // Note: vertex count equals number of vertices
                    // wgpu will interpret as lines via pipeline topology
                    render_pass.draw(
                        0..(vb_grid.size() / std::mem::size_of::<Vertex>() as u64) as u32,
                        0..1,
                    );
                    log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_grid_ok axes_index={}",
                        axes_index
                    );
                }
            }

            // Use direct pipelines for precise 2D mapping inside the viewport
            let use_direct_for_triangles = is_2d;
            let use_direct_for_lines = is_2d;
            let direct_tri_pipeline = if use_direct_for_triangles && bounds_opt.is_some() {
                self.wgpu_renderer
                    .direct_triangle_pipeline
                    .as_ref()
                    .map(|p| p as *const wgpu::RenderPipeline)
            } else {
                None
            };
            let direct_line_pipeline = if use_direct_for_lines && bounds_opt.is_some() {
                self.wgpu_renderer
                    .direct_line_pipeline
                    .as_ref()
                    .map(|p| p as *const wgpu::RenderPipeline)
            } else {
                None
            };
            let direct_point_pipeline = if is_2d && bounds_opt.is_some() {
                self.wgpu_renderer
                    .direct_point_pipeline
                    .as_ref()
                    .map(|p| p as *const wgpu::RenderPipeline)
            } else {
                None
            };

            for (idx, (render_data, vertex_buffer, index_buffer)) in render_items.iter().enumerate()
            {
                let is_triangles = matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Triangles
                );
                let is_lines =
                    matches!(render_data.pipeline_type, crate::core::PipelineType::Lines);
                let is_points = matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
                );
                let is_textured = matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Textured
                );
                // Use direct mapping for lines/triangles/points for correct pixel-sized markers in GUI
                let use_direct = is_2d
                    && ((use_direct_for_triangles && is_triangles)
                        || (use_direct_for_lines && is_lines)
                        || is_points)
                    && bounds_opt.is_some();
                if render_item_logs {
                    log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_item_start axes_index={} item_index={} pipeline={:?} use_direct={} textured={} indexed={} draw_calls={} point_buffer={} ",
                        axes_index,
                        idx,
                        render_data.pipeline_type,
                        use_direct,
                        is_textured,
                        index_buffer.is_some(),
                        render_data.draw_calls.len(),
                        point_buffers[idx].is_some()
                    );
                }

                if use_direct {
                    // Safe because we only read pointers here within pass
                    let pipeline_ref: &wgpu::RenderPipeline = unsafe {
                        if is_triangles {
                            direct_tri_pipeline.unwrap().as_ref().unwrap()
                        } else if is_lines {
                            direct_line_pipeline.unwrap().as_ref().unwrap()
                        } else {
                            direct_point_pipeline.unwrap().as_ref().unwrap()
                        }
                    };
                    let uniform_bg = self
                        .wgpu_renderer
                        .get_direct_uniform_bind_group_for_axes(axes_index);
                    render_pass.set_pipeline(pipeline_ref);
                    render_pass.set_bind_group(0, uniform_bg, &[]);
                    if is_points {
                        if let Some(ref bg) = point_style_bind_groups[idx] {
                            render_pass.set_bind_group(1, bg, &[]);
                        }
                    }
                    if render_item_logs {
                        log::debug!(
                            "runmat-plot: renderer.camera_to_target_viewport.draw_item_pipeline_ready axes_index={} item_index={} branch=direct",
                            axes_index,
                            idx
                        );
                    }
                } else if is_textured {
                    let pipeline = self
                        .wgpu_renderer
                        .get_pipeline(crate::core::PipelineType::Textured);
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(
                        0,
                        self.wgpu_renderer
                            .get_direct_uniform_bind_group_for_axes(axes_index),
                        &[],
                    );
                    if let Some(ref bg) = image_bind_groups[idx] {
                        render_pass.set_bind_group(1, bg, &[]);
                    }
                    if render_item_logs {
                        log::debug!(
                            "runmat-plot: renderer.camera_to_target_viewport.draw_item_pipeline_ready axes_index={} item_index={} branch=textured",
                            axes_index,
                            idx
                        );
                    }
                } else {
                    let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(
                        0,
                        self.wgpu_renderer
                            .get_uniform_bind_group_for_axes(axes_index),
                        &[],
                    );
                    if is_points {
                        if let Some(ref bg) = point_style_bind_groups[idx] {
                            render_pass.set_bind_group(1, bg, &[]);
                        }
                        render_pass.set_bind_group(
                            2,
                            self.wgpu_renderer
                                .get_marker_screen_bind_group_for_axes(axes_index),
                            &[],
                        );
                    }
                    if render_item_logs {
                        log::debug!(
                            "runmat-plot: renderer.camera_to_target_viewport.draw_item_pipeline_ready axes_index={} item_index={} branch=standard",
                            axes_index,
                            idx
                        );
                    }
                }

                if is_points && use_direct {
                    if let Some((ref buf, len)) = point_buffers[idx] {
                        render_pass.set_vertex_buffer(0, buf.slice(..));
                        render_pass.draw(0..len as u32, 0..1);
                        if render_item_logs {
                            log::debug!(
                                "runmat-plot: renderer.camera_to_target_viewport.draw_item_ok axes_index={} item_index={} mode=direct_points vertices={}",
                                axes_index,
                                idx,
                                len
                            );
                        }
                        continue;
                    }
                } else if is_points {
                    if let Some((ref buf, _len)) = point_buffers[idx] {
                        render_pass.set_vertex_buffer(0, buf.slice(..));
                    } else {
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    }
                } else {
                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                }
                if let Some(index_buffer_ref) = index_buffer {
                    render_pass
                        .set_index_buffer(index_buffer_ref.slice(..), wgpu::IndexFormat::Uint32);
                    if let Some(indices) = &render_data.indices {
                        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                        if render_item_logs {
                            log::debug!(
                                "runmat-plot: renderer.camera_to_target_viewport.draw_item_ok axes_index={} item_index={} mode=indexed indices={}",
                                axes_index,
                                idx,
                                indices.len()
                            );
                        }
                    }
                } else {
                    if is_points {
                        if let Some((_, len)) = point_buffers[idx] {
                            render_pass.draw(0..len as u32, 0..1);
                            continue;
                        }
                    }
                    for dc in &render_data.draw_calls {
                        render_pass.draw(
                            dc.vertex_offset as u32..(dc.vertex_offset + dc.vertex_count) as u32,
                            0..dc.instance_count as u32,
                        );
                        if render_item_logs {
                            log::debug!(
                                "runmat-plot: renderer.camera_to_target_viewport.draw_call_ok axes_index={} item_index={} mode=draw vertex_offset={} vertex_count={} instances={}",
                                axes_index,
                                idx,
                                dc.vertex_offset,
                                dc.vertex_count,
                                dc.instance_count
                            );
                        }
                    }
                }
            }

            // Draw procedural 3D grid plane after data, depth-tested (no depth writes).
            if let Some((ref vb, ref ib)) = grid_plane_buffers {
                if let Some(pipeline) = self.wgpu_renderer.grid_plane_pipeline() {
                    log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_grid_plane_start axes_index={}",
                        axes_index
                    );
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(
                        0,
                        self.wgpu_renderer
                            .get_uniform_bind_group_for_axes(axes_index),
                        &[],
                    );
                    render_pass.set_bind_group(
                        1,
                        self.wgpu_renderer
                            .get_grid_uniform_bind_group_for_axes(axes_index),
                        &[],
                    );
                    render_pass.set_vertex_buffer(0, vb.slice(..));
                    render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                    log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_grid_plane_ok axes_index={}",
                        axes_index
                    );
                }
            }
        }

        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.ok axes_index={} total_vertices={} total_triangles={}",
            axes_index,
            total_vertices,
            total_triangles
        );

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            axes_viewports_px: vec![(sx, sy, sw, sh)],
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Render all axes of a subplot grid into their respective viewport rectangles.
    /// `axes_viewports` is a vector of (x, y, w, h) in physical pixels, length equals rows*cols.
    pub fn render_axes_to_viewports(
        &mut self,
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
            .ensure_axes_uniform_capacity(axes_viewports.len().max(1));

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
            Some(self.wgpu_renderer.ensure_msaa_color_view())
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
    fn create_default_camera() -> Camera {
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

    /// Overlay configuration getters
    pub fn overlay_show_grid(&self) -> bool {
        self.figure_show_grid
    }

    pub fn set_overlay_grid_enabled(&mut self, enabled: bool) {
        self.figure_show_grid = enabled;
        self.needs_update = true;
    }

    pub fn overlay_show_minor_grid(&self) -> bool {
        self.figure_show_minor_grid
    }
    pub fn overlay_show_grid_for_axes(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.grid_enabled)
            .unwrap_or(self.figure_show_grid)
    }
    pub fn overlay_show_minor_grid_for_axes(&self, axes_index: usize) -> bool {
        Self::minor_grid_for_axes(
            self.last_figure.as_ref(),
            self.figure_show_minor_grid,
            axes_index,
        )
    }

    fn minor_grid_for_axes(
        last_figure: Option<&crate::plots::Figure>,
        figure_show_minor_grid: bool,
        axes_index: usize,
    ) -> bool {
        last_figure
            .map(|f| f.minor_grid_enabled_for_axes(axes_index))
            .unwrap_or(figure_show_minor_grid)
    }

    fn push_vertical_grid_lines(
        vertices: &mut Vec<Vertex>,
        x_bounds: (f64, f64),
        y_bounds: (f64, f64),
        step: f64,
        color: Vec4,
        skip_major: Option<(f64, f64)>,
    ) {
        Self::for_each_grid_position(x_bounds.0, x_bounds.1, step, skip_major, |x| {
            let x = x as f32;
            let y0 = y_bounds.0 as f32;
            let y1 = y_bounds.1 as f32;
            if x.is_finite() && y0.is_finite() && y1.is_finite() {
                vertices.push(Vertex::new(Vec3::new(x, y0, 0.0), color));
                vertices.push(Vertex::new(Vec3::new(x, y1, 0.0), color));
            }
        });
    }

    fn push_horizontal_grid_lines(
        vertices: &mut Vec<Vertex>,
        y_bounds: (f64, f64),
        x_bounds: (f64, f64),
        step: f64,
        color: Vec4,
        skip_major: Option<(f64, f64)>,
    ) {
        Self::for_each_grid_position(y_bounds.0, y_bounds.1, step, skip_major, |y| {
            let y = y as f32;
            let x0 = x_bounds.0 as f32;
            let x1 = x_bounds.1 as f32;
            if y.is_finite() && x0.is_finite() && x1.is_finite() {
                vertices.push(Vertex::new(Vec3::new(x0, y, 0.0), color));
                vertices.push(Vertex::new(Vec3::new(x1, y, 0.0), color));
            }
        });
    }

    fn for_each_grid_position(
        min: f64,
        max: f64,
        step: f64,
        skip_major: Option<(f64, f64)>,
        mut visit: impl FnMut(f64),
    ) {
        if !min.is_finite() || !max.is_finite() || !step.is_finite() || step <= 0.0 || min > max {
            return;
        }
        let start = (min / step).ceil() * step;
        if !start.is_finite() {
            return;
        }

        let mut value = start;
        for _ in 0..MAX_2D_GRID_LINES_PER_AXIS {
            if value > max {
                break;
            }
            let is_major = skip_major
                .map(|(major_step, tolerance)| {
                    major_step.is_finite()
                        && major_step > 0.0
                        && (value - (value / major_step).round() * major_step).abs() <= tolerance
                })
                .unwrap_or(false);
            if !is_major {
                visit(value);
            }

            let next = value + step;
            if !next.is_finite() || next <= value {
                break;
            }
            value = next;
        }
    }
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
    pub fn geometry_xray_enabled(&self) -> bool {
        self.geometry_xray_enabled
    }
    pub fn geometry_scene_presentation(&self) -> &GeometryScenePresentation {
        &self.geometry_presentation
    }
    pub fn set_geometry_scene_presentation(&mut self, presentation: GeometryScenePresentation) {
        if self.geometry_presentation == presentation {
            return;
        }
        self.geometry_presentation = presentation;
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
        self.apply_geometry_visibility_filter();
        self.needs_update = true;
    }
    fn refresh_geometry_scene_render_data(&mut self, geometry_scene: &GeometryScene) {
        for (index, chunk) in geometry_scene.chunks.iter().enumerate() {
            let node_id = geometry_scene.chunk_node_id(index, &chunk.chunk_id);
            if let Some(node) = self.scene.get_node_mut(node_id) {
                node.render_data =
                    Some(chunk.render_data_with_presentation(&self.geometry_presentation));
                node.bounds = chunk.bounds;
                node.visible = chunk.visible;
            }
        }
        self.apply_geometry_xray_to_nodes();
        self.apply_geometry_visibility_filter();
        self.scene_buffer_cache.borrow_mut().clear();
        self.needs_update = true;
    }
    fn apply_geometry_visibility_filter(&mut self) {
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
    fn apply_geometry_xray_to_nodes(&mut self) {
        let alpha = if self.geometry_xray_enabled {
            0.38
        } else {
            1.0
        };
        self.scene.for_each_node_mut(|node| {
            if let Some(render_data) = node.render_data.as_mut() {
                if matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Triangles
                ) {
                    render_data.material.albedo.w = render_data.material.albedo.w.min(alpha);
                    render_data.material.alpha_mode = if render_data.material.albedo.w < 0.98 {
                        crate::core::AlphaMode::Blend
                    } else {
                        crate::core::AlphaMode::Opaque
                    };
                    for vertex in &mut render_data.vertices {
                        vertex.color[3] = vertex.color[3].min(alpha);
                    }
                }
            }
        });
    }
    pub fn overlay_sg_title(&self) -> Option<&String> {
        self.figure_sg_title.as_ref()
    }
    pub fn overlay_sg_title_style(&self) -> &TextStyle {
        &self.figure_sg_title_style
    }
    pub fn overlay_title_for_axes(&self, axes_index: usize) -> Option<&String> {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .and_then(|m| m.title.as_ref())
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
        self.figure_colormap
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
        fig.colormap = self.figure_colormap;
        fig.colorbar_enabled = self.figure_colorbar_enabled;
        let (rows, cols) = self.figure_axes_grid();
        fig.set_subplot_grid(rows, cols);
        fig
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plots::{figure::PlotElement, Figure, PatchPlot};

    fn patch_element(vertices: Vec<Vec3>) -> PlotElement {
        PlotElement::Patch(PatchPlot::new(vertices, vec![vec![0, 1, 2]]).unwrap())
    }

    #[test]
    fn patch_3d_detection_preserves_small_scene_depth() {
        let patch = patch_element(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0e-4, 0.0, 0.0),
            Vec3::new(0.0, 1.0e-4, 1.0e-8),
        ]);

        assert!(PlotRenderer::plot_element_is_3d(&patch));
    }

    #[test]
    fn patch_3d_detection_ignores_large_scene_relative_noise() {
        let patch = patch_element(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0e6, 0.0, 0.0),
            Vec3::new(0.0, 1.0e6, 0.5),
        ]);

        assert!(!PlotRenderer::plot_element_is_3d(&patch));
    }

    #[test]
    fn applies_z_limits_to_3d_display_bounds() {
        let mut figure = Figure::new();
        figure.set_axes_z_limits(0, Some((-2.0, 3.0)));
        let bounds = BoundingBox::new(Vec3::new(-1.0, -1.0, -10.0), Vec3::new(1.0, 1.0, 10.0));

        let limited = PlotRenderer::apply_3d_display_limits_to_bounds(bounds, Some(&figure), 0);

        assert_eq!(limited.min.z, -2.0);
        assert_eq!(limited.max.z, 3.0);
        assert_eq!(limited.min.x, -1.0);
        assert_eq!(limited.max.x, 1.0);
    }

    #[test]
    fn degenerate_z_bounds_remain_finite_for_3d_display_bounds() {
        let figure = Figure::new();
        let bounds = BoundingBox::new(Vec3::new(-2.0, -2.0, 0.0), Vec3::new(2.0, 2.0, 0.0));

        let display = PlotRenderer::apply_3d_display_limits_to_bounds(bounds, Some(&figure), 0);

        assert!(PlotRenderer::bounds_are_finite(display));
        assert_eq!(display.min.z, 0.0);
        assert_eq!(display.max.z, 0.0);
    }

    #[test]
    fn axes_view_contract_tracks_subplot_limits() {
        let mut base = Figure::new();
        base.set_subplot_grid(2, 1);
        let mut limited = base.clone();
        limited.set_axes_limits(0, Some((0.0, 30.0)), None);
        limited.set_axes_limits(1, Some((200.0, 450.0)), None);

        let base_contract = PlotRenderer::axes_view_contract_for_figure(&base);
        let limited_contract = PlotRenderer::axes_view_contract_for_figure(&limited);

        assert_ne!(base_contract, limited_contract);
        assert_eq!(limited_contract.axes[0].x_limits, Some((0.0, 30.0)));
        assert_eq!(limited_contract.axes[1].x_limits, Some((200.0, 450.0)));
    }

    #[test]
    fn figure_level_minor_grid_survives_default_axes_metadata() {
        let mut figure = Figure::new();
        figure.minor_grid_enabled = true;

        assert!(!figure.axes_metadata(0).unwrap().minor_grid_enabled);
        assert!(PlotRenderer::minor_grid_for_axes(
            Some(&figure),
            figure.minor_grid_enabled,
            0
        ));
    }

    #[test]
    fn explicit_axes_minor_grid_false_overrides_figure_level_minor_grid() {
        let mut figure = Figure::new();
        figure.minor_grid_enabled = true;
        figure.set_axes_minor_grid_enabled(0, false);

        let meta = figure.axes_metadata(0).unwrap();
        assert!(!meta.minor_grid_enabled);
        assert!(meta.minor_grid_explicit);
        assert!(!PlotRenderer::minor_grid_for_axes(
            Some(&figure),
            figure.minor_grid_enabled,
            0
        ));
    }
}

/// High-level plotting utilities that use the unified renderer
pub mod plot_utils {
    pub fn generate_major_ticks(min: f64, max: f64) -> Vec<f64> {
        if !(min.is_finite() && max.is_finite()) || max <= min {
            return Vec::new();
        }
        let range = (max - min).max(1e-9);
        let step = calculate_tick_interval(range);
        if !(step.is_finite() && step > 0.0) {
            return Vec::new();
        }

        let mut ticks = Vec::new();
        let mut value = (min / step).ceil() * step;
        let epsilon = range * 1e-6 + step * 1e-6;
        while value <= max + epsilon {
            let snapped = if value.abs() < epsilon { 0.0 } else { value };
            ticks.push(snapped);
            value += step;
            if ticks.len() > 64 {
                break;
            }
        }

        let endpoint_tol = step * 0.18;
        let near_min = ticks.iter().any(|t| (*t - min).abs() <= endpoint_tol);
        let near_max = ticks.iter().any(|t| (*t - max).abs() <= endpoint_tol);
        if !near_min {
            ticks.insert(0, min);
        }
        if !near_max {
            ticks.push(max);
        }

        ticks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ticks.dedup_by(|a, b| (*a - *b).abs() <= endpoint_tol * 0.5);
        ticks
    }

    /// Calculate nice tick intervals for axis labeling
    pub fn calculate_tick_interval(range: f64) -> f64 {
        let magnitude = 10.0_f64.powf(range.log10().floor());
        let normalized = range / magnitude;

        let nice_interval = if normalized <= 1.0 {
            0.2
        } else if normalized <= 2.0 {
            0.5
        } else if normalized <= 5.0 {
            1.0
        } else {
            2.0
        };

        nice_interval * magnitude
    }

    /// Format a tick label value for display
    pub fn format_tick_label(value: f64) -> String {
        fn trim_fixed(mut s: String) -> String {
            if s.contains('.') {
                while s.ends_with('0') {
                    s.pop();
                }
                if s.ends_with('.') {
                    s.pop();
                }
            }
            if s == "-0" {
                "0".to_string()
            } else {
                s
            }
        }

        if value.abs() < 0.001 {
            "0".to_string()
        } else if value.abs() >= 1000.0 || value.fract().abs() < 0.0005 {
            format!("{value:.0}")
        } else if value.abs() < 0.1 {
            trim_fixed(format!("{value:.3}"))
        } else if value.abs() < 10.0 {
            trim_fixed(format!("{value:.2}"))
        } else {
            trim_fixed(format!("{value:.1}"))
        }
    }

    /// Generate grid lines for plotting
    pub fn generate_grid_lines(
        bounds: (f64, f64, f64, f64),
        plot_rect: (f32, f32, f32, f32), // (left, right, bottom, top)
    ) -> Vec<(f32, f32, f32, f32)> {
        // Vector of (x1, y1, x2, y2) line segments
        let (x_min, x_max, y_min, y_max) = bounds;
        let (left, right, bottom, top) = plot_rect;

        let mut lines = Vec::new();

        // X-axis grid lines
        let x_range = x_max - x_min;
        let x_interval = calculate_tick_interval(x_range);
        let mut x_val = (x_min / x_interval).ceil() * x_interval;

        while x_val <= x_max {
            let x_screen = left + ((x_val - x_min) / x_range) as f32 * (right - left);
            lines.push((x_screen, bottom, x_screen, top));
            x_val += x_interval;
        }

        // Y-axis grid lines
        let y_range = y_max - y_min;
        let y_interval = calculate_tick_interval(y_range);
        let mut y_val = (y_min / y_interval).ceil() * y_interval;

        while y_val <= y_max {
            let y_screen = bottom + ((y_val - y_min) / y_range) as f32 * (top - bottom);
            lines.push((left, y_screen, right, y_screen));
            y_val += y_interval;
        }

        lines
    }
}
