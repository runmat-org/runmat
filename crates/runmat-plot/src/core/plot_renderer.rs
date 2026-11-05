//! Unified plot rendering pipeline for both interactive GUI and static export
//!
//! This module provides the core rendering logic that is shared between
//! interactive plotting windows and static file exports, ensuring consistent
//! high-quality output across all use cases.

use crate::core::renderer::{DirectUniforms, Vertex};
use crate::core::{
    BoundingBox, Camera, CameraViewPreset, ClipPolicy, DepthMode, Scene, WgpuRenderer,
};
#[cfg(feature = "egui-overlay")]
use crate::geometry_scene::GeometrySceneAnnotationKind;
use crate::geometry_scene::{
    GeometryScene, GeometrySceneCacheKey, GeometrySceneOverlay, GeometryScenePresentation,
    GeometrySceneViewPreset,
};
use crate::plots::figure::{LegendEntry, TextStyle};
use crate::plots::surface::ColorMap;
use crate::plots::{AxesKind, Figure};
use glam::{Mat4, Vec3, Vec4};
use runmat_time::Instant;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

type ViewBounds2D = (f64, f64, f64, f64);
type PerAxesViewBounds = Vec<Option<ViewBounds2D>>;
#[path = "plot_renderer/figure_camera_fit.rs"]
mod figure_camera_fit;
#[path = "plot_renderer/y_direction.rs"]
mod y_direction;
use figure_camera_fit::FigureCameraFitState;

fn indexed_draw_ranges(
    draw_calls: &[crate::core::DrawCall],
    available_indices: usize,
) -> Vec<(Range<u32>, Range<u32>)> {
    draw_calls
        .iter()
        .filter_map(|call| {
            let start = call.index_offset.unwrap_or(0).min(available_indices);
            let count = call.index_count.unwrap_or(available_indices - start);
            let end = start.saturating_add(count).min(available_indices);
            (start < end && call.instance_count > 0)
                .then_some((start as u32..end as u32, 0..call.instance_count as u32))
        })
        .collect()
}

#[derive(Clone, Debug)]
struct CachedSceneBuffers {
    vertex_signature: (usize, usize),
    vertex_buffer: runmat_accelerate_api::WgpuBufferOwner,
    index_signature: Option<(usize, usize)>,
    index_buffer: Option<runmat_accelerate_api::WgpuBufferOwner>,
}

#[cfg(feature = "egui-overlay")]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GeometryScreenAnnotation {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub kind: GeometrySceneAnnotationKind,
    pub label: String,
    /// Projected vector extent reserved for the actual arrow, even without a caption.
    pub vector_extent: Option<[[f32; 2]; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
struct AxesViewContract {
    rows: usize,
    cols: usize,
    axes: Vec<AxesViewContractEntry>,
}

#[derive(Clone, Debug, PartialEq)]
struct AxesViewContractEntry {
    y_direction: crate::plots::figure::YDirection,
    has_3d_content: bool,
    axes_kind: AxesKind,
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
    presence: crate::collaboration::PresenceFrame,
    geometry_xray_enabled: bool,
    geometry_node_owner_ids: HashMap<u64, Vec<String>>,
    geometry_hidden_owner_node_ids: BTreeSet<String>,

    /// Last surface extent (in pixels) that was used to build viewport-dependent geometry.
    /// Used so we can rebuild the scene after the canvas is resized (common on wasm).
    last_scene_viewport_px: Option<(u32, u32)>,
    /// Per-axes tick step and length of each 3-D origin axis, as last drawn, so the overlay
    /// labels the same ticks.
    world_axis_triads: std::collections::HashMap<usize, ([f64; 3], [f32; 3])>,
    /// Last per-axes plot viewport sizes used to build viewport-dependent geometry.
    last_axes_plot_sizes_px: Option<Vec<(u32, u32)>>,
    /// Last per-axes orthographic view bounds used for viewport-dependent 2D stroke geometry.
    last_axes_view_bounds: Option<PerAxesViewBounds>,
    /// Last figure view contract used to decide whether script-owned axes state changed.
    last_axes_view_contract: Option<AxesViewContract>,

    /// If false, do not auto-fit camera when the figure updates (user has interacted).
    camera_auto_fit: bool,
    /// Automatically fitted Figure cameras follow viewport size until user or host control.
    figure_camera_fit: Vec<FigureCameraFitState>,
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
    pub fn on_surface_config_updated(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) {
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
        self.set_figure(emission, figure);
    }

    fn prepare_buffers_for_render_data(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        node_id: u64,
        render_data: &crate::core::RenderData,
    ) -> Option<(
        runmat_accelerate_api::WgpuBufferOwner,
        Option<runmat_accelerate_api::WgpuBufferOwner>,
    )> {
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

        let vertex_buffer = self.wgpu_renderer.vertex_buffer_from_sources(
            emission,
            render_data.gpu_vertices.as_ref(),
            &render_data.vertices,
        )?;
        let index_buffer = render_data.indices.as_ref().map(|indices| {
            runmat_accelerate_api::WgpuBufferOwner::new(
                self.wgpu_renderer.create_index_buffer(emission, indices),
            )
        });

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
        access: runmat_accelerate_api::WgpuDeviceAccess,
        surface_config: wgpu::SurfaceConfiguration,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let wgpu_renderer =
            access.with_shared(|emission| WgpuRenderer::new(emission, surface_config))?;
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
            presence: crate::collaboration::PresenceFrame::default(),
            geometry_xray_enabled: false,
            geometry_node_owner_ids: HashMap::new(),
            geometry_hidden_owner_node_ids: BTreeSet::new(),
            last_scene_viewport_px: None,
            world_axis_triads: std::collections::HashMap::new(),
            last_axes_plot_sizes_px: None,
            last_axes_view_bounds: None,
            last_axes_view_contract: None,
            camera_auto_fit: true,
            figure_camera_fit: vec![FigureCameraFitState::Inactive],
            axes_2d_camera_user_controlled: vec![false],
            axes_applied_view_revisions: vec![None],
            scene_buffer_cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn axes_has_3d_content(&self, axes_index: usize) -> bool {
        // Geometry scenes own one spatial axes without a backing Figure. Camera
        // fitting, interaction, and overlays must use the same 3D classification.
        if self.last_geometry_scene.is_some() {
            return axes_index == 0;
        }
        self.last_figure
            .as_ref()
            .map(|figure| {
                figure
                    .plots()
                    .zip(figure.plot_axes_indices().iter().copied())
                    .any(|(plot, plot_axes_index)| {
                        plot_axes_index == axes_index && plot.is_3d_content()
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
            if axes_index < axes_count && plot.is_3d_content() {
                has_3d_content[axes_index] = true;
            }
        }
        let axes = (0..axes_count)
            .map(|axes_index| {
                let meta = figure.axes_metadata(axes_index);
                AxesViewContractEntry {
                    y_direction: meta.map(|m| m.y_direction).unwrap_or_default(),
                    has_3d_content: has_3d_content[axes_index],
                    axes_kind: meta.map(|m| m.axes_kind).unwrap_or(AxesKind::Cartesian),
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
        self.figure_camera_fit.fill(FigureCameraFitState::Inactive);
    }

    /// An explicit camera override owns its view; an initial Figure fit must not replace it.
    pub fn cancel_figure_viewport_fit(&mut self, axes_index: usize) {
        if let Some(state) = self.figure_camera_fit.get_mut(axes_index) {
            *state = FigureCameraFitState::Inactive;
        }
    }

    pub fn fit_pending_figure_camera_for_viewport(
        &mut self,
        axes_index: usize,
        viewport: (u32, u32),
    ) {
        if !self
            .figure_camera_fit
            .get(axes_index)
            .copied()
            .unwrap_or_default()
            .pending_for(viewport)
            || !self.axes_has_3d_content(axes_index)
        {
            return;
        }
        let Some(bounds) = self.display_bounds_3d_for_axes(axes_index) else {
            return;
        };
        if let Some(camera) = self.axes_cameras.get_mut(axes_index) {
            if camera.fit_figure_perspective_bounds(bounds.min, bounds.max, viewport) {
                if let Some(state) = self.figure_camera_fit.get_mut(axes_index) {
                    state.fitted(viewport);
                }
            }
        }
    }

    pub fn fit_pending_figure_cameras_for_viewports(&mut self, viewports: &[(u32, u32)]) {
        for (axes_index, viewport) in viewports.iter().copied().enumerate() {
            self.fit_pending_figure_camera_for_viewport(axes_index, viewport);
        }
    }

    /// Non-overlay Figure rendering uses the same per-axes tile extents as scene drawing.
    pub fn fit_pending_figure_cameras_for_surface(&mut self, width: u32, height: u32) {
        let viewports = self.compute_axes_viewports_px(width.max(1), height.max(1));
        for (axes_index, (_, _, tile_width, tile_height)) in viewports.into_iter().enumerate() {
            self.fit_pending_figure_camera_for_viewport(axes_index, (tile_width, tile_height));
        }
    }

    fn arm_figure_camera_fits(&mut self) {
        self.figure_camera_fit
            .resize(self.axes_cameras.len(), FigureCameraFitState::Inactive);
        for axes_index in 0..self.axes_cameras.len() {
            self.figure_camera_fit[axes_index] = if self.axes_has_3d_content(axes_index) {
                FigureCameraFitState::Auto {
                    last_viewport: None,
                }
            } else {
                FigureCameraFitState::Inactive
            };
        }
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
        let geometry_xray_enabled = geometry_scene.chunks.iter().any(|chunk| {
            chunk.render_data.pipeline_type == crate::core::PipelineType::Triangles
                && (chunk.material.albedo.w < 0.95
                    || matches!(chunk.material.alpha_mode, crate::core::AlphaMode::Blend))
        });
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
    fn add_figure_to_scene(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        figure: Figure,
    ) {
        self.add_figure_to_scene_with_axes_plot_sizes(emission, figure, None);
    }

    fn add_figure_to_scene_with_axes_plot_sizes(
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

    fn axes_view_bounds_for_count(&self, axes_count: usize) -> PerAxesViewBounds {
        (0..axes_count)
            .map(|idx| self.view_bounds_for_axes(idx))
            .collect()
    }

    fn direct_log_flags_for_axes(&self, axes_index: usize) -> [u32; 2] {
        [
            u32::from(self.overlay_x_log_for_axes(axes_index)),
            u32::from(self.overlay_y_log_for_axes(axes_index)),
        ]
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

    fn apply_3d_display_limits_to_bounds(
        bounds: BoundingBox,
        figure: Option<&Figure>,
        axes_index: usize,
    ) -> BoundingBox {
        crate::plots::figure::display_bounds_3d(
            bounds,
            figure.and_then(|fig| fig.axes_metadata(axes_index)),
        )
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
    fn axes_3d_axis_scale(&self, axes_index: usize, bounds: BoundingBox) -> Vec3 {
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
        let fixed_aspect =
            meta.is_some_and(|m| m.axis_equal || m.data_aspect_ratio_mode == "manual");
        if fixed_aspect || has_geometry_mesh {
            return Vec3::splat(1.0 / largest);
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

    fn fit_cameras_to_axes_data(&mut self) -> bool {
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
                cam.position = cam.target + Vec3::new(1.0, -1.0, 1.0);
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
            cam.axis_scale = self.axes_3d_axis_scale(0, bounds);
            cam.target = center * cam.axis_scale;
            cam.up = Vec3::Z;
            cam.position = cam.target + Vec3::new(1.0, -1.0, 1.0);
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
        if self.last_figure.is_some() {
            self.arm_figure_camera_fits();
        }
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

        if self.last_figure.is_some() {
            self.arm_figure_camera_fits();
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
            let center = (bounds.min + bounds.max) * 0.5 * camera.axis_scale;
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
        let data_centers: Vec<Option<Vec3>> = (0..self.axes_cameras.len())
            .map(|idx| {
                if self.axes_has_3d_content(idx) {
                    self.display_bounds_3d_for_axes(idx)
                } else {
                    self.axes_bounds(idx)
                }
                .map(|b| (b.min + b.max) * 0.5)
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
                c.reset_perspective_orientation(data_centers[idx]);
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
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
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
                    emission,
                    render_data.gpu_vertices.as_ref(),
                    &render_data.vertices,
                ) {
                    self.wgpu_renderer
                        .ensure_pipeline(emission, render_data.pipeline_type);

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
            .configure_cad_lighting(cam.view_matrix(), self.last_geometry_scene.is_some());

        let model_matrix = self.axes_model_matrix(0);
        self.wgpu_renderer
            .update_uniforms(emission, view_proj_matrix, model_matrix);

        // Create render pass (respect MSAA)
        let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
        let msaa_view_opt = if use_msaa {
            let tex = emission.device().create_texture(&wgpu::TextureDescriptor {
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
        let direction = self.overlay_y_direction_for_axes(0);
        let (ndc_bottom, ndc_top) = (
            direction.orient(ndc_bottom, ndc_bottom, ndc_top),
            direction.orient(ndc_top, ndc_bottom, ndc_top),
        );

        // data_bounds passed in from caller: (x_min, y_min, x_max, y_max)
        let (x_min, y_min, x_max, y_max) = (0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64);
        self.wgpu_renderer.update_direct_uniforms(
            emission,
            DirectUniforms::new(
                [x_min as f32, y_min as f32],
                [x_max as f32, y_max as f32],
                [ndc_left, ndc_bottom],
                [ndc_right, ndc_top],
                [sw, sh],
                [0, 0],
            ),
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
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
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
        self.wgpu_renderer
            .configure_cad_lighting(cam.view_matrix(), self.last_geometry_scene.is_some());
        let model_matrix = self.axes_model_matrix(0);
        self.wgpu_renderer
            .update_uniforms(emission, view_proj_matrix, model_matrix);

        // Collect all render data and create vertex buffers first (outside render pass)
        let mut render_items = Vec::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some((vertex_buffer, index_buffer)) =
                    self.prepare_buffers_for_render_data(emission, node.id, render_data)
                {
                    self.wgpu_renderer
                        .ensure_pipeline(emission, render_data.pipeline_type);
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
            self.wgpu_renderer.ensure_image_pipeline(emission);
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
            let (ndc_y_min, ndc_y_max) = self.overlay_y_direction_for_axes(0).ndc_endpoints();
            self.wgpu_renderer.update_direct_uniforms(
                emission,
                DirectUniforms::new(
                    [x_min as f32, y_min as f32],
                    [x_max as f32, y_max as f32],
                    [-1.0, ndc_y_min],
                    [1.0, ndc_y_max],
                    [config.width as f32, config.height as f32],
                    [0, 0],
                ),
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
                        .create_image_texture_and_bind_group(emission, *width, *height, data);
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
                let (_buf, bg) = self
                    .wgpu_renderer
                    .create_point_style_bind_group(emission, style);
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
                let buffer = self.wgpu_renderer.create_vertex_buffer(emission, &expanded);
                point_buffers.push(Some((buffer, expanded.len())));
            } else {
                point_buffers.push(None);
            }
        }
        self.wgpu_renderer.update_marker_screen_uniforms(
            emission,
            [config.width.max(1) as f32, config.height.max(1) as f32],
        );

        // Create render pass
        {
            let depth_view = self.wgpu_renderer.ensure_depth_view(emission);
            let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
            let mut cached_msaa_view: Option<Arc<wgpu::TextureView>> = None;

            let (color_view, resolve_target) = if use_msaa {
                if let Some(explicit_resolve_target) = target.resolve_target {
                    (target.view, Some(explicit_resolve_target))
                } else {
                    cached_msaa_view = Some(self.wgpu_renderer.ensure_msaa_color_view(emission));
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
                    } else if render_data.pipeline_type
                        == crate::core::PipelineType::AnnotationStroke
                    {
                        render_pass.set_bind_group(
                            1,
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
                        for (index_range, instance_range) in
                            indexed_draw_ranges(&render_data.draw_calls, indices.len())
                        {
                            log::trace!(target: "runmat_plot", "draw indexed count={}", index_range.len());
                            render_pass.draw_indexed(index_range, 0, instance_range);
                        }
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
            // Drop render_pass at end of scope.
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
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        config: &PlotRenderConfig,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let (rows, cols) = self.figure_axes_grid();
        let axes_count = self.figure_axes_count();
        log::debug!(
            "runmat-plot: renderer.scene_to_target.start rows={} cols={} axes_count={} width={} height={}",
            rows,
            cols,
            axes_count,
            config.width,
            config.height
        );
        if axes_count <= 1
            && !self.axes_position_explicit_for_axes(0)
            && self.axes_grid_cell_for_axes(0).is_none()
        {
            log::debug!("runmat-plot: renderer.scene_to_target.branch_single_axes");
            return self.render(
                emission,
                encoder,
                RenderTarget {
                    view: target_view,
                    resolve_target: None,
                },
                config,
            );
        }

        let viewports = self.compute_axes_viewports_px(config.width.max(1), config.height.max(1));
        log::debug!(
            "runmat-plot: renderer.scene_to_target.branch_subplot_axes viewports={}",
            viewports.len()
        );
        self.render_axes_to_viewports(
            emission,
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

    pub fn compute_axes_viewports_px(
        &self,
        total_width: u32,
        total_height: u32,
    ) -> Vec<(u32, u32, u32, u32)> {
        self.last_figure
            .as_ref()
            .map(|figure| {
                crate::core::axes_layout::figure_viewports_px(figure, total_width, total_height)
            })
            .unwrap_or_else(|| vec![(0, 0, total_width.max(1), total_height.max(1))])
    }

    /// Render using the camera-based pipeline into a viewport region with a scissor rectangle.
    /// This preserves existing contents (Load) and draws only inside the viewport rectangle.
    #[allow(clippy::too_many_arguments)]
    pub fn render_camera_to_viewport(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
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
            Some(self.wgpu_renderer.ensure_msaa_color_view(emission))
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
            emission,
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
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
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
        let depth_view = self.wgpu_renderer.ensure_depth_view(emission);

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
        self.wgpu_renderer
            .configure_cad_lighting(cam.view_matrix(), self.last_geometry_scene.is_some());
        let model_matrix = self.axes_model_matrix(axes_index);
        self.wgpu_renderer.update_uniforms_for_axes(
            emission,
            axes_index,
            view_proj_matrix,
            model_matrix,
        );
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
                if let Some((vb, ib)) =
                    self.prepare_buffers_for_render_data(emission, node.id, render_data)
                {
                    self.wgpu_renderer
                        .ensure_pipeline(emission, render_data.pipeline_type);
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

        // 3D helpers: a horizontal world grid and world-axis ticks. Keeping the
        // grid on XY gives it a stable meaning as the camera moves.
        if !is_2d && self.overlay_show_grid_for_axes(axes_index) {
            let view_proj = view_proj_matrix * model_matrix;
            let inv_view_proj = view_proj.inverse();
            // The camera lives in axis-scaled space (3-D axes stretch to a cube); this helper is
            // built in data units, so read the camera's target and position back in data units.
            let axis_scale = cam.axis_scale.max(Vec3::splat(1.0e-12));
            let target_data = cam.target / axis_scale;
            let position_data = cam.position / axis_scale;
            let scene_bounds = self
                .scene
                .get_visible_nodes()
                .into_iter()
                .filter(|node| node.axes_index == axes_index)
                .filter_map(|node| node.render_data.as_ref().and_then(|data| data.bounds))
                .reduce(|a, b| a.union(&b));
            let plane_level = self
                .last_geometry_scene
                .as_ref()
                .map(|scene| scene.bounds)
                .or(scene_bounds)
                .filter(|bounds| bounds.min.is_finite() && bounds.max.is_finite())
                .map(|bounds| bounds.min.z - bounds.size().length().max(1.0) * 0.001)
                .unwrap_or(target_data.z);

            let on_grid = |x: f32, y: f32| -> Vec3 { Vec3::new(x, y, plane_level) };

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

            let ray_intersect_grid = |ndc_x: f32, ndc_y: f32| -> Option<Vec3> {
                // Use a near/far pair in clip space to form a ray.
                let p0 = unproject(ndc_x, ndc_y, -1.0)?;
                let p1 = unproject(ndc_x, ndc_y, 1.0)?;
                let dir = p1 - p0;
                if !dir.z.is_finite() || dir.z.abs() < 1e-8 {
                    return None;
                }
                let t = (plane_level - p0.z) / dir.z;
                if !t.is_finite() || t <= 0.0 {
                    return None;
                }
                Some(p0 + dir * t)
            };

            let mut plane_pts: Vec<Vec3> = Vec::new();
            for (nx, ny) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                if let Some(p) = ray_intersect_grid(nx, ny) {
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
                min_x = target_data.x - extent / axis_scale.x;
                max_x = target_data.x + extent / axis_scale.x;
                min_y = target_data.y - extent / axis_scale.y;
                max_y = target_data.y + extent / axis_scale.y;
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

            // Use the more visible ground direction for spacing. A nearly
            // edge-on X or Y axis should not make every major cell enormous.
            let center = on_grid(target_data.x, target_data.y);
            let px_per_world = {
                let a = project_to_px(center);
                let bx = project_to_px(center + Vec3::X);
                let by = project_to_px(center + Vec3::Y);
                match (a, bx, by) {
                    (Some((ax, ay)), Some((bx, by)), Some((cx, cy))) => (bx - ax)
                        .hypot(by - ay)
                        .max((cx - ax).hypot(cy - ay))
                        .max(1e-3),
                    _ => 1.0,
                }
            };
            let desired_major_px = 96.0_f64;
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

            // Procedural ground grid (depth-tested, no depth writes).
            let show_major_grid = self.overlay_show_grid_for_axes(axes_index);
            let show_minor_grid = self.overlay_show_minor_grid_for_axes(axes_index);
            if show_major_grid || show_minor_grid {
                let theme = self.theme.build_theme();
                let bg = theme.get_background_color();
                let grid = theme.get_grid_color();
                let bg_luma = 0.2126 * bg.x + 0.7152 * bg.y + 0.0722 * bg.z;
                let mut major_rgb = [grid.x, grid.y, grid.z];
                let mut minor_rgb = [grid.x, grid.y, grid.z];
                let mut major_alpha = grid.w.clamp(0.22, 0.34);
                let mut minor_alpha = (grid.w * 0.55).clamp(0.07, 0.16);
                if bg_luma <= 0.62 {
                    major_rgb = [grid.x * 0.80, grid.y * 0.80, grid.z * 0.80];
                    minor_rgb = [grid.x * 0.68, grid.y * 0.68, grid.z * 0.68];
                }
                if bg_luma > 0.62 {
                    major_rgb = [grid.x * 0.45, grid.y * 0.45, grid.z * 0.45];
                    minor_rgb = [grid.x * 0.33, grid.y * 0.33, grid.z * 0.33];
                    major_alpha = major_alpha.max(0.32);
                    minor_alpha = minor_alpha.max(0.14);
                }
                if !show_major_grid {
                    major_alpha = 0.0;
                }
                if !show_minor_grid {
                    minor_alpha = 0.0;
                }
                self.wgpu_renderer.ensure_grid_plane_pipeline(emission);
                self.wgpu_renderer.update_grid_uniforms_for_axes(
                    emission,
                    axes_index,
                    crate::core::renderer::GridUniforms {
                        major_step: major_step as f32,
                        minor_step: minor_step as f32,
                        fade_start: (0.85 * dx.max(dy)).max(major_step as f32),
                        fade_end: (1.40 * dx.max(dy)).max((major_step as f32) * 2.0),
                        camera_pos: position_data.to_array(),
                        _pad0: 0.0,
                        target_pos: [target_data.x, target_data.y, plane_level],
                        _pad1: 0.0,
                        major_color: [major_rgb[0], major_rgb[1], major_rgb[2], major_alpha],
                        minor_color: [minor_rgb[0], minor_rgb[1], minor_rgb[2], minor_alpha],
                    },
                );

                let quad_vertices = [
                    Vertex::new(on_grid(min_x, min_y), Vec4::ONE),
                    Vertex::new(on_grid(max_x, min_y), Vec4::ONE),
                    Vertex::new(on_grid(max_x, max_y), Vec4::ONE),
                    Vertex::new(on_grid(min_x, max_y), Vec4::ONE),
                ];
                let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
                let vb = self
                    .wgpu_renderer
                    .create_vertex_buffer(emission, &quad_vertices);
                let ib = self
                    .wgpu_renderer
                    .create_index_buffer(emission, &quad_indices);
                grid_plane_buffers = Some((vb, ib));
            }

            // World-axis helper for spatial awareness at model scale. Each axis has its own
            // tick step, since 3-D axes may scale each data axis differently.
            // Each axis reaches the far edge of the data from the origin, with about four
            // ticks; without data bounds, the step follows the on-screen scale instead.
            let (axis_steps, axis_lens) = match scene_bounds {
                Some(bounds) if bounds.min.is_finite() && bounds.max.is_finite() => {
                    let reach = bounds.min.abs().max(bounds.max.abs());
                    let reach = [reach.x, reach.y, reach.z];
                    let steps = reach.map(|r| nice_step((r as f64 / 4.0).max(1e-9)));
                    let lens = [0, 1, 2].map(|k| reach[k].max(steps[k] as f32));
                    (steps, lens)
                }
                _ => {
                    let steps = Self::world_axis_steps(&project_to_px, center, &nice_step);
                    let lens = steps.map(|step| {
                        (step as f32 * 5.0)
                            .clamp((step as f32).min(0.5), (dx.max(dy) * 0.6).max(step as f32))
                    });
                    (steps, lens)
                }
            };
            let origin = Vec3::new(0.0, 0.0, 0.0);
            let col_x = Vec4::new(0.92, 0.25, 0.25, 0.85);
            let col_y = Vec4::new(0.35, 0.90, 0.45, 0.85);
            let col_z = Vec4::new(0.35, 0.62, 0.98, 0.85);
            push_line(origin, origin + Vec3::new(axis_lens[0], 0.0, 0.0), col_x);
            push_line(origin, origin + Vec3::new(0.0, axis_lens[1], 0.0), col_y);
            push_line(origin, origin + Vec3::new(0.0, 0.0, axis_lens[2]), col_z);

            // Dynamic tick marks on the origin triad (major step only). Labels are drawn in the
            // overlay so they stay crisp; these marks provide a depth-correct anchor in the scene.
            self.world_axis_triads
                .insert(axes_index, (axis_steps, axis_lens));
            let max_ticks = 6usize;
            let mut add_ticks = |axis: Vec3, perp: Vec3, step: f64, axis_len: f32, col: Vec4| {
                if step <= 0.0 {
                    return;
                }
                // `f32::clamp` panics if min > max; keep the minimum below the maximum.
                let tick_max = (step as f32 * 0.25).max(1.0e-6);
                let tick_min = 0.01_f32.min(tick_max);
                let tick_len = (axis_len * 0.04).clamp(tick_min, tick_max);
                for i in 1..=max_ticks {
                    let t = (i as f32) * (step as f32);
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
            add_ticks(Vec3::X, Vec3::Y, axis_steps[0], axis_lens[0], col_x);
            add_ticks(Vec3::Y, Vec3::X, axis_steps[1], axis_lens[1], col_y);
            add_ticks(Vec3::Z, Vec3::X, axis_steps[2], axis_lens[2], col_z);

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
                let vb = runmat_accelerate_api::WgpuBufferOwner::new(
                    self.wgpu_renderer
                        .create_vertex_buffer(emission, &owned_render_data[idx].vertices),
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
                let buf = self.wgpu_renderer.create_vertex_buffer(emission, &expanded);
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
            self.wgpu_renderer.ensure_image_pipeline(emission);
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
                        .create_image_texture_and_bind_group(emission, *width, *height, data);
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
                let (_buf, bg) = self
                    .wgpu_renderer
                    .create_point_style_bind_group(emission, style);
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
        let log_flags = self.direct_log_flags_for_axes(axes_index);
        if is_2d && (show_major_grid || show_minor_grid) {
            if let Some((mut l, mut r, mut b, mut t)) = self.view_bounds_for_axes(axes_index) {
                if self.overlay_axes_kind_for_axes(axes_index) == AxesKind::Polar {
                    let radius = l.abs().max(r.abs()).max(b.abs()).max(t.abs()).max(1e-6);
                    l = -radius;
                    r = radius;
                    b = -radius;
                    t = radius;
                }
                // Reflect output coordinates after log mapping; limits stay ordered.
                let (ndc_y_min, ndc_y_max) = self
                    .overlay_y_direction_for_axes(axes_index)
                    .ndc_endpoints();
                // Update direct uniforms mapping for viewport
                self.wgpu_renderer.update_direct_uniforms_for_axes(
                    emission,
                    axes_index,
                    DirectUniforms::new(
                        [l as f32, b as f32],
                        [r as f32, t as f32],
                        [-1.0, ndc_y_min],
                        [1.0, ndc_y_max],
                        [sw.max(1) as f32, sh.max(1) as f32],
                        log_flags,
                    ),
                );
                self.wgpu_renderer.ensure_direct_line_pipeline(emission);

                let x_range = (r - l).max(1e-6);
                let y_range = (t - b).max(1e-6);
                let x_step = plot_utils::calculate_tick_interval(x_range);
                let y_step = plot_utils::calculate_tick_interval(y_range);
                let mut grid_vertices: Vec<Vertex> = Vec::new();
                // Custom ticks (xticks/yticks) set where the overlay draws tick labels, so major
                // grid lines must use the same values; the evenly spaced minor grid is skipped.
                let explicit_x_ticks = self.overlay_x_ticks_for_axes(axes_index);
                let explicit_y_ticks = self.overlay_y_ticks_for_axes(axes_index);
                // Log axes are labeled at whole decades; the grid follows those, with minor
                // lines at 2..9 times each decade. Even spacing in data units would bunch up.
                let (x_log, y_log) = (log_flags[0] != 0, log_flags[1] != 0);
                let log_major_x = (x_log && explicit_x_ticks.is_none())
                    .then(|| Self::log_grid_values(l, r, false));
                let log_major_y = (y_log && explicit_y_ticks.is_none())
                    .then(|| Self::log_grid_values(b, t, false));
                let explicit_x_ticks = explicit_x_ticks.or(log_major_x);
                let explicit_y_ticks = explicit_y_ticks.or(log_major_y);
                let g = 80.0_f32 / 255.0_f32;
                let major_col = Vec4::new(g, g, g, 1.0);
                let minor_col = Vec4::new(g, g, g, 0.42);
                if self.overlay_axes_kind_for_axes(axes_index) == AxesKind::Polar {
                    let radius = l.abs().max(r.abs()).max(b.abs()).max(t.abs()).max(1e-6);
                    let step = plot_utils::calculate_tick_interval(radius);
                    Self::push_polar_grid_lines(
                        &mut grid_vertices,
                        radius,
                        step,
                        show_major_grid,
                        show_minor_grid,
                        major_col,
                        minor_col,
                    );
                } else {
                    if show_minor_grid && x_log {
                        Self::push_vertical_grid_lines_at(
                            &mut grid_vertices,
                            &Self::log_grid_values(l, r, true),
                            (l, r),
                            (b, t),
                            minor_col,
                        );
                    } else if show_minor_grid
                        && explicit_x_ticks.is_none()
                        && x_step.is_finite()
                        && x_step > 0.0
                    {
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
                    if show_minor_grid && y_log {
                        Self::push_horizontal_grid_lines_at(
                            &mut grid_vertices,
                            &Self::log_grid_values(b, t, true),
                            (b, t),
                            (l, r),
                            minor_col,
                        );
                    } else if show_minor_grid
                        && explicit_y_ticks.is_none()
                        && y_step.is_finite()
                        && y_step > 0.0
                    {
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
                    if let (true, Some(ticks)) = (show_major_grid, explicit_x_ticks.as_ref()) {
                        Self::push_vertical_grid_lines_at(
                            &mut grid_vertices,
                            ticks,
                            (l, r),
                            (b, t),
                            major_col,
                        );
                    } else if show_major_grid && x_step.is_finite() && x_step > 0.0 {
                        Self::push_vertical_grid_lines(
                            &mut grid_vertices,
                            (l, r),
                            (b, t),
                            x_step,
                            major_col,
                            None,
                        );
                    }
                    if let (true, Some(ticks)) = (show_major_grid, explicit_y_ticks.as_ref()) {
                        Self::push_horizontal_grid_lines_at(
                            &mut grid_vertices,
                            ticks,
                            (b, t),
                            (l, r),
                            major_col,
                        );
                    } else if show_major_grid && y_step.is_finite() && y_step > 0.0 {
                        Self::push_horizontal_grid_lines(
                            &mut grid_vertices,
                            (b, t),
                            (l, r),
                            y_step,
                            major_col,
                            None,
                        );
                    }
                }
                if !grid_vertices.is_empty() {
                    grid_vb_opt = Some(
                        self.wgpu_renderer
                            .create_vertex_buffer(emission, &grid_vertices),
                    );
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
                let (ndc_y_min, ndc_y_max) = self
                    .overlay_y_direction_for_axes(axes_index)
                    .ndc_endpoints();
                self.wgpu_renderer.update_direct_uniforms_for_axes(
                    emission,
                    axes_index,
                    DirectUniforms::new(
                        [l as f32, b as f32],
                        [r as f32, t as f32],
                        [-1.0, ndc_y_min],
                        [1.0, ndc_y_max],
                        [sw.max(1) as f32, sh.max(1) as f32],
                        log_flags,
                    ),
                );
            }
            self.wgpu_renderer.ensure_direct_triangle_pipeline(emission);
            self.wgpu_renderer.ensure_direct_line_pipeline(emission);
            self.wgpu_renderer.ensure_direct_point_pipeline(emission);
        } else {
            // 3D: ensure camera-based pipelines exist so surfaces rotate with the camera.
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::Triangles);
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::Lines);
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::LinesNoDepth);
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::Points);
        }
        self.wgpu_renderer.update_marker_screen_uniforms_for_axes(
            emission,
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
                let is_lines = matches!(
                    render_data.pipeline_type,
                    crate::core::PipelineType::Lines | crate::core::PipelineType::LinesNoDepth
                );
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
                    } else if render_data.pipeline_type
                        == crate::core::PipelineType::AnnotationStroke
                    {
                        render_pass.set_bind_group(
                            1,
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
                        for (index_range, instance_range) in
                            indexed_draw_ranges(&render_data.draw_calls, indices.len())
                        {
                            render_pass.draw_indexed(index_range, 0, instance_range);
                        }
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

    pub fn overlay_axes_kind_for_axes(&self, axes_index: usize) -> AxesKind {
        self.last_figure
            .as_ref()
            .map(|f| f.axes_kind(axes_index))
            .unwrap_or(AxesKind::Cartesian)
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

    /// Vertical grid lines at explicit tick values inside `x_bounds`.
    /// Grid values on a log axis spanning `lo..hi`: whole decades, or the 2..9 multiples
    /// between them when `minor` is set. Empty unless both limits are positive.
    fn log_grid_values(lo: f64, hi: f64, minor: bool) -> Vec<f64> {
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        if !(lo > 0.0 && hi.is_finite()) {
            return Vec::new();
        }
        let first = lo.log10().floor() as i32;
        let last = hi.log10().ceil() as i32;
        let multiples: &[f64] = if minor {
            &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        } else {
            &[1.0]
        };
        (first..=last)
            .flat_map(|decade| {
                let base = 10f64.powi(decade);
                multiples.iter().map(move |m| m * base)
            })
            .filter(|value| *value >= lo && *value <= hi)
            .collect()
    }

    fn push_vertical_grid_lines_at(
        vertices: &mut Vec<Vertex>,
        ticks: &[f64],
        x_bounds: (f64, f64),
        y_bounds: (f64, f64),
        color: Vec4,
    ) {
        let (lo, hi) = (x_bounds.0.min(x_bounds.1), x_bounds.0.max(x_bounds.1));
        for &x in ticks
            .iter()
            .filter(|x| x.is_finite() && **x >= lo && **x <= hi)
        {
            vertices.push(Vertex::new(
                Vec3::new(x as f32, y_bounds.0 as f32, 0.0),
                color,
            ));
            vertices.push(Vertex::new(
                Vec3::new(x as f32, y_bounds.1 as f32, 0.0),
                color,
            ));
        }
    }

    /// Horizontal grid lines at explicit tick values inside `y_bounds`.
    fn push_horizontal_grid_lines_at(
        vertices: &mut Vec<Vertex>,
        ticks: &[f64],
        y_bounds: (f64, f64),
        x_bounds: (f64, f64),
        color: Vec4,
    ) {
        let (lo, hi) = (y_bounds.0.min(y_bounds.1), y_bounds.0.max(y_bounds.1));
        for &y in ticks
            .iter()
            .filter(|y| y.is_finite() && **y >= lo && **y <= hi)
        {
            vertices.push(Vertex::new(
                Vec3::new(x_bounds.0 as f32, y as f32, 0.0),
                color,
            ));
            vertices.push(Vertex::new(
                Vec3::new(x_bounds.1 as f32, y as f32, 0.0),
                color,
            ));
        }
    }

    fn push_polar_grid_lines(
        vertices: &mut Vec<Vertex>,
        radius: f64,
        major_step: f64,
        show_major: bool,
        show_minor: bool,
        major_color: Vec4,
        minor_color: Vec4,
    ) {
        if !radius.is_finite() || radius <= 0.0 || !major_step.is_finite() || major_step <= 0.0 {
            return;
        }

        if show_minor {
            let minor_step = (major_step / 5.0).max(f64::EPSILON);
            Self::for_each_grid_position(
                minor_step,
                radius,
                minor_step,
                Some((major_step, minor_step * 0.25)),
                |r| {
                    Self::push_circle_grid_line(vertices, r, minor_color);
                },
            );
        }
        if show_major {
            Self::for_each_grid_position(major_step, radius, major_step, None, |r| {
                Self::push_circle_grid_line(vertices, r, major_color);
            });
            for i in 0..12 {
                let theta = i as f64 * std::f64::consts::TAU / 12.0;
                let x = (theta.cos() * radius) as f32;
                let y = (theta.sin() * radius) as f32;
                if x.is_finite() && y.is_finite() {
                    vertices.push(Vertex::new(Vec3::ZERO, major_color));
                    vertices.push(Vertex::new(Vec3::new(x, y, 0.0), major_color));
                }
            }
        }
    }

    fn push_circle_grid_line(vertices: &mut Vec<Vertex>, radius: f64, color: Vec4) {
        if !radius.is_finite() || radius <= 0.0 {
            return;
        }
        const SEGMENTS: usize = 96;
        for i in 0..SEGMENTS {
            let theta0 = i as f64 * std::f64::consts::TAU / SEGMENTS as f64;
            let theta1 = (i + 1) as f64 * std::f64::consts::TAU / SEGMENTS as f64;
            let p0 = Vec3::new(
                (theta0.cos() * radius) as f32,
                (theta0.sin() * radius) as f32,
                0.0,
            );
            let p1 = Vec3::new(
                (theta1.cos() * radius) as f32,
                (theta1.sin() * radius) as f32,
                0.0,
            );
            if p0.is_finite() && p1.is_finite() {
                vertices.push(Vertex::new(p0, color));
                vertices.push(Vertex::new(p1, color));
            }
        }
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
    pub fn has_geometry_scene(&self) -> bool {
        self.last_geometry_scene.is_some()
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
    pub fn set_geometry_scene_presentation(&mut self, mut presentation: GeometryScenePresentation) {
        let requested_view_preset = presentation.view_preset.take();
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
    fn refresh_geometry_scene_render_data(&mut self, geometry_scene: &GeometryScene) {
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

    fn theme_background_is_light(&self) -> bool {
        relative_luminance(self.theme.build_theme().get_background_color()) > 0.5
    }

    fn rebuild_geometry_scene_nodes(&mut self, geometry_scene: &GeometryScene) {
        self.scene.clear();
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
    fn preserve_geometry_scene_state_when_unspecified(
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
    fn apply_geometry_presentation_visibility(&mut self) {
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

    /// The tick step and length of each 3-D origin axis (x, y, z) as last drawn.
    pub fn world_axis_triad_for_axes(&self, axes_index: usize) -> Option<([f64; 3], [f32; 3])> {
        self.world_axis_triads.get(&axes_index).copied()
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
    fn axes_grid_cell_for_axes(
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

fn geometry_scene_view_preset_to_camera(preset: GeometrySceneViewPreset) -> CameraViewPreset {
    match preset {
        GeometrySceneViewPreset::Perspective | GeometrySceneViewPreset::Isometric => {
            CameraViewPreset::Perspective
        }
        GeometrySceneViewPreset::Front => CameraViewPreset::Front,
        GeometrySceneViewPreset::Back => CameraViewPreset::Back,
        GeometrySceneViewPreset::Left => CameraViewPreset::Left,
        GeometrySceneViewPreset::Right => CameraViewPreset::Right,
        GeometrySceneViewPreset::Top => CameraViewPreset::Top,
        GeometrySceneViewPreset::Bottom => CameraViewPreset::Bottom,
    }
}

fn relative_luminance(color: Vec4) -> f32 {
    0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z
}

/// Scene line colors are stored with the scene, independent of theme. Near-white lines (such
/// as the undeformed outline) read well on a dark background but vanish on a light one, so on
/// a light background they are darkened to a slate of the same hue and made a little more opaque.
fn darken_pale_lines_for_light_background(render_data: &mut crate::core::RenderData) {
    const PALE: f32 = 0.8;
    const TARGET: f32 = 0.38;
    // GPU-resident vertices can't be recolored here; scene lines are CPU data.
    if render_data.gpu_vertices.is_some()
        || !matches!(
            render_data.pipeline_type,
            crate::core::PipelineType::Lines | crate::core::PipelineType::LinesNoDepth
        )
    {
        return;
    }
    for vertex in &mut render_data.vertices {
        let color = Vec4::from_array(vertex.color);
        let luminance = relative_luminance(color);
        if luminance <= PALE || color.w <= 0.0 {
            continue;
        }
        let scale = TARGET / luminance;
        vertex.color = [
            color.x * scale,
            color.y * scale,
            color.z * scale,
            color.w.max(0.5),
        ];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DrawCall;

    #[test]
    fn pale_scene_lines_darken_on_light_background_only() {
        let line = |color: [f32; 4]| Vertex::new(Vec3::ZERO, Vec4::from_array(color));
        let mut render_data = crate::core::RenderData {
            pipeline_type: crate::core::PipelineType::Lines,
            vertices: vec![
                line([0.92, 0.92, 1.0, 0.35]),
                line([0.58, 0.72, 0.88, 0.95]),
            ],
            indices: None,
            gpu_vertices: None,
            bounds: None,
            material: crate::core::Material::default(),
            draw_calls: Vec::new(),
            image: None,
        };
        darken_pale_lines_for_light_background(&mut render_data);
        let outline = Vec4::from_array(render_data.vertices[0].color);
        assert!((relative_luminance(outline) - 0.38).abs() < 1e-3);
        assert!(outline.z > outline.x, "keeps its blue tint");
        assert_eq!(outline.w, 0.5);
        // Mid-tone wireframe edges already contrast with white and stay as saved.
        assert_eq!(render_data.vertices[1].color, [0.58, 0.72, 0.88, 0.95]);
    }

    #[test]
    fn log_grid_values_follow_decades_inside_limits() {
        assert_eq!(
            PlotRenderer::log_grid_values(10.0, 1000.0, false),
            vec![10.0, 100.0, 1000.0]
        );
        assert_eq!(
            PlotRenderer::log_grid_values(15.0, 45.0, true),
            vec![20.0, 30.0, 40.0]
        );
        assert!(PlotRenderer::log_grid_values(-1.0, 100.0, false).is_empty());
    }
    use crate::plots::{figure::PlotElement, Figure, PatchPlot};

    #[test]
    fn indexed_renderer_respects_cleared_draw_calls_and_index_ranges() {
        assert!(indexed_draw_ranges(&[], 6).is_empty());
        let calls = [DrawCall {
            vertex_offset: 0,
            vertex_count: 4,
            index_offset: Some(2),
            index_count: Some(3),
            instance_count: 2,
        }];
        assert_eq!(indexed_draw_ranges(&calls, 6), vec![(2..5, 0..2)]);
    }

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

        assert!(patch.is_3d_content());
    }

    #[test]
    fn patch_3d_detection_ignores_large_scene_relative_noise() {
        let patch = patch_element(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0e6, 0.0, 0.0),
            Vec3::new(0.0, 1.0e6, 0.5),
        ]);

        assert!(!patch.is_3d_content());
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
    fn applies_data_aspect_ratio_to_3d_display_bounds() {
        let mut figure = Figure::new();
        figure.set_axes_data_aspect_ratio(0, [1.0, 2.0, 4.0], "manual");
        let bounds = BoundingBox::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 1.0));

        let display = PlotRenderer::apply_3d_display_limits_to_bounds(bounds, Some(&figure), 0);

        assert_eq!(display.min, Vec3::new(0.0, 0.0, -1.5));
        assert_eq!(display.max, Vec3::new(1.0, 2.0, 2.5));
    }

    #[test]
    fn degenerate_z_bounds_remain_finite_for_3d_display_bounds() {
        let figure = Figure::new();
        let bounds = BoundingBox::new(Vec3::new(-2.0, -2.0, 0.0), Vec3::new(2.0, 2.0, 0.0));

        // Figure metadata applies its default tick-aligned axis limits. The
        // metadata-free fallback instead supplies the minimum camera span.
        for (figure, z_min, z_max) in [(Some(&figure), -0.6, 0.6), (None, -0.05, 0.05)] {
            let display = PlotRenderer::apply_3d_display_limits_to_bounds(bounds, figure, 0);
            assert!(PlotRenderer::bounds_are_finite(display));
            assert_eq!(display.min.z, z_min);
            assert_eq!(display.max.z, z_max);
            assert_eq!(display.min.x, bounds.min.x);
            assert_eq!(display.max.x, bounds.max.x);
            assert_eq!(display.min.y, bounds.min.y);
            assert_eq!(display.max.y, bounds.max.y);
        }
    }

    #[test]
    fn geometry_scene_view_presets_map_to_camera_presets() {
        assert_eq!(
            geometry_scene_view_preset_to_camera(GeometrySceneViewPreset::Isometric),
            CameraViewPreset::Perspective
        );
        assert_eq!(
            geometry_scene_view_preset_to_camera(GeometrySceneViewPreset::Front),
            CameraViewPreset::Front
        );
        assert_eq!(
            geometry_scene_view_preset_to_camera(GeometrySceneViewPreset::Top),
            CameraViewPreset::Top
        );
    }

    #[test]
    fn positive_log_bounds_preserve_valid_positive_range() {
        assert_eq!(
            crate::plots::figure::positive_log_bounds(1.0, 100.0),
            (1.0, 100.0)
        );
    }

    #[test]
    fn positive_log_bounds_recover_from_nonpositive_lower_bound() {
        let (lo, hi) = crate::plots::figure::positive_log_bounds(-10.0, 100.0);

        assert!(lo > 0.0);
        assert_eq!(hi, 100.0);
        assert!(lo < hi);
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

        // Only the nice values are ticks; MATLAB does not label an axis limit that falls
        // between them.
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

        // Tick values are multiples of a nice step, so ten significant digits show each one
        // exactly (12.65, 0.0002) while trimming the floating-point tail (0.30000000000000004).
        if !value.is_finite() {
            return format!("{value}");
        }
        if value.abs() < 1.0e-12 {
            return "0".to_string();
        }
        let magnitude = value.abs().log10().floor() as i32;
        let decimals = (9 - magnitude).clamp(0, 15) as usize;
        trim_fixed(format!("{value:.decimals$}"))
    }

    pub fn canonical_tick_label_format(format: &str) -> String {
        match format.trim().to_ascii_lowercase().as_str() {
            "auto" => "%g".to_string(),
            "usd" => "$%,.2f".to_string(),
            "eur" => "\u{20AC}%,.2f".to_string(),
            "gbp" => "\u{00A3}%,.2f".to_string(),
            "jpy" => "\u{00A5}%,d".to_string(),
            "degrees" => "%g\u{00B0}".to_string(),
            "percentage" => "%g%%".to_string(),
            _ => format.to_string(),
        }
    }

    pub fn format_tick_label_with_format(value: f64, format: Option<&str>) -> String {
        TickLabelFormatter::new(format).format(value)
    }

    #[derive(Debug, Clone)]
    pub struct TickLabelFormatter {
        spec: Option<NumericTickFormat>,
    }

    impl TickLabelFormatter {
        pub fn new(format: Option<&str>) -> Self {
            let spec = format
                .map(canonical_tick_label_format)
                .and_then(|format| NumericTickFormat::parse(&format));
            Self { spec }
        }

        pub fn format(&self, value: f64) -> String {
            let Some(spec) = &self.spec else {
                return format_tick_label(value);
            };
            spec.format(value)
        }
    }

    #[derive(Debug, Clone)]
    struct NumericTickFormat {
        prefix: String,
        suffix: String,
        flags: TickFormatFlags,
        width: Option<usize>,
        precision: Option<usize>,
        conversion: char,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct TickFormatFlags {
        thousands: bool,
        plus: bool,
        zero_pad: bool,
        left: bool,
        alternate: bool,
    }

    impl NumericTickFormat {
        fn parse(format: &str) -> Option<Self> {
            let chars: Vec<char> = format.chars().collect();
            let mut index = 0usize;
            while index < chars.len() {
                if chars[index] == '%' {
                    if chars.get(index + 1) == Some(&'%') {
                        index += 2;
                        continue;
                    }
                    break;
                }
                index += 1;
            }
            if index >= chars.len() {
                return None;
            }

            let prefix = decode_percent_literals(&chars[..index]);
            index += 1;
            let mut flags = TickFormatFlags::default();
            while index < chars.len() {
                match chars[index] {
                    ',' => flags.thousands = true,
                    '+' => flags.plus = true,
                    '0' => flags.zero_pad = true,
                    '-' => flags.left = true,
                    '#' => flags.alternate = true,
                    _ => break,
                }
                index += 1;
            }

            let width_start = index;
            while index < chars.len() && chars[index].is_ascii_digit() {
                index += 1;
            }
            let width = parse_usize(&chars[width_start..index]);

            let precision = if chars.get(index) == Some(&'.') {
                index += 1;
                let precision_start = index;
                while index < chars.len() && chars[index].is_ascii_digit() {
                    index += 1;
                }
                parse_usize(&chars[precision_start..index]).or(Some(0))
            } else {
                None
            };

            let conversion = *chars.get(index)?;
            if !matches!(conversion, 'd' | 'i' | 'f' | 'e' | 'E' | 'g' | 'G') {
                return None;
            }
            index += 1;
            let suffix = decode_percent_literals(&chars[index..]);
            Some(Self {
                prefix,
                suffix,
                flags,
                width,
                precision,
                conversion,
            })
        }

        fn format(&self, value: f64) -> String {
            let mut body = match self.conversion {
                'd' | 'i' => format_integer(value, self.precision, self.flags),
                'f' => format_fixed(value, self.precision.unwrap_or(6), self.flags),
                'e' => format_exponential(value, self.precision.unwrap_or(6), false),
                'E' => format_exponential(value, self.precision.unwrap_or(6), true),
                'g' => format_general(value, self.precision.unwrap_or(6), false, self.flags),
                'G' => format_general(value, self.precision.unwrap_or(6), true, self.flags),
                _ => format_tick_label(value),
            };
            if self.flags.plus && value.is_sign_positive() && !body.starts_with('+') {
                body.insert(0, '+');
            }
            if self.flags.thousands {
                body = add_thousands_separators(&body);
            }
            body = apply_width(body, self.width, self.flags);
            format!("{}{}{}", self.prefix, body, self.suffix)
        }
    }

    fn parse_usize(chars: &[char]) -> Option<usize> {
        if chars.is_empty() {
            return None;
        }
        chars.iter().collect::<String>().parse().ok()
    }

    fn decode_percent_literals(chars: &[char]) -> String {
        let mut out = String::new();
        let mut index = 0usize;
        while index < chars.len() {
            if chars[index] == '%' && chars.get(index + 1) == Some(&'%') {
                out.push('%');
                index += 2;
            } else {
                out.push(chars[index]);
                index += 1;
            }
        }
        out
    }

    fn format_integer(value: f64, precision: Option<usize>, flags: TickFormatFlags) -> String {
        let rounded = if value.is_finite() {
            value.round()
        } else {
            value
        };
        let mut text = format!("{rounded:.0}");
        if let Some(precision) = precision {
            let negative = text.starts_with('-');
            let digits = if negative { &text[1..] } else { &text };
            if digits.len() < precision {
                let mut padded = "0".repeat(precision - digits.len());
                padded.push_str(digits);
                text = if negative {
                    format!("-{padded}")
                } else {
                    padded
                };
            }
        }
        if flags.alternate && !text.contains('.') {
            text.push('.');
        }
        text
    }

    fn format_fixed(value: f64, precision: usize, flags: TickFormatFlags) -> String {
        let mut text = format!("{value:.precision$}");
        if precision == 0 && flags.alternate && !text.contains('.') {
            text.push('.');
        }
        text
    }

    fn format_exponential(value: f64, precision: usize, upper: bool) -> String {
        let text = format!("{value:.precision$e}");
        if upper {
            text.to_ascii_uppercase()
        } else {
            text
        }
    }

    fn format_general(value: f64, precision: usize, upper: bool, flags: TickFormatFlags) -> String {
        if value == 0.0 {
            return "0".to_string();
        }
        let abs = value.abs();
        let use_exp = abs < 1e-4 || abs >= 10f64.powi(precision as i32);
        let text = if use_exp {
            let decimals = precision.saturating_sub(1);
            format_exponential(value, decimals, upper)
        } else {
            let integer_digits = abs.log10().floor().max(0.0) as usize + 1;
            let decimals = precision.saturating_sub(integer_digits);
            format_fixed(value, decimals, flags)
        };
        if flags.alternate {
            text
        } else {
            trim_general_zeros(text)
        }
    }

    fn trim_general_zeros(mut text: String) -> String {
        let exp_index = text.find(['e', 'E']);
        let suffix = exp_index.map(|idx| text.split_off(idx));
        if text.contains('.') {
            while text.ends_with('0') {
                text.pop();
            }
            if text.ends_with('.') {
                text.pop();
            }
        }
        if let Some(suffix) = suffix {
            text.push_str(&suffix);
        }
        text
    }

    fn add_thousands_separators(text: &str) -> String {
        let (sign, rest) = match text.strip_prefix('-') {
            Some(rest) => ("-", rest),
            None => match text.strip_prefix('+') {
                Some(rest) => ("+", rest),
                None => ("", text),
            },
        };
        let (integer, suffix) = rest
            .find(['.', 'e', 'E'])
            .map(|idx| rest.split_at(idx))
            .unwrap_or((rest, ""));
        let chars: Vec<char> = integer.chars().collect();
        let mut out = String::new();
        for (idx, ch) in chars.iter().enumerate() {
            if idx > 0 && (chars.len() - idx).is_multiple_of(3) {
                out.push(',');
            }
            out.push(*ch);
        }
        format!("{sign}{out}{suffix}")
    }

    fn apply_width(mut text: String, width: Option<usize>, flags: TickFormatFlags) -> String {
        let Some(width) = width else {
            return text;
        };
        let len = text.chars().count();
        if len >= width {
            return text;
        }
        let pad = width - len;
        if flags.left {
            text.push_str(&" ".repeat(pad));
            text
        } else if flags.zero_pad {
            let mut chars = text.chars();
            match chars.next() {
                Some(sign @ ('-' | '+')) => {
                    let rest = chars.collect::<String>();
                    format!("{sign}{}{rest}", "0".repeat(pad))
                }
                _ => format!("{}{text}", "0".repeat(pad)),
            }
        } else {
            format!("{}{text}", " ".repeat(pad))
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
