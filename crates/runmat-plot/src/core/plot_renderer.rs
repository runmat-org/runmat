//! Unified plot rendering pipeline for both interactive GUI and static export
//!
//! This module provides the core rendering logic that is shared between
//! interactive plotting windows and static file exports, ensuring consistent
//! high-quality output across all use cases.

#[path = "plot_renderer/scene_setup.rs"]
mod scene_setup;
#[path = "plot_renderer/view_fit.rs"]
mod view_fit;
#[path = "plot_renderer/render_frame.rs"]
mod render_frame;
#[path = "plot_renderer/camera_pass.rs"]
mod camera_pass;
#[path = "plot_renderer/axes_viewports.rs"]
mod axes_viewports;
#[path = "plot_renderer/grid_lines.rs"]
mod grid_lines;
#[path = "plot_renderer/geometry_presentation.rs"]
mod geometry_presentation;
#[path = "plot_renderer/overlay_queries.rs"]
mod overlay_queries;
#[path = "plot_renderer/data_primitives.rs"]
mod data_primitives;
pub use data_primitives::DataPrimitive;
#[path = "plot_renderer/axis_frame.rs"]
mod axis_frame;
pub use axis_frame::*;
#[path = "plot_renderer/free_helpers.rs"]
mod free_helpers;
#[cfg(test)]
#[path = "plot_renderer/tests.rs"]
mod tests;
#[path = "plot_renderer/plot_utils.rs"]
pub mod plot_utils;
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
#[path = "plot_renderer/world_helpers.rs"]
mod world_helpers;
#[path = "plot_renderer/grid_2d.rs"]
mod grid_2d;
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
}

#[allow(unused_imports)]
pub use axes_viewports::*;
#[allow(unused_imports)]
pub use camera_pass::*;
#[allow(unused_imports)]
pub use free_helpers::*;
#[allow(unused_imports)]
pub use geometry_presentation::*;
#[allow(unused_imports)]
pub use grid_lines::*;
#[allow(unused_imports)]
pub use overlay_queries::*;
#[allow(unused_imports)]
pub use render_frame::*;
#[allow(unused_imports)]
pub use scene_setup::*;
#[allow(unused_imports)]
pub use view_fit::*;
