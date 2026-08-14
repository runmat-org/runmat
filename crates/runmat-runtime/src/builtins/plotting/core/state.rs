use glam::Vec4;
use once_cell::sync::OnceCell;
use runmat_builtins::Tensor;
use runmat_plot::plots::{
    surface::ColorMap, surface::ShadingMode, AxesKind, Figure, LegendStyle, LineStyle, PlotElement,
    TextStyle,
};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::ops::{Deref, DerefMut};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::MutexGuard;
use std::sync::{Arc, Mutex};
use thiserror::Error;

use super::common::{default_figure, ERR_PLOTTING_UNAVAILABLE};
#[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
use super::engine::render_figure;
use super::web::current_plot_theme_config;
use super::{plotting_error, plotting_error_with_source};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::{BuiltinResult, RuntimeError};

type AxisLimitSnapshot = (Option<(f64, f64)>, Option<(f64, f64)>);
type AxisTickSnapshot = (Option<Vec<f64>>, Option<Vec<f64>>);
type AxisTickLabelSnapshot = (Option<Vec<String>>, Option<Vec<String>>);
type AxisTickFormatSnapshot = (Option<String>, Option<String>);
type AxisTickAngleSnapshot = (Option<f64>, Option<f64>);
type AxisDisplayBoundsSnapshot = Option<(f64, f64, f64, f64)>;
const DEFAULT_COLORMAP_LENGTH: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkAxesMode {
    X,
    Y,
    XY,
}

impl LinkAxesMode {
    fn links_x(self) -> bool {
        matches!(self, Self::X | Self::XY)
    }

    fn links_y(self) -> bool {
        matches!(self, Self::Y | Self::XY)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinkAxesAxis {
    X,
    Y,
}

#[derive(Clone, Debug)]
struct LinkAxesGroup {
    axis: LinkAxesAxis,
    targets: Vec<(FigureHandle, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FigureHandle(u32);

impl FigureHandle {
    pub fn as_u32(self) -> u32 {
        self.0
    }

    fn next(self) -> FigureHandle {
        FigureHandle(self.0 + 1)
    }
}

impl From<u32> for FigureHandle {
    fn from(value: u32) -> Self {
        FigureHandle(value.max(1))
    }
}

impl Default for FigureHandle {
    fn default() -> Self {
        FigureHandle(1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoomMotion {
    Both,
    Horizontal,
    Vertical,
}

impl ZoomMotion {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::Horizontal => "horizontal",
            Self::Vertical => "vertical",
        }
    }
}

impl Default for ZoomMotion {
    fn default() -> Self {
        Self::Both
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoomDirection {
    In,
    Out,
}

impl ZoomDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoomRightClickAction {
    PostContextMenu,
    InverseZoom,
}

impl ZoomRightClickAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PostContextMenu => "PostContextMenu",
            Self::InverseZoom => "InverseZoom",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoomModeState {
    pub enabled: bool,
    pub motion: ZoomMotion,
    pub direction: ZoomDirection,
    pub right_click_action: ZoomRightClickAction,
    pub use_legacy_exploration_modes: bool,
}

impl Default for ZoomModeState {
    fn default() -> Self {
        Self {
            enabled: false,
            motion: ZoomMotion::Both,
            direction: ZoomDirection::In,
            right_click_action: ZoomRightClickAction::PostContextMenu,
            use_legacy_exploration_modes: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoomModeCommand {
    On,
    Off,
    Toggle,
    XOn,
    YOn,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomStateSnapshot {
    pub figure: FigureHandle,
    pub axes_index: Option<usize>,
    pub mode: ZoomModeState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanModeState {
    pub enabled: bool,
    pub motion: ZoomMotion,
}

impl Default for PanModeState {
    fn default() -> Self {
        Self {
            enabled: false,
            motion: ZoomMotion::Both,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanStateSnapshot {
    pub figure: FigureHandle,
    pub axes_index: Option<usize>,
    pub mode: PanModeState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanModeCommand {
    On,
    Off,
    Toggle,
    XOn,
    YOn,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataCursorModeState {
    pub enabled: bool,
    pub snap_to_data_vertex: bool,
    pub display_style: String,
}

impl Default for DataCursorModeState {
    fn default() -> Self {
        Self {
            enabled: false,
            snap_to_data_vertex: true,
            display_style: "datatip".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataCursorStateSnapshot {
    pub figure: FigureHandle,
    pub mode: DataCursorModeState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaitbarState {
    pub progress: f64,
    pub message: String,
}

const DEFAULT_LINE_STYLE_ORDER: [LineStyle; 1] = [LineStyle::Solid];

#[derive(Clone)]
struct LineStyleCycle {
    order: Vec<LineStyle>,
    cursor: usize,
}

impl Default for LineStyleCycle {
    fn default() -> Self {
        Self {
            order: DEFAULT_LINE_STYLE_ORDER.to_vec(),
            cursor: 0,
        }
    }
}

impl LineStyleCycle {
    fn next(&mut self) -> LineStyle {
        if self.order.is_empty() {
            self.order = DEFAULT_LINE_STYLE_ORDER.to_vec();
        }
        let style = self.order[self.cursor % self.order.len()];
        self.cursor = (self.cursor + 1) % self.order.len();
        style
    }

    fn set_order(&mut self, order: &[LineStyle]) {
        if order.is_empty() {
            self.order = DEFAULT_LINE_STYLE_ORDER.to_vec();
        } else {
            self.order = order.to_vec();
        }
        self.cursor = 0;
    }

    fn reset_cursor(&mut self) {
        self.cursor = 0;
    }
}

#[derive(Clone, Default)]
struct LineColorCycle {
    order: Option<Vec<Vec4>>,
    cursor: usize,
}

impl LineColorCycle {
    fn next(&mut self) -> Vec4 {
        let color = match self.order.as_deref() {
            Some(order) if !order.is_empty() => order[self.cursor % order.len()],
            _ => line_color_for_series_index(self.cursor),
        };
        self.cursor = self.cursor.saturating_add(1);
        color
    }

    fn color_at(&self, index: usize) -> Vec4 {
        match self.order.as_deref() {
            Some(order) if !order.is_empty() => order[index % order.len()],
            _ => line_color_for_series_index(index),
        }
    }

    fn set_order(&mut self, order: &[Vec4]) {
        self.order = if order.is_empty() {
            None
        } else {
            Some(order.to_vec())
        };
        self.cursor = 0;
    }

    fn reset_cursor(&mut self) {
        self.cursor = 0;
    }
}

#[derive(Default)]
struct FigureState {
    figure: Figure,
    active_axes: usize,
    tag: String,
    hold_per_axes: HashMap<usize, bool>,
    line_style_cycles: HashMap<usize, LineStyleCycle>,
    line_color_cycles: HashMap<usize, LineColorCycle>,
    figure_color_order: Option<Vec<Vec4>>,
    colormap_lengths: HashMap<usize, usize>,
    zoom_mode: ZoomModeState,
    last_enabled_zoom_motion: ZoomMotion,
    zoom_axes_modes: HashMap<usize, ZoomModeState>,
    zoom_baselines: HashMap<usize, AxisLimitSnapshot>,
    pan_mode: PanModeState,
    last_enabled_pan_motion: ZoomMotion,
    pan_axes_modes: HashMap<usize, PanModeState>,
    data_cursor_mode: DataCursorModeState,
    waitbar: Option<WaitbarState>,
    revision: u64,
}

impl FigureState {
    fn new(handle: FigureHandle) -> Self {
        let title = format!("Figure {}", handle.as_u32());
        let figure = default_figure(&title, "X", "Y");
        Self {
            figure,
            active_axes: 0,
            tag: String::new(),
            hold_per_axes: HashMap::new(),
            line_style_cycles: HashMap::new(),
            line_color_cycles: HashMap::new(),
            figure_color_order: None,
            colormap_lengths: HashMap::new(),
            zoom_mode: ZoomModeState::default(),
            last_enabled_zoom_motion: ZoomMotion::Both,
            zoom_axes_modes: HashMap::new(),
            zoom_baselines: HashMap::new(),
            pan_mode: PanModeState::default(),
            last_enabled_pan_motion: ZoomMotion::Both,
            pan_axes_modes: HashMap::new(),
            data_cursor_mode: DataCursorModeState::default(),
            waitbar: None,
            revision: 0,
        }
    }

    fn hold(&self) -> bool {
        *self.hold_per_axes.get(&self.active_axes).unwrap_or(&false)
    }

    fn set_hold(&mut self, hold: bool) {
        self.hold_per_axes.insert(self.active_axes, hold);
    }

    fn cycle_for_axes_mut(&mut self, axes_index: usize) -> &mut LineStyleCycle {
        self.line_style_cycles.entry(axes_index).or_default()
    }

    fn color_cycle_for_axes_mut(&mut self, axes_index: usize) -> &mut LineColorCycle {
        match self.line_color_cycles.entry(axes_index) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let mut cycle = LineColorCycle::default();
                if let Some(order) = self.figure_color_order.as_deref() {
                    cycle.set_order(order);
                }
                entry.insert(cycle)
            }
        }
    }

    fn reset_cycle(&mut self, axes_index: usize) {
        if let Some(cycle) = self.line_style_cycles.get_mut(&axes_index) {
            cycle.reset_cursor();
        }
        if let Some(cycle) = self.line_color_cycles.get_mut(&axes_index) {
            cycle.reset_cursor();
        }
    }
}

struct ActiveAxesContext {
    axes_index: usize,
    style_cycle_ptr: *mut LineStyleCycle,
    color_cycle_ptr: *mut LineColorCycle,
}

struct AxesContextGuard {
    _private: (),
}

impl AxesContextGuard {
    fn install(state: &mut FigureState, axes_index: usize) -> Self {
        let style_cycle_ptr = state.cycle_for_axes_mut(axes_index) as *mut LineStyleCycle;
        let color_cycle_ptr = state.color_cycle_for_axes_mut(axes_index) as *mut LineColorCycle;
        ACTIVE_AXES_CONTEXT.with(|ctx| {
            debug_assert!(
                ctx.borrow().is_none(),
                "plot axes context already installed"
            );
            ctx.borrow_mut().replace(ActiveAxesContext {
                axes_index,
                style_cycle_ptr,
                color_cycle_ptr,
            });
        });
        Self { _private: () }
    }
}

impl Drop for AxesContextGuard {
    fn drop(&mut self) {
        ACTIVE_AXES_CONTEXT.with(|ctx| {
            ctx.borrow_mut().take();
        });
    }
}

fn with_active_style_cycle<R>(
    axes_index: usize,
    f: impl FnOnce(&mut LineStyleCycle) -> R,
) -> Option<R> {
    ACTIVE_AXES_CONTEXT.with(|ctx| {
        let guard = ctx.borrow();
        let active = guard.as_ref()?;
        if active.axes_index != axes_index {
            return None;
        }
        let cycle = unsafe { &mut *active.style_cycle_ptr };
        Some(f(cycle))
    })
}

fn with_active_color_cycle<R>(
    axes_index: usize,
    f: impl FnOnce(&mut LineColorCycle) -> R,
) -> Option<R> {
    ACTIVE_AXES_CONTEXT.with(|ctx| {
        let guard = ctx.borrow();
        let active = guard.as_ref()?;
        if active.axes_index != axes_index {
            return None;
        }
        let cycle = unsafe { &mut *active.color_cycle_ptr };
        Some(f(cycle))
    })
}

struct PlotRegistry {
    current: FigureHandle,
    next_handle: FigureHandle,
    figures: HashMap<FigureHandle, FigureState>,
    root_defaults: HashMap<String, RootPropertyEntry>,
    root_units: String,
    root_show_hidden_handles: bool,
    next_plot_child_handle: u64,
    plot_children: HashMap<u64, PlotChildHandleState>,
    link_axes_groups: Vec<LinkAxesGroup>,
}

#[derive(Clone, Debug)]
pub struct RootPropertyEntry {
    pub display_name: String,
    pub value: RootPropertyValue,
}

#[derive(Clone, Debug)]
pub enum RootPropertyValue {
    Bool(bool),
    Num(f64),
    String(String),
    Tensor(Tensor),
    StringArray {
        rows: usize,
        cols: usize,
        shape: Vec<usize>,
        data: Vec<String>,
    },
}

#[derive(Clone, Debug)]
pub struct HistogramHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub bin_edges: Vec<f64>,
    pub raw_counts: Vec<f64>,
    pub normalization: String,
    pub normalization_denominator: f64,
    pub display_name: Option<String>,
    pub metadata: HistogramHandleMetadata,
}

#[derive(Clone, Debug)]
pub struct HistogramHandleMetadata {
    pub data: Option<Tensor>,
    pub display_style: String,
    pub face_color: String,
    pub face_alpha: f64,
    pub edge_color: String,
    pub bin_width: f64,
    pub bin_limits: (f64, f64),
    pub is_polar: bool,
}

impl HistogramHandleMetadata {
    pub fn new(bin_edges: &[f64]) -> Self {
        let bin_width = bin_edges
            .windows(2)
            .next()
            .map(|pair| pair[1] - pair[0])
            .unwrap_or(0.0);
        let bin_limits = (
            bin_edges.first().copied().unwrap_or(0.0),
            bin_edges.last().copied().unwrap_or(0.0),
        );
        Self {
            data: None,
            display_style: "bar".into(),
            face_color: "auto".into(),
            face_alpha: 1.0,
            edge_color: "auto".into(),
            bin_width,
            bin_limits,
            is_polar: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Histogram2HandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub values: Tensor,
    pub raw_counts: Tensor,
    pub x_bin_edges: Vec<f64>,
    pub y_bin_edges: Vec<f64>,
    pub normalization: String,
    pub normalization_denominator: f64,
    pub display_style: crate::builtins::plotting::histogram2::Histogram2DisplayStyle,
    pub show_empty_bins: bool,
    pub face_alpha: f64,
    pub display_name: Option<String>,
    pub data: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct StemHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
}

#[derive(Clone, Debug)]
pub struct SimplePlotHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
}

#[derive(Clone, Debug)]
pub struct AnimatedLineHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub is_3d: bool,
    pub maximum_num_points: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct ErrorBarHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
}

#[derive(Clone, Debug)]
pub struct QuiverHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub is_3d: bool,
}

#[derive(Clone, Debug)]
pub struct ImageHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub c_data: Option<Tensor>,
    pub c_data_mapping: String,
}

#[derive(Clone, Debug)]
pub struct HeatmapHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub x_labels: Vec<String>,
    pub y_labels: Vec<String>,
    pub color_data: Tensor,
    pub color_limits: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct BinscatterHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub values: Tensor,
    pub x_bin_edges: Vec<f64>,
    pub y_bin_edges: Vec<f64>,
    pub x_data: Tensor,
    pub y_data: Tensor,
    pub num_bins: [usize; 2],
    pub auto_bins: bool,
    pub x_limits_option: Option<Tensor>,
    pub y_limits_option: Option<Tensor>,
    pub x_limits: (f64, f64),
    pub y_limits: (f64, f64),
    pub show_empty_bins: bool,
    pub face_alpha: f64,
    pub display_name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FunctionSurfaceHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub mesh_density: usize,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub function: FunctionSurfaceFunctionState,
}

#[derive(Clone, Debug)]
pub enum FunctionSurfaceFunctionState {
    Explicit(FunctionSurfaceFunctionRef),
    Parametric {
        x: FunctionSurfaceFunctionRef,
        y: FunctionSurfaceFunctionRef,
        z: FunctionSurfaceFunctionRef,
    },
}

#[derive(Clone, Debug)]
pub enum FunctionSurfaceFunctionRef {
    FunctionHandle(String),
    ExternalFunctionHandle(String),
    MethodFunctionHandle(String),
    BoundFunctionHandle {
        name: String,
        function: usize,
    },
    ClosureSummary {
        function_name: String,
        bound_function: Option<usize>,
    },
}

#[derive(Clone, Debug)]
pub struct FunctionContourHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub mesh_density: usize,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub function: FunctionSurfaceFunctionRef,
    pub fill: bool,
}

#[derive(Clone, Debug)]
pub struct AreaHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
}

#[derive(Clone, Debug)]
pub struct TextAnnotationHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub annotation_index: usize,
}

#[derive(Clone, Debug)]
pub struct TextScatterHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub annotation_indices: Vec<usize>,
    pub marker_plot_index: Option<usize>,
    pub is_3d: bool,
    pub text_data: Vec<String>,
    pub text_density_percentage: f64,
    pub max_text_length: usize,
    pub marker_color: TextScatterMarkerColor,
    pub marker_size: f64,
    pub color_data: Option<Vec<glam::Vec4>>,
    pub colors: Vec<glam::Vec4>,
    pub visible: bool,
    pub base_style: runmat_plot::plots::TextStyle,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TextScatterMarkerColor {
    Auto,
    None,
    Color(glam::Vec4),
}

#[derive(Clone, Debug)]
pub struct WordCloudHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub annotation_indices: Vec<usize>,
    pub word_data: Vec<String>,
    pub size_data: Vec<f64>,
    pub word_variable: String,
    pub size_variable: String,
    pub max_display_words: usize,
    pub color: Vec<glam::Vec4>,
    pub highlight_color: glam::Vec4,
    pub shape: String,
    pub layout_num: usize,
    pub size_power: f64,
    pub title: String,
    pub title_font_name: String,
    pub visible: bool,
    pub font_name: String,
    pub box_visible: bool,
    pub units: String,
    pub position: [f64; 4],
    pub handle_visibility: String,
    pub display_name: String,
    pub tag: String,
}

#[derive(Clone, Debug)]
pub struct StackedSourceTableSnapshot {
    pub classes: Vec<String>,
    pub variable_names: Vec<Vec<String>>,
}

#[derive(Clone, Debug)]
pub struct StackedPlotHandleState {
    pub figure: FigureHandle,
    pub axes_indices: Vec<usize>,
    pub line_plot_indices: Vec<usize>,
    pub line_group_counts: Vec<usize>,
    pub line_labels: Vec<String>,
    pub x_data: Vec<f64>,
    pub y_data: Vec<Vec<f64>>,
    pub display_variables: Vec<String>,
    pub source_table: Option<StackedSourceTableSnapshot>,
    pub x_variable: Vec<String>,
    pub combine_matching_names: bool,
    pub x_label: String,
    pub title: String,
    pub appearance: crate::builtins::plotting::style::LineAppearance,
    pub visible: bool,
    pub grid_visible: bool,
    pub x_limits: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub enum PlotChildHandleState {
    Histogram(HistogramHandleState),
    Histogram2(Histogram2HandleState),
    Line(SimplePlotHandleState),
    AnimatedLine(AnimatedLineHandleState),
    Scatter(SimplePlotHandleState),
    Bar(SimplePlotHandleState),
    Stem(StemHandleState),
    ErrorBar(ErrorBarHandleState),
    Stairs(SimplePlotHandleState),
    Quiver(QuiverHandleState),
    Image(ImageHandleState),
    Heatmap(HeatmapHandleState),
    Binscatter(BinscatterHandleState),
    FunctionSurface(FunctionSurfaceHandleState),
    FunctionContour(FunctionContourHandleState),
    Area(AreaHandleState),
    Surface(SimplePlotHandleState),
    Patch(SimplePlotHandleState),
    Line3(SimplePlotHandleState),
    Scatter3(SimplePlotHandleState),
    Contour(SimplePlotHandleState),
    ContourFill(SimplePlotHandleState),
    ReferenceLine(SimplePlotHandleState),
    Pie(SimplePlotHandleState),
    Text(TextAnnotationHandleState),
    TextScatter(TextScatterHandleState),
    WordCloud(WordCloudHandleState),
    StackedPlot(StackedPlotHandleState),
}

impl PlotChildHandleState {
    pub fn figure_axes(&self) -> (FigureHandle, usize) {
        match self {
            Self::Histogram(state) => (state.figure, state.axes_index),
            Self::Histogram2(state) => (state.figure, state.axes_index),
            Self::Line(state)
            | Self::Scatter(state)
            | Self::Bar(state)
            | Self::Stairs(state)
            | Self::Surface(state)
            | Self::Patch(state)
            | Self::Line3(state)
            | Self::Scatter3(state)
            | Self::Contour(state)
            | Self::ContourFill(state)
            | Self::ReferenceLine(state)
            | Self::Pie(state) => (state.figure, state.axes_index),
            Self::AnimatedLine(state) => (state.figure, state.axes_index),
            Self::Stem(state) => (state.figure, state.axes_index),
            Self::ErrorBar(state) => (state.figure, state.axes_index),
            Self::Quiver(state) => (state.figure, state.axes_index),
            Self::Image(state) => (state.figure, state.axes_index),
            Self::Heatmap(state) => (state.figure, state.axes_index),
            Self::Binscatter(state) => (state.figure, state.axes_index),
            Self::FunctionSurface(state) => (state.figure, state.axes_index),
            Self::FunctionContour(state) => (state.figure, state.axes_index),
            Self::Area(state) => (state.figure, state.axes_index),
            Self::Text(state) => (state.figure, state.axes_index),
            Self::TextScatter(state) => (state.figure, state.axes_index),
            Self::WordCloud(state) => (state.figure, state.axes_index),
            Self::StackedPlot(state) => (
                state.figure,
                state.axes_indices.first().copied().unwrap_or(0),
            ),
        }
    }

    pub fn plot_index(&self) -> Option<usize> {
        Some(match self {
            Self::Histogram(state) => state.plot_index,
            Self::Histogram2(state) => state.plot_index,
            Self::Line(state)
            | Self::Scatter(state)
            | Self::Bar(state)
            | Self::Stairs(state)
            | Self::Surface(state)
            | Self::Patch(state)
            | Self::Line3(state)
            | Self::Scatter3(state)
            | Self::Contour(state)
            | Self::ContourFill(state)
            | Self::ReferenceLine(state)
            | Self::Pie(state) => state.plot_index,
            Self::AnimatedLine(state) => state.plot_index,
            Self::Stem(state) => state.plot_index,
            Self::ErrorBar(state) => state.plot_index,
            Self::Quiver(state) => state.plot_index,
            Self::Image(state) => state.plot_index,
            Self::Heatmap(state) => state.plot_index,
            Self::Binscatter(state) => state.plot_index,
            Self::FunctionSurface(state) => state.plot_index,
            Self::FunctionContour(state) => state.plot_index,
            Self::Area(state) => state.plot_index,
            Self::Text(_) => return None,
            Self::TextScatter(state) => return state.marker_plot_index,
            Self::WordCloud(_) => return None,
            Self::StackedPlot(state) => return state.line_plot_indices.first().copied(),
        })
    }

    pub fn with_plot_location(
        &self,
        figure: FigureHandle,
        axes_index: usize,
        plot_index: usize,
    ) -> Option<Self> {
        Some(match self {
            Self::Histogram(state) => Self::Histogram(HistogramHandleState {
                figure,
                axes_index,
                plot_index,
                bin_edges: state.bin_edges.clone(),
                raw_counts: state.raw_counts.clone(),
                normalization: state.normalization.clone(),
                normalization_denominator: state.normalization_denominator,
                display_name: state.display_name.clone(),
                metadata: state.metadata.clone(),
            }),
            Self::Histogram2(state) => Self::Histogram2(Histogram2HandleState {
                figure,
                axes_index,
                plot_index,
                values: state.values.clone(),
                raw_counts: state.raw_counts.clone(),
                x_bin_edges: state.x_bin_edges.clone(),
                y_bin_edges: state.y_bin_edges.clone(),
                normalization: state.normalization.clone(),
                normalization_denominator: state.normalization_denominator,
                display_style: state.display_style,
                show_empty_bins: state.show_empty_bins,
                face_alpha: state.face_alpha,
                display_name: state.display_name.clone(),
                data: state.data.clone(),
            }),
            Self::Line(_) => Self::Line(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Scatter(_) => Self::Scatter(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Bar(_) => Self::Bar(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Stairs(_) => Self::Stairs(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Surface(_) => Self::Surface(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Patch(_) => Self::Patch(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Line3(_) => Self::Line3(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Scatter3(_) => Self::Scatter3(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Contour(_) => Self::Contour(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::ContourFill(_) => Self::ContourFill(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::ReferenceLine(_) => Self::ReferenceLine(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Pie(_) => Self::Pie(SimplePlotHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::AnimatedLine(state) => Self::AnimatedLine(AnimatedLineHandleState {
                figure,
                axes_index,
                plot_index,
                is_3d: state.is_3d,
                maximum_num_points: state.maximum_num_points,
            }),
            Self::Stem(_) => Self::Stem(StemHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::ErrorBar(_) => Self::ErrorBar(ErrorBarHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Quiver(state) => Self::Quiver(QuiverHandleState {
                figure,
                axes_index,
                plot_index,
                is_3d: state.is_3d,
            }),
            Self::Image(state) => Self::Image(ImageHandleState {
                figure,
                axes_index,
                plot_index,
                c_data: state.c_data.clone(),
                c_data_mapping: state.c_data_mapping.clone(),
            }),
            Self::Heatmap(state) => Self::Heatmap(HeatmapHandleState {
                figure,
                axes_index,
                plot_index,
                x_labels: state.x_labels.clone(),
                y_labels: state.y_labels.clone(),
                color_data: state.color_data.clone(),
                color_limits: state.color_limits.clone(),
            }),
            Self::Binscatter(state) => Self::Binscatter(BinscatterHandleState {
                figure,
                axes_index,
                plot_index,
                values: state.values.clone(),
                x_bin_edges: state.x_bin_edges.clone(),
                y_bin_edges: state.y_bin_edges.clone(),
                x_data: state.x_data.clone(),
                y_data: state.y_data.clone(),
                num_bins: state.num_bins,
                auto_bins: state.auto_bins,
                x_limits_option: state.x_limits_option.clone(),
                y_limits_option: state.y_limits_option.clone(),
                x_limits: state.x_limits,
                y_limits: state.y_limits,
                show_empty_bins: state.show_empty_bins,
                face_alpha: state.face_alpha,
                display_name: state.display_name.clone(),
            }),
            Self::FunctionSurface(state) => Self::FunctionSurface(FunctionSurfaceHandleState {
                figure,
                axes_index,
                plot_index,
                mesh_density: state.mesh_density,
                x_range: state.x_range,
                y_range: state.y_range,
                function: state.function.clone(),
            }),
            Self::FunctionContour(state) => Self::FunctionContour(FunctionContourHandleState {
                figure,
                axes_index,
                plot_index,
                mesh_density: state.mesh_density,
                x_range: state.x_range,
                y_range: state.y_range,
                function: state.function.clone(),
                fill: state.fill,
            }),
            Self::Area(_) => Self::Area(AreaHandleState {
                figure,
                axes_index,
                plot_index,
            }),
            Self::Text(_) => return None,
            Self::TextScatter(_) => return None,
            Self::WordCloud(_) => return None,
            Self::StackedPlot(_) => return None,
        })
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Histogram(_) => "histogram",
            Self::Histogram2(_) => "histogram2",
            Self::Line(_) | Self::Line3(_) => "line",
            Self::AnimatedLine(_) => "animatedline",
            Self::Scatter(_) | Self::Scatter3(_) => "scatter",
            Self::Bar(_) => "bar",
            Self::Stem(_) => "stem",
            Self::ErrorBar(_) => "errorbar",
            Self::Stairs(_) => "stairs",
            Self::Quiver(_) => "quiver",
            Self::Image(_) => "image",
            Self::Heatmap(_) => "heatmap",
            Self::Binscatter(_) => "binscatter",
            Self::FunctionSurface(_) => "functionsurface",
            Self::FunctionContour(_) => "functioncontour",
            Self::Area(_) => "area",
            Self::Surface(_) => "surface",
            Self::Patch(_) => "patch",
            Self::Contour(_) | Self::ContourFill(_) => "contour",
            Self::ReferenceLine(_) => "constantline",
            Self::Pie(_) => "pie",
            Self::Text(_) => "text",
            Self::TextScatter(_) => "textscatter",
            Self::WordCloud(_) => "wordcloud",
            Self::StackedPlot(_) => "stackedplot",
        }
    }
}

impl Default for PlotRegistry {
    fn default() -> Self {
        Self {
            current: FigureHandle::default(),
            next_handle: FigureHandle::default().next(),
            figures: HashMap::new(),
            root_defaults: HashMap::new(),
            root_units: "pixels".to_string(),
            root_show_hidden_handles: false,
            next_plot_child_handle: 1u64 << 40,
            plot_children: HashMap::new(),
            link_axes_groups: Vec::new(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
static REGISTRY: OnceCell<Mutex<PlotRegistry>> = OnceCell::new();

static TEST_PLOT_REGISTRY_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static TEST_PLOT_OUTER_LOCK_HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[doc(hidden)]
pub struct PlotTestLockGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
    disable_previous: Option<std::ffi::OsString>,
    host_previous: Option<std::ffi::OsString>,
}

impl Drop for PlotTestLockGuard {
    fn drop(&mut self) {
        restore_env_var(
            "RUNMAT_DISABLE_INTERACTIVE_PLOTS",
            self.disable_previous.take(),
        );
        restore_env_var("RUNMAT_HOST_MANAGED_PLOTS", self.host_previous.take());
        TEST_PLOT_OUTER_LOCK_HELD.with(|flag| flag.set(false));
    }
}

impl PlotTestLockGuard {
    #[doc(hidden)]
    pub fn disable_host_managed_plot_env(&self) -> HostManagedPlotEnvGuard<'_> {
        // The returned guard borrows `self`, so TEST_PLOT_REGISTRY_LOCK stays held
        // for the full duration of this process-env override.
        let previous = std::env::var_os("RUNMAT_HOST_MANAGED_PLOTS");
        unsafe {
            std::env::remove_var("RUNMAT_HOST_MANAGED_PLOTS");
        }
        HostManagedPlotEnvGuard {
            _plot_guard: self,
            previous,
        }
    }
}

#[doc(hidden)]
pub fn lock_plot_test_registry() -> PlotTestLockGuard {
    let guard = TEST_PLOT_REGISTRY_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    TEST_PLOT_OUTER_LOCK_HELD.with(|flag| flag.set(true));
    let disable_previous = std::env::var_os("RUNMAT_DISABLE_INTERACTIVE_PLOTS");
    let host_previous = std::env::var_os("RUNMAT_HOST_MANAGED_PLOTS");
    set_plot_test_env_vars();
    PlotTestLockGuard {
        _guard: guard,
        disable_previous,
        host_previous,
    }
}

#[doc(hidden)]
pub struct HostManagedPlotEnvGuard<'a> {
    _plot_guard: &'a PlotTestLockGuard,
    previous: Option<std::ffi::OsString>,
}

impl Drop for HostManagedPlotEnvGuard<'_> {
    fn drop(&mut self) {
        restore_env_var("RUNMAT_HOST_MANAGED_PLOTS", self.previous.take());
    }
}

fn set_plot_test_env_vars() {
    unsafe {
        std::env::set_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS", "1");
        std::env::set_var("RUNMAT_HOST_MANAGED_PLOTS", "1");
    }
}

fn restore_env_var(key: &'static str, previous: Option<std::ffi::OsString>) {
    unsafe {
        if let Some(previous) = previous {
            std::env::set_var(key, previous);
        } else {
            std::env::remove_var(key);
        }
    }
}

#[cfg(target_arch = "wasm32")]
runmat_thread_local! {
    static REGISTRY: RefCell<PlotRegistry> = RefCell::new(PlotRegistry::default());
}

#[cfg(not(target_arch = "wasm32"))]
type RegistryBackendGuard<'a> = MutexGuard<'a, PlotRegistry>;
#[cfg(target_arch = "wasm32")]
type RegistryBackendGuard<'a> = std::cell::RefMut<'a, PlotRegistry>;

struct PlotRegistryGuard<'a> {
    inner: RegistryBackendGuard<'a>,
    #[cfg(test)]
    _test_lock: Option<std::sync::MutexGuard<'static, ()>>,
}

impl<'a> PlotRegistryGuard<'a> {
    #[cfg(test)]
    fn new(
        inner: RegistryBackendGuard<'a>,
        _test_lock: Option<std::sync::MutexGuard<'static, ()>>,
    ) -> Self {
        Self { inner, _test_lock }
    }

    #[cfg(not(test))]
    fn new(inner: RegistryBackendGuard<'a>) -> Self {
        Self { inner }
    }
}

impl<'a> Deref for PlotRegistryGuard<'a> {
    type Target = PlotRegistry;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> DerefMut for PlotRegistryGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

const AXES_INDEX_BITS: u32 = 20;
const AXES_INDEX_MASK: u64 = (1 << AXES_INDEX_BITS) - 1;
const OBJECT_KIND_BITS: u32 = 4;
const OBJECT_KIND_MASK: u64 = (1 << OBJECT_KIND_BITS) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlotObjectKind {
    Title = 1,
    XLabel = 2,
    YLabel = 3,
    ZLabel = 4,
    Legend = 5,
    SuperTitle = 6,
    Subtitle = 7,
    XAxis = 8,
    YAxis = 9,
}

impl PlotObjectKind {
    fn from_u64(value: u64) -> Option<Self> {
        match value {
            1 => Some(Self::Title),
            2 => Some(Self::XLabel),
            3 => Some(Self::YLabel),
            4 => Some(Self::ZLabel),
            5 => Some(Self::Legend),
            6 => Some(Self::SuperTitle),
            7 => Some(Self::Subtitle),
            8 => Some(Self::XAxis),
            9 => Some(Self::YAxis),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum FigureError {
    #[error("figure handle {0} does not exist")]
    InvalidHandle(u32),
    #[error("subplot grid dimensions must be positive (rows={rows}, cols={cols})")]
    InvalidSubplotGrid { rows: usize, cols: usize },
    #[error("subplot index {index} is out of range for a {rows}x{cols} grid")]
    InvalidSubplotIndex {
        rows: usize,
        cols: usize,
        index: usize,
    },
    #[error("invalid axes handle")]
    InvalidAxesHandle,
    #[error("invalid plot object handle")]
    InvalidPlotObjectHandle,
    #[error("failed to render figure snapshot: {source}")]
    RenderFailure {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

fn map_figure_error(builtin: &'static str, err: FigureError) -> RuntimeError {
    let message = format!("{builtin}: {err}");
    plotting_error_with_source(builtin, message, err)
}

pub(crate) fn clear_figure_with_builtin(
    builtin: &'static str,
    target: Option<FigureHandle>,
) -> BuiltinResult<FigureHandle> {
    clear_figure(target).map_err(|err| map_figure_error(builtin, err))
}

pub(crate) fn close_figure_with_builtin(
    builtin: &'static str,
    target: Option<FigureHandle>,
) -> BuiltinResult<FigureHandle> {
    close_figure(target).map_err(|err| map_figure_error(builtin, err))
}

pub fn set_grid_enabled(enabled: bool) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_grid_enabled(axes, enabled);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_grid_and_minor_grid_enabled(grid_enabled: bool, minor_grid_enabled: Option<bool>) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_grid_enabled(axes, grid_enabled);
        if let Some(minor_enabled) = minor_grid_enabled {
            state
                .figure
                .set_axes_minor_grid_enabled(axes, minor_enabled);
        }
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_grid_enabled_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_grid_enabled(axes_index, enabled);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_minor_grid_enabled_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state
            .figure
            .set_axes_minor_grid_enabled(axes_index, enabled);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn toggle_grid() -> bool {
    let (handle, figure_clone, enabled) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        let next = !state
            .figure
            .axes_metadata(axes)
            .map(|m| m.grid_enabled)
            .unwrap_or(true);
        state.figure.set_axes_grid_enabled(axes, next);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone(), next)
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    enabled
}

pub fn toggle_minor_grid() -> bool {
    let (handle, figure_clone, enabled) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        let next = !state.figure.minor_grid_enabled_for_axes(axes);
        state.figure.set_axes_minor_grid_enabled(axes, next);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone(), next)
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    enabled
}

pub fn set_box_enabled(enabled: bool) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_box_enabled(axes, enabled);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_box_enabled_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_box_enabled(axes_index, enabled);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_hidden_line_removal_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state
            .figure
            .set_axes_hidden_line_removal(axes_index, enabled);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_axes_style_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    style: TextStyle,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_style(axes_index, style);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn default_color_order() -> Vec<Vec4> {
    let theme = current_plot_theme_config().build_theme();
    (0..8).map(|idx| theme.get_data_color(idx)).collect()
}

pub fn color_order_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<Vec<Vec4>, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    if let Some(order) = state
        .figure
        .axes_metadata(axes_index)
        .and_then(|meta| meta.color_order.clone())
    {
        return Ok(order);
    }
    Ok(state
        .figure_color_order
        .clone()
        .unwrap_or_else(default_color_order))
}

pub fn color_order_for_figure(handle: FigureHandle) -> Result<Vec<Vec4>, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    Ok(state
        .figure_color_order
        .clone()
        .unwrap_or_else(default_color_order))
}

pub fn set_color_order_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    colors: &[Vec4],
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = get_state_mut(&mut reg, handle);
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.color_cycle_for_axes_mut(axes_index).set_order(colors);
        state
            .figure
            .set_axes_color_order(axes_index, colors.to_vec());
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_color_order_for_figure(
    handle: FigureHandle,
    colors: &[Vec4],
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = get_state_mut(&mut reg, handle);
        state.figure_color_order = Some(colors.to_vec());
        let total_axes = axes_count(state);
        for axes_index in 0..total_axes {
            state.color_cycle_for_axes_mut(axes_index).set_order(colors);
        }
        state.figure.set_all_axes_color_order(colors.to_vec());
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_figure_title_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    title: &str,
    style: TextStyle,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_title(axes_index, title.to_string());
        state.figure.set_axes_title_style(axes_index, style);
        encode_plot_object_handle(handle, axes_index, PlotObjectKind::Title)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn set_figure_subtitle_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    subtitle: &str,
    style: TextStyle,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state
            .figure
            .set_axes_subtitle(axes_index, subtitle.to_string());
        state.figure.set_axes_subtitle_style(axes_index, style);
        encode_plot_object_handle(handle, axes_index, PlotObjectKind::Subtitle)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn set_sg_title_for_figure(
    handle: FigureHandle,
    title: &str,
    style: TextStyle,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_figure_mut(handle, |state| {
        state.figure.set_sg_title(title.to_string());
        state.figure.set_sg_title_style(style);
        encode_plot_object_handle(handle, 0, PlotObjectKind::SuperTitle)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn set_sg_title_properties_for_figure(
    handle: FigureHandle,
    text: Option<String>,
    style: Option<TextStyle>,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_figure_mut(handle, |state| {
        if let Some(text) = text {
            state.figure.set_sg_title(text);
        }
        if let Some(style) = style {
            state.figure.set_sg_title_style(style);
        }
        encode_plot_object_handle(handle, 0, PlotObjectKind::SuperTitle)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn set_figure_name(handle: FigureHandle, name: String) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.figure.set_name(name);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_figure_tag(handle: FigureHandle, tag: String) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.tag = tag;
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_figure_number_title(handle: FigureHandle, enabled: bool) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.figure.set_number_title(enabled);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_figure_visible(
    handle: FigureHandle,
    visible: bool,
) -> Result<(bool, Figure), FigureError> {
    let ((was_visible, now_visible), figure_clone) = with_figure_mut(handle, |state| {
        let was_visible = state.figure.visible;
        state.figure.set_visible(visible);
        (was_visible, state.figure.visible)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok((!was_visible && now_visible, figure_clone))
}

pub fn set_figure_position(handle: FigureHandle, position: [f64; 4]) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.figure.set_position(position);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_figure_background_color(handle: FigureHandle, color: Vec4) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.figure.set_background_color(color);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_text_properties_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    kind: PlotObjectKind,
    text: Option<String>,
    style: Option<TextStyle>,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        if let Some(text) = text {
            match kind {
                PlotObjectKind::Title => state.figure.set_axes_title(axes_index, text),
                PlotObjectKind::Subtitle => state.figure.set_axes_subtitle(axes_index, text),
                PlotObjectKind::XLabel => state.figure.set_axes_xlabel(axes_index, text),
                PlotObjectKind::YLabel => state.figure.set_axes_ylabel(axes_index, text),
                PlotObjectKind::ZLabel => state.figure.set_axes_zlabel(axes_index, text),
                PlotObjectKind::Legend | PlotObjectKind::XAxis | PlotObjectKind::YAxis => {}
                PlotObjectKind::SuperTitle => state.figure.set_sg_title(text),
            }
        }
        if let Some(style) = style {
            match kind {
                PlotObjectKind::Title => state.figure.set_axes_title_style(axes_index, style),
                PlotObjectKind::Subtitle => state.figure.set_axes_subtitle_style(axes_index, style),
                PlotObjectKind::XLabel => state.figure.set_axes_xlabel_style(axes_index, style),
                PlotObjectKind::YLabel => state.figure.set_axes_ylabel_style(axes_index, style),
                PlotObjectKind::ZLabel => state.figure.set_axes_zlabel_style(axes_index, style),
                PlotObjectKind::Legend | PlotObjectKind::XAxis | PlotObjectKind::YAxis => {}
                PlotObjectKind::SuperTitle => state.figure.set_sg_title_style(style),
            }
        }
        encode_plot_object_handle(handle, axes_index, kind)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn set_xlabel_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    label: &str,
    style: TextStyle,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_xlabel(axes_index, label.to_string());
        state.figure.set_axes_xlabel_style(axes_index, style);
        encode_plot_object_handle(handle, axes_index, PlotObjectKind::XLabel)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn set_ylabel_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    label: &str,
    style: TextStyle,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_ylabel(axes_index, label.to_string());
        state.figure.set_axes_ylabel_style(axes_index, style);
        encode_plot_object_handle(handle, axes_index, PlotObjectKind::YLabel)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn set_zlabel_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    label: &str,
    style: TextStyle,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_zlabel(axes_index, label.to_string());
        state.figure.set_axes_zlabel_style(axes_index, style);
        encode_plot_object_handle(handle, axes_index, PlotObjectKind::ZLabel)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn add_text_annotation_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    position: glam::Vec3,
    text: &str,
    style: TextStyle,
) -> Result<f64, FigureError> {
    let (annotation_index, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state
            .figure
            .add_axes_text_annotation(axes_index, position, text.to_string(), style)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(register_text_annotation_handle(
        handle,
        axes_index,
        annotation_index,
    ))
}

pub fn set_text_annotation_properties_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    annotation_index: usize,
    text: Option<String>,
    position: Option<glam::Vec3>,
    style: Option<TextStyle>,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        if let Some(text) = text {
            state
                .figure
                .set_axes_text_annotation_text(axes_index, annotation_index, text);
        }
        if let Some(position) = position {
            state
                .figure
                .set_axes_text_annotation_position(axes_index, annotation_index, position);
        }
        if let Some(style) = style {
            state
                .figure
                .set_axes_text_annotation_style(axes_index, annotation_index, style);
        }
        register_text_annotation_handle(handle, axes_index, annotation_index)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn toggle_box() -> bool {
    let (handle, figure_clone, enabled) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        let next = !state
            .figure
            .axes_metadata(axes)
            .map(|m| m.box_enabled)
            .unwrap_or(true);
        state.figure.set_axes_box_enabled(axes, next);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone(), next)
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    enabled
}

pub fn set_axis_equal(enabled: bool) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_axis_equal(axes, enabled);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_axis_equal_and_limits(enabled: bool, x: Option<(f64, f64)>, y: Option<(f64, f64)>) {
    let updates = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_axis_equal(axes, enabled);
        set_axes_limits_with_links(&mut reg, handle, axes, x, y)
            .expect("active axes target should be valid")
    };
    notify_figure_updates(updates);
}

pub fn set_axis_equal_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_axis_equal(axes_index, enabled);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn data_aspect_ratio_snapshot() -> ([f64; 3], String) {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes = state.active_axes;
    state
        .figure
        .axes_metadata(axes)
        .map(|meta| (meta.data_aspect_ratio, meta.data_aspect_ratio_mode.clone()))
        .unwrap_or(([1.0, 1.0, 1.0], "auto".into()))
}

pub fn data_aspect_ratio_snapshot_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<([f64; 3], String), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    validate_axes_index(state, axes_index)?;
    Ok(state
        .figure
        .axes_metadata(axes_index)
        .map(|meta| (meta.data_aspect_ratio, meta.data_aspect_ratio_mode.clone()))
        .unwrap_or(([1.0, 1.0, 1.0], "auto".into())))
}

pub fn set_data_aspect_ratio(ratio: [f64; 3], mode: &str) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state
            .figure
            .set_axes_data_aspect_ratio(axes, ratio, mode.to_string());
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_data_aspect_ratio_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    ratio: [f64; 3],
    mode: &str,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = get_state_mut(&mut reg, handle);
        validate_axes_index(state, axes_index)?;
        state
            .figure
            .set_axes_data_aspect_ratio(axes_index, ratio, mode.to_string());
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

fn clone_touched_figures(
    registry: &mut PlotRegistry,
    touched: HashSet<FigureHandle>,
) -> Vec<(FigureHandle, Figure)> {
    touched
        .into_iter()
        .filter_map(|handle| {
            let state = registry.figures.get_mut(&handle)?;
            state.revision = state.revision.wrapping_add(1);
            Some((handle, state.figure.clone()))
        })
        .collect()
}

fn set_axes_limits_with_links(
    registry: &mut PlotRegistry,
    source_handle: FigureHandle,
    source_axes: usize,
    x: Option<(f64, f64)>,
    y: Option<(f64, f64)>,
) -> Result<Vec<(FigureHandle, Figure)>, FigureError> {
    let source_state = registry
        .figures
        .get(&source_handle)
        .ok_or_else(|| FigureError::InvalidHandle(source_handle.as_u32()))?;
    if source_axes >= axes_count(source_state) {
        return Err(FigureError::InvalidSubplotIndex {
            rows: source_state.figure.axes_rows.max(1),
            cols: source_state.figure.axes_cols.max(1),
            index: source_axes,
        });
    }

    let x_group = registry
        .link_axes_groups
        .iter()
        .find(|group| {
            group.axis == LinkAxesAxis::X && group.targets.contains(&(source_handle, source_axes))
        })
        .cloned();
    let y_group = registry
        .link_axes_groups
        .iter()
        .find(|group| {
            group.axis == LinkAxesAxis::Y && group.targets.contains(&(source_handle, source_axes))
        })
        .cloned();
    if x_group.is_none() && y_group.is_none() {
        let state = registry
            .figures
            .get_mut(&source_handle)
            .expect("validated source axes target should exist");
        state.figure.set_axes_limits(source_axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        return Ok(vec![(source_handle, state.figure.clone())]);
    };

    let mut updates = HashMap::new();
    updates.insert((source_handle, source_axes), (x, y));
    if let Some(group) = x_group {
        for &(handle, axes_index) in &group.targets {
            let Some(state) = registry.figures.get(&handle) else {
                continue;
            };
            if axes_index >= axes_count(state) {
                continue;
            }
            let Some(meta) = state.figure.axes_metadata(axes_index) else {
                continue;
            };
            updates
                .entry((handle, axes_index))
                .or_insert((meta.x_limits, meta.y_limits))
                .0 = x;
        }
    }
    if let Some(group) = y_group {
        for &(handle, axes_index) in &group.targets {
            let Some(state) = registry.figures.get(&handle) else {
                continue;
            };
            if axes_index >= axes_count(state) {
                continue;
            }
            let Some(meta) = state.figure.axes_metadata(axes_index) else {
                continue;
            };
            updates
                .entry((handle, axes_index))
                .or_insert((meta.x_limits, meta.y_limits))
                .1 = y;
        }
    }

    let mut touched = HashSet::new();
    for ((handle, axes_index), (x_limits, y_limits)) in updates {
        let Some(state) = registry.figures.get_mut(&handle) else {
            continue;
        };
        if axes_index >= axes_count(state) {
            continue;
        }
        state.figure.set_axes_limits(axes_index, x_limits, y_limits);
        touched.insert(handle);
    }
    Ok(clone_touched_figures(registry, touched))
}

fn notify_figure_updates(updates: Vec<(FigureHandle, Figure)>) {
    for (handle, figure) in updates {
        notify_with_figure(handle, &figure, FigureEventKind::Updated);
    }
}

pub fn set_axis_limits(x: Option<(f64, f64)>, y: Option<(f64, f64)>) {
    let updates = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        set_axes_limits_with_links(&mut reg, handle, axes, x, y)
            .expect("active axes target should be valid")
    };
    notify_figure_updates(updates);
}

pub fn set_axis_limits_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    x: Option<(f64, f64)>,
    y: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let updates = {
        let mut reg = registry();
        let state = get_state_mut(&mut reg, handle);
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.active_axes = axes_index;
        state.figure.set_active_axes_index(axes_index);
        set_axes_limits_with_links(&mut reg, handle, axes_index, x, y)?
    };
    notify_figure_updates(updates);
    Ok(())
}

fn axis_limits_for_link_axis(
    state: &mut FigureState,
    axes_index: usize,
    axis: LinkAxesAxis,
) -> Option<(f64, f64)> {
    if let Some(meta) = state.figure.axes_metadata(axes_index) {
        let explicit = match axis {
            LinkAxesAxis::X => meta.x_limits,
            LinkAxesAxis::Y => meta.y_limits,
        };
        if let Some((lo, hi)) = explicit {
            return Some((lo.min(hi), lo.max(hi)));
        }
    }
    let display = display_bounds_for_state_axes(state, axes_index)?;
    let limits = match axis {
        LinkAxesAxis::X => (display.0, display.1),
        LinkAxesAxis::Y => (display.2, display.3),
    };
    if limits.0.is_finite() && limits.1.is_finite() {
        Some((limits.0.min(limits.1), limits.0.max(limits.1)))
    } else {
        None
    }
}

fn union_link_axis_limits(
    registry: &mut PlotRegistry,
    targets: &[(FigureHandle, usize)],
    axis: LinkAxesAxis,
) -> Option<(f64, f64)> {
    let mut union = None;
    for &(handle, axes_index) in targets {
        let Some(state) = registry.figures.get_mut(&handle) else {
            continue;
        };
        if axes_index >= axes_count(state) {
            continue;
        }
        let Some((lo, hi)) = axis_limits_for_link_axis(state, axes_index, axis) else {
            continue;
        };
        union = Some(match union {
            Some((current_lo, current_hi)) => (f64::min(current_lo, lo), f64::max(current_hi, hi)),
            None => (lo, hi),
        });
    }
    union
}

fn remove_link_axes_targets(
    registry: &mut PlotRegistry,
    target_set: &HashSet<(FigureHandle, usize)>,
    axis: Option<LinkAxesAxis>,
) {
    for group in &mut registry.link_axes_groups {
        if axis.is_none_or(|axis| group.axis == axis) {
            group.targets.retain(|target| !target_set.contains(target));
        }
    }
    registry
        .link_axes_groups
        .retain(|group| group.targets.len() >= 2);
}

fn purge_link_axes_for_figure(registry: &mut PlotRegistry, handle: FigureHandle) {
    for group in &mut registry.link_axes_groups {
        group
            .targets
            .retain(|(target_handle, _)| *target_handle != handle);
    }
    registry
        .link_axes_groups
        .retain(|group| group.targets.len() >= 2);
}

pub fn link_axes(
    targets: Vec<(FigureHandle, usize)>,
    mode: Option<LinkAxesMode>,
) -> Result<(), FigureError> {
    let updates = {
        let mut reg = registry();
        let mut seen = HashSet::new();
        let targets = targets
            .into_iter()
            .filter(|target| seen.insert(*target))
            .collect::<Vec<_>>();

        for &(handle, axes_index) in &targets {
            let state = reg
                .figures
                .get(&handle)
                .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
            if axes_index >= axes_count(state) {
                return Err(FigureError::InvalidSubplotIndex {
                    rows: state.figure.axes_rows.max(1),
                    cols: state.figure.axes_cols.max(1),
                    index: axes_index,
                });
            }
        }

        let target_set = targets.iter().copied().collect::<HashSet<_>>();
        let Some(mode) = mode else {
            remove_link_axes_targets(&mut reg, &target_set, None);
            return Ok(());
        };
        if mode.links_x() {
            remove_link_axes_targets(&mut reg, &target_set, Some(LinkAxesAxis::X));
        }
        if mode.links_y() {
            remove_link_axes_targets(&mut reg, &target_set, Some(LinkAxesAxis::Y));
        }
        if targets.len() < 2 {
            return Ok(());
        }

        let (source_handle, source_axes) = targets[0];
        let x_sync = if mode.links_x() {
            union_link_axis_limits(&mut reg, &targets, LinkAxesAxis::X)
        } else {
            None
        };
        let y_sync = if mode.links_y() {
            union_link_axis_limits(&mut reg, &targets, LinkAxesAxis::Y)
        } else {
            None
        };
        let source_meta = reg
            .figures
            .get(&source_handle)
            .and_then(|state| state.figure.axes_metadata(source_axes))
            .cloned()
            .ok_or(FigureError::InvalidAxesHandle)?;
        if mode.links_x() {
            reg.link_axes_groups.push(LinkAxesGroup {
                axis: LinkAxesAxis::X,
                targets: targets.clone(),
            });
        }
        if mode.links_y() {
            reg.link_axes_groups.push(LinkAxesGroup {
                axis: LinkAxesAxis::Y,
                targets,
            });
        }
        set_axes_limits_with_links(
            &mut reg,
            source_handle,
            source_axes,
            if mode.links_x() {
                x_sync
            } else {
                source_meta.x_limits
            },
            if mode.links_y() {
                y_sync
            } else {
                source_meta.y_limits
            },
        )?
    };
    notify_figure_updates(updates);
    Ok(())
}

pub fn set_axis_ticks(x: Option<Vec<f64>>, y: Option<Vec<f64>>) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_ticks(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_axis_ticks_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.figure.set_axes_ticks(axes_index, x, y);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_axis_tick_labels(x: Option<Vec<String>>, y: Option<Vec<String>>) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_tick_labels(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_axis_tick_labels_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    x: Option<Vec<String>>,
    y: Option<Vec<String>>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.figure.set_axes_tick_labels(axes_index, x, y);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_axis_tick_formats(x: Option<String>, y: Option<String>) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_tick_formats(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_axis_tick_formats_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    x: Option<String>,
    y: Option<String>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.figure.set_axes_tick_formats(axes_index, x, y);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_axis_tick_angles(x: Option<f64>, y: Option<f64>) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_tick_label_rotations(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_axis_tick_angles_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    x: Option<f64>,
    y: Option<f64>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.figure.set_axes_tick_label_rotations(axes_index, x, y);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn axis_limits_snapshot() -> AxisLimitSnapshot {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes = state.active_axes;
    let meta = state
        .figure
        .axes_metadata(axes)
        .cloned()
        .unwrap_or_default();
    (meta.x_limits, meta.y_limits)
}

pub fn axis_ticks_snapshot() -> AxisTickSnapshot {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes = state.active_axes;
    let meta = state
        .figure
        .axes_metadata(axes)
        .cloned()
        .unwrap_or_default();
    (meta.x_ticks, meta.y_ticks)
}

pub fn axis_tick_labels_snapshot() -> AxisTickLabelSnapshot {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes = state.active_axes;
    let meta = state
        .figure
        .axes_metadata(axes)
        .cloned()
        .unwrap_or_default();
    (meta.x_tick_labels, meta.y_tick_labels)
}

pub fn axis_tick_formats_snapshot() -> AxisTickFormatSnapshot {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes = state.active_axes;
    let meta = state
        .figure
        .axes_metadata(axes)
        .cloned()
        .unwrap_or_default();
    (meta.x_tick_format, meta.y_tick_format)
}

pub fn axis_tick_angles_snapshot() -> AxisTickAngleSnapshot {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes = state.active_axes;
    let meta = state
        .figure
        .axes_metadata(axes)
        .cloned()
        .unwrap_or_default();
    (meta.x_tick_label_rotation, meta.y_tick_label_rotation)
}

pub fn axis_ticks_snapshot_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisTickSnapshot, FigureError> {
    let reg = registry();
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    Ok((meta.x_ticks, meta.y_ticks))
}

pub fn axis_tick_labels_snapshot_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisTickLabelSnapshot, FigureError> {
    let reg = registry();
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    Ok((meta.x_tick_labels, meta.y_tick_labels))
}

pub fn axis_tick_formats_snapshot_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisTickFormatSnapshot, FigureError> {
    let reg = registry();
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    Ok((meta.x_tick_format, meta.y_tick_format))
}

pub fn axis_tick_angles_snapshot_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisTickAngleSnapshot, FigureError> {
    let reg = registry();
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    Ok((meta.x_tick_label_rotation, meta.y_tick_label_rotation))
}

pub fn axis_display_bounds_snapshot() -> AxisDisplayBoundsSnapshot {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes_index = state.active_axes;
    display_bounds_for_state_axes(state, axes_index)
}

pub fn axis_display_bounds_snapshot_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisDisplayBoundsSnapshot, FigureError> {
    let mut reg = registry();
    let state = reg
        .figures
        .get_mut(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    Ok(display_bounds_for_state_axes(state, axes_index))
}

fn display_bounds_for_state_axes(
    state: &mut FigureState,
    axes_index: usize,
) -> AxisDisplayBoundsSnapshot {
    let bounds = state.figure.data_bounds_for_axes(axes_index);
    if !(bounds.min.x.is_finite()
        && bounds.max.x.is_finite()
        && bounds.min.y.is_finite()
        && bounds.max.y.is_finite())
    {
        return None;
    }

    let mut x_min = bounds.min.x as f64;
    let mut x_max = bounds.max.x as f64;
    let mut y_min = bounds.min.y as f64;
    let mut y_max = bounds.max.y as f64;

    if let Some(meta) = state.figure.axes_metadata(axes_index) {
        if let Some((lo, hi)) = meta.x_limits {
            x_min = lo;
            x_max = hi;
        }
        if let Some((lo, hi)) = meta.y_limits {
            y_min = lo;
            y_max = hi;
        }
        if meta.axis_equal || meta.data_aspect_ratio_mode == "manual" {
            (x_min, x_max, y_min, y_max) =
                data_aspect_adjusted_bounds(x_min, x_max, y_min, y_max, meta.data_aspect_ratio);
        }
    }

    Some((x_min, x_max, y_min, y_max))
}

fn data_aspect_adjusted_bounds(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    ratio: [f64; 3],
) -> (f64, f64, f64, f64) {
    let x_ratio = ratio[0].abs().max(1.0e-12);
    let y_ratio = ratio[1].abs().max(1.0e-12);
    let cx = (x_min + x_max) * 0.5;
    let cy = (y_min + y_max) * 0.5;
    let x_span = (x_max - x_min).abs().max(0.1);
    let y_span = (y_max - y_min).abs().max(0.1);
    let units = (x_span / x_ratio).max(y_span / y_ratio).max(0.1);
    let next_x = units * x_ratio;
    let next_y = units * y_ratio;
    (
        cx - next_x * 0.5,
        cx + next_x * 0.5,
        cy - next_y * 0.5,
        cy + next_y * 0.5,
    )
}

fn axes_count(state: &FigureState) -> usize {
    state.figure.axes_count()
}

fn validate_axes_index(state: &FigureState, axes_index: usize) -> Result<(), FigureError> {
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    Ok(())
}

fn zoom_mode_for_target(state: &FigureState, axes_index: Option<usize>) -> ZoomModeState {
    axes_index
        .and_then(|index| state.zoom_axes_modes.get(&index).copied())
        .unwrap_or(state.zoom_mode)
}

fn apply_zoom_mode_command(
    mode: &mut ZoomModeState,
    last_enabled_motion: &mut ZoomMotion,
    command: ZoomModeCommand,
) {
    match command {
        ZoomModeCommand::On => {
            mode.enabled = true;
            mode.motion = ZoomMotion::Both;
            *last_enabled_motion = ZoomMotion::Both;
        }
        ZoomModeCommand::Off => {
            mode.enabled = false;
        }
        ZoomModeCommand::Toggle => {
            mode.enabled = !mode.enabled;
            if mode.enabled {
                mode.motion = *last_enabled_motion;
            }
        }
        ZoomModeCommand::XOn => {
            mode.enabled = true;
            mode.motion = ZoomMotion::Horizontal;
            *last_enabled_motion = ZoomMotion::Horizontal;
        }
        ZoomModeCommand::YOn => {
            mode.enabled = true;
            mode.motion = ZoomMotion::Vertical;
            *last_enabled_motion = ZoomMotion::Vertical;
        }
    }
}

fn axis_limit_basis_for_zoom(state: &mut FigureState, axes_index: usize) -> AxisLimitSnapshot {
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    let bounds = display_bounds_for_state_axes(state, axes_index);
    let x = meta
        .x_limits
        .or_else(|| bounds.map(|(x_min, x_max, _, _)| (x_min, x_max)))
        .or(Some((0.0, 1.0)));
    let y = meta
        .y_limits
        .or_else(|| bounds.map(|(_, _, y_min, y_max)| (y_min, y_max)))
        .or(Some((0.0, 1.0)));
    (x, y)
}

fn zoom_interval(limits: (f64, f64), factor: f64) -> (f64, f64) {
    let (lo, hi) = limits;
    let center = (lo + hi) * 0.5;
    let span = (hi - lo).abs().max(f64::EPSILON) / factor;
    (center - span * 0.5, center + span * 0.5)
}

fn zoom_factor_limits_for_state(
    state: &mut FigureState,
    axes_index: usize,
    factor: f64,
) -> AxisLimitSnapshot {
    let mode = zoom_mode_for_target(state, Some(axes_index));
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    let (basis_x, basis_y) = axis_limit_basis_for_zoom(state, axes_index);
    let x_limits = match (mode.motion, basis_x) {
        (ZoomMotion::Vertical, _) => meta.x_limits,
        (_, Some(limits)) => Some(zoom_interval(limits, factor)),
        (_, None) => meta.x_limits,
    };
    let y_limits = match (mode.motion, basis_y) {
        (ZoomMotion::Horizontal, _) => meta.y_limits,
        (_, Some(limits)) => Some(zoom_interval(limits, factor)),
        (_, None) => meta.y_limits,
    };
    (x_limits, y_limits)
}

fn zoom_baseline_limits_for_state(state: &FigureState, axes_index: usize) -> AxisLimitSnapshot {
    state
        .zoom_baselines
        .get(&axes_index)
        .copied()
        .unwrap_or((None, None))
}

pub fn zoom_state_snapshot(
    handle: FigureHandle,
    axes_index: Option<usize>,
) -> Result<ZoomStateSnapshot, FigureError> {
    let reg = registry();
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    if let Some(index) = axes_index {
        validate_axes_index(state, index)?;
    }
    Ok(ZoomStateSnapshot {
        figure: handle,
        axes_index,
        mode: zoom_mode_for_target(state, axes_index),
    })
}

pub fn set_zoom_mode_for_figure(
    handle: FigureHandle,
    command: ZoomModeCommand,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        apply_zoom_mode_command(
            &mut state.zoom_mode,
            &mut state.last_enabled_zoom_motion,
            command,
        );
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_zoom_mode_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    command: ZoomModeCommand,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        let mut mode = zoom_mode_for_target(state, Some(axes_index));
        apply_zoom_mode_command(&mut mode, &mut state.last_enabled_zoom_motion, command);
        state.zoom_axes_modes.insert(axes_index, mode);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_zoom_motion_for_figure(
    handle: FigureHandle,
    motion: ZoomMotion,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        state.zoom_mode.motion = motion;
        if state.zoom_mode.enabled {
            state.last_enabled_zoom_motion = motion;
        }
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_zoom_enabled_for_figure(handle: FigureHandle, enabled: bool) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        state.zoom_mode.enabled = enabled;
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_zoom_direction_for_figure(
    handle: FigureHandle,
    direction: ZoomDirection,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        state.zoom_mode.direction = direction;
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_zoom_right_click_action_for_figure(
    handle: FigureHandle,
    action: ZoomRightClickAction,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        state.zoom_mode.right_click_action = action;
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_zoom_legacy_for_figure(handle: FigureHandle, enabled: bool) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        state.zoom_mode.use_legacy_exploration_modes = enabled;
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn reset_zoom_baseline_for_figure(handle: FigureHandle) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        for axes_index in 0..axes_count(state) {
            let snapshot = axis_limit_basis_for_zoom(state, axes_index);
            state.zoom_baselines.insert(axes_index, snapshot);
        }
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn reset_zoom_baseline_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        let snapshot = axis_limit_basis_for_zoom(state, axes_index);
        state.zoom_baselines.insert(axes_index, snapshot);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn restore_zoom_baseline_for_figure(handle: FigureHandle) -> Result<(), FigureError> {
    let updates = {
        let mut reg = registry();
        let limits = {
            let state = reg
                .figures
                .get(&handle)
                .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
            (0..axes_count(state))
                .map(|axes_index| {
                    (
                        axes_index,
                        zoom_baseline_limits_for_state(state, axes_index),
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut updates = Vec::new();
        for (axes_index, (x_limits, y_limits)) in limits {
            updates.extend(set_axes_limits_with_links(
                &mut reg, handle, axes_index, x_limits, y_limits,
            )?);
        }
        updates
    };
    notify_figure_updates(updates);
    Ok(())
}

pub fn restore_zoom_baseline_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<(), FigureError> {
    let updates = {
        let mut reg = registry();
        let (x_limits, y_limits) = {
            let state = reg
                .figures
                .get(&handle)
                .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
            validate_axes_index(state, axes_index)?;
            zoom_baseline_limits_for_state(state, axes_index)
        };
        set_axes_limits_with_links(&mut reg, handle, axes_index, x_limits, y_limits)?
    };
    notify_figure_updates(updates);
    Ok(())
}

pub fn apply_zoom_factor_for_figure(handle: FigureHandle, factor: f64) -> Result<(), FigureError> {
    let updates = {
        let mut reg = registry();
        let (axes_index, x_limits, y_limits) = {
            let state = reg
                .figures
                .get_mut(&handle)
                .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
            let axes_index = state.active_axes;
            let (x_limits, y_limits) = zoom_factor_limits_for_state(state, axes_index, factor);
            (axes_index, x_limits, y_limits)
        };
        set_axes_limits_with_links(&mut reg, handle, axes_index, x_limits, y_limits)?
    };
    notify_figure_updates(updates);
    Ok(())
}

pub fn apply_zoom_factor_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    factor: f64,
) -> Result<(), FigureError> {
    let updates = {
        let mut reg = registry();
        let (x_limits, y_limits) = {
            let state = reg
                .figures
                .get_mut(&handle)
                .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
            validate_axes_index(state, axes_index)?;
            zoom_factor_limits_for_state(state, axes_index, factor)
        };
        set_axes_limits_with_links(&mut reg, handle, axes_index, x_limits, y_limits)?
    };
    notify_figure_updates(updates);
    Ok(())
}

fn pan_mode_for_target(state: &FigureState, axes_index: Option<usize>) -> PanModeState {
    axes_index
        .and_then(|index| state.pan_axes_modes.get(&index).copied())
        .unwrap_or(state.pan_mode)
}

fn apply_pan_mode_command(
    mode: &mut PanModeState,
    last_enabled_motion: &mut ZoomMotion,
    command: PanModeCommand,
) {
    match command {
        PanModeCommand::On => {
            mode.enabled = true;
            mode.motion = *last_enabled_motion;
        }
        PanModeCommand::Off => mode.enabled = false,
        PanModeCommand::Toggle => {
            mode.enabled = !mode.enabled;
            if mode.enabled {
                mode.motion = *last_enabled_motion;
            }
        }
        PanModeCommand::XOn => {
            mode.enabled = true;
            mode.motion = ZoomMotion::Horizontal;
            *last_enabled_motion = ZoomMotion::Horizontal;
        }
        PanModeCommand::YOn => {
            mode.enabled = true;
            mode.motion = ZoomMotion::Vertical;
            *last_enabled_motion = ZoomMotion::Vertical;
        }
    }
}

pub fn pan_state_snapshot(
    handle: FigureHandle,
    axes_index: Option<usize>,
) -> Result<PanStateSnapshot, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    if let Some(index) = axes_index {
        validate_axes_index(state, index)?;
    }
    Ok(PanStateSnapshot {
        figure: handle,
        axes_index,
        mode: pan_mode_for_target(state, axes_index),
    })
}

pub fn set_pan_mode_for_figure(
    handle: FigureHandle,
    command: PanModeCommand,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        apply_pan_mode_command(
            &mut state.pan_mode,
            &mut state.last_enabled_pan_motion,
            command,
        );
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_pan_mode_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    command: PanModeCommand,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        let mut mode = pan_mode_for_target(state, Some(axes_index));
        apply_pan_mode_command(&mut mode, &mut state.last_enabled_pan_motion, command);
        state.pan_axes_modes.insert(axes_index, mode);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_pan_enabled_for_figure(handle: FigureHandle, enabled: bool) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.pan_mode.enabled = enabled;
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_pan_motion_for_figure(
    handle: FigureHandle,
    motion: ZoomMotion,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.pan_mode.motion = motion;
        state.last_enabled_pan_motion = motion;
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn data_cursor_state_snapshot(
    handle: FigureHandle,
) -> Result<DataCursorStateSnapshot, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    Ok(DataCursorStateSnapshot {
        figure: handle,
        mode: state.data_cursor_mode.clone(),
    })
}

pub fn set_data_cursor_enabled_for_figure(
    handle: FigureHandle,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.data_cursor_mode.enabled = enabled;
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_data_cursor_snap_to_data_vertex_for_figure(
    handle: FigureHandle,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.data_cursor_mode.snap_to_data_vertex = enabled;
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_data_cursor_display_style_for_figure(
    handle: FigureHandle,
    style: String,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        state.data_cursor_mode.display_style = style;
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_waitbar_state(
    handle: FigureHandle,
    progress: f64,
    message: String,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut(handle, |state| {
        if state.tag.is_empty() {
            state.tag = "TMWWaitbar".into();
        }
        state.waitbar = Some(WaitbarState { progress, message });
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn waitbar_state_snapshot(handle: FigureHandle) -> Result<Option<WaitbarState>, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    Ok(state.waitbar.clone())
}

pub fn current_or_any_waitbar_handle() -> Option<FigureHandle> {
    let reg = registry();
    if let Some(state) = reg.figures.get(&reg.current) {
        if state.waitbar.is_some() {
            return Some(reg.current);
        }
    }
    reg.figures
        .iter()
        .find_map(|(handle, state)| state.waitbar.as_ref().map(|_| *handle))
}

pub fn z_limits_snapshot() -> Option<(f64, f64)> {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes = state.active_axes;
    state.figure.axes_metadata(axes).and_then(|m| m.z_limits)
}

pub fn color_limits_snapshot() -> Option<(f64, f64)> {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let axes = state.active_axes;
    state
        .figure
        .axes_metadata(axes)
        .and_then(|m| m.color_limits)
}

pub fn set_z_limits(limits: Option<(f64, f64)>) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_z_limits(axes, limits);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_z_limits_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    limits: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_z_limits(axes_index, limits);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_color_limits_runtime(limits: Option<(f64, f64)>) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_color_limits(axes, limits);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_color_limits_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    limits: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_color_limits(axes_index, limits);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn clear_current_axes() {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let axes_index = {
            let state = get_state_mut(&mut reg, handle);
            let axes_index = state.active_axes;
            state.figure.clear_axes(axes_index);
            state.reset_cycle(axes_index);
            state.revision = state.revision.wrapping_add(1);
            axes_index
        };
        purge_plot_children_for_axes(&mut reg, handle, axes_index);
        let figure_clone = reg
            .figures
            .get(&handle)
            .expect("figure exists")
            .figure
            .clone();
        (handle, figure_clone)
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_colorbar_enabled(enabled: bool) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_colorbar_enabled(axes, enabled);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_colorbar_enabled_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_colorbar_enabled(axes_index, enabled);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_legend_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
    labels: Option<&[String]>,
    style: Option<LegendStyle>,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_legend_enabled(axes_index, enabled);
        if let Some(labels) = labels {
            state.figure.set_labels_for_axes(axes_index, labels);
        }
        if let Some(style) = style {
            state.figure.set_axes_legend_style(axes_index, style);
        }
        encode_plot_object_handle(handle, axes_index, PlotObjectKind::Legend)
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(object_handle)
}

pub fn set_log_modes_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    x_log: bool,
    y_log: bool,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.figure.set_axes_log_modes(axes_index, x_log, y_log);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_y_axis_location_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    location: String,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_y_axis_location(axes_index, location);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_axes_position_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    position: [f64; 4],
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_position(axes_index, position);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_axes_units_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    units: String,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_units(axes_index, units);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_view_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    azimuth_deg: f32,
    elevation_deg: f32,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state
            .figure
            .set_axes_view(axes_index, azimuth_deg, elevation_deg);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn legend_entries_snapshot(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<Vec<runmat_plot::plots::LegendEntry>, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    Ok(state.figure.legend_entries_for_axes(axes_index))
}

pub fn toggle_colorbar() -> bool {
    let (handle, figure_clone, enabled) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        let next = !state
            .figure
            .axes_metadata(axes)
            .map(|m| m.colorbar_enabled)
            .unwrap_or(false);
        state.figure.set_axes_colorbar_enabled(axes, next);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone(), next)
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    enabled
}

pub fn set_colormap_with_length(colormap: ColorMap, length: usize) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let direct_images = direct_image_limits_updates(&reg.plot_children, handle, None);
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_colormap(axes, colormap);
        state.colormap_lengths.insert(axes, length);
        refresh_direct_image_limits(&direct_images, axes, length, &mut state.figure);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_colormap_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    colormap: ColorMap,
) -> Result<(), FigureError> {
    set_colormap_for_axes_with_length(handle, axes_index, colormap, DEFAULT_COLORMAP_LENGTH)
}

pub fn set_colormap_for_axes_with_length(
    handle: FigureHandle,
    axes_index: usize,
    colormap: ColorMap,
    length: usize,
) -> Result<(), FigureError> {
    with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_colormap(axes_index, colormap);
        state.colormap_lengths.insert(axes_index, length);
    })?;
    let figure_clone = {
        let mut reg = registry();
        let direct_images =
            direct_image_limits_updates(&reg.plot_children, handle, Some(axes_index));
        if let Some(state) = reg.figures.get_mut(&handle) {
            refresh_direct_image_limits(&direct_images, axes_index, length, &mut state.figure);
            state.figure.clone()
        } else {
            return Err(FigureError::InvalidHandle(handle.0));
        }
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

fn direct_image_limits_updates(
    children: &HashMap<u64, PlotChildHandleState>,
    figure: FigureHandle,
    axes_index: Option<usize>,
) -> Vec<(usize, usize, bool)> {
    children
        .values()
        .filter_map(|child| match child {
            PlotChildHandleState::Image(image)
                if image.figure == figure
                    && axes_index.is_none_or(|axes| image.axes_index == axes)
                    && image.c_data_mapping == "direct" =>
            {
                Some((
                    image.axes_index,
                    image.plot_index,
                    image
                        .c_data
                        .as_ref()
                        .is_some_and(|tensor| tensor.integer_storage().is_some()),
                ))
            }
            _ => None,
        })
        .collect()
}

fn refresh_direct_image_limits(
    direct_images: &[(usize, usize, bool)],
    axes_index: usize,
    length: usize,
    plot_figure: &mut Figure,
) {
    for &(_, plot_index, integer) in direct_images
        .iter()
        .filter(|(axes, _, _)| *axes == axes_index)
    {
        if let Some(runmat_plot::plots::figure::PlotElement::Surface(surface)) =
            plot_figure.get_plot_mut(plot_index)
        {
            surface.set_color_limits(Some(
                crate::builtins::plotting::image::direct_colormap_limits(integer, length),
            ));
        }
    }
}

pub fn colormap_length_for_axes(handle: FigureHandle, axes_index: usize) -> usize {
    let reg = registry();
    reg.figures
        .get(&handle)
        .and_then(|state| state.colormap_lengths.get(&axes_index).copied())
        .unwrap_or(DEFAULT_COLORMAP_LENGTH)
}

pub fn current_colormap_length() -> usize {
    let reg = registry();
    reg.figures
        .get(&reg.current)
        .and_then(|state| state.colormap_lengths.get(&state.active_axes).copied())
        .unwrap_or(DEFAULT_COLORMAP_LENGTH)
}

pub fn set_surface_shading(mode: ShadingMode) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let plot_count = state.figure.len();
        for idx in 0..plot_count {
            if let Some(PlotElement::Surface(surface)) = state.figure.get_plot_mut(idx) {
                *surface = surface.clone().with_shading(mode);
            }
        }
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FigureEventKind {
    Created,
    Updated,
    Cleared,
    Closed,
}

#[derive(Clone, Copy)]
pub struct FigureEventView<'a> {
    pub handle: FigureHandle,
    pub kind: FigureEventKind,
    pub revision: Option<u64>,
    pub figure: Option<&'a Figure>,
}

type FigureObserver = dyn for<'a> Fn(FigureEventView<'a>) + Send + Sync + 'static;

struct FigureObserverRegistry {
    observers: Mutex<Vec<Arc<FigureObserver>>>,
}

impl FigureObserverRegistry {
    fn new() -> Self {
        Self {
            observers: Mutex::new(Vec::new()),
        }
    }

    fn install(&self, observer: Arc<FigureObserver>) {
        let mut guard = self.observers.lock().expect("figure observers poisoned");
        guard.push(observer);
    }

    fn notify(&self, view: FigureEventView<'_>) {
        let snapshot = {
            let guard = self.observers.lock().expect("figure observers poisoned");
            guard.clone()
        };
        for observer in snapshot {
            observer(view);
        }
    }

    fn is_empty(&self) -> bool {
        self.observers
            .lock()
            .map(|guard| guard.is_empty())
            .unwrap_or(true)
    }
}

static FIGURE_OBSERVERS: OnceCell<FigureObserverRegistry> = OnceCell::new();

runmat_thread_local! {
    static RECENT_FIGURES: RefCell<HashSet<FigureHandle>> = RefCell::new(HashSet::new());
    static ACTIVE_AXES_CONTEXT: RefCell<Option<ActiveAxesContext>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug)]
pub struct FigureAxesState {
    pub handle: FigureHandle,
    pub rows: usize,
    pub cols: usize,
    pub active_index: usize,
}

pub fn encode_axes_handle(handle: FigureHandle, axes_index: usize) -> f64 {
    let encoded =
        ((handle.as_u32() as u64) << AXES_INDEX_BITS) | ((axes_index as u64) & AXES_INDEX_MASK);
    encoded as f64
}

pub fn encode_plot_object_handle(
    handle: FigureHandle,
    axes_index: usize,
    kind: PlotObjectKind,
) -> f64 {
    let encoded = (((handle.as_u32() as u64) << AXES_INDEX_BITS)
        | ((axes_index as u64) & AXES_INDEX_MASK))
        << OBJECT_KIND_BITS
        | ((kind as u64) & OBJECT_KIND_MASK);
    encoded as f64
}

pub fn decode_plot_object_handle(
    value: f64,
) -> Result<(FigureHandle, usize, PlotObjectKind), FigureError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let encoded = value.round() as u64;
    let kind = PlotObjectKind::from_u64(encoded & OBJECT_KIND_MASK)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    let base = encoded >> OBJECT_KIND_BITS;
    let figure_id = base >> AXES_INDEX_BITS;
    if figure_id == 0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let axes_index = (base & AXES_INDEX_MASK) as usize;
    Ok((FigureHandle::from(figure_id as u32), axes_index, kind))
}

pub fn register_histogram_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    bin_edges: Vec<f64>,
    raw_counts: Vec<f64>,
    normalization: String,
    normalization_denominator: f64,
    metadata: HistogramHandleMetadata,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::Histogram(HistogramHandleState {
            figure,
            axes_index,
            plot_index,
            bin_edges,
            raw_counts,
            normalization,
            normalization_denominator,
            display_name: None,
            metadata,
        }),
    );
    id as f64
}

#[allow(clippy::too_many_arguments)]
pub fn register_histogram2_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    values: Tensor,
    raw_counts: Tensor,
    x_bin_edges: Vec<f64>,
    y_bin_edges: Vec<f64>,
    normalization: String,
    normalization_denominator: f64,
    display_style: crate::builtins::plotting::histogram2::Histogram2DisplayStyle,
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<String>,
    data: Option<Tensor>,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::Histogram2(Histogram2HandleState {
            figure,
            axes_index,
            plot_index,
            values,
            raw_counts,
            x_bin_edges,
            y_bin_edges,
            normalization,
            normalization_denominator,
            display_style,
            show_empty_bins,
            face_alpha,
            display_name,
            data,
        }),
    );
    id as f64
}

fn register_simple_plot_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    constructor: fn(SimplePlotHandleState) -> PlotChildHandleState,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        constructor(SimplePlotHandleState {
            figure,
            axes_index,
            plot_index,
        }),
    );
    id as f64
}

pub fn register_line_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, PlotChildHandleState::Line)
}

pub fn register_animated_line_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    is_3d: bool,
    maximum_num_points: Option<usize>,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::AnimatedLine(AnimatedLineHandleState {
            figure,
            axes_index,
            plot_index,
            is_3d,
            maximum_num_points,
        }),
    );
    id as f64
}

pub fn register_reference_line_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
) -> f64 {
    register_simple_plot_handle(
        figure,
        axes_index,
        plot_index,
        PlotChildHandleState::ReferenceLine,
    )
}

pub fn register_scatter_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(
        figure,
        axes_index,
        plot_index,
        PlotChildHandleState::Scatter,
    )
}

pub fn register_bar_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, PlotChildHandleState::Bar)
}

pub fn register_stem_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, |state| {
        PlotChildHandleState::Stem(StemHandleState {
            figure: state.figure,
            axes_index: state.axes_index,
            plot_index: state.plot_index,
        })
    })
}

pub fn register_errorbar_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, |state| {
        PlotChildHandleState::ErrorBar(ErrorBarHandleState {
            figure: state.figure,
            axes_index: state.axes_index,
            plot_index: state.plot_index,
        })
    })
}

pub fn register_stairs_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, PlotChildHandleState::Stairs)
}

pub fn register_quiver_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, |state| {
        PlotChildHandleState::Quiver(QuiverHandleState {
            figure: state.figure,
            axes_index: state.axes_index,
            plot_index: state.plot_index,
            is_3d: false,
        })
    })
}

pub fn register_quiver3_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, |state| {
        PlotChildHandleState::Quiver(QuiverHandleState {
            figure: state.figure,
            axes_index: state.axes_index,
            plot_index: state.plot_index,
            is_3d: true,
        })
    })
}

pub fn register_image_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    c_data: Option<Tensor>,
    c_data_mapping: impl Into<String>,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::Image(ImageHandleState {
            figure,
            axes_index,
            plot_index,
            c_data,
            c_data_mapping: c_data_mapping.into(),
        }),
    );
    id as f64
}

pub fn register_heatmap_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    color_data: Tensor,
    color_limits: Option<Tensor>,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::Heatmap(HeatmapHandleState {
            figure,
            axes_index,
            plot_index,
            x_labels,
            y_labels,
            color_data,
            color_limits,
        }),
    );
    id as f64
}

pub fn set_heatmap_color_limits(handle: f64, limits: Tensor) -> Result<(), FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let mut reg = registry();
    match reg.plot_children.get_mut(&(handle.round() as u64)) {
        Some(PlotChildHandleState::Heatmap(state)) => {
            state.color_limits = Some(limits);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn register_binscatter_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    values: Tensor,
    x_bin_edges: Vec<f64>,
    y_bin_edges: Vec<f64>,
    x_data: Tensor,
    y_data: Tensor,
    num_bins: [usize; 2],
    auto_bins: bool,
    x_limits_option: Option<Tensor>,
    y_limits_option: Option<Tensor>,
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<String>,
) -> f64 {
    let x_limits = (
        *x_bin_edges.first().unwrap_or(&0.0),
        *x_bin_edges.last().unwrap_or(&1.0),
    );
    let y_limits = (
        *y_bin_edges.first().unwrap_or(&0.0),
        *y_bin_edges.last().unwrap_or(&1.0),
    );
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::Binscatter(BinscatterHandleState {
            figure,
            axes_index,
            plot_index,
            values,
            x_bin_edges,
            y_bin_edges,
            x_data,
            y_data,
            num_bins,
            auto_bins,
            x_limits_option,
            y_limits_option,
            x_limits,
            y_limits,
            show_empty_bins,
            face_alpha,
            display_name,
        }),
    );
    id as f64
}

pub fn register_function_surface_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    mesh_density: usize,
    x_range: (f64, f64),
    y_range: (f64, f64),
    function: FunctionSurfaceFunctionState,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::FunctionSurface(FunctionSurfaceHandleState {
            figure,
            axes_index,
            plot_index,
            mesh_density,
            x_range,
            y_range,
            function,
        }),
    );
    id as f64
}

pub fn register_function_contour_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    mesh_density: usize,
    x_range: (f64, f64),
    y_range: (f64, f64),
    function: FunctionSurfaceFunctionRef,
    fill: bool,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::FunctionContour(FunctionContourHandleState {
            figure,
            axes_index,
            plot_index,
            mesh_density,
            x_range,
            y_range,
            function,
            fill,
        }),
    );
    id as f64
}

pub fn update_binscatter_handle_for_plot(
    figure: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut BinscatterHandleState),
) -> Result<(), FigureError> {
    let mut updater = Some(updater);
    let mut reg = registry();
    for state in reg.plot_children.values_mut() {
        if let PlotChildHandleState::Binscatter(binscatter) = state {
            if binscatter.figure == figure && binscatter.plot_index == plot_index {
                let Some(updater) = updater.take() else {
                    return Ok(());
                };
                updater(binscatter);
                return Ok(());
            }
        }
    }
    Err(FigureError::InvalidPlotObjectHandle)
}

pub fn register_area_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, |state| {
        PlotChildHandleState::Area(AreaHandleState {
            figure: state.figure,
            axes_index: state.axes_index,
            plot_index: state.plot_index,
        })
    })
}

pub fn register_surface_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(
        figure,
        axes_index,
        plot_index,
        PlotChildHandleState::Surface,
    )
}

pub fn register_patch_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, PlotChildHandleState::Patch)
}

pub fn register_line3_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, PlotChildHandleState::Line3)
}

pub fn register_scatter3_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(
        figure,
        axes_index,
        plot_index,
        PlotChildHandleState::Scatter3,
    )
}

pub fn register_contour_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(
        figure,
        axes_index,
        plot_index,
        PlotChildHandleState::Contour,
    )
}

pub fn register_contour_fill_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
) -> f64 {
    register_simple_plot_handle(
        figure,
        axes_index,
        plot_index,
        PlotChildHandleState::ContourFill,
    )
}

pub fn register_pie_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, PlotChildHandleState::Pie)
}

pub fn register_text_annotation_handle(
    figure: FigureHandle,
    axes_index: usize,
    annotation_index: usize,
) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children.insert(
        id,
        PlotChildHandleState::Text(TextAnnotationHandleState {
            figure,
            axes_index,
            annotation_index,
        }),
    );
    id as f64
}

pub fn register_textscatter_handle(state: TextScatterHandleState) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children
        .insert(id, PlotChildHandleState::TextScatter(state));
    id as f64
}

pub fn update_textscatter_handle_state(
    handle: f64,
    state: TextScatterHandleState,
) -> Result<(), FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let mut reg = registry();
    let id = handle.round() as u64;
    match reg.plot_children.get_mut(&id) {
        Some(PlotChildHandleState::TextScatter(slot)) => {
            *slot = state;
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_textscatter_figure(
    state: &TextScatterHandleState,
    mut apply: impl FnMut(&mut Figure) -> Result<(), FigureError>,
) -> Result<(), FigureError> {
    let (result, figure_clone) =
        with_axes_target_mut(state.figure, state.axes_index, |figure_state| {
            apply(&mut figure_state.figure)
        })?;
    result?;
    notify_with_figure(state.figure, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn register_wordcloud_handle(state: WordCloudHandleState) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children
        .insert(id, PlotChildHandleState::WordCloud(state));
    id as f64
}

pub fn update_wordcloud_handle_state(
    handle: f64,
    state: WordCloudHandleState,
) -> Result<(), FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let mut reg = registry();
    let id = handle.round() as u64;
    match reg.plot_children.get_mut(&id) {
        Some(PlotChildHandleState::WordCloud(slot)) => {
            *slot = state;
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_wordcloud_figure(
    state: &WordCloudHandleState,
    mut apply: impl FnMut(&mut Figure) -> Result<(), FigureError>,
) -> Result<(), FigureError> {
    let (result, figure_clone) =
        with_axes_target_mut(state.figure, state.axes_index, |figure_state| {
            apply(&mut figure_state.figure)
        })?;
    result?;
    notify_with_figure(state.figure, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn register_stackedplot_handle(state: StackedPlotHandleState) -> f64 {
    let mut reg = registry();
    let id = reg.next_plot_child_handle;
    reg.next_plot_child_handle += 1;
    reg.plot_children
        .insert(id, PlotChildHandleState::StackedPlot(state));
    id as f64
}

pub fn update_stackedplot_handle_state(
    handle: f64,
    state: StackedPlotHandleState,
) -> Result<(), FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let mut reg = registry();
    let id = handle.round() as u64;
    match reg.plot_children.get_mut(&id) {
        Some(PlotChildHandleState::StackedPlot(slot)) => {
            *slot = state;
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_stackedplot_figure(
    state: &StackedPlotHandleState,
    mut apply: impl FnMut(&mut Figure) -> Result<(), FigureError>,
) -> Result<(), FigureError> {
    let axes_index = state.axes_indices.first().copied().unwrap_or(0);
    let (result, figure_clone) = with_axes_target_mut(state.figure, axes_index, |figure_state| {
        apply(&mut figure_state.figure)
    })?;
    result?;
    notify_with_figure(state.figure, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn render_stackedplot_chart<F>(
    builtin: &'static str,
    target: Option<FigureHandle>,
    axes_count: usize,
    mut apply: F,
) -> BuiltinResult<(FigureHandle, Vec<usize>, String)>
where
    F: FnMut(&mut Figure, &[usize]) -> BuiltinResult<()>,
{
    if axes_count == 0 {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: expected at least one plotted variable"),
        ));
    }
    let rendering_disabled = interactive_rendering_disabled();
    let host_managed_rendering = host_managed_rendering_enabled();
    let (handle, axes_indices, figure_clone) = {
        let mut reg = registry();
        let handle = target.unwrap_or(reg.current);
        let axes_indices = (0..axes_count).collect::<Vec<_>>();
        {
            let state = get_state_mut(&mut reg, handle);
            state.figure.set_subplot_grid(axes_count, 1);
            for axes_index in &axes_indices {
                state.figure.clear_axes(*axes_index);
                state.figure.set_axes_kind(*axes_index, AxesKind::Cartesian);
                state.figure.set_axes_limits(*axes_index, None, None);
                state.figure.set_axes_z_limits(*axes_index, None);
                state.figure.set_axes_grid_enabled(*axes_index, true);
                state.figure.set_axes_minor_grid_enabled(*axes_index, false);
                state.figure.set_axes_legend_enabled(*axes_index, false);
                state.reset_cycle(*axes_index);
            }
            state.active_axes = *axes_indices.last().unwrap_or(&0);
            state.figure.set_active_axes_index(state.active_axes);
        }
        for axes_index in &axes_indices {
            purge_plot_children_for_axes(&mut reg, handle, *axes_index);
        }
        {
            let state = get_state_mut(&mut reg, handle);
            apply(&mut state.figure, &axes_indices)
                .map_err(|flow| map_control_flow_with_builtin(flow, builtin))?;
            state.revision = state.revision.wrapping_add(1);
        }
        reg.current = handle;
        let figure_clone = reg
            .figures
            .get(&handle)
            .expect("figure exists")
            .figure
            .clone();
        (handle, axes_indices, figure_clone)
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    let presentation = present_figure_update_with_options(
        builtin,
        handle,
        figure_clone,
        rendering_disabled,
        host_managed_rendering,
    )?;
    Ok((handle, axes_indices, presentation))
}

#[derive(Clone, Copy, Debug)]
pub enum CopyParentTarget {
    Axes(FigureHandle, usize),
}

pub fn plot_child_handle_snapshot(handle: f64) -> Result<PlotChildHandleState, FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let reg = registry();
    reg.plot_children
        .get(&(handle.round() as u64))
        .cloned()
        .ok_or(FigureError::InvalidPlotObjectHandle)
}

pub fn validate_plot_child_copy_source(source_handle: f64) -> Result<(), FigureError> {
    if !source_handle.is_finite() || source_handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let reg = registry();
    let source_state = reg
        .plot_children
        .get(&(source_handle.round() as u64))
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    let source_plot_index = source_state
        .plot_index()
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    let (source_figure, _) = source_state.figure_axes();
    reg.figures
        .get(&source_figure)
        .and_then(|state| state.figure.plots().nth(source_plot_index))
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    Ok(())
}

pub fn copy_plot_child_to_parent(
    source_handle: f64,
    target: CopyParentTarget,
) -> Result<f64, FigureError> {
    if !source_handle.is_finite() || source_handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }

    let (new_handle, target_figure, figure_clone) = {
        let mut reg = registry();
        let source_key = source_handle.round() as u64;
        let source_state = reg
            .plot_children
            .get(&source_key)
            .cloned()
            .ok_or(FigureError::InvalidPlotObjectHandle)?;
        let source_plot_index = source_state
            .plot_index()
            .ok_or(FigureError::InvalidPlotObjectHandle)?;
        let (source_figure, _) = source_state.figure_axes();
        let plot = reg
            .figures
            .get(&source_figure)
            .and_then(|state| state.figure.plots().nth(source_plot_index))
            .cloned()
            .ok_or(FigureError::InvalidPlotObjectHandle)?;

        let CopyParentTarget::Axes(target_figure, target_axes) = target;

        let (new_plot_index, figure_clone) = {
            let target_state = get_state_mut(&mut reg, target_figure);
            target_state.figure.ensure_axes(target_axes);
            target_state.figure.set_active_axes_index(target_axes);
            target_state.active_axes = target_axes;
            let new_plot_index = target_state
                .figure
                .add_plot_element_on_axes(plot, target_axes);
            target_state.revision = target_state.revision.wrapping_add(1);
            (new_plot_index, target_state.figure.clone())
        };

        let remapped_state = source_state
            .with_plot_location(target_figure, target_axes, new_plot_index)
            .ok_or(FigureError::InvalidPlotObjectHandle)?;
        let new_key = reg.next_plot_child_handle;
        reg.next_plot_child_handle += 1;
        reg.plot_children.insert(new_key, remapped_state);

        (new_key as f64, target_figure, figure_clone)
    };

    notify_with_figure(target_figure, &figure_clone, FigureEventKind::Updated);
    Ok(new_handle)
}

pub fn set_heatmap_display_labels(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    x_labels: Option<Vec<String>>,
    y_labels: Option<Vec<String>>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let (current_x_labels, current_y_labels) = {
            let state = reg.plot_children.values_mut().find(|state| match state {
                PlotChildHandleState::Heatmap(heatmap) => {
                    heatmap.figure == figure
                        && heatmap.axes_index == axes_index
                        && heatmap.plot_index == plot_index
                }
                _ => false,
            });
            let PlotChildHandleState::Heatmap(heatmap) =
                state.ok_or(FigureError::InvalidPlotObjectHandle)?
            else {
                return Err(FigureError::InvalidPlotObjectHandle);
            };
            if let Some(labels) = x_labels {
                heatmap.x_labels = labels;
            }
            if let Some(labels) = y_labels {
                heatmap.y_labels = labels;
            }
            (heatmap.x_labels.clone(), heatmap.y_labels.clone())
        };

        let state = get_state_mut(&mut reg, figure);
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.active_axes = axes_index;
        state.figure.set_active_axes_index(axes_index);
        state.figure.set_axes_tick_labels(
            axes_index,
            Some(current_x_labels),
            Some(current_y_labels),
        );
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(figure, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn update_histogram_handle_for_plot(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    normalization: String,
    raw_counts: Vec<f64>,
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = reg.plot_children.values_mut().find(|state| match state {
        PlotChildHandleState::Histogram(hist) => {
            hist.figure == figure && hist.axes_index == axes_index && hist.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Histogram(hist) => {
            hist.normalization = normalization;
            hist.raw_counts = raw_counts;
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_histogram2_handle_for_plot(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    updater: impl FnOnce(&mut Histogram2HandleState),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = reg.plot_children.values_mut().find(|state| match state {
        PlotChildHandleState::Histogram2(hist) => {
            hist.figure == figure && hist.axes_index == axes_index && hist.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Histogram2(hist) => {
            updater(hist);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_image_handle_for_plot(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    updater: impl FnOnce(&mut ImageHandleState),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = reg.plot_children.values_mut().find(|state| match state {
        PlotChildHandleState::Image(image) => {
            image.figure == figure
                && image.axes_index == axes_index
                && image.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Image(image) => {
            updater(image);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn set_histogram_handle_display_name(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    display_name: Option<String>,
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = reg.plot_children.values_mut().find(|state| match state {
        PlotChildHandleState::Histogram(hist) => {
            hist.figure == figure && hist.axes_index == axes_index && hist.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Histogram(hist) => {
            hist.display_name = display_name;
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_histogram_handle_metadata_for_plot(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    updater: impl FnOnce(&mut HistogramHandleMetadata),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = reg.plot_children.values_mut().find(|state| match state {
        PlotChildHandleState::Histogram(hist) => {
            hist.figure == figure && hist.axes_index == axes_index && hist.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Histogram(hist) => {
            updater(&mut hist.metadata);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_errorbar_plot(
    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::ErrorBar),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::ErrorBar(errorbar) => {
            updater(errorbar);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_histogram_plot_data(
    figure_handle: FigureHandle,
    plot_index: usize,
    labels: Vec<String>,
    values: Vec<f64>,
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Bar(bar) => {
            bar.set_data(labels, values)
                .map_err(|_| FigureError::InvalidPlotObjectHandle)?;
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_stem_plot(
    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::StemPlot),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Stem(stem) => {
            updater(stem);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_quiver_plot(
    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::QuiverPlot),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Quiver(quiver) => {
            updater(quiver);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_image_plot(
    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::SurfacePlot),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Surface(surface) if surface.image_mode => {
            updater(surface);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_area_plot(
    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::AreaPlot),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Area(area) => {
            updater(area);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub fn update_plot_element(
    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::figure::PlotElement),
) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    updater(plot);
    Ok(())
}

pub fn append_points_to_animated_line(
    handle: &AnimatedLineHandleState,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Option<Vec<f64>>,
) -> Result<(), String> {
    if x.len() != y.len() || z.as_ref().is_some_and(|z| z.len() != x.len()) {
        return Err("coordinate vectors must have the same length".to_string());
    }
    if x.is_empty() {
        return Ok(());
    }

    let (figure_clone, became_3d) = {
        let mut reg = registry();
        let mut became_3d = false;
        let figure_clone = {
            let state = get_state_mut(&mut reg, handle.figure);
            let plot = state
                .figure
                .get_plot_mut(handle.plot_index)
                .ok_or_else(|| "invalid animated line handle".to_string())?;

            match z {
                None => match plot {
                    PlotElement::Line(line) => {
                        let (mut next_x, mut next_y) = line
                            .host_xy_f64()
                            .map_err(|err| format!("failed to read animated line: {err}"))?
                            .ok_or_else(|| "animated line data is unavailable".to_string())?;
                        next_x.extend(x);
                        next_y.extend(y);
                        trim_oldest_xy(handle.maximum_num_points, &mut next_x, &mut next_y);
                        line.update_data(next_x, next_y)
                            .map_err(|err| format!("failed to update animated line: {err}"))?;
                    }
                    PlotElement::Line3(_) => {
                        return Err(
                            "3-D animated lines require X, Y, and Z coordinates".to_string()
                        );
                    }
                    _ => return Err("invalid animated line handle".to_string()),
                },
                Some(z) => match plot {
                    PlotElement::Line(line) => {
                        if line.marker.is_some() {
                            return Err("3-D animated lines do not support marker properties yet"
                                .to_string());
                        }
                        let (mut next_x, mut next_y) = line
                            .host_xy_f64()
                            .map_err(|err| format!("failed to read animated line: {err}"))?
                            .ok_or_else(|| "animated line data is unavailable".to_string())?;
                        let mut next_z = vec![0.0; next_x.len()];
                        next_x.extend(x);
                        next_y.extend(y);
                        next_z.extend(z);
                        trim_oldest_xyz(
                            handle.maximum_num_points,
                            &mut next_x,
                            &mut next_y,
                            &mut next_z,
                        );
                        let mut line3 = runmat_plot::plots::Line3Plot::new(next_x, next_y, next_z)
                            .map_err(|err| format!("failed to update animated line: {err}"))?;
                        line3.color = line.color;
                        line3.line_width = line.line_width;
                        line3.line_style = line.line_style;
                        line3.label = line.label.clone();
                        line3.visible = line.visible;
                        *plot = PlotElement::Line3(line3);
                        became_3d = true;
                    }
                    PlotElement::Line3(line) => {
                        let (mut next_x, mut next_y, mut next_z) = line
                            .host_xyz_f64()
                            .map_err(|err| format!("failed to read animated line: {err}"))?
                            .ok_or_else(|| "animated line data is unavailable".to_string())?;
                        next_x.extend(x);
                        next_y.extend(y);
                        next_z.extend(z);
                        trim_oldest_xyz(
                            handle.maximum_num_points,
                            &mut next_x,
                            &mut next_y,
                            &mut next_z,
                        );
                        line.update_data(next_x, next_y, next_z)
                            .map_err(|err| format!("failed to update animated line: {err}"))?;
                    }
                    _ => return Err("invalid animated line handle".to_string()),
                },
            }

            state.revision = state.revision.wrapping_add(1);
            state.figure.clone()
        };

        if became_3d {
            for child in reg.plot_children.values_mut() {
                if let PlotChildHandleState::AnimatedLine(animated) = child {
                    if animated.figure == handle.figure && animated.plot_index == handle.plot_index
                    {
                        animated.is_3d = true;
                    }
                }
            }
        }
        (figure_clone, became_3d)
    };
    let _ = became_3d;
    notify_with_figure(handle.figure, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn set_animated_line_maximum_num_points(
    handle: &AnimatedLineHandleState,
    maximum_num_points: Option<usize>,
) -> Result<(), String> {
    let figure_clone = {
        let mut reg = registry();
        let figure_clone = {
            let state = get_state_mut(&mut reg, handle.figure);
            let plot = state
                .figure
                .get_plot_mut(handle.plot_index)
                .ok_or_else(|| "invalid animated line handle".to_string())?;
            match plot {
                PlotElement::Line(line) => {
                    let (mut x, mut y) = line
                        .host_xy_f64()
                        .map_err(|err| format!("failed to read animated line: {err}"))?
                        .ok_or_else(|| "animated line data is unavailable".to_string())?;
                    trim_oldest_xy(maximum_num_points, &mut x, &mut y);
                    line.update_data(x, y)
                        .map_err(|err| format!("failed to update animated line: {err}"))?;
                }
                PlotElement::Line3(line) => {
                    let (mut x, mut y, mut z) = line
                        .host_xyz_f64()
                        .map_err(|err| format!("failed to read animated line: {err}"))?
                        .ok_or_else(|| "animated line data is unavailable".to_string())?;
                    trim_oldest_xyz(maximum_num_points, &mut x, &mut y, &mut z);
                    line.update_data(x, y, z)
                        .map_err(|err| format!("failed to update animated line: {err}"))?;
                }
                _ => return Err("invalid animated line handle".to_string()),
            }
            state.revision = state.revision.wrapping_add(1);
            state.figure.clone()
        };

        for child in reg.plot_children.values_mut() {
            if let PlotChildHandleState::AnimatedLine(animated) = child {
                if animated.figure == handle.figure && animated.plot_index == handle.plot_index {
                    animated.maximum_num_points = maximum_num_points;
                }
            }
        }
        figure_clone
    };
    notify_with_figure(handle.figure, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

fn trim_oldest_xy(maximum: Option<usize>, x: &mut Vec<f64>, y: &mut Vec<f64>) {
    let Some(maximum) = maximum else {
        return;
    };
    if x.len() <= maximum {
        return;
    }
    let drop = x.len() - maximum;
    x.drain(0..drop);
    y.drain(0..drop);
}

fn trim_oldest_xyz(maximum: Option<usize>, x: &mut Vec<f64>, y: &mut Vec<f64>, z: &mut Vec<f64>) {
    let Some(maximum) = maximum else {
        return;
    };
    if x.len() <= maximum {
        return;
    }
    let drop = x.len() - maximum;
    x.drain(0..drop);
    y.drain(0..drop);
    z.drain(0..drop);
}

fn purge_plot_children_for_figure(reg: &mut PlotRegistry, handle: FigureHandle) {
    reg.plot_children.retain(|_, state| match state {
        PlotChildHandleState::Histogram(hist) => hist.figure != handle,
        PlotChildHandleState::Histogram2(hist) => hist.figure != handle,
        PlotChildHandleState::Line(plot)
        | PlotChildHandleState::Scatter(plot)
        | PlotChildHandleState::Bar(plot)
        | PlotChildHandleState::Stairs(plot)
        | PlotChildHandleState::Surface(plot)
        | PlotChildHandleState::Patch(plot)
        | PlotChildHandleState::Line3(plot)
        | PlotChildHandleState::Scatter3(plot)
        | PlotChildHandleState::Contour(plot)
        | PlotChildHandleState::ContourFill(plot)
        | PlotChildHandleState::ReferenceLine(plot)
        | PlotChildHandleState::Pie(plot) => plot.figure != handle,
        PlotChildHandleState::AnimatedLine(animated) => animated.figure != handle,
        PlotChildHandleState::Stem(stem) => stem.figure != handle,
        PlotChildHandleState::ErrorBar(err) => err.figure != handle,
        PlotChildHandleState::Quiver(quiver) => quiver.figure != handle,
        PlotChildHandleState::Image(image) => image.figure != handle,
        PlotChildHandleState::Heatmap(heatmap) => heatmap.figure != handle,
        PlotChildHandleState::Binscatter(binscatter) => binscatter.figure != handle,
        PlotChildHandleState::FunctionSurface(function_surface) => {
            function_surface.figure != handle
        }
        PlotChildHandleState::FunctionContour(function_contour) => {
            function_contour.figure != handle
        }
        PlotChildHandleState::Area(area) => area.figure != handle,
        PlotChildHandleState::Text(text) => text.figure != handle,
        PlotChildHandleState::TextScatter(textscatter) => textscatter.figure != handle,
        PlotChildHandleState::WordCloud(wordcloud) => wordcloud.figure != handle,
        PlotChildHandleState::StackedPlot(stacked) => stacked.figure != handle,
    });
}

fn purge_plot_children_for_axes(reg: &mut PlotRegistry, handle: FigureHandle, axes_index: usize) {
    reg.plot_children.retain(|_, state| match state {
        PlotChildHandleState::Histogram(hist) => {
            !(hist.figure == handle && hist.axes_index == axes_index)
        }
        PlotChildHandleState::Histogram2(hist) => {
            !(hist.figure == handle && hist.axes_index == axes_index)
        }
        PlotChildHandleState::Line(plot)
        | PlotChildHandleState::Scatter(plot)
        | PlotChildHandleState::Bar(plot)
        | PlotChildHandleState::Stairs(plot)
        | PlotChildHandleState::Surface(plot)
        | PlotChildHandleState::Patch(plot)
        | PlotChildHandleState::Line3(plot)
        | PlotChildHandleState::Scatter3(plot)
        | PlotChildHandleState::Contour(plot)
        | PlotChildHandleState::ContourFill(plot)
        | PlotChildHandleState::ReferenceLine(plot)
        | PlotChildHandleState::Pie(plot) => {
            !(plot.figure == handle && plot.axes_index == axes_index)
        }
        PlotChildHandleState::AnimatedLine(animated) => {
            !(animated.figure == handle && animated.axes_index == axes_index)
        }
        PlotChildHandleState::Stem(stem) => {
            !(stem.figure == handle && stem.axes_index == axes_index)
        }
        PlotChildHandleState::ErrorBar(err) => {
            !(err.figure == handle && err.axes_index == axes_index)
        }
        PlotChildHandleState::Quiver(quiver) => {
            !(quiver.figure == handle && quiver.axes_index == axes_index)
        }
        PlotChildHandleState::Image(image) => {
            !(image.figure == handle && image.axes_index == axes_index)
        }
        PlotChildHandleState::Heatmap(heatmap) => {
            !(heatmap.figure == handle && heatmap.axes_index == axes_index)
        }
        PlotChildHandleState::Binscatter(binscatter) => {
            !(binscatter.figure == handle && binscatter.axes_index == axes_index)
        }
        PlotChildHandleState::FunctionSurface(function_surface) => {
            !(function_surface.figure == handle && function_surface.axes_index == axes_index)
        }
        PlotChildHandleState::FunctionContour(function_contour) => {
            !(function_contour.figure == handle && function_contour.axes_index == axes_index)
        }
        PlotChildHandleState::Area(area) => {
            !(area.figure == handle && area.axes_index == axes_index)
        }
        PlotChildHandleState::Text(text) => {
            !(text.figure == handle && text.axes_index == axes_index)
        }
        PlotChildHandleState::TextScatter(textscatter) => {
            !(textscatter.figure == handle && textscatter.axes_index == axes_index)
        }
        PlotChildHandleState::WordCloud(wordcloud) => {
            !(wordcloud.figure == handle && wordcloud.axes_index == axes_index)
        }
        PlotChildHandleState::StackedPlot(stacked) => {
            !(stacked.figure == handle && stacked.axes_indices.contains(&axes_index))
        }
    });
}

pub fn decode_axes_handle(value: f64) -> Result<(FigureHandle, usize), FigureError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(FigureError::InvalidAxesHandle);
    }
    let encoded = value.round() as u64;
    let figure_id = encoded >> AXES_INDEX_BITS;
    if figure_id == 0 {
        return Err(FigureError::InvalidAxesHandle);
    }
    let axes_index = (encoded & AXES_INDEX_MASK) as usize;
    Ok((FigureHandle::from(figure_id as u32), axes_index))
}

#[cfg(not(target_arch = "wasm32"))]
fn registry() -> PlotRegistryGuard<'static> {
    #[cfg(test)]
    let test_lock = TEST_PLOT_OUTER_LOCK_HELD.with(|flag| {
        if flag.get() {
            None
        } else {
            Some(
                TEST_PLOT_REGISTRY_LOCK
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        }
    });
    let guard = REGISTRY
        .get_or_init(|| Mutex::new(PlotRegistry::default()))
        .lock()
        .expect("plot registry poisoned");
    #[cfg(test)]
    {
        PlotRegistryGuard::new(guard, test_lock)
    }
    #[cfg(not(test))]
    {
        PlotRegistryGuard::new(guard)
    }
}

#[cfg(target_arch = "wasm32")]
fn registry() -> PlotRegistryGuard<'static> {
    REGISTRY.with(|cell| {
        let guard = cell.borrow_mut();
        // SAFETY: the thread-local RefCell lives for the program lifetime and the borrow
        // guard is dropped when PlotRegistryGuard is dropped, so extending the lifetime
        // to 'static is sound.
        let guard_static: std::cell::RefMut<'static, PlotRegistry> =
            unsafe { std::mem::transmute::<std::cell::RefMut<'_, PlotRegistry>, _>(guard) };
        #[cfg(test)]
        {
            let test_lock = TEST_PLOT_OUTER_LOCK_HELD.with(|flag| {
                if flag.get() {
                    None
                } else {
                    Some(
                        TEST_PLOT_REGISTRY_LOCK
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()),
                    )
                }
            });
            PlotRegistryGuard::new(guard_static, test_lock)
        }
        #[cfg(not(test))]
        {
            PlotRegistryGuard::new(guard_static)
        }
    })
}

fn get_state_mut(registry: &mut PlotRegistry, handle: FigureHandle) -> &mut FigureState {
    registry
        .figures
        .entry(handle)
        .or_insert_with(|| FigureState::new(handle))
}

fn observer_registry() -> &'static FigureObserverRegistry {
    FIGURE_OBSERVERS.get_or_init(FigureObserverRegistry::new)
}

pub fn install_figure_observer(observer: Arc<FigureObserver>) -> BuiltinResult<()> {
    observer_registry().install(observer);
    Ok(())
}

fn notify_event<'a>(view: FigureEventView<'a>) {
    note_recent_figure(view.handle);
    if let Some(registry) = FIGURE_OBSERVERS.get() {
        if registry.is_empty() {
            return;
        }
        registry.notify(view);
    }
}

fn notify_with_figure(handle: FigureHandle, figure: &Figure, kind: FigureEventKind) {
    notify_event(FigureEventView {
        handle,
        kind,
        revision: current_figure_revision(handle),
        figure: Some(figure),
    });
}

fn notify_without_figure(handle: FigureHandle, kind: FigureEventKind) {
    notify_event(FigureEventView {
        handle,
        kind,
        revision: current_figure_revision(handle),
        figure: None,
    });
}

fn note_recent_figure(handle: FigureHandle) {
    RECENT_FIGURES.with(|set| {
        set.borrow_mut().insert(handle);
    });
}

pub fn record_recent_figure(handle: FigureHandle) {
    note_recent_figure(handle);
}

pub fn reset_recent_figures() {
    RECENT_FIGURES.with(|set| set.borrow_mut().clear());
}

pub fn reset_plot_state() {
    {
        let mut reg = registry();
        *reg = PlotRegistry::default();
    }
    reset_recent_figures();
}

pub fn take_recent_figures() -> Vec<FigureHandle> {
    RECENT_FIGURES.with(|set| set.borrow_mut().drain().collect())
}

pub fn select_figure(handle: FigureHandle) {
    let mut reg = registry();
    reg.current = handle;
    let maybe_new = match reg.figures.entry(handle) {
        Entry::Occupied(entry) => {
            let _ = entry.into_mut();
            None
        }
        Entry::Vacant(vacant) => {
            let state = vacant.insert(FigureState::new(handle));
            Some(state.figure.clone())
        }
    };
    drop(reg);
    if let Some(figure_clone) = maybe_new {
        notify_with_figure(handle, &figure_clone, FigureEventKind::Created);
    }
}

pub fn new_figure_handle() -> FigureHandle {
    let mut reg = registry();
    let handle = reg.next_handle;
    reg.next_handle = reg.next_handle.next();
    reg.current = handle;
    let figure_clone = {
        let state = get_state_mut(&mut reg, handle);
        state.figure.clone()
    };
    drop(reg);
    notify_with_figure(handle, &figure_clone, FigureEventKind::Created);
    handle
}

pub fn current_figure_handle() -> FigureHandle {
    registry().current
}

pub fn current_figure_handle_if_exists() -> Option<FigureHandle> {
    let reg = registry();
    if reg.figures.contains_key(&reg.current) {
        Some(reg.current)
    } else {
        None
    }
}

pub fn select_current_figure_if_exists(handle: FigureHandle) -> Result<(), FigureError> {
    let mut reg = registry();
    if !reg.figures.contains_key(&handle) {
        return Err(FigureError::InvalidHandle(handle.as_u32()));
    }
    reg.current = handle;
    Ok(())
}

pub fn current_axes_state() -> FigureAxesState {
    let mut reg = registry();
    let handle = reg.current;
    // Ensure a default figure exists even if nothing has rendered yet (common on wasm/web).
    let state = get_state_mut(&mut reg, handle);
    FigureAxesState {
        handle,
        rows: state.figure.axes_rows.max(1),
        cols: state.figure.axes_cols.max(1),
        active_index: state.active_axes,
    }
}

pub fn axes_handle_exists(handle: FigureHandle, axes_index: usize) -> bool {
    let reg = registry();
    reg.figures
        .get(&handle)
        .map(|state| axes_index < axes_count(state))
        .unwrap_or(false)
}

pub fn figure_handle_exists(handle: FigureHandle) -> bool {
    let reg = registry();
    reg.figures.contains_key(&handle)
}

pub fn axes_metadata_snapshot(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<runmat_plot::plots::AxesMetadata, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .ok_or(FigureError::InvalidAxesHandle)
}

pub fn axes_state_snapshot(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<FigureAxesState, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    Ok(FigureAxesState {
        handle,
        rows: state.figure.axes_rows.max(1),
        cols: state.figure.axes_cols.max(1),
        active_index: axes_index,
    })
}

pub fn current_axes_handle_for_figure(handle: FigureHandle) -> Result<f64, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    Ok(encode_axes_handle(handle, state.active_axes))
}

pub fn axes_handles_for_figure(handle: FigureHandle) -> Result<Vec<f64>, FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    let total_axes = axes_count(state);
    Ok((0..total_axes)
        .map(|idx| encode_axes_handle(handle, idx))
        .collect())
}

pub fn plot_child_handles_for_axes(handle: FigureHandle, axes_index: usize) -> Vec<f64> {
    let mut handles = registry()
        .plot_children
        .iter()
        .filter_map(|(id, state)| {
            let (figure, axes) = state.figure_axes();
            (figure == handle && axes == axes_index).then_some(*id as f64)
        })
        .collect::<Vec<_>>();
    handles.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    handles
}

pub fn select_axes_for_figure(handle: FigureHandle, axes_index: usize) -> Result<(), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    reg.current = handle;
    let state = get_state_mut(&mut reg, handle);
    state.active_axes = axes_index;
    state.figure.set_active_axes_index(axes_index);
    Ok(())
}

pub fn create_axes_for_figure(
    target: Option<FigureHandle>,
) -> Result<(FigureHandle, usize), FigureError> {
    let mut reg = registry();
    let handle = target.unwrap_or(reg.current);
    let axes_index = {
        let state = get_state_mut(&mut reg, handle);
        let axes_index = axes_count(state);
        state.figure.ensure_axes(axes_index);
        state.figure.set_active_axes_index(axes_index);
        state.active_axes = axes_index;
        state.reset_cycle(axes_index);
        axes_index
    };
    reg.current = handle;
    Ok((handle, axes_index))
}

fn with_axes_target_mut<R>(
    handle: FigureHandle,
    axes_index: usize,
    f: impl FnOnce(&mut FigureState) -> R,
) -> Result<(R, Figure), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    state.active_axes = axes_index;
    state.figure.set_active_axes_index(axes_index);
    let result = f(state);
    state.revision = state.revision.wrapping_add(1);
    Ok((result, state.figure.clone()))
}

fn with_figure_mut<R>(
    handle: FigureHandle,
    f: impl FnOnce(&mut FigureState) -> R,
) -> Result<(R, Figure), FigureError> {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    let result = f(state);
    state.revision = state.revision.wrapping_add(1);
    Ok((result, state.figure.clone()))
}

pub fn current_hold_enabled() -> bool {
    let mut reg = registry();
    let handle = reg.current;
    // Ensure a default figure exists even if nothing has rendered yet (common on wasm/web).
    let state = get_state_mut(&mut reg, handle);
    *state
        .hold_per_axes
        .get(&state.active_axes)
        .unwrap_or(&false)
}

/// Reset hold state for all figures/axes.
///
/// In the IDE, we want re-running code to behave like a fresh plotting run unless the code
/// explicitly enables `hold on` again. Without this, a prior `hold on` will cause subsequent
/// runs to keep appending to the same axes (surprising for typical "Run" workflows).
pub fn reset_hold_state_for_run() {
    let mut reg = registry();
    for state in reg.figures.values_mut() {
        state.hold_per_axes.clear();
    }
}

pub fn figure_handles() -> Vec<FigureHandle> {
    let reg = registry();
    reg.figures.keys().copied().collect()
}

pub fn root_figure_handles() -> Vec<FigureHandle> {
    let mut handles = figure_handles();
    handles.sort_by_key(|handle| handle.as_u32());
    handles
}

pub fn root_default_properties() -> Vec<(String, RootPropertyValue)> {
    let reg = registry();
    let mut properties: Vec<_> = reg
        .root_defaults
        .iter()
        .map(|(key, entry)| (key.clone(), entry.display_name.clone(), entry.value.clone()))
        .collect();
    properties.sort_by(|(left, _, _), (right, _, _)| left.cmp(right));
    properties
        .into_iter()
        .map(|(_, display_name, value)| (display_name, value))
        .collect()
}

pub fn root_default_property(name: &str) -> Option<RootPropertyValue> {
    registry()
        .root_defaults
        .get(name)
        .map(|entry| entry.value.clone())
}

pub fn set_root_default_property(name: String, display_name: String, value: RootPropertyValue) {
    registry().root_defaults.insert(
        name,
        RootPropertyEntry {
            display_name,
            value,
        },
    );
}

pub fn root_units() -> String {
    registry().root_units.clone()
}

pub fn set_root_units(units: String) {
    registry().root_units = units;
}

pub fn root_show_hidden_handles() -> bool {
    registry().root_show_hidden_handles
}

pub fn set_root_show_hidden_handles(enabled: bool) {
    registry().root_show_hidden_handles = enabled;
}

pub fn clone_figure(handle: FigureHandle) -> Option<Figure> {
    let reg = registry();
    reg.figures.get(&handle).map(|state| state.figure.clone())
}

pub fn figure_tag(handle: FigureHandle) -> Option<String> {
    let reg = registry();
    reg.figures.get(&handle).map(|state| state.tag.clone())
}

pub fn figure_has_sg_title(handle: FigureHandle) -> bool {
    let reg = registry();
    reg.figures
        .get(&handle)
        .map(|state| state.figure.sg_title.is_some())
        .unwrap_or(false)
}

pub fn import_figure(figure: Figure) -> FigureHandle {
    let mut reg = registry();
    let handle = reg.next_handle;
    reg.next_handle = reg.next_handle.next();
    reg.current = handle;
    let figure_clone = figure.clone();
    reg.figures.insert(
        handle,
        FigureState {
            figure,
            ..FigureState::new(handle)
        },
    );
    drop(reg);
    notify_with_figure(handle, &figure_clone, FigureEventKind::Created);
    handle
}

pub fn clear_figure(target: Option<FigureHandle>) -> Result<FigureHandle, FigureError> {
    let mut reg = registry();
    let handle = target.unwrap_or(reg.current);
    {
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        *state = FigureState::new(handle);
    }
    purge_link_axes_for_figure(&mut reg, handle);
    purge_plot_children_for_figure(&mut reg, handle);
    let figure_clone = reg
        .figures
        .get(&handle)
        .expect("figure exists")
        .figure
        .clone();
    drop(reg);
    notify_with_figure(handle, &figure_clone, FigureEventKind::Cleared);
    Ok(handle)
}

pub fn close_figure(target: Option<FigureHandle>) -> Result<FigureHandle, FigureError> {
    let mut reg = registry();
    let handle = target.unwrap_or(reg.current);
    let existed = reg.figures.remove(&handle);
    if existed.is_none() {
        return Err(FigureError::InvalidHandle(handle.as_u32()));
    }
    purge_link_axes_for_figure(&mut reg, handle);
    purge_plot_children_for_figure(&mut reg, handle);

    if reg.current == handle {
        if let Some((&next_handle, _)) = reg.figures.iter().next() {
            reg.current = next_handle;
        } else {
            let default = FigureHandle::default();
            reg.current = default;
            reg.next_handle = default.next();
            drop(reg);
            notify_without_figure(handle, FigureEventKind::Closed);
            return Ok(handle);
        }
    }

    drop(reg);
    notify_without_figure(handle, FigureEventKind::Closed);
    Ok(handle)
}

#[derive(Clone)]
pub struct PlotRenderOptions<'a> {
    pub title: &'a str,
    pub x_label: &'a str,
    pub y_label: &'a str,
    pub grid: bool,
    pub axis_equal: bool,
}

impl<'a> Default for PlotRenderOptions<'a> {
    fn default() -> Self {
        Self {
            title: "",
            x_label: "X",
            y_label: "Y",
            grid: true,
            axis_equal: false,
        }
    }
}

pub enum HoldMode {
    On,
    Off,
    Toggle,
}

pub fn set_hold(mode: HoldMode) -> bool {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    let current = state.hold();
    let new_value = match mode {
        HoldMode::On => true,
        HoldMode::Off => false,
        HoldMode::Toggle => !current,
    };
    state.set_hold(new_value);
    new_value
}

pub fn configure_subplot(rows: usize, cols: usize, index: usize) -> Result<(), FigureError> {
    if rows == 0 || cols == 0 {
        return Err(FigureError::InvalidSubplotGrid { rows, cols });
    }
    let total_axes = rows
        .checked_mul(cols)
        .ok_or(FigureError::InvalidSubplotGrid { rows, cols })?;
    if index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex { rows, cols, index });
    }
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.figure.set_subplot_grid(rows, cols);
    state.active_axes = index;
    state.figure.set_active_axes_index(index);
    Ok(())
}

pub fn prepare_plotyy_axes() -> Result<(FigureHandle, usize, usize, f64, f64), FigureError> {
    let mut reg = registry();
    let handle = reg.current;
    let (left_axes, right_axes) = {
        let state = get_state_mut(&mut reg, handle);
        let left_axes = state.active_axes;
        let right_axes = state.figure.ensure_overlay_axes(left_axes);
        state.figure.clear_axes(left_axes);
        state.figure.clear_axes(right_axes);
        state.figure.set_axes_kind(left_axes, AxesKind::Cartesian);
        state.figure.set_axes_kind(right_axes, AxesKind::Cartesian);
        state.figure.set_axes_limits(left_axes, None, None);
        state.figure.set_axes_limits(right_axes, None, None);
        state.figure.set_axes_z_limits(left_axes, None);
        state.figure.set_axes_z_limits(right_axes, None);
        state.figure.set_axes_y_axis_location(left_axes, "left");
        state.figure.set_axes_y_axis_location(right_axes, "right");
        state.figure.set_axes_grid_enabled(right_axes, false);
        state.figure.set_axes_minor_grid_enabled(right_axes, false);
        state.figure.set_axes_legend_enabled(right_axes, false);
        state.reset_cycle(left_axes);
        state.reset_cycle(right_axes);
        state.active_axes = left_axes;
        state.figure.set_active_axes_index(left_axes);
        (left_axes, right_axes)
    };
    purge_plot_children_for_axes(&mut reg, handle, left_axes);
    purge_plot_children_for_axes(&mut reg, handle, right_axes);
    Ok((
        handle,
        left_axes,
        right_axes,
        encode_axes_handle(handle, left_axes),
        encode_axes_handle(handle, right_axes),
    ))
}

pub fn render_active_plot<F>(
    builtin: &'static str,
    opts: PlotRenderOptions<'_>,
    mut apply: F,
) -> BuiltinResult<String>
where
    F: FnMut(&mut Figure, usize) -> BuiltinResult<()>,
{
    let rendering_disabled = interactive_rendering_disabled();
    let host_managed_rendering = host_managed_rendering_enabled();
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let axes_index = { get_state_mut(&mut reg, handle).active_axes };
        let should_clear = { !get_state_mut(&mut reg, handle).hold() };
        {
            let state = get_state_mut(&mut reg, handle);
            state.figure.set_active_axes_index(axes_index);
            if should_clear {
                state.figure.clear_axes(axes_index);
                state.figure.set_axes_kind(axes_index, AxesKind::Cartesian);
                state.figure.set_axes_limits(axes_index, None, None);
                state.figure.set_axes_z_limits(axes_index, None);
                state.reset_cycle(axes_index);
            }
        }
        if should_clear {
            purge_plot_children_for_axes(&mut reg, handle, axes_index);
        }
        {
            let state = get_state_mut(&mut reg, handle);
            if !opts.title.is_empty() {
                state.figure.set_axes_title(axes_index, opts.title);
            }
            if !opts.x_label.is_empty() || !opts.y_label.is_empty() {
                state
                    .figure
                    .set_axes_labels(axes_index, opts.x_label, opts.y_label);
            }
            state.figure.set_grid(opts.grid);
            state.figure.set_axis_equal(opts.axis_equal);

            let _axes_context = AxesContextGuard::install(state, axes_index);
            apply(&mut state.figure, axes_index)
                .map_err(|flow| map_control_flow_with_builtin(flow, builtin))?;

            // Increment revision after a successful mutation so surfaces can avoid
            // re-rendering unchanged figures when "presenting" an already-loaded handle.
            state.revision = state.revision.wrapping_add(1);
        }
        let figure_clone = reg
            .figures
            .get(&handle)
            .expect("figure exists")
            .figure
            .clone();
        (handle, figure_clone)
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);

    present_figure_update_with_options(
        builtin,
        handle,
        figure_clone,
        rendering_disabled,
        host_managed_rendering,
    )
}

pub fn append_active_plot<F>(
    builtin: &'static str,
    opts: PlotRenderOptions<'_>,
    mut apply: F,
) -> BuiltinResult<String>
where
    F: FnMut(&mut Figure, usize) -> BuiltinResult<()>,
{
    let rendering_disabled = interactive_rendering_disabled();
    let host_managed_rendering = host_managed_rendering_enabled();
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let axes_index = { get_state_mut(&mut reg, handle).active_axes };
        {
            let state = get_state_mut(&mut reg, handle);
            state.figure.set_active_axes_index(axes_index);
            if !opts.title.is_empty() {
                state.figure.set_axes_title(axes_index, opts.title);
            }
            if !opts.x_label.is_empty() || !opts.y_label.is_empty() {
                state
                    .figure
                    .set_axes_labels(axes_index, opts.x_label, opts.y_label);
            }
            state.figure.set_grid(opts.grid);
            state.figure.set_axis_equal(opts.axis_equal);

            let _axes_context = AxesContextGuard::install(state, axes_index);
            apply(&mut state.figure, axes_index)
                .map_err(|flow| map_control_flow_with_builtin(flow, builtin))?;
            state.revision = state.revision.wrapping_add(1);
        }
        let figure_clone = reg
            .figures
            .get(&handle)
            .expect("figure exists")
            .figure
            .clone();
        (handle, figure_clone)
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);

    present_figure_update_with_options(
        builtin,
        handle,
        figure_clone,
        rendering_disabled,
        host_managed_rendering,
    )
}

pub fn present_figure_update(
    builtin: &'static str,
    handle: FigureHandle,
    figure_clone: Figure,
) -> BuiltinResult<String> {
    present_figure_update_with_options(
        builtin,
        handle,
        figure_clone,
        interactive_rendering_disabled(),
        host_managed_rendering_enabled(),
    )
}

fn present_figure_update_with_options(
    builtin: &'static str,
    handle: FigureHandle,
    figure_clone: Figure,
    rendering_disabled: bool,
    host_managed_rendering: bool,
) -> BuiltinResult<String> {
    let updated = || format!("Figure {} updated", handle.as_u32());
    if !figure_clone.visible {
        #[cfg(all(
            feature = "gui",
            not(all(target_arch = "wasm32", feature = "plot-web"))
        ))]
        {
            if !rendering_disabled && !host_managed_rendering {
                let rendered = render_figure(handle, figure_clone)
                    .map_err(|flow| map_control_flow_with_builtin(flow, builtin))?;
                return Ok(format!("{}: {rendered}", updated()));
            }
        }
        return Ok(updated());
    }

    if rendering_disabled {
        if host_managed_rendering {
            return Ok(updated());
        }
        return Err(plotting_error(builtin, ERR_PLOTTING_UNAVAILABLE));
    }

    if host_managed_rendering {
        return Ok(updated());
    }

    // On Web/WASM we deliberately decouple "mutate figure state" from "present pixels".
    // The host coalesces figure events and presents on a frame cadence, and `drawnow()` /
    // `pause()` provide explicit "flush" boundaries for scripts.
    #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
    {
        let _ = figure_clone;
        Ok(updated())
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
    {
        let rendered = render_figure(handle, figure_clone)
            .map_err(|flow| map_control_flow_with_builtin(flow, builtin))?;
        Ok(format!("{}: {rendered}", updated()))
    }
}

/// Monotonic revision counter that increments on each successful mutation of the figure.
/// Used by web surface presentation logic to avoid redundant `render_figure` calls when
/// a surface is already up-to-date for a handle.
pub fn current_figure_revision(handle: FigureHandle) -> Option<u64> {
    let reg = registry();
    reg.figures.get(&handle).map(|state| state.revision)
}

fn interactive_rendering_disabled() -> bool {
    std::env::var_os("RUNMAT_DISABLE_INTERACTIVE_PLOTS").is_some()
}

fn host_managed_rendering_enabled() -> bool {
    std::env::var_os("RUNMAT_HOST_MANAGED_PLOTS").is_some()
}

#[cfg(test)]
pub(crate) fn disable_rendering_for_tests() {
    set_plot_test_env_vars();
}

pub fn set_line_style_order_for_axes(axes_index: usize, order: &[LineStyle]) {
    if with_active_style_cycle(axes_index, |cycle| cycle.set_order(order)).is_some() {
        return;
    }
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.cycle_for_axes_mut(axes_index).set_order(order);
}

pub fn next_line_style_for_axes(axes_index: usize) -> LineStyle {
    if let Some(style) = with_active_style_cycle(axes_index, |cycle| cycle.next()) {
        return style;
    }
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.cycle_for_axes_mut(axes_index).next()
}

pub fn line_color_for_series_index(series_index: usize) -> Vec4 {
    let theme = current_plot_theme_config().build_theme();
    theme.get_data_color(series_index)
}

pub fn line_color_for_axes_series_index(axes_index: usize, series_index: usize) -> Vec4 {
    if let Some(color) = with_active_color_cycle(axes_index, |cycle| cycle.color_at(series_index)) {
        return color;
    }
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state
        .color_cycle_for_axes_mut(axes_index)
        .color_at(series_index)
}

pub fn line_color_for_target_axes_series_index(
    handle: FigureHandle,
    axes_index: usize,
    series_index: usize,
) -> Vec4 {
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    state
        .color_cycle_for_axes_mut(axes_index)
        .color_at(series_index)
}

pub fn next_line_color_for_axes(axes_index: usize) -> Vec4 {
    if let Some(color) = with_active_color_cycle(axes_index, |cycle| cycle.next()) {
        return color;
    }
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.color_cycle_for_axes_mut(axes_index).next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;

    #[cfg(test)]
    pub(crate) fn reset_for_tests() {
        let mut reg = registry();
        reg.figures.clear();
        reg.current = FigureHandle::default();
        reg.next_handle = FigureHandle::default().next();
    }

    #[test]
    fn closing_last_figure_leaves_no_visible_figures() {
        let _guard = lock_plot_test_registry();
        ensure_plot_test_env();
        reset_for_tests();

        let handle = new_figure_handle();
        assert_eq!(figure_handles(), vec![handle]);

        close_figure(Some(handle)).expect("close figure");

        assert!(
            figure_handles().is_empty(),
            "closing the last figure should not recreate a default visible figure"
        );
    }

    #[test]
    fn hidden_figure_update_does_not_require_interactive_renderer() {
        let _guard = lock_plot_test_registry();
        ensure_plot_test_env();
        reset_for_tests();

        let handle = new_figure_handle();
        let mut figure = clone_figure(handle).expect("figure exists");
        figure.set_visible(false);

        let result = present_figure_update_with_options("plot", handle, figure, true, false)
            .expect("hidden figure should not try to present");
        assert_eq!(result, format!("Figure {} updated", handle.as_u32()));
    }

    #[cfg(all(
        feature = "gui",
        not(all(target_arch = "wasm32", feature = "plot-web"))
    ))]
    #[test]
    fn hidden_figure_update_uses_native_close_path_when_rendering_available() {
        let _guard = lock_plot_test_registry();
        ensure_plot_test_env();
        reset_for_tests();

        let handle = new_figure_handle();
        let mut figure = clone_figure(handle).expect("figure exists");
        figure.set_visible(false);

        let result = present_figure_update_with_options("plot", handle, figure, false, false)
            .expect("hidden figure should request native close");
        assert_eq!(
            result,
            format!(
                "Figure {} updated: Figure {} is hidden",
                handle.as_u32(),
                handle.as_u32()
            )
        );
    }

    #[test]
    fn hidden_figure_update_does_not_render_when_host_managed() {
        let _guard = lock_plot_test_registry();
        ensure_plot_test_env();
        reset_for_tests();

        let handle = new_figure_handle();
        let mut figure = clone_figure(handle).expect("figure exists");
        figure.set_visible(false);

        let result = present_figure_update_with_options("plot", handle, figure, false, true)
            .expect("hidden host-managed figure should not render directly");
        assert_eq!(result, format!("Figure {} updated", handle.as_u32()));
    }

    #[test]
    fn toggle_minor_grid_uses_effective_inherited_state() {
        let _guard = lock_plot_test_registry();
        ensure_plot_test_env();
        reset_for_tests();

        let handle = new_figure_handle();
        {
            let mut reg = registry();
            let state = get_state_mut(&mut reg, handle);
            state.figure.minor_grid_enabled = true;
            assert!(state.figure.minor_grid_enabled_for_axes(state.active_axes));
            assert!(
                !state
                    .figure
                    .axes_metadata(state.active_axes)
                    .expect("active axes metadata")
                    .minor_grid_explicit
            );
        }

        assert!(!toggle_minor_grid());

        let figure = clone_figure(handle).expect("figure exists");
        assert!(!figure.minor_grid_enabled_for_axes(0));
        let meta = figure.axes_metadata(0).expect("active axes metadata");
        assert!(meta.minor_grid_explicit);
        assert!(!meta.minor_grid_enabled);
    }
}
