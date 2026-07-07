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
type AxisDisplayBoundsSnapshot = Option<(f64, f64, f64, f64)>;

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
    hold_per_axes: HashMap<usize, bool>,
    line_style_cycles: HashMap<usize, LineStyleCycle>,
    line_color_cycles: HashMap<usize, LineColorCycle>,
    figure_color_order: Option<Vec<Vec4>>,
    zoom_mode: ZoomModeState,
    last_enabled_zoom_motion: ZoomMotion,
    zoom_axes_modes: HashMap<usize, ZoomModeState>,
    zoom_baselines: HashMap<usize, AxisLimitSnapshot>,
    revision: u64,
}

impl FigureState {
    fn new(handle: FigureHandle) -> Self {
        let title = format!("Figure {}", handle.as_u32());
        let figure = default_figure(&title, "X", "Y");
        Self {
            figure,
            active_axes: 0,
            hold_per_axes: HashMap::new(),
            line_style_cycles: HashMap::new(),
            line_color_cycles: HashMap::new(),
            figure_color_order: None,
            zoom_mode: ZoomModeState::default(),
            last_enabled_zoom_motion: ZoomMotion::Both,
            zoom_axes_modes: HashMap::new(),
            zoom_baselines: HashMap::new(),
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
    pub display_name: Option<String>,
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
}

#[derive(Clone, Debug)]
pub struct ImageHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
}

#[derive(Clone, Debug)]
pub struct HeatmapHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub x_labels: Vec<String>,
    pub y_labels: Vec<String>,
    pub color_data: Tensor,
}

#[derive(Clone, Debug)]
pub struct BinscatterHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub plot_index: usize,
    pub values: Tensor,
    pub x_bin_edges: Vec<f64>,
    pub y_bin_edges: Vec<f64>,
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
    pub num_bins: [usize; 2],
    pub auto_bins: bool,
    pub x_limits_option: Option<(f64, f64)>,
    pub y_limits_option: Option<(f64, f64)>,
    pub x_limits: (f64, f64),
    pub y_limits: (f64, f64),
    pub show_empty_bins: bool,
    pub face_alpha: f64,
    pub display_name: Option<String>,
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
pub enum PlotChildHandleState {
    Histogram(HistogramHandleState),
    Line(SimplePlotHandleState),
    Scatter(SimplePlotHandleState),
    Bar(SimplePlotHandleState),
    Stem(StemHandleState),
    ErrorBar(ErrorBarHandleState),
    Stairs(SimplePlotHandleState),
    Quiver(QuiverHandleState),
    Image(ImageHandleState),
    Heatmap(HeatmapHandleState),
    Binscatter(BinscatterHandleState),
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
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_axis_equal(axes, enabled);
        state.figure.set_axes_limits(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
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

pub fn set_axis_limits(x: Option<(f64, f64)>, y: Option<(f64, f64)>) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_limits(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
}

pub fn set_axis_limits_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    x: Option<(f64, f64)>,
    y: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_limits(axes_index, x, y);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
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

    Some((x_min, x_max, y_min, y_max))
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

fn apply_zoom_factor_to_state(state: &mut FigureState, axes_index: usize, factor: f64) {
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
    state.figure.set_axes_limits(axes_index, x_limits, y_limits);
}

fn restore_zoom_baseline_for_state(state: &mut FigureState, axes_index: usize) {
    let (x_limits, y_limits) = state
        .zoom_baselines
        .get(&axes_index)
        .copied()
        .unwrap_or((None, None));
    state.figure.set_axes_limits(axes_index, x_limits, y_limits);
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
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        for axes_index in 0..axes_count(state) {
            restore_zoom_baseline_for_state(state, axes_index);
        }
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn restore_zoom_baseline_for_axes(
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
        restore_zoom_baseline_for_state(state, axes_index);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn apply_zoom_factor_for_figure(handle: FigureHandle, factor: f64) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        apply_zoom_factor_to_state(state, state.active_axes, factor);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
}

pub fn apply_zoom_factor_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    factor: f64,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        apply_zoom_factor_to_state(state, axes_index, factor);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
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

pub fn set_colormap(colormap: ColorMap) {
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_colormap(axes, colormap);
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
    let ((), figure_clone) = with_axes_target_mut(handle, axes_index, |state| {
        state.figure.set_axes_colormap(axes_index, colormap);
    })?;
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);
    Ok(())
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
            display_name: None,
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
        })
    })
}

pub fn register_image_handle(figure: FigureHandle, axes_index: usize, plot_index: usize) -> f64 {
    register_simple_plot_handle(figure, axes_index, plot_index, |state| {
        PlotChildHandleState::Image(ImageHandleState {
            figure: state.figure,
            axes_index: state.axes_index,
            plot_index: state.plot_index,
        })
    })
}

pub fn register_heatmap_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    color_data: Tensor,
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
        }),
    );
    id as f64
}

pub fn register_binscatter_handle(
    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    values: Tensor,
    x_bin_edges: Vec<f64>,
    y_bin_edges: Vec<f64>,
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    num_bins: [usize; 2],
    auto_bins: bool,
    x_limits_option: Option<(f64, f64)>,
    y_limits_option: Option<(f64, f64)>,
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

fn purge_plot_children_for_figure(reg: &mut PlotRegistry, handle: FigureHandle) {
    reg.plot_children.retain(|_, state| match state {
        PlotChildHandleState::Histogram(hist) => hist.figure != handle,
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
        PlotChildHandleState::Stem(stem) => stem.figure != handle,
        PlotChildHandleState::ErrorBar(err) => err.figure != handle,
        PlotChildHandleState::Quiver(quiver) => quiver.figure != handle,
        PlotChildHandleState::Image(image) => image.figure != handle,
        PlotChildHandleState::Heatmap(heatmap) => heatmap.figure != handle,
        PlotChildHandleState::Binscatter(binscatter) => binscatter.figure != handle,
        PlotChildHandleState::Area(area) => area.figure != handle,
        PlotChildHandleState::Text(text) => text.figure != handle,
    });
}

fn purge_plot_children_for_axes(reg: &mut PlotRegistry, handle: FigureHandle, axes_index: usize) {
    reg.plot_children.retain(|_, state| match state {
        PlotChildHandleState::Histogram(hist) => {
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
        PlotChildHandleState::Area(area) => {
            !(area.figure == handle && area.axes_index == axes_index)
        }
        PlotChildHandleState::Text(text) => {
            !(text.figure == handle && text.axes_index == axes_index)
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
    let mut reg = registry();
    let state = get_state_mut(&mut reg, handle);
    axes_index < axes_count(state)
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
