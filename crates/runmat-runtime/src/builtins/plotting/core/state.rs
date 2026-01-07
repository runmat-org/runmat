use once_cell::sync::OnceCell;
use runmat_plot::plots::{Figure, LineStyle};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::fmt;
use std::ops::{Deref, DerefMut};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::MutexGuard;
#[cfg(test)]
use std::sync::Once;
use std::sync::{Arc, Mutex};

use super::common::{default_figure, ERR_PLOTTING_UNAVAILABLE};
use super::engine::render_figure;

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

const DEFAULT_LINE_STYLE_ORDER: [LineStyle; 4] = [
    LineStyle::Solid,
    LineStyle::Dashed,
    LineStyle::Dotted,
    LineStyle::DashDot,
];

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

#[derive(Default)]
struct FigureState {
    figure: Figure,
    active_axes: usize,
    hold_per_axes: HashMap<usize, bool>,
    line_style_cycles: HashMap<usize, LineStyleCycle>,
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

    fn reset_cycle(&mut self, axes_index: usize) {
        if let Some(cycle) = self.line_style_cycles.get_mut(&axes_index) {
            cycle.reset_cursor();
        }
    }
}

struct ActiveAxesContext {
    axes_index: usize,
    cycle_ptr: *mut LineStyleCycle,
}

struct AxesContextGuard {
    _private: (),
}

impl AxesContextGuard {
    fn install(state: &mut FigureState, axes_index: usize) -> Self {
        let cycle_ptr = state.cycle_for_axes_mut(axes_index) as *mut LineStyleCycle;
        ACTIVE_AXES_CONTEXT.with(|ctx| {
            debug_assert!(
                ctx.borrow().is_none(),
                "plot axes context already installed"
            );
            ctx.borrow_mut().replace(ActiveAxesContext {
                axes_index,
                cycle_ptr,
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

fn with_active_cycle<R>(axes_index: usize, f: impl FnOnce(&mut LineStyleCycle) -> R) -> Option<R> {
    ACTIVE_AXES_CONTEXT.with(|ctx| {
        let guard = ctx.borrow();
        let active = guard.as_ref()?;
        if active.axes_index != axes_index {
            return None;
        }
        let cycle = unsafe { &mut *active.cycle_ptr };
        Some(f(cycle))
    })
}

struct PlotRegistry {
    current: FigureHandle,
    next_handle: FigureHandle,
    figures: HashMap<FigureHandle, FigureState>,
}

impl Default for PlotRegistry {
    fn default() -> Self {
        Self {
            current: FigureHandle::default(),
            next_handle: FigureHandle::default().next(),
            figures: HashMap::new(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
static REGISTRY: OnceCell<Mutex<PlotRegistry>> = OnceCell::new();

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
}

impl<'a> PlotRegistryGuard<'a> {
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

#[derive(Clone, Debug)]
pub enum FigureError {
    InvalidHandle(u32),
    InvalidSubplotGrid {
        rows: usize,
        cols: usize,
    },
    InvalidSubplotIndex {
        rows: usize,
        cols: usize,
        index: usize,
    },
    InvalidAxesHandle,
    RenderFailure(String),
}

impl fmt::Display for FigureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FigureError::InvalidHandle(handle) => {
                write!(f, "figure handle {handle} does not exist")
            }
            FigureError::InvalidSubplotGrid { rows, cols } => write!(
                f,
                "subplot grid dimensions must be positive (rows={rows}, cols={cols})"
            ),
            FigureError::InvalidSubplotIndex { rows, cols, index } => write!(
                f,
                "subplot index {index} is out of range for a {rows}x{cols} grid"
            ),
            FigureError::InvalidAxesHandle => write!(f, "invalid axes handle"),
            FigureError::RenderFailure(message) => {
                write!(f, "failed to render figure snapshot: {message}")
            }
        }
    }
}

impl std::error::Error for FigureError {}

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

#[allow(dead_code)]
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
    let guard = REGISTRY
        .get_or_init(|| Mutex::new(PlotRegistry::default()))
        .lock()
        .expect("plot registry poisoned");
    PlotRegistryGuard::new(guard)
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
        PlotRegistryGuard::new(guard_static)
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

pub fn install_figure_observer(observer: Arc<FigureObserver>) -> Result<(), String> {
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
        figure: Some(figure),
    });
}

fn notify_without_figure(handle: FigureHandle, kind: FigureEventKind) {
    notify_event(FigureEventView {
        handle,
        kind,
        figure: None,
    });
}

fn note_recent_figure(handle: FigureHandle) {
    RECENT_FIGURES.with(|set| {
        set.borrow_mut().insert(handle);
    });
}

pub fn reset_recent_figures() {
    RECENT_FIGURES.with(|set| set.borrow_mut().clear());
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

pub fn current_axes_state() -> FigureAxesState {
    let reg = registry();
    let handle = reg.current;
    let state = reg.figures.get(&handle).expect("current figure must exist");
    FigureAxesState {
        handle,
        rows: state.figure.axes_rows.max(1),
        cols: state.figure.axes_cols.max(1),
        active_index: state.active_axes,
    }
}

pub fn figure_handles() -> Vec<FigureHandle> {
    let reg = registry();
    reg.figures.keys().copied().collect()
}

pub fn clone_figure(handle: FigureHandle) -> Option<Figure> {
    let reg = registry();
    reg.figures.get(&handle).map(|state| state.figure.clone())
}

pub fn clear_figure(target: Option<FigureHandle>) -> Result<FigureHandle, FigureError> {
    let mut reg = registry();
    let handle = target.unwrap_or(reg.current);
    let state = reg
        .figures
        .get_mut(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    *state = FigureState::new(handle);
    let figure_clone = state.figure.clone();
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

    if reg.current == handle {
        if let Some((&next_handle, _)) = reg.figures.iter().next() {
            reg.current = next_handle;
        } else {
            let default = FigureHandle::default();
            reg.current = default;
            reg.next_handle = default.next();
            reg.figures.insert(default, FigureState::new(default));
            let figure_clone = reg
                .figures
                .get(&default)
                .expect("default figure inserted")
                .figure
                .clone();
            drop(reg);
            notify_without_figure(handle, FigureEventKind::Closed);
            notify_with_figure(default, &figure_clone, FigureEventKind::Created);
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
    Ok(())
}

pub fn render_active_plot<F>(opts: PlotRenderOptions<'_>, mut apply: F) -> Result<String, String>
where
    F: FnMut(&mut Figure, usize) -> Result<(), String>,
{
    let rendering_disabled = interactive_rendering_disabled();
    let (handle, figure_clone) = {
        let mut reg = registry();
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes_index = state.active_axes;

        if !state.hold() {
            state.figure.clear_axes(axes_index);
            state.reset_cycle(axes_index);
        }

        if !opts.title.is_empty() {
            state.figure.set_title(opts.title);
        }
        state.figure.set_axis_labels(opts.x_label, opts.y_label);
        state.figure.set_grid(opts.grid);
        state.figure.set_axis_equal(opts.axis_equal);

        let _axes_context = AxesContextGuard::install(state, axes_index);
        apply(&mut state.figure, axes_index)?;

        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure_clone, FigureEventKind::Updated);

    if rendering_disabled {
        return Err(ERR_PLOTTING_UNAVAILABLE.to_string());
    }

    let rendered = render_figure(handle, figure_clone)?;
    Ok(format!("Figure {} updated: {rendered}", handle.as_u32()))
}

fn interactive_rendering_disabled() -> bool {
    std::env::var_os("RUNMAT_DISABLE_INTERACTIVE_PLOTS").is_some()
}

#[cfg(test)]
pub(crate) fn disable_rendering_for_tests() {
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        std::env::set_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS", "1");
    });
}

pub fn set_line_style_order_for_axes(axes_index: usize, order: &[LineStyle]) {
    if with_active_cycle(axes_index, |cycle| cycle.set_order(order)).is_some() {
        return;
    }
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.cycle_for_axes_mut(axes_index).set_order(order);
}

pub fn next_line_style_for_axes(axes_index: usize) -> LineStyle {
    if let Some(style) = with_active_cycle(axes_index, |cycle| cycle.next()) {
        return style;
    }
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.cycle_for_axes_mut(axes_index).next()
}
