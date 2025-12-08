use once_cell::sync::OnceCell;
use runmat_plot::plots::{Figure, LineStyle};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use super::common::default_figure;
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
        self.line_style_cycles
            .entry(axes_index)
            .or_insert_with(LineStyleCycle::default)
    }

    fn reset_cycle(&mut self, axes_index: usize) {
        if let Some(cycle) = self.line_style_cycles.get_mut(&axes_index) {
            cycle.reset_cursor();
        }
    }
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

static REGISTRY: OnceCell<Mutex<PlotRegistry>> = OnceCell::new();
type FigureObserver = dyn Fn(u32, &Figure) + Send + Sync + 'static;
static FIGURE_OBSERVER: OnceCell<Arc<FigureObserver>> = OnceCell::new();

fn registry() -> MutexGuard<'static, PlotRegistry> {
    REGISTRY
        .get_or_init(|| Mutex::new(PlotRegistry::default()))
        .lock()
        .expect("plot registry poisoned")
}

fn get_state_mut<'a>(registry: &'a mut PlotRegistry, handle: FigureHandle) -> &'a mut FigureState {
    registry
        .figures
        .entry(handle)
        .or_insert_with(|| FigureState::new(handle))
}

pub fn install_figure_observer(observer: Arc<FigureObserver>) -> Result<(), String> {
    FIGURE_OBSERVER
        .set(observer)
        .map_err(|_| "figure observer already installed".to_string())
}

fn notify_figure_observer(handle: FigureHandle, figure: &Figure) {
    if let Some(observer) = FIGURE_OBSERVER.get() {
        observer(handle.as_u32(), figure);
    }
}

pub fn select_figure(handle: FigureHandle) {
    let mut reg = registry();
    reg.current = handle;
    get_state_mut(&mut reg, handle);
}

pub fn new_figure_handle() -> FigureHandle {
    let mut reg = registry();
    let handle = reg.next_handle;
    reg.next_handle = reg.next_handle.next();
    reg.current = handle;
    get_state_mut(&mut reg, handle);
    handle
}

pub fn current_figure_handle() -> FigureHandle {
    registry().current
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

pub fn configure_subplot(rows: usize, cols: usize, index: usize) {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.figure.set_subplot_grid(rows, cols);
    let total_axes = state.figure.axes_rows.max(1) * state.figure.axes_cols.max(1);
    let clamped = index.min(total_axes.saturating_sub(1));
    state.active_axes = clamped;
}

pub fn render_active_plot<F>(opts: PlotRenderOptions<'_>, mut apply: F) -> Result<String, String>
where
    F: FnMut(&mut Figure, usize) -> Result<(), String>,
{
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

        apply(&mut state.figure, axes_index)?;

        (handle, state.figure.clone())
    };

    notify_figure_observer(handle, &figure_clone);
    let rendered = render_figure(handle, figure_clone)?;
    Ok(format!("Figure {} updated: {rendered}", handle.as_u32()))
}

pub fn set_line_style_order_for_axes(axes_index: usize, order: &[LineStyle]) {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.cycle_for_axes_mut(axes_index).set_order(order);
}

pub fn next_line_style_for_axes(axes_index: usize) -> LineStyle {
    let mut reg = registry();
    let handle = reg.current;
    let state = get_state_mut(&mut reg, handle);
    state.cycle_for_axes_mut(axes_index).next()
}
