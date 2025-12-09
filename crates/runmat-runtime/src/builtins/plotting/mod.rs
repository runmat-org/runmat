//! Plotting builtins backed by the runmat-plot renderer.

#[path = "core/common.rs"]
mod common;
#[path = "core/context.rs"]
pub mod context;
#[path = "core/engine.rs"]
mod engine;
#[path = "core/gpu_helpers.rs"]
mod gpu_helpers;
#[path = "core/perf.rs"]
mod perf;
#[path = "core/point.rs"]
mod point;
#[path = "core/state.rs"]
mod state;
#[path = "core/style.rs"]
mod style;
#[path = "core/web.rs"]
pub mod web;

#[path = "ops/bar.rs"]
mod bar;
#[path = "ops/clf.rs"]
mod clf;
#[path = "ops/close.rs"]
mod close;
#[path = "ops/contour.rs"]
mod contour;
#[path = "ops/contourf.rs"]
mod contourf;
#[path = "ops/figure.rs"]
mod figure;
#[path = "ops/gca.rs"]
mod gca;
#[path = "ops/gcf.rs"]
mod gcf;
#[path = "ops/handle_args.rs"]
mod handle_args;
#[path = "ops/hist.rs"]
pub mod hist;
#[path = "ops/hold.rs"]
mod hold;
#[path = "ops/mesh.rs"]
mod mesh;
#[path = "ops/meshc.rs"]
mod meshc;
#[path = "ops/plot.rs"]
mod plot;
#[path = "ops/scatter.rs"]
mod scatter;
#[path = "ops/scatter3.rs"]
mod scatter3;
#[path = "ops/stairs.rs"]
mod stairs;
#[path = "ops/subplot.rs"]
mod subplot;
#[path = "ops/surf.rs"]
mod surf;
#[path = "ops/surfc.rs"]
mod surfc;

pub use perf::{set_scatter_target_points, set_surface_vertex_budget};
pub use state::{
    clear_figure, clone_figure, close_figure, configure_subplot, current_axes_state,
    current_figure_handle, install_figure_observer, new_figure_handle, reset_recent_figures,
    select_figure, set_hold, take_recent_figures, FigureAxesState, FigureError, FigureEventKind,
    FigureEventView, FigureHandle, HoldMode,
};
pub use web::{install_web_renderer, install_web_renderer_for_handle, web_renderer_ready};

#[cfg(feature = "plot-core")]
pub use engine::{render_figure_png_bytes, render_figure_snapshot};

pub mod ops {
    pub use super::hist;
}
