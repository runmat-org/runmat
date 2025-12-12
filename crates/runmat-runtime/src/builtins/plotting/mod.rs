//! Plotting builtins backed by the runmat-plot renderer.

#[path = "core/common.rs"]
pub(crate) mod common;
#[path = "core/context.rs"]
pub mod context;
#[path = "core/engine.rs"]
pub(crate) mod engine;
#[path = "core/gpu_helpers.rs"]
pub(crate) mod gpu_helpers;
#[path = "core/perf.rs"]
pub(crate) mod perf;
#[path = "core/point.rs"]
pub(crate) mod point;
#[path = "core/state.rs"]
pub(crate) mod state;
#[path = "core/style.rs"]
pub(crate) mod style;
#[path = "core/web.rs"]
pub mod web;

#[path = "ops/bar.rs"]
pub(crate) mod bar;
#[path = "ops/clf.rs"]
pub(crate) mod clf;
#[path = "ops/close.rs"]
pub(crate) mod close;
#[path = "ops/contour.rs"]
pub(crate) mod contour;
#[path = "ops/contourf.rs"]
pub(crate) mod contourf;
#[path = "ops/figure.rs"]
pub(crate) mod figure;
#[path = "ops/gca.rs"]
pub(crate) mod gca;
#[path = "ops/gcf.rs"]
pub(crate) mod gcf;
#[path = "ops/handle_args.rs"]
pub(crate) mod handle_args;
#[path = "ops/hist.rs"]
pub mod hist;
#[path = "ops/hold.rs"]
pub(crate) mod hold;
#[path = "ops/mesh.rs"]
pub(crate) mod mesh;
#[path = "ops/meshc.rs"]
pub(crate) mod meshc;
#[path = "ops/plot.rs"]
pub(crate) mod plot;
#[path = "ops/scatter.rs"]
pub(crate) mod scatter;
#[path = "ops/scatter3.rs"]
pub(crate) mod scatter3;
#[path = "ops/stairs.rs"]
pub(crate) mod stairs;
#[path = "ops/subplot.rs"]
pub(crate) mod subplot;
#[path = "ops/surf.rs"]
pub(crate) mod surf;
#[path = "ops/surfc.rs"]
pub(crate) mod surfc;

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

#[cfg(test)]
pub(crate) mod tests {
    use super::state;
    use std::sync::Once;

    pub(crate) fn ensure_plot_test_env() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            state::disable_rendering_for_tests();
        });
    }
}
