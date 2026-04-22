#![cfg(target_arch = "wasm32")]

// Keep the crate root as a compact audit point for the JS-facing export surface.
mod api;
mod runtime;
mod wire;

pub use api::init::{init_runmat, register_fs_provider};
// Plotting and figure APIs.
pub use api::plot::{
    bind_surface_to_figure, create_plot_surface, deregister_figure_canvas, deregister_plot_canvas,
    destroy_plot_surface, fit_plot_surface_extents, get_plot_surface_camera_state,
    handle_plot_surface_event, on_figure_event, plot_renderer_ready, present_figure_on_surface,
    present_surface, register_figure_canvas, register_plot_canvas, render_current_figure_scene,
    reset_plot_surface_camera, resize_figure_canvas, resize_plot_surface,
    set_plot_surface_camera_state, set_plot_theme_config, wasm_clear_figure, wasm_close_figure,
    wasm_configure_subplot, wasm_current_axes_info, wasm_current_figure_handle,
    wasm_new_figure_handle, wasm_render_figure_image, wasm_render_figure_image_with_camera_state,
    wasm_render_figure_image_with_textmark, wasm_select_figure, wasm_set_hold_mode,
};
// Stateful session wrapper.
pub use api::session::RunMatWasm;
// Log and trace streams.
pub use api::streams::{
    set_log_filter, subscribe_runtime_log, subscribe_stdout, subscribe_trace_events,
    unsubscribe_runtime_log, unsubscribe_stdout, unsubscribe_trace_events,
};
