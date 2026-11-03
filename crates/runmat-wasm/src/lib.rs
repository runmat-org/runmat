#![cfg(target_arch = "wasm32")]

// Keep the crate root as a compact audit point for the JS-facing export surface.
mod api;
mod runtime;
// The full host also uses execution error/preview codecs in this shared module.
#[cfg_attr(feature = "graphics-replay", allow(dead_code))]
mod wire;

#[cfg(not(feature = "graphics-replay"))]
pub use api::cluster_program::{prepare_cluster_graph, prepare_cluster_program};
#[cfg(not(feature = "graphics-replay"))]
pub use api::config::{migrate_legacy_runmat_config, patch_runmat_config, resolve_runmat_config};
#[cfg(not(feature = "graphics-replay"))]
pub use api::execution_artifact::{
    build_execution_bundle, materialize_execution_bundle, BrowserExecutionRecipient,
};
#[cfg(not(feature = "graphics-replay"))]
pub use api::execution_program::{execute_program_artifact, execute_worker_program};
#[cfg(not(feature = "graphics-replay"))]
pub use api::init::{init_runmat, string_element_wire_schema_version};
#[cfg(not(feature = "graphics-replay"))]
pub use api::observability::{
    flush_open_telemetry, open_telemetry_stats, send_open_telemetry_test, shutdown_open_telemetry,
};
#[cfg(not(feature = "graphics-replay"))]
pub use api::package::{
    build_git_snapshot, package_cache_acquire_lease, package_cache_gc, package_cache_release_lease,
    package_cache_renew_lease, package_cache_status, plan_git_acquisition, resolve_project,
    validate_git_snapshot,
};
#[cfg(not(feature = "graphics-replay"))]
pub use api::project_entrypoints::prepare_named_cluster_program;
#[cfg(not(feature = "graphics-replay"))]
pub use api::settings::{edit_user_runtime_settings, resolve_runtime_settings};
#[cfg(not(feature = "graphics-replay"))]
pub use api::source_check::check_source;
// Plotting and figure APIs.
pub use api::plot::{
    bind_surface_to_figure, create_plot_surface, destroy_plot_surface, fit_plot_surface_extents,
    get_plot_surface_camera_state, handle_plot_surface_event, on_figure_event, plot_renderer_ready,
    present_figure_on_surface, present_surface, render_current_figure_scene,
    reset_plot_surface_camera, resize_plot_surface, set_plot_surface_camera_state,
    set_plot_theme_config, wasm_clear_figure, wasm_close_figure, wasm_configure_subplot,
    wasm_current_axes_info, wasm_current_figure_handle, wasm_new_figure_handle,
    wasm_render_figure_image, wasm_render_figure_image_with_camera_state,
    wasm_render_figure_image_with_textmark, wasm_select_figure, wasm_set_hold_mode,
};
#[cfg(not(feature = "graphics-replay"))]
pub use api::project::{project_handoff, project_revision, validate_project_handoff};
// Stateful session wrapper.
#[cfg(not(feature = "graphics-replay"))]
pub use api::session::RunMatWasm;
#[cfg(not(feature = "graphics-replay"))]
pub use api::testing::{
    freeze_test_snapshot, project_test_layout, run_tests, run_tests_with_events,
};
// Log and trace streams.
#[cfg(not(feature = "graphics-replay"))]
pub use api::streams::{
    set_log_filter, subscribe_runtime_log, subscribe_stdout, subscribe_trace_events,
    unsubscribe_runtime_log, unsubscribe_stdout, unsubscribe_trace_events,
};

#[cfg(not(feature = "graphics-replay"))]
pub use api::fea_execution::execute_prepared_fea_study;
#[cfg(not(feature = "graphics-replay"))]
pub use api::fea_step_worker::execute_owned_fea_study_step;
