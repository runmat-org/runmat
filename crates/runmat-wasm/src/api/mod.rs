// The stable wasm-bindgen boundary exposed to JavaScript.
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod config;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod settings;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod source_check;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod project_entrypoints;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod cluster_program;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod execution_artifact;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod execution_program;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod execution_remote;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod init;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod observability;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod package;
pub(crate) mod plot;
pub(crate) mod plot_replay;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod project;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod session;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod streams;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod testing;

#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod fea_execution;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod fea_run_context;
#[cfg(not(feature = "graphics-replay"))]
mod fea_lease;
#[cfg(not(feature = "graphics-replay"))]
mod fea_meshing;
#[cfg(not(feature = "graphics-replay"))]
mod fea_async_host;
#[cfg(not(feature = "graphics-replay"))]
mod fea_analysis_host;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod fea_step_worker;
