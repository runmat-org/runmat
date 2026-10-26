// The stable wasm-bindgen boundary exposed to JavaScript.
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod config;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod execution_artifact;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod execution_program;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod execution_remote;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod init;
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
