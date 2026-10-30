// Shared implementation support consumed by the wasm API layer.
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod config;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod settings;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod execution;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod filesystem;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod gpu;
// Shared with the full host; playback does not install execution stream callbacks.
#[cfg_attr(feature = "graphics-replay", allow(dead_code))]
pub(crate) mod logging;
#[cfg(not(feature = "graphics-replay"))]
pub(crate) mod package_cache;
#[cfg_attr(feature = "graphics-replay", allow(dead_code))]
pub(crate) mod state;
