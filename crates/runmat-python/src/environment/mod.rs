#[cfg(not(target_arch = "wasm32"))]
mod discover;
mod model;

#[cfg(not(target_arch = "wasm32"))]
pub use discover::discover_python;
pub use model::{
    PythonDiscoveryRequest, PythonExecutionMode, PythonInstallation, PythonStatus, PythonVersion,
};
