#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(not(target_arch = "wasm32"))]
pub use native::{RemoteFsConfig, RemoteFsProvider};
#[cfg(target_arch = "wasm32")]
pub use wasm::{RemoteFsConfig, RemoteFsProvider};
