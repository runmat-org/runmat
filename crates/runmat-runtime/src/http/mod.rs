//! Runtime-owned HTTP wire contract and host-service adapters.

mod types;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod browser;

#[cfg(target_arch = "wasm32")]
pub use browser::BrowserHttpService;
#[cfg(not(target_arch = "wasm32"))]
pub use native::NativeHttpService;
pub use types::{HttpMethod, HttpRequest, HttpResponse, TransportError, TransportErrorKind};
