//! Runtime-owned HTTP wire contract and host-service adapters.

mod types;
mod stream;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod browser;

#[cfg(target_arch = "wasm32")]
pub use browser::BrowserHttpService;
#[cfg(not(target_arch = "wasm32"))]
pub use native::NativeHttpService;
pub use stream::{HttpBodyStream, HttpResponseHead};
pub use types::{HttpMethod, HttpRequest, HttpResponse, TransportError, TransportErrorKind};
