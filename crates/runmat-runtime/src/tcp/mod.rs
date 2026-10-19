//! Typed TCP transport failures and native host adapter.

mod error;
#[cfg(not(target_arch = "wasm32"))]
mod native;

pub use error::{TcpTransportError, TcpTransportErrorKind, TcpWriteProgress};
#[cfg(not(target_arch = "wasm32"))]
pub use native::NativeTcpService;
