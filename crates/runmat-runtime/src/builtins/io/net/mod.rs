//! Networking builtins.

pub mod accept;
pub(crate) mod client_state;
pub(crate) mod control;
pub mod close;
pub mod read;
pub mod readline;
pub(crate) mod server_state;
pub mod tcpclient;
pub mod tcpserver;
mod transport;
pub mod write;

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod test_support;
#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests;
