//! Networking builtins.

pub mod accept;
pub(crate) mod control;
mod error;
mod receiver;
pub mod close;
pub mod read;
pub mod readline;
pub mod tcpclient;
pub mod tcpserver;
mod transport;
pub mod write;

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod test_support;
#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests;
