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
pub mod write;

#[cfg(test)]
pub(crate) mod test_support;
