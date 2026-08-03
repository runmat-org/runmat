pub mod endpoint;
pub mod frame;
pub mod handshake;
pub mod hidden;
pub mod stdio;

#[cfg(unix)]
pub mod unix;
#[cfg(windows)]
pub mod windows;

pub use endpoint::LocalEndpoint;
pub use frame::{read_frame, read_payload, write_frame, write_payload, FrameLimits};
pub use handshake::{negotiate_handshake, HostHandshake};
