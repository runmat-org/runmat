pub mod endpoint;
pub mod frame;
pub mod handshake;
pub mod hidden;
pub mod session;
pub mod stdio;

#[cfg(unix)]
pub mod unix;
#[cfg(windows)]
pub mod windows;

pub use endpoint::LocalEndpoint;
pub use frame::{
    read_frame, read_payload, read_payload_blocking, write_frame, write_payload,
    write_payload_blocking, FrameLimits,
};
pub use handshake::{negotiate_handshake, HostHandshake};
pub use session::{
    authenticate_driver, authenticate_host, authenticate_host_blocking, AuthenticatedSession,
    SessionSecret, INITIAL_SESSION_LIMITS,
};
