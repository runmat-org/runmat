pub mod endpoint;
pub mod frame;
pub mod handshake;
pub mod hidden;
mod parent;
pub mod stdio;
pub use parent::{WorkerParentGuard, WORKER_PARENT_MONITOR_STACK_BYTES};

#[cfg(unix)]
pub mod unix;
#[cfg(windows)]
pub mod windows;

pub use endpoint::LocalEndpoint;
pub use frame::{
    read_frame, read_payload, write_frame, write_payload, write_payload_sync, FrameLimits,
};
pub use handshake::{negotiate_handshake, HostHandshake};

mod meter;
pub use meter::{IpcByteCounter, MeteredIo};
