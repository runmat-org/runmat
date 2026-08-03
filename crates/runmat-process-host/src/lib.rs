//! Domain-neutral native child-process and local IPC infrastructure.

pub mod capability;
pub mod child;
pub mod command;
pub mod environment;
pub mod error;
pub mod ipc;
pub mod shared_memory;

pub use capability::HostCapability;
pub use child::{ChildProcess, ChildStdio, ProcessExit};
pub use command::{HostCommand, StdioPolicy};
pub use error::{ProcessHostError, ProcessHostResult};
pub use ipc::hidden::{HiddenMode, HiddenModeRegistry};
