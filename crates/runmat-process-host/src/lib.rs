//! Domain-neutral native child-process and local IPC infrastructure.

pub mod capability;
pub mod child;
pub mod command;
pub mod environment;
pub mod error;
pub mod ipc;
pub mod shared_memory;

pub use capability::HostCapability;
pub use child::{
    is_process_alive, release_startup_stderr, terminate_process_tree, ChildProcess, ChildStdio,
    ProcessExit, ResourceLimits, StderrCompletion, StderrSnapshot,
};
pub use command::{ChildLifetime, HostCommand, StdioPolicy};
pub use error::{ProcessHostError, ProcessHostResult};
pub use ipc::hidden::{HiddenMode, HiddenModeRegistry};
