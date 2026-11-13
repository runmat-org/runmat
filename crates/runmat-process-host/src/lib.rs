//! Domain-neutral native child-process and local IPC infrastructure.

pub mod capability;
pub mod child;
pub mod clock;
pub mod durable_file;
pub mod command;
pub mod environment;
pub mod error;
pub mod ipc;
pub mod private_bytes;
pub mod shared_memory;

pub use capability::HostCapability;
pub use child::{
    is_process_alive, release_startup_stderr, terminate_process_tree, BootstrapBoundChild,
    CapturedProcessOutput, ChildOutput, ChildProcess, ChildStdio, ContainedChildRetirement,
    MemberRetirementBoundChild, ProcessExit, ProcessOutputCaptureWriter, ProcessOutputCompletion,
    ProcessOutputSnapshot, ProcessRetirementLease, ProcessSpawnEvidence, ResourceLimits,
    RetirementBoundChild,
};
pub use command::{
    ChildLifetime, CompleteProcessOutput, HostCommand, ProcessGroupPlacement, StdioPolicy,
};
pub use error::{ProcessHostError, ProcessHostResult};
pub use ipc::hidden::{HiddenMode, HiddenModeRegistry};
