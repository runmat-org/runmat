mod lifecycle;
#[cfg(unix)]
mod exit_observation;
mod limits;
mod process_tree;
mod retirement;
mod spawn;
mod stderr;

pub use lifecycle::{ChildProcess, ChildStdio, ProcessExit};
pub use limits::ResourceLimits;
pub use process_tree::{is_alive as is_process_alive, terminate_id as terminate_process_tree};
pub use retirement::{
    BootstrapBoundChild, ContainedChildRetirement, MemberRetirementBoundChild,
    ProcessRetirementLease, ProcessSpawnEvidence, RetirementBoundChild,
};
pub use spawn::{spawn, spawn_observed};
pub use stderr::CapturedStderr;
