mod lifecycle;
#[cfg(unix)]
mod exit_observation;
mod limits;
mod process_tree;
mod retirement;
mod retirement_observation;
mod spawn;
mod startup;
mod output_capture;

pub use lifecycle::{ChildOutput, ChildProcess, ChildStdio, ProcessExit};
pub use limits::ResourceLimits;
pub use output_capture::{
    CapturedProcessOutput, ProcessOutputCaptureWriter, ProcessOutputCompletion,
    ProcessOutputSnapshot,
};
pub use process_tree::{is_alive as is_process_alive, terminate_id as terminate_process_tree};
pub use retirement::{
    BootstrapBoundChild, ContainedChildRetirement, MemberRetirementBoundChild,
    ProcessRetirementLease, ProcessSpawnEvidence, RetirementBoundChild,
};
pub use spawn::{spawn, spawn_observed};
pub use startup::release_startup_stderr;
