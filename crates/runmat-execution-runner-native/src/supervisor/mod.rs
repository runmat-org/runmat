mod auth;
mod client;
mod driver;
mod filesystem;
mod model;
mod protocol;
mod service;
mod store;
#[cfg(all(test, unix))]
mod tests;

pub use client::LocalSupervisorClient;
pub use driver::{
    complete_batch_driver, complete_batch_driver_with_value, execute_program_batch,
    prepare_batch_driver,
};
pub use model::{
    BatchDriverInvocation, BatchSubmission, JobAttachment, LocalJobRecord, LocalJobState,
    ProgramBatchSubmission,
};
pub use service::{run_local_supervisor, LocalSupervisor, LocalSupervisorConfig};
pub use store::SupervisorPaths;
