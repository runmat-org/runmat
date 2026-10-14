mod auth;
mod client;
mod driver;
mod filesystem;
pub(crate) use filesystem::unix_millis;
mod model;
mod protocol;
mod service;
mod store;
#[cfg(all(test, unix))]
mod tests;

pub use client::LocalSupervisorClient;
pub use driver::{
    complete_batch_driver, complete_batch_driver_with_response, execute_graph_batch,
    execute_program_batch, prepare_batch_driver,
};
pub use model::{
    BatchDriverInvocation, BatchSubmission, GraphBatchSubmission, JobAttachment, LocalJobRecord,
    LocalJobState, LocalJobWorkloadKind, ProgramBatchSubmission, MIN_RETENTION_MILLIS,
};
pub use service::{run_local_supervisor, LocalSupervisor, LocalSupervisorConfig};
pub use store::SupervisorPaths;
