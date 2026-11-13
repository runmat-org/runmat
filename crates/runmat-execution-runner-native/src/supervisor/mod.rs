mod auth;
mod bootstrap;
mod cancellation;
mod client;
mod config;
mod console_access;
mod raw_output_access;
mod raw_output_model;
mod console_journal;
mod control;
mod driver;
mod driver_files;
mod execution;
pub(crate) mod filesystem;
pub(crate) use filesystem::unix_millis;
mod model;
mod ownership;
mod program_store;
mod retained;
pub(crate) mod protocol;
mod scheduler;
mod server;
mod service;
mod store;
#[cfg(all(test, unix))]
mod tests;
#[cfg(test)]
mod test_support;

pub use client::{JobSubmissionDisposition, LocalSupervisorClient};
pub use config::LocalSupervisorConfig;
pub use driver::{
    complete_batch_driver, complete_batch_driver_with_response, execute_and_complete_graph_batch,
    execute_and_complete_program_batch, execute_graph_batch, prepare_batch_driver,
};
pub use model::{
    BatchDriverInvocation, BatchSubmission, DurableExecutionContext, DurableProgramAttempt,
    GraphBatchSubmission, JobAttachment, LocalJobRecord, LocalJobState, LocalJobWorkloadKind,
    ProgramBatchExecution, ProgramBatchSubmission, MIN_RETENTION_MILLIS,
};
pub use raw_output_model::{
    RawProcessOutputAttempt, RawProcessOutputCursor, RawProcessOutputLookup, RawProcessOutputPage,
};
pub use server::run_local_supervisor;
pub use service::LocalSupervisor;
pub use store::SupervisorPaths;

pub use ownership::DurableAttemptLease;
