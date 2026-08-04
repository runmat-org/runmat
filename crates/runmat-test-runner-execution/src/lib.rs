//! Test-owned adapter from [`runmat_test_runner::worker::WorkerBackend`] to the
//! portable RunMat execution scheduler.
//!
//! The adapter deliberately knows only how to identify and resource one exact
//! test attempt. Discovery, fixtures, retries, coverage, events, artifacts, and
//! result semantics remain owned by `runmat-test` and `runmat-test-runner`.

mod backend;
mod capability;
mod request;
mod session;
mod workload;

pub use backend::ExecutionWorkerBackend;
pub use capability::ExecutionBackendConfig;
pub use session::ExecutionWorkerSession;
pub use workload::{
    decode_execution, encode_execution, TestAttemptWorkload, TEST_ATTEMPT_EXECUTION_MODE,
    TEST_ATTEMPT_TARGET_PROFILE,
};
