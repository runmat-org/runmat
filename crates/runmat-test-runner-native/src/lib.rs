#![allow(clippy::result_large_err)]

pub mod artifact;
pub mod host;
pub mod snapshot;
pub mod telemetry;
pub mod transport;
pub mod worker;

mod error;

pub use error::{NativeRunnerError, NativeRunnerResult};
pub use worker::{
    run_core_worker_stdio, LocalBackend, LocalBackendConfig, LocalSession, ProcessBackend,
    ProcessBackendConfig, ProcessSession,
};
