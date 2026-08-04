//! Native process composition for the portable RunMat execution driver.

mod config;
mod driver;
mod durable;
mod error;
mod local_store;
mod protocol;
mod remote;
mod service;
pub mod supervisor;
mod test_workload;
mod worker;

pub use config::NativeExecutionConfig;
pub use error::{NativeExecutionError, NativeExecutionResult};
pub use protocol::WorkerResponse;
pub use remote::{
    run_remote_driver_from_env, run_remote_worker_from_env, run_remote_worker_quic,
    run_remote_worker_relay, QuicRemoteWorkerChannel, RelayRemoteWorkerChannel, RemoteAttempt,
    RemoteBundleReceipt, RemotePoolDriver, RemoteTaskCompletion, RemoteValueReceipt,
    RemoteWorkerChannel,
};
pub use service::NativeExecutionService;
pub use test_workload::execute_host_program_request;
pub use worker::run_worker_stdio;
