//! Native process composition for the portable RunMat execution driver.

mod config;
mod driver;
mod durable;
mod error;
mod local_store;
mod materialized_project;
mod meshing_host;
mod object_store;
mod program_session;
mod protocol;
mod remote;
mod service;
pub mod supervisor;
mod test_workload;
mod worker;

pub use config::NativeExecutionConfig;
pub use driver::NATIVE_OBJECT_STORE_ROOT_ENV;
pub use error::{NativeExecutionError, NativeExecutionResult};
pub use meshing_host::{
    execute_meshing_program_request, run_meshing_worker_stdio, NativeMeshingHostLimits,
};
pub use object_store::NativeObjectStore;
pub use program_session::{NativeProgramSession, NativeProgramTask};
pub use protocol::{ProgramProgress, WorkerResponse};
pub use remote::{
    run_remote_driver_from_env, run_remote_worker_from_env, run_remote_worker_quic,
    run_remote_worker_relay, QuicRemoteWorkerChannel, RelayRemoteWorkerChannel, RemoteAttempt,
    RemoteBundleReceipt, RemoteObjectReceipt, RemotePoolDriver, RemoteTaskCompletion,
    RemoteValueReceipt, RemoteWorkerChannel, RemoteWorkerRelayRequest,
};
pub use service::NativeExecutionService;
pub use test_workload::{execute_host_program_request, execute_host_program_request_with_project};
pub use worker::run_worker_stdio;
