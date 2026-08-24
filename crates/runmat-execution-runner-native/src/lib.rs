//! Native process composition for the portable RunMat execution driver.

mod config;
mod driver;
mod durable;
mod error;
mod exact_geometry_admission;
mod exact_meshing_executor;
mod exact_meshing_job;
mod local_store;
mod materialized_project;
mod meshing_evaluator;
mod meshing_host;
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
pub use exact_geometry_admission::{
    admit_prepared_exact_geometry, prepare_exact_geometry_admission, ExactGeometryAdmissionError,
    PreparedExactGeometryAdmission,
};
pub use exact_meshing_executor::{NativeExactMeshingExecutor, NativeMeshingExecutionPolicy};
pub use exact_meshing_job::{
    mesh_exact_geometry, NativeExactMeshingJob, NativeExactMeshingJobError,
    NativeExactMeshingResult,
};
pub use meshing_evaluator::{native_meshing_kernel_dispatcher, NativeMeshingEvaluatorProvider};
pub use meshing_host::{
    execute_meshing_program_request, run_meshing_worker_stdio, NativeMeshingHostLimits,
};
pub use program_session::{NativeProgramSession, NativeProgramTask};
pub use protocol::{ProgramProgress, WorkerResponse};
pub use remote::{
    run_remote_driver_from_env, run_remote_meshing_worker_quic, run_remote_worker_from_env,
    run_remote_worker_quic, run_remote_worker_relay, QuicRemoteWorkerChannel,
    RelayRemoteWorkerChannel, RemoteAttempt, RemoteBundleReceipt, RemoteMeshingWorkerQuicRequest,
    RemoteObjectReceipt, RemotePoolDriver, RemoteTaskCompletion, RemoteValueReceipt,
    RemoteWorkerChannel, RemoteWorkerChannelConfig, RemoteWorkerQuicRequest,
    RemoteWorkerRelayRequest,
};
pub use runmat_execution_artifact::cache::FilesystemObjectStore;
pub use service::NativeExecutionService;
pub use test_workload::{execute_host_program_request, execute_host_program_request_with_project};
pub use worker::run_worker_stdio;
