//! Native process composition for the portable RunMat execution driver.

mod analysis_host;
mod analysis_kernel;
mod config;
mod driver;
mod durable;
mod error;
mod exact_geometry_admission;
mod exact_meshing_executor;
mod exact_meshing_job;
mod exact_meshing_root;
mod local_store;
mod materialized_project;
mod meshing_evaluator;
mod meshing_host;
mod meshing_platform;
mod program_session;
mod protocol;
mod remote;
mod remote_exact_meshing_executor;
mod runtime_meshing_provider;
mod service;
pub mod supervisor;
mod test_workload;
mod worker;

pub use analysis_host::{execute_analysis_program_request, NativeAnalysisHostLimits};
pub use analysis_kernel::NativeAnalysisKernel;
pub use config::NativeExecutionConfig;
pub use driver::NATIVE_OBJECT_STORE_ROOT_ENV;
pub use error::{NativeExecutionError, NativeExecutionResult};
pub use exact_geometry_admission::{
    admit_prepared_exact_geometry, prepare_exact_geometry_admission, ExactGeometryAdmissionError,
    NativeMeshingContactDefinition, PreparedExactGeometryAdmission,
};
pub use exact_meshing_executor::{MeshingExecutionPolicy, NativeExactMeshingExecutor};
pub use exact_meshing_job::{
    mesh_exact_geometry, NativeExactMeshingJob, NativeExactMeshingJobError,
    NativeExactMeshingResult,
};
pub use exact_meshing_root::{
    decode_exact_meshing_root, prepare_exact_meshing_root, DecodedExactMeshingRoot,
    ExactMeshingRootPreparation, ExactMeshingRootRequest, PreparedExactMeshingRoot,
    EXACT_MESHING_ROOT_EXECUTION_MODE, EXACT_MESHING_ROOT_TARGET_PROFILE,
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
pub use remote_exact_meshing_executor::RemoteExactMeshingExecutor;
pub use runmat_execution_artifact::cache::FilesystemObjectStore;
pub use runtime_meshing_provider::{
    mesh_source_with_native_execution, native_meshing_execution_provider,
    resolve_native_meshing_request, NativeMeshingPartitionPolicy, NativeMeshingSourceError,
    NativeMeshingSourceRequest,
};
pub use service::NativeExecutionService;
pub use test_workload::{execute_host_program_request, execute_host_program_request_with_project};
pub use worker::run_worker_stdio;
