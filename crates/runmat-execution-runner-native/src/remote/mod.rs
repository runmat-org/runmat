mod attempt_reply;
mod bundle_cache;
mod channel;
mod checkpoint;
mod clock_observer;
pub(crate) use clock_observer::RemoteObservedWave;
mod collective;
mod config;
mod crypto;
mod direct_admission;
mod driver;
mod fea_admission;
mod fea_control;
pub(crate) mod exact_meshing;
pub(crate) mod exact_structural_study;
pub(crate) mod exact_structural_sweep;
mod family_study;
mod gang;
pub(crate) mod input_materialization;
mod object_transfer;
mod pool;
mod pool_allocation;
mod pool_checkpoint;
pub(crate) mod pool_execution;
mod pool_objects;
mod pool_progress;
mod pool_reconcile;
mod pool_resources;
pub(crate) mod pool_tasks;
mod pool_values;
mod pool_wait;
mod program_staging;
mod protocol;
mod protocol_schema;
mod quic_channel;
mod recovery;
mod relay_channel;
mod result_objects;
mod route;
pub(crate) mod structural_study;
pub(crate) mod value_transfer;
mod worker_bundle;
mod worker_attempt_journal;
pub(crate) mod runtime_settings;
mod worker_entry;
mod worker_env;
mod worker_execution;
mod worker_graph_stage;
mod worker_protocol;
mod worker_server;

pub use channel::{
    AuthenticatedPhysicalRetirement, CheckpointedAttempt, RemoteAttempt, RemoteAttemptProgram,
    RemoteBundleReceipt, RemoteGraphProgramReceipt, RemoteObjectReceipt,
    RemoteStagedGraphReference, RemoteValueReceipt, RemoteWorkerChannel,
};
#[cfg(test)]
pub(crate) use driver::execute_driver_analysis;
pub use driver::run_remote_driver_from_env;
pub use gang::{RemoteSpmdGangCompletion, RemoteSpmdGangProgram};
pub use pool::{
    RemoteGraphTransfer, RemotePoolCheckpoint, RemotePoolDriver, RemotePoolRestoreAuthority,
};
pub use pool_progress::RemoteTaskCompletion;
pub use quic_channel::{QuicRemoteWorkerChannel, RemoteWorkerChannelConfig};
pub use relay_channel::RelayRemoteWorkerChannel;
pub use worker_entry::{
    run_remote_meshing_worker_quic, run_remote_worker_quic, run_remote_worker_quic_ready_v1,
    run_remote_worker_relay, RemoteMeshingWorkerQuicRequest, RemoteWorkerQuicRequest,
    RemoteWorkerRelayRequest,
};
pub use worker_env::run_remote_worker_from_env;

#[cfg(all(test, feature = "occt-native"))]
mod driver_replacement_tests;
#[cfg(test)]
mod pool_tests;
#[cfg(test)]
mod tests;
