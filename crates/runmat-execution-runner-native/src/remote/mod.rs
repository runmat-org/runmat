mod bundle_cache;
mod channel;
mod config;
mod crypto;
mod driver;
pub(crate) mod exact_meshing;
pub(crate) mod exact_structural_study;
pub(crate) mod exact_structural_sweep;
mod object_transfer;
mod pool;
mod pool_allocation;
pub(crate) mod pool_execution;
mod pool_objects;
mod pool_progress;
mod pool_reconcile;
mod pool_resources;
mod pool_tasks;
mod pool_values;
mod pool_wait;
mod protocol;
mod quic_channel;
mod relay_channel;
mod result_objects;
mod route;
mod value_transfer;
mod worker_bundle;
mod worker_entry;
mod worker_env;
mod worker_execution;
mod worker_protocol;
mod worker_server;

pub use channel::{
    RemoteAttempt, RemoteBundleReceipt, RemoteObjectReceipt, RemoteValueReceipt,
    RemoteWorkerChannel,
};
pub use driver::run_remote_driver_from_env;
pub use pool::RemotePoolDriver;
pub use pool_progress::RemoteTaskCompletion;
pub use quic_channel::{QuicRemoteWorkerChannel, RemoteWorkerChannelConfig};
pub use relay_channel::RelayRemoteWorkerChannel;
pub use worker_entry::{
    run_remote_meshing_worker_quic, run_remote_worker_quic, run_remote_worker_relay,
    RemoteMeshingWorkerQuicRequest, RemoteWorkerQuicRequest, RemoteWorkerRelayRequest,
};
pub use worker_env::run_remote_worker_from_env;

#[cfg(test)]
mod pool_tests;
#[cfg(test)]
mod tests;
