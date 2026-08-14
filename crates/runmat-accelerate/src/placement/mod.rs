//! Executor-neutral placement observation and policy scaffolding.

mod event;
mod fusion;
mod provider_contract;
mod recorder;

#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::wgpu_capability_snapshot;
pub(crate) use provider_contract::{
    dense_feasibility, dense_output_representation, in_process_capability_snapshot,
    provider_rejection_token, tensor_representation,
};

pub use fusion::fusion_operation_token;
pub(crate) use fusion::FusionPlacementObserver;

pub use event::{
    PlacementAttribute, PlacementCorrelationId, PlacementEvent, PlacementEventKind,
    PlacementReport, PlacementTrace, PlacementVariant, PLACEMENT_REPORT_SCHEMA_VERSION,
};
pub use recorder::{
    begin_trace, complete_trace, record_event, report, reset, PlacementTraceContext,
};
