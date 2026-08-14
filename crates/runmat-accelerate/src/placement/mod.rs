//! Executor-neutral placement observation and policy scaffolding.

mod event;
mod fusion;
mod local;
mod planner;
mod provider_contract;
#[cfg(feature = "wgpu")]
mod provider_cost;
mod recorder;
mod residency;

#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::wgpu_capability_snapshot;
pub(crate) use provider_contract::{
    dense_feasibility, dense_output_representation, in_process_capability_snapshot,
    provider_rejection_token, tensor_representation,
};
#[cfg(feature = "wgpu")]
pub(crate) use provider_cost::wgpu_cost_estimate;

pub use fusion::fusion_operation_token;
pub(crate) use fusion::FusionPlacementObserver;
pub(crate) use local::{plan_local, LocalPlacementOutcome, LocalPlacementRequest};

pub use event::{
    PlacementAttribute, PlacementCorrelationId, PlacementEvent, PlacementEventKind,
    PlacementReport, PlacementTrace, PlacementVariant, PLACEMENT_REPORT_SCHEMA_VERSION,
};
pub(crate) use planner::{select_candidate, PlacementPolicy};
pub use recorder::{
    begin_trace, complete_trace, record_event, report, reset, PlacementTraceContext,
};
pub(crate) use residency::summarize_values;
pub use residency::{CoherencyRecord, CoherencyState};
