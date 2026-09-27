//! Executor-neutral placement observation and policy scaffolding.

mod event;
mod fusion;
mod local;
mod partition;
mod planner;
mod provider_contract;
#[cfg(feature = "wgpu")]
mod provider_cost;
mod recorder;
mod residency;
mod session;

pub(crate) use provider_contract::{
    dense_output_representation, in_process_capability_snapshot, in_process_feasibility,
    provider_rejection_token, tensor_representation,
};
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::{wgpu_capability_snapshot, wgpu_feasibility};
#[cfg(feature = "wgpu")]
pub(crate) use provider_cost::wgpu_cost_estimate;

pub use fusion::fusion_operation_token;
pub(crate) use fusion::FusionPlacementObserver;
pub(crate) use local::{
    plan_local, LocalPlacementFeedback, LocalPlacementOutcome, LocalPlacementRequest,
};
pub use session::{
    ExplorationMode, PlacementFeedbackSeries, PlacementPlanError, PlacementProfileSnapshot,
    PlacementSession, PlacementSessionConfig, PLACEMENT_PROFILE_SCHEMA_VERSION,
};

pub use event::{
    PlacementAttribute, PlacementCorrelationId, PlacementEvent, PlacementEventKind,
    PlacementReport, PlacementTrace, PlacementVariant, PLACEMENT_REPORT_SCHEMA_VERSION,
};
pub(crate) use planner::{select_candidate, PlacementPolicy};
#[cfg(test)]
pub(crate) use recorder::ScopedRecorder;
pub use recorder::{
    begin_trace, complete_trace, record_event, report, reset, PlacementTraceContext,
};
pub(crate) use residency::summarize_values;
pub use residency::{CoherencyRecord, CoherencyState};
