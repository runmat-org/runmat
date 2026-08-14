//! Executor-neutral placement observation and policy scaffolding.

mod event;
mod provider_contract;
mod recorder;

#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::wgpu_capability_snapshot;
pub(crate) use provider_contract::{dense_feasibility, in_process_capability_snapshot};

pub use event::{
    PlacementAttribute, PlacementCorrelationId, PlacementEvent, PlacementEventKind,
    PlacementReport, PlacementTrace, PlacementVariant, PLACEMENT_REPORT_SCHEMA_VERSION,
};
pub use recorder::{
    begin_trace, complete_trace, record_event, report, reset, PlacementTraceContext,
};
