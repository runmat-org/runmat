//! Executor-neutral placement observation and policy scaffolding.

mod completion;
mod event;
pub(crate) use completion::publish_provider_feedback_vm_full_call;
pub(crate) use completion::{publish_provider_feedback, publish_provider_feedback_full_call};
mod fusion;
mod local;
#[cfg(test)]
pub(crate) use local::LocalPlacementFeedbackAuthority;
mod partition;
mod planner;
mod provider_contract;
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::floating_conversion::build_for_input as floating_cast_work;
#[cfg(feature = "wgpu")]
mod provider_cost;
mod recorder;
mod residency;
mod revision;
mod session;
mod typed_reduction_inputs;
pub(crate) use provider_contract::matmul::scalar_work_units as matmul_scalar_work_units;

#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::convolution::observation_capability as convolution_observation_capability;
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::extrema::observation_capability as extrema_observation_capability;
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::floating_conversion::{
    build_for_input as floating_cast_build_for_input,
    observation_capability as floating_cast_observation_capability,
    physical_work as floating_cast_physical_work,
    unresolved_adaptation as floating_cast_unresolved_adaptation,
};
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::fspecial::observation_capability as fspecial_observation_capability;
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::fusion_elementwise::observation_capability as provider_observation_capability;
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::fusion_elementwise::physical_work as fusion_physical_work;
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::fusion_reduction::{
    observation_capability as typed_reduction_observation_capability,
    physical_work as typed_reduction_physical_work,
};
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::imfilter::{
    observation_capability as imfilter_observation_capability,
    physical_evidence as imfilter_physical_evidence, shader_disabled as imfilter_shader_disabled,
};
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::sum::observation_capability as sum_observation_capability;
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::sum::{physical_work as sum_physical_work, SumKernelRoute};
pub(crate) use provider_contract::{
    dense_output_representation, in_process_capability_snapshot, in_process_feasibility,
    provider_rejection_token, tensor_representation,
};
#[cfg(feature = "wgpu")]
pub(crate) use provider_contract::{wgpu_capability_snapshot, wgpu_feasibility};
#[cfg(feature = "wgpu")]
pub(crate) use provider_cost::wgpu_cost_estimate_with_geometry;

pub use fusion::fusion_operation_token;
pub(crate) use fusion::{FusionLeafIndexWork, FusionPlacementObserver, FusionScalarEvidenceWork};
pub(crate) use local::{
    plan_local_multi, plan_local_multi_bounded, plan_local_vm_cpu_with_provider, LocalCpuAdmission,
    LocalCpuCandidate, LocalMultiPlacementRequest, LocalPhysicalCandidate, LocalPlacementFeedback,
    LocalPlacementOutcome, LocalPlacementRequest, LocalProviderAdmission,
    LocalPublicationSemantics, LocalSourceDecline, PreparedPhysicalAuthority,
    TotalCandidateObservationContract,
};
pub use session::{
    ExplorationMode, ManagedPlacementCapacity, PlacementFeedbackSeries, PlacementPlanError,
    PlacementProfileSnapshot, PlacementSession, PlacementSessionConfig,
    PLACEMENT_PROFILE_SCHEMA_VERSION,
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
pub(crate) use typed_reduction_inputs::{
    typed_reduction_input_resources, TypedReductionInputResources,
};
