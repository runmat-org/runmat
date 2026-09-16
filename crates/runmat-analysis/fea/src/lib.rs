//! FEA pipeline scaffolding for assembly, solve orchestration, and diagnostics.

pub mod adaptation;
pub mod admission;
pub mod algebra;
pub mod assembly;
pub mod constraints;
pub mod contracts;
pub mod diagnostics;
pub mod excitation;
pub mod execution;
pub mod fixtures;
pub mod frequency;
pub mod parity;
pub mod physics;
pub mod pipeline;
pub mod planning;
pub mod post;
pub mod progress;
pub mod results;
pub mod signal_processing;
pub mod solve;
pub mod topology;

pub use admission::AnalysisRunIdentity;
pub use contracts::*;
pub use frequency::{
    FrequencyResponsePlan, FrequencyResponsePlanEntry, FrequencyResponsePlanEntryIdentity,
    FrequencyResponsePlanIdentity,
};
pub use pipeline::electromagnetic::{run_electromagnetic, run_electromagnetic_with_options};
pub use pipeline::electromagnetic_frequency::{
    run_electromagnetic_frequency_response, ElectromagneticFrequencySummary,
    FeaElectromagneticFrequencyResponse,
};
pub use pipeline::linear_static::{
    run_linear_static, run_linear_static_with_options,
    run_linear_static_with_prepared_constraint_plan,
    run_linear_static_with_prepared_constraint_plan_and_linearization, LinearStaticExecution,
};
pub use pipeline::modal::{
    execute_structural_modal_workload, execute_structural_prestressed_modal_workload,
    recover_completed_structural_modal_run, recover_completed_structural_prestressed_modal_run,
    run_modal, run_modal_with_options,
};
pub use pipeline::modal_basis::{
    ModalBasisArtifact, ModalBasisIdentity, ModalBasisMode, ModalDirectionalMetric,
};
pub use pipeline::modal_governance::{
    ModalConvergencePolicy, ModalEigenspaceEvidence, ModalTruncationEvidence, ModalTruncationPolicy,
};
pub use pipeline::modal_residual::ModalResidualVector;
pub use pipeline::nonlinear::{run_nonlinear, run_nonlinear_with_options};
pub use pipeline::structural_buckling::{
    execute_structural_buckling_workload, recover_completed_structural_buckling_run,
};
pub use pipeline::thermal::run_thermal_with_options;
pub use pipeline::transient::{run_transient, run_transient_with_options};
pub use progress::{
    check_cancelled as check_fea_cancelled, emit_phase as emit_fea_progress_phase,
    emit_progress as emit_fea_progress, is_cancelled as is_fea_cancelled,
    replace_fea_progress_context, FeaCancellationToken, FeaProgressContext,
    FeaProgressContextGuard, FeaProgressEvent, FeaProgressHandler, FeaProgressPhase,
    FeaProgressStatus,
};
pub use signal_processing::{construct_shock_response_spectrum, ShockResponseSpectrumRequest};

#[cfg(test)]
mod tests;
