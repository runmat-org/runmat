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
pub mod parity;
pub mod physics;
pub mod pipeline;
pub mod planning;
pub mod post;
pub mod progress;
pub mod results;
pub mod solve;
pub mod topology;

pub use admission::AnalysisRunIdentity;
pub use contracts::*;
pub use pipeline::electromagnetic::{run_electromagnetic, run_electromagnetic_with_options};
pub use pipeline::electromagnetic_frequency::{
    run_electromagnetic_frequency_response, ElectromagneticFrequencyPlan,
    ElectromagneticFrequencyPlanEntry, ElectromagneticFrequencyPlanEntryIdentity,
    ElectromagneticFrequencyPlanIdentity, ElectromagneticFrequencySummary,
    FeaElectromagneticFrequencyResponse,
};
pub use pipeline::linear_static::{
    run_linear_static, run_linear_static_with_options,
    run_linear_static_with_prepared_constraint_plan,
};
pub use pipeline::modal::{run_modal, run_modal_with_options};
pub use pipeline::modal_basis::{
    ModalBasisArtifact, ModalBasisIdentity, ModalBasisMode, ModalDirectionalMetric,
};
pub use pipeline::modal_governance::{
    ModalEigenspaceEvidence, ModalTruncationEvidence, ModalTruncationPolicy,
};
pub use pipeline::nonlinear::{run_nonlinear, run_nonlinear_with_options};
pub use pipeline::thermal::run_thermal_with_options;
pub use pipeline::transient::{run_transient, run_transient_with_options};
pub use progress::{
    check_cancelled as check_fea_cancelled, emit_phase as emit_fea_progress_phase,
    emit_progress as emit_fea_progress, is_cancelled as is_fea_cancelled,
    replace_fea_progress_context, FeaCancellationToken, FeaProgressContext,
    FeaProgressContextGuard, FeaProgressEvent, FeaProgressHandler, FeaProgressPhase,
    FeaProgressStatus,
};

#[cfg(test)]
mod tests;
