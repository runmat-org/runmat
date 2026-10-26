use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, ProcedureStep, ProcedureStepId};
use runmat_analysis_fea::execution::StructuralStudyWorkloads;
use runmat_analysis_fea::results::observation::ObservationExecutionPolicy;
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_meshing_core::SolverMeshArtifact;

use crate::operations::OperationContext;

use super::{
    AnalysisExecutionStep, AnalysisStudySpec, PreparedStructuralLinearRun,
    StructuralStudyNumericalResult,
};

pub(in crate::analysis) mod lifecycle;
pub(in crate::analysis) mod publication;

pub(in crate::analysis) use lifecycle::prepare_structural_study_intent_completion;
pub use lifecycle::{
    analysis_complete_structural_study_intent_op, analysis_prepare_structural_study_intent_op,
    structural_study_execution_failure,
};
pub(super) use lifecycle::{
    complete_buckling_study, complete_frequency_study, complete_modal_study,
    complete_modal_transient_study, complete_prestressed_frequency_study,
    complete_prestressed_modal_study, complete_random_response_study,
    complete_response_spectrum_modal_options, complete_response_spectrum_study,
    prepare_structural_buckling_template, prepare_structural_frequency_workload,
    prepare_structural_modal_transient_template, prepare_structural_modal_workload,
    prepare_structural_prestressed_frequency_template,
    prepare_structural_prestressed_modal_template, prepare_structural_random_response_template,
    prepare_structural_response_spectrum_template,
};
pub(super) use publication::{
    complete_study_run, persist_content_addressed_study_evidence, study_completion_from_prepared,
    study_evidence_root, CompletedStudyStep, CompletedStudyStepRun, StudyEvidenceReadError,
};
#[cfg(test)]
pub(super) use publication::{persist_immutable_study_evidence, run_evidence_stage};
pub(super) use publication::{standalone_completion, standalone_thermal_completion};

pub const STRUCTURAL_STUDY_INTENT_MAX_BYTES: usize = 64 * 1024 * 1024;

use super::AnalysisStudyCompletionIdentity;

#[derive(Debug, Clone)]
pub(super) struct PreparedStudySequence {
    pub(super) spec: AnalysisStudySpec,
    pub(super) context: OperationContext,
    pub(super) study_fingerprint: String,
    pub(super) model: AnalysisModel,
    pub(super) steps: Vec<PreparedStudyStep>,
    pub(super) operation_sequence: Vec<String>,
    pub(super) solver_mesh_artifact_path: Option<String>,
    pub(super) meshing_evidence_artifact_path: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedStudyStep {
    pub(super) authored_execution: AnalysisExecutionStep,
    pub(super) procedure_step: ProcedureStep,
}

/// A structural study whose authored model, topology targets, mesh, and numerical request have
/// been resolved. It is safe to retain this value while its path-free worker payload is placed.
#[derive(Debug, Clone)]
pub struct PreparedStructuralStudy {
    pub(super) study: PreparedStudySequence,
    pub(super) execution: PreparedStructuralStudyExecution,
    pub(super) solver_mesh: SolverMeshArtifact,
}

#[derive(Debug, Clone)]
pub(super) enum PreparedStructuralStudyExecution {
    LinearSequence {
        linears: BTreeMap<ProcedureStepId, PreparedStructuralLinearRun>,
    },
    StructuralFrequency {
        workload: Box<runmat_analysis_fea::execution::StructuralFrequencyWorkload>,
    },
    StructuralModal {
        workload: Box<runmat_analysis_fea::execution::StructuralModalWorkload>,
    },
    StructuralModalTransient {
        workloads:
            BTreeMap<ProcedureStepId, runmat_analysis_fea::execution::StructuralStepWorkload>,
    },
    StructuralModalResponseSpectrum {
        workloads:
            BTreeMap<ProcedureStepId, runmat_analysis_fea::execution::StructuralStepWorkload>,
    },
    StructuralRandomResponse {
        workloads:
            BTreeMap<ProcedureStepId, runmat_analysis_fea::execution::StructuralStepWorkload>,
    },
    StructuralPrestressedModal {
        workloads:
            BTreeMap<ProcedureStepId, runmat_analysis_fea::execution::StructuralStepWorkload>,
    },
    StructuralPrestressedFrequency {
        workloads:
            BTreeMap<ProcedureStepId, runmat_analysis_fea::execution::StructuralStepWorkload>,
    },
    StructuralBuckling {
        workloads:
            BTreeMap<ProcedureStepId, runmat_analysis_fea::execution::StructuralStepWorkload>,
    },
}

/// Validated structural study semantics before geometry is meshed.
///
/// This value deliberately contains neither a geometry source path nor a solver-mesh artifact
/// path. A remote study coordinator can therefore checkpoint and restore it independently of the
/// machine that authored the study. Geometry source bytes and the eventual canonical solver mesh
/// remain separate, content-addressed execution inputs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PreparedStructuralStudyIntent {
    pub(super) study_id: String,
    pub(super) geometry_source_digest: runmat_execution::Digest,
    pub(super) geometry_unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence,
    pub(super) meshing_request: Option<runmat_meshing_core::MeshingRequest>,
    pub(super) meshing_contacts: Vec<crate::meshing::MeshingContactDefinition>,
    pub(super) context: OperationContext,
    pub(super) authored_study_digest: runmat_execution::Digest,
    pub(super) intent_digest: runmat_execution::Digest,
    pub(super) model: AnalysisModel,
    pub(super) steps: Vec<PreparedStructuralStudyIntentStep>,
    pub(super) workloads: StructuralStudyWorkloads,
    pub(super) operation_sequence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedStructuralStudyIntentStep {
    pub(super) authored_execution: AnalysisExecutionStep,
    pub(super) procedure_step: ProcedureStep,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralStudyExecutionFailure {
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

pub type StructuralStudyExecutionOutcome =
    Result<StructuralStudyNumericalResult, StructuralStudyExecutionFailure>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralStudyIntentContinuation {
    schema_version: u16,
    study_id: String,
    geometry_source_digest: runmat_execution::Digest,
    geometry_unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence,
    meshing_request: Option<runmat_meshing_core::MeshingRequest>,
    meshing_contacts: Vec<crate::meshing::MeshingContactDefinition>,
    context: OperationContext,
    authored_study_digest: runmat_execution::Digest,
    intent_digest: runmat_execution::Digest,
    model: AnalysisModel,
    steps: Vec<PreparedStructuralStudyIntentStep>,
    workloads: StructuralStudyWorkloads,
    operation_sequence: Vec<String>,
}

mod intent;

mod alignment;
use alignment::linear_workload_matches_prepared_execution;

pub(super) fn resolve_intent_meshing_request(
    settings: Option<runmat_meshing_core::MeshingRequestSettings>,
) -> Result<Option<runmat_meshing_core::MeshingRequest>, String> {
    let Some(settings) = settings else {
        return Ok(None);
    };
    let import_options = runmat_geometry_io::ExactCadImportOptions::default();
    import_options
        .validate()
        .map_err(|error| error.to_string())?;
    let tolerance = runmat_geometry_core::GeometryTolerancePolicy {
        source_tolerance_m: runmat_types::physical::Length::new(0.0),
        absolute_floor_m: runmat_types::physical::Length::new(
            import_options
                .analysis
                .absolute_tolerance_floor_m
                .canonical_value(),
        ),
        model_relative_term: import_options.analysis.model_relative_tolerance,
        requested_deviation_m: runmat_types::physical::Length::new(
            import_options
                .analysis
                .requested_deviation_m
                .canonical_value(),
        ),
        maximum_healing_displacement_m: runmat_types::physical::Length::new(
            import_options
                .analysis
                .maximum_healing_displacement_m
                .canonical_value(),
        ),
    };
    runmat_meshing_core::resolve_meshing_request(tolerance, settings)
        .map(Some)
        .map_err(|error| error.to_string())
}

impl PreparedStructuralStudy {
    pub fn workloads(&self) -> StructuralStudyWorkloads {
        match &self.execution {
            PreparedStructuralStudyExecution::LinearSequence { linears } => {
                StructuralStudyWorkloads::LinearSequence {
                    workloads: linears
                        .iter()
                        .map(|(step_id, linear)| (step_id.clone(), linear.workload.clone()))
                        .collect(),
                }
            }
            PreparedStructuralStudyExecution::StructuralFrequency { workload } => {
                StructuralStudyWorkloads::StructuralFrequency {
                    workload: workload.clone(),
                }
            }
            PreparedStructuralStudyExecution::StructuralModal { workload } => {
                StructuralStudyWorkloads::StructuralModal {
                    workload: workload.clone(),
                }
            }
            PreparedStructuralStudyExecution::StructuralModalTransient { workloads } => {
                StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
            }
            PreparedStructuralStudyExecution::StructuralModalResponseSpectrum { workloads } => {
                StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
            }
            PreparedStructuralStudyExecution::StructuralRandomResponse { workloads } => {
                StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
            }
            PreparedStructuralStudyExecution::StructuralPrestressedModal { workloads } => {
                StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
            }
            PreparedStructuralStudyExecution::StructuralPrestressedFrequency { workloads } => {
                StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
            }
            PreparedStructuralStudyExecution::StructuralBuckling { workloads } => {
                StructuralStudyWorkloads::HeterogeneousSequence {
                    workloads: workloads.clone(),
                }
            }
        }
    }

    pub fn solver_mesh(&self) -> &SolverMeshArtifact {
        &self.solver_mesh
    }
}
