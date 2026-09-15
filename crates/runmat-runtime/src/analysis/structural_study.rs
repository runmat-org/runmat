use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, ProcedureStep, ProcedureStepId};
use runmat_analysis_fea::execution::StructuralStudyWorkloads;
use runmat_analysis_fea::results::observation::ObservationExecutionPolicy;
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_meshing_core::SolverMeshArtifact;

use crate::operations::OperationContext;

use super::{
    AnalysisExecutionStep, AnalysisModelTemplate, AnalysisStudySpec, PreparedStructuralLinearRun,
    StructuralStudyNumericalResult,
};

mod lifecycle;
mod publication;

pub use lifecycle::{
    analysis_complete_structural_study_intent_op, analysis_prepare_structural_study_intent_op,
    structural_study_execution_failure,
};
pub(super) use lifecycle::{
    complete_frequency_study, complete_modal_study, complete_modal_transient_study,
    prepare_structural_frequency_workload, prepare_structural_modal_transient_template,
    prepare_structural_modal_workload,
};
pub(super) use publication::{
    complete_study_run, persist_content_addressed_study_evidence, study_completion_from_prepared,
    study_evidence_root, CompletedStudyStep, CompletedStudyStepRun,
};
#[cfg(test)]
pub(super) use publication::{persist_immutable_study_evidence, run_evidence_stage};

pub const STRUCTURAL_STUDY_INTENT_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Server allocation identity used to fence local materialization of a remote study result.
///
/// The server remains content-blind: this value names an admitted execution and is combined with
/// the authored study digest before it becomes a deterministic local run identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralStudyCompletionIdentity(String);

impl StructuralStudyCompletionIdentity {
    pub fn new(server_run_id: impl Into<String>) -> Result<Self, String> {
        let server_run_id = server_run_id.into();
        if server_run_id.is_empty()
            || server_run_id.len() > 512
            || !server_run_id.is_ascii()
            || server_run_id.trim() != server_run_id
            || server_run_id.chars().any(char::is_control)
        {
            return Err("remote structural study completion identity is invalid".into());
        }
        Ok(Self(server_run_id))
    }

    pub(super) fn local_run_id(
        &self,
        authored_study_digest: runmat_execution::Digest,
        step_id: &ProcedureStepId,
    ) -> String {
        let digest = runmat_execution::Digest::sha256(
            [
                b"runmat-remote-structural-study-completion/v1\0".as_slice(),
                self.0.as_bytes(),
                authored_study_digest.bytes(),
                step_id.as_str().as_bytes(),
            ]
            .concat(),
        );
        format!(
            "run_remote_{}",
            digest.to_string().trim_start_matches("sha256:")
        )
    }
}

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
}

/// Validated structural study semantics before geometry is meshed.
///
/// This value deliberately contains neither a geometry source path nor a solver-mesh artifact
/// path. A remote study coordinator can therefore checkpoint and restore it independently of the
/// machine that authored the study. Geometry source bytes and the eventual canonical solver mesh
/// remain separate, content-addressed execution inputs.
#[derive(Debug, Clone)]
pub struct PreparedStructuralStudyIntent {
    pub(super) study_id: String,
    pub(super) model_template: AnalysisModelTemplate,
    pub(super) geometry_source_digest: runmat_execution::Digest,
    pub(super) geometry_units: runmat_geometry_core::UnitSystem,
    pub(super) meshing_settings: Option<runmat_meshing_core::MeshingRequestSettings>,
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
    model_template: AnalysisModelTemplate,
    geometry_source_digest: runmat_execution::Digest,
    geometry_units: runmat_geometry_core::UnitSystem,
    meshing_settings: Option<runmat_meshing_core::MeshingRequestSettings>,
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

impl PreparedStructuralStudyIntent {
    pub fn authored_study_digest(&self) -> runmat_execution::Digest {
        self.authored_study_digest
    }

    pub fn study_id(&self) -> &str {
        &self.study_id
    }

    pub fn operation_context(&self) -> &OperationContext {
        &self.context
    }

    pub fn model_template(&self) -> AnalysisModelTemplate {
        self.model_template
    }

    pub fn model(&self) -> &AnalysisModel {
        &self.model
    }

    pub fn workloads(&self) -> StructuralStudyWorkloads {
        self.workloads.clone()
    }

    pub fn operation_sequence(&self) -> &[String] {
        &self.operation_sequence
    }

    pub fn meshing_settings(&self) -> Option<&runmat_meshing_core::MeshingRequestSettings> {
        self.meshing_settings.as_ref()
    }

    pub fn meshing_request(&self) -> Option<&runmat_meshing_core::MeshingRequest> {
        self.meshing_request.as_ref()
    }

    pub fn geometry_units(&self) -> runmat_geometry_core::UnitSystem {
        self.geometry_units
    }

    pub fn meshing_contacts(&self) -> &[crate::meshing::MeshingContactDefinition] {
        &self.meshing_contacts
    }

    pub fn geometry_source_digest(&self) -> runmat_execution::Digest {
        self.geometry_source_digest
    }

    pub fn continuation_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(&StructuralStudyIntentContinuation {
            schema_version: 1,
            study_id: self.study_id.clone(),
            model_template: self.model_template,
            geometry_source_digest: self.geometry_source_digest,
            geometry_units: self.geometry_units,
            meshing_settings: self.meshing_settings.clone(),
            meshing_request: self.meshing_request.clone(),
            meshing_contacts: self.meshing_contacts.clone(),
            context: self.context.clone(),
            authored_study_digest: self.authored_study_digest,
            intent_digest: self.intent_digest,
            model: self.model.clone(),
            steps: self.steps.clone(),
            workloads: self.workloads.clone(),
            operation_sequence: self.operation_sequence.clone(),
        })
        .map_err(|error| error.to_string())?;
        if bytes.len() > STRUCTURAL_STUDY_INTENT_MAX_BYTES {
            return Err("structural study intent continuation exceeds its bound".into());
        }
        Ok(bytes)
    }

    pub fn from_continuation_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > STRUCTURAL_STUDY_INTENT_MAX_BYTES {
            return Err("structural study intent continuation exceeds its bound".into());
        }
        let continuation: StructuralStudyIntentContinuation =
            serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if continuation.schema_version != 1 {
            return Err("unsupported structural study intent continuation schema".into());
        }
        let intent = Self {
            study_id: continuation.study_id,
            model_template: continuation.model_template,
            geometry_source_digest: continuation.geometry_source_digest,
            geometry_units: continuation.geometry_units,
            meshing_settings: continuation.meshing_settings,
            meshing_request: continuation.meshing_request,
            meshing_contacts: continuation.meshing_contacts,
            context: continuation.context,
            authored_study_digest: continuation.authored_study_digest,
            intent_digest: continuation.intent_digest,
            model: continuation.model,
            steps: continuation.steps,
            workloads: continuation.workloads,
            operation_sequence: continuation.operation_sequence,
        };
        intent.validate()?;
        Ok(intent)
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        if self.study_id.trim().is_empty()
            || self.geometry_source_digest.bytes() == &[0_u8; 32]
            || self.authored_study_digest.bytes() == &[0_u8; 32]
            || self.meshing_request
                != resolve_intent_meshing_request(
                    self.geometry_units,
                    self.meshing_settings.clone(),
                )?
            || self.meshing_contacts
                != super::meshing::meshing_contact_definitions_for_model(&self.model)
                    .collect::<Vec<_>>()
        {
            return Err("structural study intent semantics are inconsistent".into());
        }
        if self.steps.len() != self.model.sequence.steps.len() {
            return Err("structural study intent does not cover every sequence step".into());
        }
        self.workloads
            .validate()
            .map_err(|error| error.to_string())?;
        if self
            .workloads
            .authored_model()
            .map_err(|error| error.to_string())?
            != &self.model
        {
            return Err("structural study workload model differs from its intent model".into());
        }
        self.validate_workload_alignment()?;
        for (index, step) in self.steps.iter().enumerate() {
            let authored = self
                .model
                .sequence
                .steps
                .get(index)
                .ok_or_else(|| "structural study intent step order is invalid".to_string())?;
            if step.procedure_step != *authored
                || step.authored_execution.step_id != authored.step_id
            {
                return Err("structural study intent step semantics are inconsistent".into());
            }
        }
        let semantic_digest = self.semantic_digest()?;
        if self.intent_digest != semantic_digest {
            return Err("structural study intent digest differs from its identity".into());
        }
        Ok(())
    }

    fn validate_workload_alignment(&self) -> Result<(), String> {
        match &self.workloads {
            StructuralStudyWorkloads::HeterogeneousSequence { workloads } => {
                if workloads.len() != self.steps.len() {
                    return Err(
                        "heterogeneous structural workloads do not cover the prepared intent steps"
                            .into(),
                    );
                }
                for step in &self.steps {
                    let workload =
                        workloads.get(&step.procedure_step.step_id).ok_or_else(|| {
                            "heterogeneous structural workload is missing its step".to_string()
                        })?;
                    match (workload, &step.authored_execution.solver_options) {
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::ModalExtraction {
                                workload,
                            },
                            super::AnalysisSolverOptions::StructuralModes(options),
                        ) if workload.step_id() == &step.procedure_step.step_id
                            && workload.backend() == step.authored_execution.backend
                            && workload.formulation_requirement()
                                == step.authored_execution.requirement
                            && workload.observation_execution()
                                == ObservationExecutionPolicy::scalable()
                            && workload.truncation() == options.truncation
                            && workload.convergence() == options.convergence
                            && options.solver_mesh_artifact_path.is_none() => {}
                        (
                            runmat_analysis_fea::execution::StructuralStepWorkload::ModalTransient {
                                template,
                                block_plan,
                            },
                            super::AnalysisSolverOptions::StructuralTransient(options),
                        ) if template.step_id == step.procedure_step.step_id
                            && template.backend == step.authored_execution.backend
                            && template.formulation_requirement
                                == step.authored_execution.requirement
                            && template.time_step_policy.interval_count()
                                == options.time_step_policy.interval_count()
                            && template.residual_target == options.residual_target
                            && block_plan.maximum_intervals_per_block()
                                == options.maximum_intervals_per_block
                            && options.solver_mesh_artifact_path.is_none() => {}
                        _ => {
                            return Err(
                                "heterogeneous structural workload differs from its prepared execution"
                                    .into(),
                            )
                        }
                    }
                }
            }
            StructuralStudyWorkloads::LinearSequence { workloads } => {
                if workloads.len() != self.steps.len() {
                    return Err(
                        "structural linear workloads do not cover the prepared intent steps".into(),
                    );
                }
                for step in &self.steps {
                    let super::AnalysisSolverOptions::StructuralLinear(options) =
                        &step.authored_execution.solver_options
                    else {
                        return Err(
                            "structural linear workload has non-linear authored options".into()
                        );
                    };
                    let workload =
                        workloads.get(&step.procedure_step.step_id).ok_or_else(|| {
                            "structural linear workload is missing its step".to_string()
                        })?;
                    let expected_preconditioner = match options.preconditioner_mode {
                        super::PreconditionerMode::Auto | super::PreconditionerMode::Jacobi => {
                            SpdPreconditionerKind::Jacobi
                        }
                        super::PreconditionerMode::Ilu => SpdPreconditionerKind::Ilu0,
                        super::PreconditionerMode::Amg => SpdPreconditionerKind::Jacobi,
                    };
                    let backend_matches = match step.authored_execution.backend {
                        runmat_analysis_fea::ComputeBackend::Cpu => {
                            workload.algebra_backend == LinearAlgebraBackendKind::CpuReference
                        }
                        runmat_analysis_fea::ComputeBackend::Gpu => matches!(
                            workload.algebra_backend,
                            LinearAlgebraBackendKind::CpuReference
                                | LinearAlgebraBackendKind::Accelerator
                        ),
                    };
                    if workload.backend != step.authored_execution.backend
                        || workload.formulation_requirement != step.authored_execution.requirement
                        || workload.preconditioner != expected_preconditioner
                        || !backend_matches
                        || workload.observation_execution != ObservationExecutionPolicy::scalable()
                        || !workload.require_solver_mesh_for_solid
                        || workload.initial_guess_transition.is_some()
                        || options.solver_mesh_artifact_path.is_some()
                    {
                        return Err(
                            "structural linear workload differs from its prepared execution".into(),
                        );
                    }
                }
            }
            StructuralStudyWorkloads::StructuralTransient {
                workload,
                block_plan,
            } => {
                let [step] = self.steps.as_slice() else {
                    return Err(
                        "direct structural transient requires exactly one prepared step".into(),
                    );
                };
                let super::AnalysisSolverOptions::StructuralTransient(options) =
                    &step.authored_execution.solver_options
                else {
                    return Err(
                        "direct structural transient workload has non-transient authored options"
                            .into(),
                    );
                };
                let expected_time_step_policy = match options.time_step_policy {
                    super::AnalysisStructuralTimeStepPolicy::Fixed { interval_count } => {
                        runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy::Fixed {
                            interval_count,
                        }
                    }
                };
                if workload.step_id != step.procedure_step.step_id
                    || workload.backend != step.authored_execution.backend
                    || workload.formulation_requirement != step.authored_execution.requirement
                    || workload.solver
                        != runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct
                    || workload.time_step_policy != expected_time_step_policy
                    || workload.residual_target != options.residual_target
                    || block_plan.maximum_intervals_per_block()
                        != options.maximum_intervals_per_block
                    || options.solver_mesh_artifact_path.is_some()
                {
                    return Err(
                        "direct structural transient workload differs from its prepared execution"
                            .into(),
                    );
                }
            }
            StructuralStudyWorkloads::StructuralFrequency { workload } => {
                let [step] = self.steps.as_slice() else {
                    return Err(
                        "direct structural frequency execution requires exactly one prepared step"
                            .into(),
                    );
                };
                let super::AnalysisSolverOptions::StructuralFrequency(options) =
                    &step.authored_execution.solver_options
                else {
                    return Err(
                        "structural frequency workload has non-frequency authored options".into(),
                    );
                };
                let runmat_analysis_fea::execution::StructuralFrequencySolverPlan::Direct {
                    policy,
                } = workload.solver()
                else {
                    return Err(
                        "standalone structural frequency execution requires a direct solver".into(),
                    );
                };
                if workload.step_id() != &step.procedure_step.step_id
                    || workload.backend() != step.authored_execution.backend
                    || workload.formulation_requirement() != step.authored_execution.requirement
                    || workload.observation_execution() != ObservationExecutionPolicy::scalable()
                    || *policy != options.direct_solver
                    || options.solver_mesh_artifact_path.is_some()
                {
                    return Err(
                        "structural frequency workload differs from its prepared execution".into(),
                    );
                }
            }
            StructuralStudyWorkloads::StructuralModal { workload } => {
                let [step] = self.steps.as_slice() else {
                    return Err(
                        "structural modal extraction requires exactly one prepared step".into(),
                    );
                };
                let super::AnalysisSolverOptions::StructuralModes(options) =
                    &step.authored_execution.solver_options
                else {
                    return Err("structural modal workload has non-modal authored options".into());
                };
                if workload.step_id() != &step.procedure_step.step_id
                    || workload.backend() != step.authored_execution.backend
                    || workload.formulation_requirement() != step.authored_execution.requirement
                    || workload.observation_execution() != ObservationExecutionPolicy::scalable()
                    || workload.truncation() != options.truncation
                    || workload.convergence() != options.convergence
                    || options.solver_mesh_artifact_path.is_some()
                {
                    return Err(
                        "structural modal workload differs from its prepared execution".into(),
                    );
                }
            }
        }
        Ok(())
    }

    pub(super) fn semantic_digest(&self) -> Result<runmat_execution::Digest, String> {
        let bytes = serde_json::to_vec(&(
            &self.study_id,
            self.model_template,
            self.geometry_source_digest,
            self.geometry_units,
            &self.meshing_settings,
            &self.meshing_request,
            &self.meshing_contacts,
            &self.model,
            &self.steps,
            &self.workloads,
            &self.operation_sequence,
        ))
        .map_err(|error| error.to_string())?;
        Ok(runmat_execution::Digest::sha256(&bytes))
    }
}

pub(super) fn resolve_intent_meshing_request(
    source_units: runmat_geometry_core::UnitSystem,
    settings: Option<runmat_meshing_core::MeshingRequestSettings>,
) -> Result<Option<runmat_meshing_core::MeshingRequest>, String> {
    let Some(settings) = settings else {
        return Ok(None);
    };
    let import_options = runmat_geometry_io::ExactCadImportOptions {
        source_units,
        ..runmat_geometry_io::ExactCadImportOptions::default()
    };
    import_options
        .validate()
        .map_err(|error| error.to_string())?;
    let tolerance = runmat_geometry_core::GeometryTolerancePolicy {
        source_tolerance_m: 0.0,
        absolute_floor_m: import_options.analysis.absolute_tolerance_floor_m,
        model_relative_term: import_options.analysis.model_relative_tolerance,
        requested_deviation_m: import_options.analysis.requested_deviation_m,
        maximum_healing_displacement_m: import_options.analysis.maximum_healing_displacement_m,
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
        }
    }

    pub fn solver_mesh(&self) -> &SolverMeshArtifact {
        &self.solver_mesh
    }
}
