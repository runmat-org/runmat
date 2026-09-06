use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_analysis_fea::execution::StructuralLinearWorkload;
use runmat_meshing_core::SolverMeshArtifact;

use crate::operations::OperationContext;

use super::{
    AnalysisExecutionStep, AnalysisModelTemplate, AnalysisRunOptions, AnalysisSolverOptions,
    AnalysisStudySpec, PreparedStructuralLinearRun,
};

mod lifecycle;
mod publication;

pub use lifecycle::{
    analysis_complete_structural_study_intent_op, analysis_prepare_structural_study_intent_op,
};
#[cfg(test)]
pub(super) use publication::run_evidence_stage;
pub(super) use publication::{
    complete_study_run, persist_immutable_study_evidence, study_completion_from_prepared,
    study_evidence_root,
};

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

    pub(super) fn local_run_id(&self, authored_study_digest: runmat_execution::Digest) -> String {
        let digest = runmat_execution::Digest::sha256(
            [
                b"runmat-remote-structural-study-completion/v1\0".as_slice(),
                self.0.as_bytes(),
                authored_study_digest.bytes(),
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
pub(super) struct PreparedStudyRun {
    pub(super) spec: AnalysisStudySpec,
    pub(super) context: OperationContext,
    pub(super) study_fingerprint: String,
    pub(super) model: AnalysisModel,
    pub(super) authored_execution: AnalysisExecutionStep,
    pub(super) procedure_step: ProcedureStep,
    pub(super) operation_sequence: Vec<String>,
    pub(super) solver_mesh_artifact_path: Option<String>,
    pub(super) meshing_evidence_artifact_path: Option<String>,
}

/// A structural study whose authored model, topology targets, mesh, and numerical request have
/// been resolved. It is safe to retain this value while its path-free worker payload is placed.
#[derive(Debug, Clone)]
pub struct PreparedStructuralStudy {
    pub(super) study: PreparedStudyRun,
    pub(super) linear: PreparedStructuralLinearRun,
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
    pub(super) authored_execution: AnalysisExecutionStep,
    pub(super) procedure_step: ProcedureStep,
    pub(super) operation_sequence: Vec<String>,
    pub(super) workload_template: StructuralLinearWorkload,
    pub(super) options: AnalysisRunOptions,
}

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
    authored_execution: AnalysisExecutionStep,
    procedure_step: ProcedureStep,
    operation_sequence: Vec<String>,
    workload_template: StructuralLinearWorkload,
    options: AnalysisRunOptions,
}

impl PreparedStructuralStudyIntent {
    pub fn authored_study_digest(&self) -> runmat_execution::Digest {
        self.authored_study_digest
    }

    pub fn study_id(&self) -> &str {
        &self.study_id
    }

    pub fn model_template(&self) -> AnalysisModelTemplate {
        self.model_template
    }

    pub fn model(&self) -> &AnalysisModel {
        &self.model
    }

    pub fn procedure_step(&self) -> &ProcedureStep {
        &self.procedure_step
    }

    pub fn execution(&self) -> &AnalysisExecutionStep {
        &self.authored_execution
    }

    pub fn workload_template(&self) -> &StructuralLinearWorkload {
        &self.workload_template
    }

    pub fn run_options(&self) -> &AnalysisRunOptions {
        &self.options
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
            authored_execution: self.authored_execution.clone(),
            procedure_step: self.procedure_step.clone(),
            operation_sequence: self.operation_sequence.clone(),
            workload_template: self.workload_template.clone(),
            options: self.options.clone(),
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
            authored_execution: continuation.authored_execution,
            procedure_step: continuation.procedure_step,
            operation_sequence: continuation.operation_sequence,
            workload_template: continuation.workload_template,
            options: continuation.options,
        };
        intent.validate()?;
        Ok(intent)
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        if self.study_id.trim().is_empty()
            || self.geometry_source_digest.bytes() == &[0_u8; 32]
            || self.authored_study_digest.bytes() == &[0_u8; 32]
            || self.options.solver_mesh_artifact_path.is_some()
            || self.workload_template.model != self.model
            || self.meshing_request
                != resolve_intent_meshing_request(
                    self.geometry_units,
                    self.meshing_settings.clone(),
                )?
            || self.meshing_contacts
                != super::meshing::meshing_contact_definitions_for_model(&self.model)
                    .collect::<Vec<_>>()
            || self.workload_template.backend != self.authored_execution.backend
            || self.workload_template.formulation_requirement != self.authored_execution.requirement
            || self.authored_execution.step_id != self.procedure_step.step_id
            || self.workload_template.model.sequence.steps.as_slice()
                != std::slice::from_ref(&self.procedure_step)
        {
            return Err("structural study intent semantics are inconsistent".into());
        }
        let AnalysisSolverOptions::StructuralLinear(mut expected_options) =
            self.authored_execution.solver_options.clone()
        else {
            return Err("structural study intent is not structural-linear".into());
        };
        expected_options.solver_mesh_artifact_path = None;
        if self.options != expected_options {
            return Err("structural study intent options differ from authored execution".into());
        }
        self.workload_template
            .validate()
            .map_err(|error| error.to_string())?;
        let semantic_digest = self.semantic_digest()?;
        if self.intent_digest != semantic_digest {
            return Err("structural study intent digest differs from its identity".into());
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
            &self.authored_execution,
            &self.procedure_step,
            &self.operation_sequence,
            &self.workload_template,
            &self.options,
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
    pub fn workload(&self) -> &StructuralLinearWorkload {
        self.linear.workload()
    }

    pub fn solver_mesh(&self) -> &SolverMeshArtifact {
        self.linear
            .solver_mesh()
            .expect("prepared structural studies always contain a canonical solver mesh")
    }
}
