use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_analysis_fea::execution::StructuralLinearWorkload;
use runmat_meshing_core::SolverMeshArtifact;

use crate::operations::OperationContext;

use super::{
    analysis_study_content_digest, execution_operation, solver_mesh_artifact, study_fingerprint,
    study_operation_sequence, AnalysisExecutionStep, AnalysisRunOptions, AnalysisSolverOptions,
    AnalysisStudySpec, PreparedStructuralLinearRun,
};

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralStudyContinuation {
    schema_version: u16,
    study: PreparedStudyRunContinuation,
    workload: StructuralLinearWorkload,
    options: AnalysisRunOptions,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedStudyRunContinuation {
    spec: AnalysisStudySpec,
    context: OperationContext,
    study_fingerprint: String,
    model: AnalysisModel,
    authored_execution: AnalysisExecutionStep,
    procedure_step: ProcedureStep,
    operation_sequence: Vec<String>,
    solver_mesh_artifact_path: String,
    meshing_evidence_artifact_path: Option<String>,
    solver_mesh_digest: runmat_meshing_core::StableDigest,
}

impl PreparedStructuralStudy {
    pub fn authored_study_digest(&self) -> runmat_execution::Digest {
        analysis_study_content_digest(&self.study.spec)
    }

    pub fn workload(&self) -> &StructuralLinearWorkload {
        self.linear.workload()
    }

    pub fn solver_mesh(&self) -> &SolverMeshArtifact {
        self.linear
            .solver_mesh()
            .expect("prepared structural studies always contain a canonical solver mesh")
    }

    pub fn continuation_bytes(&self) -> Result<Vec<u8>, String> {
        let solver_mesh_artifact_path = self
            .study
            .solver_mesh_artifact_path
            .clone()
            .ok_or_else(|| "prepared structural study has no persisted solver mesh".to_string())?;
        let continuation = StructuralStudyContinuation {
            schema_version: 1,
            study: PreparedStudyRunContinuation {
                spec: self.study.spec.clone(),
                context: self.study.context.clone(),
                study_fingerprint: self.study.study_fingerprint.clone(),
                model: self.study.model.clone(),
                authored_execution: self.study.authored_execution.clone(),
                procedure_step: self.study.procedure_step.clone(),
                operation_sequence: self.study.operation_sequence.clone(),
                solver_mesh_artifact_path,
                meshing_evidence_artifact_path: self.study.meshing_evidence_artifact_path.clone(),
                solver_mesh_digest: self.solver_mesh().canonical_digest,
            },
            workload: self.linear.workload.clone(),
            options: self.linear.options.clone(),
        };
        serde_json::to_vec(&continuation).map_err(|error| error.to_string())
    }

    pub fn from_continuation_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > 16 * 1024 * 1024 {
            return Err("structural study continuation exceeds its bound".into());
        }
        let continuation: StructuralStudyContinuation =
            serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if continuation.schema_version != 1 {
            return Err("unsupported structural study continuation schema".into());
        }
        let mesh = solver_mesh_artifact::load_solver_mesh_artifact(std::path::Path::new(
            &continuation.study.solver_mesh_artifact_path,
        ))
        .map_err(|error| error.to_string())?;
        if mesh.canonical_digest != continuation.study.solver_mesh_digest {
            return Err("structural study continuation mesh identity changed".into());
        }
        continuation
            .workload
            .validate()
            .map_err(|error| error.to_string())?;
        let [authored_execution] = continuation.study.spec.execution.steps.as_slice() else {
            return Err("structural study continuation requires one execution step".into());
        };
        let procedure_step = continuation
            .study
            .spec
            .execution
            .procedure_step(&continuation.study.model.sequence, authored_execution)?;
        let planned_operation = execution_operation(authored_execution, procedure_step)
            .ok_or_else(|| "structural study continuation operation is unsupported".to_string())?;
        let AnalysisSolverOptions::StructuralLinear(mut expected_options) =
            authored_execution.solver_options.clone()
        else {
            return Err("structural study continuation is not structural-linear".into());
        };
        if expected_options.solver_mesh_artifact_path.is_none() {
            expected_options.solver_mesh_artifact_path =
                Some(continuation.study.solver_mesh_artifact_path.clone());
        }
        if continuation.workload.model != continuation.study.model
            || continuation.study.study_fingerprint != study_fingerprint(&continuation.study.spec)
            || continuation.study.spec.create_model_intent.model_id
                != continuation.study.model.model_id.0
            || authored_execution != &continuation.study.authored_execution
            || procedure_step != &continuation.study.procedure_step
            || continuation.study.operation_sequence
                != study_operation_sequence(&continuation.study.spec, planned_operation.version())
            || continuation.workload.backend != authored_execution.backend
            || continuation.workload.formulation_requirement != authored_execution.requirement
            || continuation.options != expected_options
            || mesh.geometry.geometry_revision
                != u64::from(continuation.study.model.geometry_revision)
        {
            return Err("structural study continuation semantics changed".into());
        }
        let study = PreparedStudyRun {
            spec: continuation.study.spec,
            context: continuation.study.context.clone(),
            study_fingerprint: continuation.study.study_fingerprint,
            model: continuation.study.model,
            authored_execution: continuation.study.authored_execution,
            procedure_step: continuation.study.procedure_step,
            operation_sequence: continuation.study.operation_sequence,
            solver_mesh_artifact_path: Some(continuation.study.solver_mesh_artifact_path),
            meshing_evidence_artifact_path: continuation.study.meshing_evidence_artifact_path,
        };
        Ok(Self {
            linear: PreparedStructuralLinearRun {
                workload: continuation.workload,
                solver_mesh: Some(mesh),
                options: continuation.options,
                context: continuation.study.context,
            },
            study,
        })
    }
}
