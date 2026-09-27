use super::*;

pub(in crate::analysis) struct PreparedStudyCompletion {
    pub(in crate::analysis) thermal_prefix: Option<Box<ThermalPrefixCompletion>>,
    pub(in crate::analysis::structural_study) context: OperationContext,
    pub(in crate::analysis::structural_study) study_id: String,
    pub(in crate::analysis::structural_study) study_fingerprint: String,
    pub(in crate::analysis::structural_study) authored_study_digest: runmat_execution::Digest,
    pub(in crate::analysis::structural_study) model: AnalysisModel,
    pub(in crate::analysis::structural_study) steps: Vec<PreparedStudyStep>,
    pub(in crate::analysis::structural_study) operation_sequence: Vec<String>,
    pub(in crate::analysis::structural_study) solver_mesh_artifact_path: Option<String>,
    pub(in crate::analysis::structural_study) meshing_evidence_artifact_path: Option<String>,
}

pub(in crate::analysis) fn study_completion_from_prepared(
    prepared: PreparedStudySequence,
) -> PreparedStudyCompletion {
    let authored_study_digest = crate::analysis::analysis_study_content_digest(&prepared.spec);
    PreparedStudyCompletion {
        thermal_prefix: None,
        context: prepared.context,
        study_id: prepared.spec.study_id,
        study_fingerprint: prepared.study_fingerprint,
        authored_study_digest,
        model: prepared.model,
        steps: prepared.steps,
        operation_sequence: prepared.operation_sequence,
        solver_mesh_artifact_path: prepared.solver_mesh_artifact_path,
        meshing_evidence_artifact_path: prepared.meshing_evidence_artifact_path,
    }
}

pub(in crate::analysis) struct CompletedStudyStep {
    pub(in crate::analysis) step_id: ProcedureStepId,
    pub(in crate::analysis) solver_result_identity: SolverResultIdentity,
    pub(in crate::analysis) run: CompletedStudyStepRun,
    pub(in crate::analysis) resolved_solver_options: AnalysisSolverOptions,
}

pub(in crate::analysis) struct ThermalPrefixCompletion {
    pub authored: PreparedStudyStep,
    pub completed: CompletedStudyStep,
    pub states: Vec<CanonicalSemanticState>,
    pub observation_source: crate::analysis::ThermalStudyObjectSource,
    pub thermal: Box<crate::analysis::ThermalStudyCompletion>,
}

pub(in crate::analysis) enum CompletedStudyStepRun {
    Standard(Box<OperationEnvelope<AnalysisRunResult>>),
    StructuralFrequency(Box<CompletedStructuralFrequencyStep>),
    StructuralResponseSpectrum(Box<CompletedStructuralResponseSpectrumStep>),
    StructuralRandomResponse(Box<CompletedStructuralRandomResponseStep>),
}
