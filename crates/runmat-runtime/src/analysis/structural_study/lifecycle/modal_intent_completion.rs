//! Resolve portable modal result identities before shared original-state completion.
use super::modal_completion::prepared_completion;
use super::*;

pub(super) fn complete_structural_modal(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workload: runmat_analysis_fea::execution::StructuralModalWorkload,
    numerical: StructuralStudyModalNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "structural modal completion requires exactly one authored procedure step",
        ));
    };
    let assigned_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &step.procedure_step.step_id,
    );
    complete_modal_study(
        prepared_completion(intent, study_model, persisted_mesh),
        solver_mesh,
        authored_workload,
        numerical,
        assigned_run_id,
        preparation,
    )
}

pub(super) fn complete_structural_modal_transient(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    solver_mesh: runmat_meshing_core::SolverMeshArtifact,
    authored_workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralStepWorkload,
    >,
    numerical: StructuralStudyModalTransientNumericalResult,
    completion_identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let [modal_step, transient_step] = intent.steps.as_slice() else {
        return Err(intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.STEP_INVENTORY_MISMATCH",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            "modal-transient completion requires exactly two authored procedure steps",
        ));
    };
    let modal_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &modal_step.procedure_step.step_id,
    );
    let transient_run_id = completion_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &transient_step.procedure_step.step_id,
    );
    complete_modal_transient_study(
        prepared_completion(intent, study_model, persisted_mesh),
        solver_mesh,
        authored_workloads,
        numerical,
        modal_run_id,
        transient_run_id,
        preparation,
    )
}
