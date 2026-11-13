use super::*;

pub(super) struct AdmittedPrestressedModalReduction {
    pub run: OperationEnvelope<crate::analysis::AnalysisRunResult>,
    pub result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub basis_state_identity: runmat_analysis_fea::execution::StateIdentity,
}

pub(super) struct PrestressedModalReductionAdmission<'a> {
    pub prepared: &'a PreparedStudyCompletion,
    pub solver_mesh: &'a runmat_meshing_core::SolverMeshArtifact,
    pub prestress_state: &'a runmat_analysis_fea::execution::CanonicalSemanticState,
    pub frequency_template:
        &'a runmat_analysis_fea::execution::StructuralPrestressedFrequencyTemplate,
    pub modal_template: runmat_analysis_fea::execution::StructuralPrestressedModalTemplate,
    pub modal_result: &'a runmat_analysis_fea::execution::StructuralModalSolverResult,
    pub modal_receipt: &'a runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
    pub basis_state: &'a runmat_analysis_fea::execution::CanonicalSemanticState,
    pub modal_options: &'a mut crate::analysis::AnalysisModalRunOptions,
    pub run_id: &'a str,
    pub topology_digest: runmat_execution::Digest,
}

pub(super) fn admit_prestressed_modal_reduction(
    input: PrestressedModalReductionAdmission<'_>,
) -> Result<AdmittedPrestressedModalReduction, OperationErrorEnvelope> {
    let mut modal_template = input.modal_template;
    modal_template.model = input.prepared.model.clone();
    let modal_transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &input.prepared.model,
        &modal_template.step_id,
        input.prestress_state,
        runmat_analysis_core::StateUse::Prestress,
        input.topology_digest,
    )
    .map_err(|error| {
        completion_error(
            input.prepared,
            "RM.FEA.RUN_STUDY.PRESTRESS_STATE_INVALID",
            error.to_string(),
        )
    })?;
    let modal_workload = modal_template.bind(modal_transition).map_err(|error| {
        completion_error(
            input.prepared,
            "RM.FEA.RUN_STUDY.MODAL_WORKLOAD_INVALID",
            error.to_string(),
        )
    })?;
    let recovered = runmat_analysis_fea::recover_completed_structural_prestressed_modal_run(
        &modal_workload,
        input.solver_mesh,
        input.prestress_state,
        input.modal_result,
        input.modal_receipt,
    )
    .map_err(|error| {
        completion_error(
            input.prepared,
            "RM.FEA.RUN_STUDY.MODAL_RESULT_ADMISSION_FAILED",
            error.to_string(),
        )
    })?;
    input.modal_options.solver_mesh_artifact_path =
        input.prepared.solver_mesh_artifact_path.clone();
    let run = analysis_admit_modal_run_with_id(
        recovered,
        modal_workload.backend(),
        input.modal_options.clone(),
        input.prepared.context.clone(),
        Some(input.solver_mesh),
        Some(input.run_id),
    )?;
    input.basis_state.validate().map_err(|error| {
        completion_error(
            input.prepared,
            "RM.FEA.RUN_STUDY.MODAL_BASIS_STATE_INVALID",
            error.to_string(),
        )
    })?;
    if input.basis_state.source_result != input.modal_receipt.source_result()
        || input
            .basis_state
            .payload
            .modal_basis()
            .map(|basis| basis.identity())
            != Some(input.modal_result.basis_identity())
    {
        return Err(completion_error(
            input.prepared,
            "RM.FEA.RUN_STUDY.MODAL_BASIS_STATE_INVALID",
            "modal-basis state differs from its admitted modal producer",
        ));
    }
    runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &input.prepared.model,
        &input.frequency_template.step_id,
        input.basis_state,
        runmat_analysis_core::StateUse::ModalBasis,
        input.topology_digest,
    )
    .map_err(|error| {
        completion_error(
            input.prepared,
            "RM.FEA.RUN_STUDY.MODAL_BASIS_STATE_INVALID",
            error.to_string(),
        )
    })?;
    Ok(AdmittedPrestressedModalReduction {
        run,
        result_identity: input.modal_receipt.source_result(),
        basis_state_identity: input.basis_state.identity,
    })
}
