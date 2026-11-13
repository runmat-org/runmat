use super::*;

pub(super) struct PrestressedFrequencyStatePublication<'a> {
    pub prepared: &'a PreparedStudyCompletion,
    pub preload_step: &'a super::super::PreparedStudyStep,
    pub modal_step: Option<&'a super::super::PreparedStudyStep>,
    pub frequency_step: &'a super::super::PreparedStudyStep,
    pub prestress_state: &'a runmat_analysis_fea::execution::CanonicalSemanticState,
    pub preload_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
    pub reduction: &'a crate::analysis::StructuralStudyPrestressedFrequencyReductionResult,
    pub frequency_publication:
        &'a runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest,
    pub preload_fields: &'a [runmat_analysis_core::AnalysisField],
    pub topology_digest: runmat_execution::Digest,
    pub frequency_result_identity: runmat_analysis_fea::execution::SolverResultIdentity,
}

pub(super) fn prestressed_frequency_states(
    input: PrestressedFrequencyStatePublication<'_>,
) -> Result<Vec<runmat_analysis_fea::execution::CanonicalSemanticState>, OperationErrorEnvelope> {
    let mut states = Vec::with_capacity(
        input.preload_step.procedure_step.state_outputs.len()
            + input
                .modal_step
                .map_or(0, |step| step.procedure_step.state_outputs.len())
            + input.frequency_step.procedure_step.state_outputs.len(),
    );
    for declaration in &input.preload_step.procedure_step.state_outputs {
        if declaration.state_id == input.prestress_state.state_id {
            states.push(input.prestress_state.clone());
        } else {
            states.push(
                runmat_analysis_fea::execution::CanonicalSemanticState::new(
                    runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                        producer: input.prestress_state.producer,
                        declaration: declaration.clone(),
                        topology_digest: input.topology_digest,
                        source_result: input.preload_result_identity,
                        payload:
                            runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                                fields: input.preload_fields.to_vec(),
                            },
                    },
                )
                .map_err(|error| {
                    completion_error(
                        input.prepared,
                        "RM.FEA.RUN_STUDY.STATE_INVALID",
                        error.to_string(),
                    )
                })?,
            );
        }
    }
    if let crate::analysis::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(
        reduction,
    ) = input.reduction
    {
        states.push(reduction.basis_state.clone());
    }
    let sequence_identity = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &input.prepared.model.sequence,
    )
    .map_err(|error| {
        completion_error(
            input.prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence_identity,
        &input.frequency_step.procedure_step,
    )
    .map_err(|error| {
        completion_error(
            input.prepared,
            "RM.FEA.RUN_STUDY.STATE_INVALID",
            error.to_string(),
        )
    })?;
    for declaration in &input.frequency_step.procedure_step.state_outputs {
        states.push(
            runmat_analysis_fea::execution::CanonicalSemanticState::new(
                runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                    producer,
                    declaration: declaration.clone(),
                    topology_digest: input.topology_digest,
                    source_result: input.frequency_result_identity,
                    payload: runmat_analysis_fea::execution::CanonicalSemanticStatePayload::StructuralFrequency {
                        publication: Box::new(input.frequency_publication.clone()),
                    },
                },
            )
            .map_err(|error| {
                completion_error(
                    input.prepared,
                    "RM.FEA.RUN_STUDY.STATE_INVALID",
                    error.to_string(),
                )
            })?,
        );
    }
    Ok(states)
}
