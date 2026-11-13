//! Shared global and explicitly entry-scoped authored state publication admission.
use super::*;

pub(super) fn validate_entry_scoped_acoustic_states(
    context: &OperationContext,
    authored: &runmat_analysis_core::ProcedureStep,
    result: &crate::analysis::study_publication::AcousticFrequencyCompletionResults,
) -> Result<(), OperationErrorEnvelope> {
    let invalid = |message: &str| {
        completion_error(
            context,
            "RM.FEA.RUN_STUDY.STATE_INVENTORY_MISMATCH",
            message,
        )
    };
    let entries = result
        .workload
        .plan()
        .entries()
        .map_err(|e| invalid(&e.to_string()))?;
    if entries.len() != result.entries.len()
        || result.entries.len() != result.result.results().len()
    {
        return Err(invalid(
            "acoustic state inventory does not cover every exact frequency entry",
        ));
    }
    // The published record keeps only the manifest's authority; it must be the one the complete
    // numerical result implies.
    if result.result.workload() != &result.workload
        || result.publication.authority()
            != &result
                .result
                .publication_authority()
                .map_err(|e| invalid(&e.to_string()))?
    {
        return Err(invalid(
            "acoustic frequency publication authority differs from its numerical result",
        ));
    }
    let mut seen = BTreeSet::new();
    for ((planned, entry), numerical) in entries
        .iter()
        .zip(&result.entries)
        .zip(result.result.results())
    {
        if planned != &entry.entry
            || numerical != &entry.result
            || entry.workload
                != result
                    .workload
                    .entry(*planned)
                    .map_err(|e| invalid(&e.to_string()))?
            || entry.workload.step_id != authored.step_id
            || entry.workload.model.sequence.steps.as_slice() != [authored.clone()]
            || entry.states.len() != authored.state_outputs.len()
        {
            return Err(invalid(
                "acoustic entry-scoped state inventory differs from its authored declaration",
            ));
        }
        let requests = numerical
            .state_publication_requests()
            .map_err(|e| invalid(&e.to_string()))?;
        if requests.len() != entry.states.len() {
            return Err(invalid(
                "acoustic frequency state inventory differs from original result",
            ));
        }
        for ((state, declaration), request) in entry
            .states
            .iter()
            .zip(&authored.state_outputs)
            .zip(&requests)
        {
            state.validate().map_err(|e| invalid(&e.to_string()))?;
            let payload = state
                .payload
                .acoustic()
                .ok_or_else(|| invalid("acoustic entry has a different state payload"))?;
            if state.producer != request.producer
                || state.source_result != request.source_result
                || state.topology_digest != request.topology_digest
                || state.state_id != request.declaration.state_id
                || state.kind != request.declaration.kind
                || state.completeness != request.declaration.completeness
                || state.persistence != request.declaration.persistence
                || payload.ordered_node_ids != numerical.ordered_node_ids()
                || payload.ordered_element_ids != numerical.ordered_element_ids()
                || payload.recovery_request
                    != numerical
                        .native_field_request_from_result()
                        .map_err(|e| invalid(&e.to_string()))?
                || !numerical.native_fields_match(payload.fields.fields())
            {
                return Err(invalid(
                    "acoustic frequency state differs from its exact entry source",
                ));
            }
            if state.producer != numerical.producer()
                || state.source_result
                    != numerical
                        .source_result_identity()
                        .map_err(|e| invalid(&e.to_string()))?
                || state.state_id != declaration.state_id
                || state.kind != declaration.kind
                || state.completeness != declaration.completeness
                || state.persistence != declaration.persistence
                || payload.frequency.canonical_value() != planned.frequency_hz()
                || !seen.insert((planned.ordinal(), declaration.state_id.clone()))
            {
                return Err(invalid(
                    "acoustic state differs from its exact frequency/declaration/result source",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_and_persist_states(
    context: &OperationContext,
    study_fingerprint: &str,
    sequence_identity: SequenceIdentity,
    model: &AnalysisModel,
    solver_results: &BTreeMap<ProcedureStepId, SolverResultIdentity>,
    states: Vec<CanonicalSemanticState>,
    entry_scoped_state_steps: &BTreeSet<ProcedureStepId>,
    staging: &storage::AnalysisPublicationBuilder,
) -> Result<Vec<AnalysisStudyStatePublicationData>, OperationErrorEnvelope> {
    let expected = model
        .sequence
        .steps
        .iter()
        .filter(|step| !entry_scoped_state_steps.contains(&step.step_id))
        .flat_map(|step| {
            step.state_outputs
                .iter()
                .map(move |declaration| (step, declaration))
        })
        .collect::<Vec<_>>();
    if states.len() != expected.len() {
        return Err(completion_error(
            context,
            "RM.FEA.RUN_STUDY.STATE_INVENTORY_MISMATCH",
            "published states do not exactly cover the authored declarations",
        ));
    }
    let mut seen = BTreeSet::new();
    for (state, (producer, declaration)) in states.iter().zip(&expected) {
        state
            .validate()
            .map_err(|error| completion_error(context, "RM.FEA.RUN_STUDY.STATE_INVALID", error))?;
        let producer_identity = StepIdentity::from_solver_contract(sequence_identity, producer)
            .map_err(|error| completion_error(context, "RM.FEA.RUN_STUDY.STATE_INVALID", error))?;
        if state.producer != producer_identity
            || state.state_id != declaration.state_id
            || state.kind != declaration.kind
            || state.completeness != declaration.completeness
            || state.persistence != declaration.persistence
            || solver_results.get(&producer.step_id) != Some(&state.source_result)
            || !seen.insert((producer.step_id.clone(), declaration.state_id.clone()))
        {
            return Err(completion_error(
                context,
                "RM.FEA.RUN_STUDY.STATE_INVENTORY_MISMATCH",
                "published state identity, order, declaration, or solver lineage changed",
            ));
        }
    }
    let mut results = Vec::with_capacity(states.len());
    for (state, (producer, _)) in states.into_iter().zip(expected) {
        let artifact_path =
            persist_semantic_state(study_fingerprint, &state, staging).map_err(|error| {
                completion_storage_error(
                    context,
                    "RM.FEA.RUN_STUDY.STATE_ARTIFACT_STORE_FAILED",
                    error,
                )
            })?;
        results.push(AnalysisStudyStatePublicationData {
            producer_step: producer.step_id.clone(),
            state_id: state.state_id,
            kind: state.kind,
            completeness: state.completeness,
            persistence: state.persistence,
            identity: state.identity,
            content_digest: state.content_digest,
            topology_digest: state.topology_digest,
            source_result: state.source_result,
            artifact_path,
        });
    }
    Ok(results)
}

fn persist_semantic_state(
    study_fingerprint: &str,
    state: &CanonicalSemanticState,
    staging: &storage::AnalysisPublicationBuilder,
) -> Result<String, storage::AnalysisPublicationError> {
    staging.persist_state(study_fingerprint, state)
}
