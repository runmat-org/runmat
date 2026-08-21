use std::collections::BTreeSet;

use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use crate::adaptation::SolverFieldTransferMethod;

use super::{
    SolverFieldTransferErrorEvidence, StructuralAdaptationIterationError,
    StructuralAdaptationSolverResult, StructuralRecoveryEstimate, StructuralTargetQuantity,
};

const MAX_TRANSFER_FIELDS: usize = 4096;

pub(super) fn validate_transfer_errors(
    errors: &[SolverFieldTransferErrorEvidence],
    source_digest: StableDigest,
    target_digest: StableDigest,
) -> Result<(), StructuralAdaptationIterationError> {
    if errors.is_empty() || errors.len() > MAX_TRANSFER_FIELDS {
        return Err(StructuralAdaptationIterationError::InvalidTransferEvidence);
    }
    let mut previous = None;
    for error in errors {
        let key = (&error.transfer.topology_id, error.transfer.location);
        if error.transfer.source_artifact_digest != source_digest
            || error.transfer.target_artifact_digest != target_digest
            || error.transferred_field_digest == StableDigest::ZERO
            || error.reference_field_digest == StableDigest::ZERO
            || error.transfer.component_count == 0
            || error.transfer.methods.is_empty()
            || !error.absolute_l2_error.is_finite()
            || error.absolute_l2_error < 0.0
            || error
                .relative_l2_error
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            || previous.is_some_and(|prior| prior >= key)
        {
            return Err(StructuralAdaptationIterationError::InvalidTransferEvidence);
        }
        previous = Some(key);
    }
    Ok(())
}

pub(super) fn validate_estimator_against(
    estimator: &StructuralRecoveryEstimate,
    target: &SolverMeshArtifact,
) -> Result<(), StructuralAdaptationIterationError> {
    let expected = target
        .topology
        .volume_elements
        .iter()
        .map(|element| element.stable_identity)
        .collect::<BTreeSet<_>>();
    let actual = estimator
        .indicators
        .iter()
        .map(|indicator| indicator.element_stable_identity)
        .collect::<BTreeSet<_>>();
    if estimator.solver_artifact_digest != target.canonical_digest || actual != expected {
        return Err(StructuralAdaptationIterationError::InvalidEstimator);
    }
    Ok(())
}

pub(super) fn validate_transfer_errors_against(
    errors: &[SolverFieldTransferErrorEvidence],
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
) -> Result<(), StructuralAdaptationIterationError> {
    for error in errors {
        let source_topology = source
            .topology
            .field_topologies
            .iter()
            .find(|topology| topology.topology_id == error.transfer.topology_id);
        let target_topology = target
            .topology
            .field_topologies
            .iter()
            .find(|topology| topology.topology_id == error.transfer.topology_id);
        let methods = error
            .transfer
            .methods
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if source_topology.is_none_or(|topology| topology.location != error.transfer.location)
            || target_topology.is_none_or(|topology| topology.location != error.transfer.location)
            || target_topology.is_none_or(|topology| {
                error
                    .transfer
                    .copied_entity_count
                    .checked_add(error.transfer.projected_entity_count)
                    != Some(topology.ordered_entity_ids.len())
            })
            || methods.len() != error.transfer.methods.len()
            || error.transfer.methods.first() != Some(&SolverFieldTransferMethod::StableIdentity)
        {
            return Err(StructuralAdaptationIterationError::InvalidTransferEvidence);
        }
    }
    Ok(())
}

pub(super) fn validate_solver_result(
    result: &StructuralAdaptationSolverResult,
) -> Result<(), StructuralAdaptationIterationError> {
    if result.result_digest == StableDigest::ZERO
        || result.iteration_count == 0
        || !result.normalized_residual.is_finite()
        || result.normalized_residual < 0.0
    {
        return Err(StructuralAdaptationIterationError::InvalidSolverResult);
    }
    Ok(())
}

pub(super) fn validate_target_quantity(
    target: &StructuralTargetQuantity,
) -> Result<(), StructuralAdaptationIterationError> {
    if target.quantity_id.is_empty()
        || target.quantity_id.len() > 256
        || !target.quantity_id.is_ascii()
        || target.quantity_id.chars().any(char::is_control)
        || !target.value.is_finite()
    {
        return Err(StructuralAdaptationIterationError::InvalidTargetQuantity);
    }
    Ok(())
}
