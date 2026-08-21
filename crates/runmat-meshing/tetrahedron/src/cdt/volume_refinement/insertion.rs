use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    validate_delaunay_volume_refinement_candidate, DelaunayVolumeRefinementCandidate,
    DelaunayVolumeRefinementCandidateError, DelaunayVolumeRefinementCandidateErrorKind,
    DelaunayVolumeRefinementInput, DelaunayVolumeRefinementStep, DelaunayVolumeRefinementStepError,
    DelaunayVolumeRefinementStepErrorKind, DelaunayVolumeRefinementStepOptions,
};
use crate::cdt::{
    evaluate_delaunay_volume_quality, insertion::insert_delaunay_volume_node_with_barriers,
    insertion::validate_constrained_delaunay_volume_topology, validate_delaunay_volume_provenance,
    validate_delaunay_volume_quality, DelaunayInsertionError, DelaunayInsertionErrorKind,
    DelaunayVolumeProvenanceError, DelaunayVolumeProvenanceErrorKind, DelaunayVolumeQualityError,
    DelaunayVolumeQualityErrorKind,
};

pub fn insert_delaunay_volume_refinement_candidate(
    input: DelaunayVolumeRefinementInput<'_>,
    candidate: &DelaunayVolumeRefinementCandidate,
    options: DelaunayVolumeRefinementStepOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeRefinementStep, DelaunayVolumeRefinementStepError> {
    validate_delaunay_volume_refinement_candidate(
        input,
        &Some(candidate.clone()),
        options.candidate,
        cancellation,
    )
    .map_err(candidate_error)?;
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    validate_constrained_delaunay_volume_topology(
        input.topology,
        &protected_faces,
        options.insertion,
        cancellation,
    )
    .map_err(insertion_error)?;
    let topology = insert_delaunay_volume_node_with_barriers(
        input.topology.clone(),
        candidate.node,
        &protected_faces,
        options.insertion,
        cancellation,
    )
    .map_err(insertion_error)?;
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        input.metric_request,
        input.provenance,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    let step = DelaunayVolumeRefinementStep { topology, quality };
    validate_delaunay_volume_refinement_step(input, candidate, &step, options, cancellation)?;
    Ok(step)
}

pub fn validate_delaunay_volume_refinement_step(
    input: DelaunayVolumeRefinementInput<'_>,
    candidate: &DelaunayVolumeRefinementCandidate,
    step: &DelaunayVolumeRefinementStep,
    options: DelaunayVolumeRefinementStepOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeRefinementStepError> {
    validate_delaunay_volume_refinement_candidate(
        input,
        &Some(candidate.clone()),
        options.candidate,
        cancellation,
    )
    .map_err(candidate_error)?;
    if step.topology.nodes.len() != input.topology.nodes.len() + 1
        || step
            .topology
            .nodes
            .binary_search_by_key(&candidate.node.identity, |node| node.identity)
            .ok()
            .and_then(|index| step.topology.nodes.get(index))
            != Some(&candidate.node)
        || input.topology.nodes.iter().any(|old| {
            step.topology
                .nodes
                .binary_search_by_key(&old.identity, |node| node.identity)
                .ok()
                .and_then(|index| step.topology.nodes.get(index))
                != Some(old)
        })
    {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
            "refinement must retain every old node and add exactly the selected candidate",
        ));
    }
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    validate_constrained_delaunay_volume_topology(
        &step.topology,
        &protected_faces,
        options.insertion,
        cancellation,
    )
    .map_err(insertion_error)?;
    validate_delaunay_volume_provenance(
        &step.topology,
        input.provenance,
        input.quality_options.provenance,
        cancellation,
    )
    .map_err(provenance_error)?;
    validate_delaunay_volume_quality(
        &step.topology,
        input.metric_request,
        input.provenance,
        &step.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    Ok(())
}

pub(super) fn candidate_error(
    failure: DelaunayVolumeRefinementCandidateError,
) -> DelaunayVolumeRefinementStepError {
    let kind = match failure.kind {
        DelaunayVolumeRefinementCandidateErrorKind::InvalidOptions => {
            DelaunayVolumeRefinementStepErrorKind::InvalidOptions
        }
        DelaunayVolumeRefinementCandidateErrorKind::InvalidTopology => {
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology
        }
        DelaunayVolumeRefinementCandidateErrorKind::InvalidQuality => {
            DelaunayVolumeRefinementStepErrorKind::InvalidInput
        }
        DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate
        | DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure => {
            DelaunayVolumeRefinementStepErrorKind::InvalidCandidate
        }
        DelaunayVolumeRefinementCandidateErrorKind::ResourceLimit => {
            DelaunayVolumeRefinementStepErrorKind::ResourceLimit
        }
        DelaunayVolumeRefinementCandidateErrorKind::Cancelled => {
            DelaunayVolumeRefinementStepErrorKind::Cancelled
        }
    };
    error(kind, failure.to_string())
}

pub(super) fn insertion_error(
    failure: DelaunayInsertionError,
) -> DelaunayVolumeRefinementStepError {
    let kind = match failure.kind {
        DelaunayInsertionErrorKind::InvalidOptions => {
            DelaunayVolumeRefinementStepErrorKind::InvalidOptions
        }
        DelaunayInsertionErrorKind::InvalidNode => {
            DelaunayVolumeRefinementStepErrorKind::InvalidCandidate
        }
        DelaunayInsertionErrorKind::InvalidTopology
        | DelaunayInsertionErrorKind::PointOutsideTopology => {
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology
        }
        DelaunayInsertionErrorKind::ResourceLimit => {
            DelaunayVolumeRefinementStepErrorKind::ResourceLimit
        }
        DelaunayInsertionErrorKind::Cancelled => DelaunayVolumeRefinementStepErrorKind::Cancelled,
    };
    error(kind, failure.to_string())
}

fn provenance_error(failure: DelaunayVolumeProvenanceError) -> DelaunayVolumeRefinementStepError {
    let kind = match failure.kind {
        DelaunayVolumeProvenanceErrorKind::InvalidOptions => {
            DelaunayVolumeRefinementStepErrorKind::InvalidOptions
        }
        DelaunayVolumeProvenanceErrorKind::InvalidTopology => {
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology
        }
        DelaunayVolumeProvenanceErrorKind::InvalidProvenance => {
            DelaunayVolumeRefinementStepErrorKind::InvalidProvenance
        }
        DelaunayVolumeProvenanceErrorKind::ResourceLimit => {
            DelaunayVolumeRefinementStepErrorKind::ResourceLimit
        }
        DelaunayVolumeProvenanceErrorKind::Cancelled => {
            DelaunayVolumeRefinementStepErrorKind::Cancelled
        }
    };
    error(kind, failure.to_string())
}

pub(super) fn quality_error(
    failure: DelaunayVolumeQualityError,
) -> DelaunayVolumeRefinementStepError {
    let kind = match failure.kind {
        DelaunayVolumeQualityErrorKind::InvalidOptions => {
            DelaunayVolumeRefinementStepErrorKind::InvalidOptions
        }
        DelaunayVolumeQualityErrorKind::InvalidTopology => {
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology
        }
        DelaunayVolumeQualityErrorKind::InvalidMetric
        | DelaunayVolumeQualityErrorKind::InvalidMetricContext => {
            DelaunayVolumeRefinementStepErrorKind::InvalidProvenance
        }
        DelaunayVolumeQualityErrorKind::InvalidQuality
        | DelaunayVolumeQualityErrorKind::NumericalFailure => {
            DelaunayVolumeRefinementStepErrorKind::InvalidQuality
        }
        DelaunayVolumeQualityErrorKind::ResourceLimit => {
            DelaunayVolumeRefinementStepErrorKind::ResourceLimit
        }
        DelaunayVolumeQualityErrorKind::Cancelled => {
            DelaunayVolumeRefinementStepErrorKind::Cancelled
        }
    };
    error(kind, failure.to_string())
}

pub(super) fn error(
    kind: DelaunayVolumeRefinementStepErrorKind,
    reason: impl Into<String>,
) -> DelaunayVolumeRefinementStepError {
    DelaunayVolumeRefinementStepError {
        kind,
        reason: reason.into(),
    }
}
