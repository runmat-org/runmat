use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    batch::{insert_independent_refinement_batch, RefinementBatchOutcome},
    insertion::insert_delaunay_volume_refinement_candidate_from_admitted_input,
    insertion::{error, insertion_error, quality_error},
    select_delaunay_volume_refinement_candidate_excluding_from_admitted_input,
    DelaunayVolumeRefinement, DelaunayVolumeRefinementInput, DelaunayVolumeRefinementOptions,
    DelaunayVolumeRefinementStepError, DelaunayVolumeRefinementStepErrorKind,
};
use crate::cdt::{
    insertion::validate_conforming_volume_topology, validate_delaunay_volume_quality,
};

const MAXIMUM_REFINEMENT_BATCH_SIZE: u64 = 128;
const PROGRESS_INSERTION_INTERVAL: u64 = 32;

pub fn refine_delaunay_volume(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeRefinement, DelaunayVolumeRefinementStepError> {
    let progress = refine_delaunay_volume_until_stalled(input, options, cancellation)?;
    if progress.stalled {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::ResourceLimit,
            "volume refinement exhausted every admissible construction before convergence",
        ));
    }
    validate_delaunay_volume_refinement(input, &progress.refinement, options, cancellation)?;
    Ok(progress.refinement)
}

pub(in crate::cdt) struct DelaunayVolumeRefinementProgress {
    pub(in crate::cdt) refinement: DelaunayVolumeRefinement,
    pub(in crate::cdt) stalled: bool,
}

pub(in crate::cdt) fn refine_delaunay_volume_until_stalled(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeRefinementProgress, DelaunayVolumeRefinementStepError> {
    validate_options(options)?;
    let mut topology = input.topology.clone();
    let mut quality = input.quality.clone();
    let mut insertions = Vec::new();
    let mut unusable_node_identities = BTreeSet::new();
    let mut insertion_count = 0u64;
    let mut maximum_batch_size = MAXIMUM_REFINEMENT_BATCH_SIZE;
    let mut next_progress_insertion = PROGRESS_INSERTION_INTERVAL;
    emit_progress_checkpoint(insertion_count, &topology, &quality, "started");
    loop {
        let current = DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality: &quality,
            quality_options: input.quality_options,
            fixed_node_identities: input.fixed_node_identities,
        };
        let remaining_insertions = options.maximum_insertions.saturating_sub(insertion_count);
        if remaining_insertions >= 2 {
            let maximum_batch = remaining_insertions.min(maximum_batch_size) as usize;
            match insert_independent_refinement_batch(
                current,
                options.step,
                &mut unusable_node_identities,
                maximum_batch,
                cancellation,
            )? {
                RefinementBatchOutcome::Admitted(batch) => {
                    maximum_batch_size = maximum_batch_size
                        .saturating_mul(2)
                        .min(MAXIMUM_REFINEMENT_BATCH_SIZE);
                    insertion_count += batch.inserted_nodes.len() as u64;
                    insertions.extend(batch.inserted_nodes);
                    topology = batch.step.topology;
                    quality = batch.step.quality;
                    // Candidate admission depends on the current Delaunay cavity.
                    // A point rejected against the preceding topology must be
                    // reconsidered after an admitted mutation changes that cavity.
                    unusable_node_identities.clear();
                    if insertion_count >= next_progress_insertion {
                        emit_progress_checkpoint(insertion_count, &topology, &quality, "refining");
                        next_progress_insertion =
                            insertion_count.saturating_add(PROGRESS_INSERTION_INTERVAL);
                    }
                    continue;
                }
                RefinementBatchOutcome::RetrySmaller if maximum_batch > 2 => {
                    maximum_batch_size = ((maximum_batch / 2).max(2)) as u64;
                    continue;
                }
                RefinementBatchOutcome::Unavailable | RefinementBatchOutcome::RetrySmaller => {}
            }
        }
        let candidate =
            match select_delaunay_volume_refinement_candidate_excluding_from_admitted_input(
                current,
                options.step.candidate,
                &unusable_node_identities,
                cancellation,
            ) {
                Ok(candidate) => candidate,
                Err(failure)
                    if failure.kind
                        == crate::cdt::DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure =>
                {
                    return Ok(DelaunayVolumeRefinementProgress {
                        refinement: DelaunayVolumeRefinement {
                            topology,
                            quality,
                            insertions,
                        },
                        stalled: true,
                    });
                }
                Err(failure) => return Err(super::insertion::candidate_error(failure)),
            };
        let Some(candidate) = candidate else {
            emit_progress_checkpoint(insertion_count, &topology, &quality, "completed");
            return Ok(DelaunayVolumeRefinementProgress {
                refinement: DelaunayVolumeRefinement {
                    topology,
                    quality,
                    insertions,
                },
                stalled: false,
            });
        };
        if insertion_count >= options.maximum_insertions {
            break;
        }
        let step = match insert_delaunay_volume_refinement_candidate_from_admitted_input(
            current,
            &candidate,
            options.step,
            cancellation,
        ) {
            Ok(step) => step,
            Err(failure)
                if failure.kind == DelaunayVolumeRefinementStepErrorKind::InvalidQuality =>
            {
                unusable_node_identities.insert(candidate.node.identity);
                continue;
            }
            Err(failure) => return Err(failure),
        };
        insertion_count += 1;
        insertions.push(candidate.node);
        topology = step.topology;
        quality = step.quality;
        unusable_node_identities.clear();
        if insertion_count >= next_progress_insertion {
            emit_progress_checkpoint(insertion_count, &topology, &quality, "refining");
            next_progress_insertion = insertion_count.saturating_add(PROGRESS_INSERTION_INTERVAL);
        }
    }
    let last_violation = quality
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.requires_refinement())
        .map(|tetrahedron| tetrahedron.refinement_violation_ratio)
        .max_by(f64::total_cmp)
        .unwrap_or(1.0);
    emit_progress_checkpoint(insertion_count, &topology, &quality, "exhausted");
    Err(error(
        DelaunayVolumeRefinementStepErrorKind::ResourceLimit,
        format!(
            "volume refinement exhausted {} insertions with worst violation ratio {last_violation}",
            options.maximum_insertions
        ),
    ))
}

fn emit_progress_checkpoint(
    insertion_count: u64,
    topology: &crate::cdt::DelaunayVolumeTopology,
    quality: &crate::cdt::DelaunayVolumeQuality,
    state: &'static str,
) {
    let violation_count = quality
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.requires_refinement())
        .count();
    let worst = quality.tetrahedra.iter().max_by(|left, right| {
        left.refinement_violation_ratio
            .total_cmp(&right.refinement_violation_ratio)
            .then_with(|| right.node_identities.cmp(&left.node_identities))
    });
    let worst_violation_ratio = worst.map_or(0.0, |cell| cell.refinement_violation_ratio);
    let worst_violation_count = quality
        .tetrahedra
        .iter()
        .filter(|cell| cell.refinement_violation_ratio.to_bits() == worst_violation_ratio.to_bits())
        .count();
    let boundary_nodes = topology
        .incidence
        .boundary_facets
        .iter()
        .flat_map(|facet| facet.vertex_indices)
        .collect::<BTreeSet<_>>();
    let worst_boundary_node_count = worst.map_or(0, |cell| {
        cell.node_identities
            .iter()
            .filter(|identity| {
                topology
                    .nodes
                    .binary_search_by_key(identity, |node| &node.identity)
                    .is_ok_and(|index| boundary_nodes.contains(&(index as u32)))
            })
            .count()
    });
    let worst_maximum_metric_edge_length =
        worst.map_or(0.0, |cell| cell.maximum_metric_edge_length);
    let worst_metric_radius_edge_ratio = worst.map_or(0.0, |cell| cell.metric_radius_edge_ratio);
    let worst_metric_scaled_jacobian = worst.map_or(0.0, |cell| cell.metric_scaled_jacobian);
    let span = tracing::info_span!(
        "meshing.tetrahedron.refinement_checkpoint",
        state,
        insertion_count,
        node_count = topology.nodes.len(),
        tetrahedron_count = topology.tetrahedra.len(),
        violation_count,
        worst_violation_ratio,
        worst_violation_count,
        worst_boundary_node_count,
        worst_maximum_metric_edge_length,
        worst_metric_radius_edge_ratio,
        worst_metric_scaled_jacobian,
    );
    let _guard = span.enter();
    tracing::info!(
        state,
        insertion_count,
        node_count = topology.nodes.len(),
        tetrahedron_count = topology.tetrahedra.len(),
        violation_count,
        worst_violation_ratio,
        worst_violation_count,
        worst_boundary_node_count,
        worst_maximum_metric_edge_length,
        worst_metric_radius_edge_ratio,
        worst_metric_scaled_jacobian,
        "tetrahedral refinement checkpoint"
    );
}

pub fn validate_delaunay_volume_refinement(
    input: DelaunayVolumeRefinementInput<'_>,
    refinement: &DelaunayVolumeRefinement,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeRefinementStepError> {
    validate_options(options)?;
    if refinement.quality.worst_refinement_tetrahedron.is_some() {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidQuality,
            "completed volume refinement must satisfy every configured quality bound",
        ));
    }
    if input.quality.worst_refinement_tetrahedron.is_none()
        && (refinement.topology != *input.topology
            || refinement.quality != *input.quality
            || !refinement.insertions.is_empty())
    {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidInput,
            "already converged input must remain unchanged",
        ));
    }
    validate_mutation_lineage(input, refinement, options)?;
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    validate_conforming_volume_topology(
        &refinement.topology,
        &protected_faces,
        options.step.insertion,
        cancellation,
    )
    .map_err(insertion_error)?;
    validate_delaunay_volume_quality(
        &refinement.topology,
        input.metric_request,
        input.provenance,
        &refinement.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    Ok(())
}

fn validate_options(
    options: DelaunayVolumeRefinementOptions,
) -> Result<(), DelaunayVolumeRefinementStepError> {
    if options.maximum_insertions == 0 {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidOptions,
            "maximum volume-refinement insertions must be nonzero",
        ));
    }
    super::super::insertion::validate_options(options.step.insertion).map_err(insertion_error)?;
    Ok(())
}

fn validate_mutation_lineage(
    input: DelaunayVolumeRefinementInput<'_>,
    refinement: &DelaunayVolumeRefinement,
    options: DelaunayVolumeRefinementOptions,
) -> Result<(), DelaunayVolumeRefinementStepError> {
    if refinement.insertions.len() as u64 > options.maximum_insertions {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
            "refinement mutation evidence exceeds its configured hard limits",
        ));
    }
    let mut nodes = input
        .topology
        .nodes
        .iter()
        .map(|node| (node.identity, *node))
        .collect::<BTreeMap<_, _>>();
    for node in &refinement.insertions {
        if node.identity == StableDigest::ZERO
            || node.coordinates_m.iter().any(|value| !value.is_finite())
            || nodes.insert(node.identity, *node).is_some()
        {
            return Err(invalid_lineage());
        }
    }
    if nodes.values().copied().collect::<Vec<_>>() != refinement.topology.nodes {
        return Err(invalid_lineage());
    }
    Ok(())
}

fn invalid_lineage() -> DelaunayVolumeRefinementStepError {
    error(
        DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
        "refinement topology does not exactly match its ordered insertion lineage",
    )
}
