mod checkpoint;
use checkpoint::emit_progress_checkpoint;

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

mod entry;
pub use entry::refine_delaunay_volume;
#[cfg(test)]
pub(in crate::cdt) use entry::refine_owned_until_stalled;
pub(in crate::cdt) use entry::{DelaunayVolumeRefinementProgress, RefinementSchedule};

pub(in crate::cdt) fn refine_owned_with_schedule(
    mut topology: crate::cdt::DelaunayVolumeTopology,
    mut quality: crate::cdt::DelaunayVolumeQuality,
    context: super::context::RefinementContext<'_>,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
    schedule: RefinementSchedule,
) -> Result<DelaunayVolumeRefinementProgress, DelaunayVolumeRefinementStepError> {
    validate_options(options)?;
    let mut insertions = Vec::new();
    let mut unusable_node_identities = BTreeSet::new();
    let mut insertion_count = 0u64;
    let mut maximum_batch_size = MAXIMUM_REFINEMENT_BATCH_SIZE;
    let mut next_progress_insertion = PROGRESS_INSERTION_INTERVAL;
    emit_progress_checkpoint(insertion_count, &topology, &quality, "started");
    if quality.worst_refinement_tetrahedron.is_none() {
        emit_progress_checkpoint(insertion_count, &topology, &quality, "completed");
        return Ok(DelaunayVolumeRefinementProgress {
            refinement: DelaunayVolumeRefinement {
                topology,
                quality,
                insertions,
            },
            stalled: false,
        });
    }
    let mut queries = super::CandidateQueries::new(&topology, cancellation)
        .map_err(super::insertion::candidate_error)?;
    let mut frontier = super::batch::sources::Frontier::default();
    loop {
        let current = context.with_state(&topology, &quality);
        let remaining_insertions = options.maximum_insertions.saturating_sub(insertion_count);
        if remaining_insertions >= 2 {
            let maximum_batch = remaining_insertions.min(maximum_batch_size) as usize;
            let prior_frontier = frontier;
            match insert_independent_refinement_batch(
                current,
                &queries,
                super::batch::RefinementBatchOptions {
                    step: options.step,
                    maximum_insertions: maximum_batch,
                    frontier: &mut frontier,
                },
                &mut unusable_node_identities,
                cancellation,
            )? {
                RefinementBatchOutcome::Admitted(batch) => {
                    queries
                        .append(&batch.inserted_nodes, cancellation)
                        .map_err(super::insertion::candidate_error)?;
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
                    if schedule == RefinementSchedule::Cooperative {
                        return Ok(DelaunayVolumeRefinementProgress {
                            refinement: DelaunayVolumeRefinement {
                                topology,
                                quality,
                                insertions,
                            },
                            stalled: false,
                        });
                    }
                    continue;
                }
                RefinementBatchOutcome::MoreSources => continue,
                RefinementBatchOutcome::RetrySmaller if maximum_batch > 2 => {
                    frontier = prior_frontier;
                    maximum_batch_size = ((maximum_batch / 2).max(2)) as u64;
                    continue;
                }
                RefinementBatchOutcome::Unavailable | RefinementBatchOutcome::RetrySmaller => {}
            }
        }
        // A shape defect can block every floor-preserving batch candidate.
        // Hand it back to the shape owner; this is an intermediate checkpoint,
        // never a convergence or publication verdict.
        if schedule == RefinementSchedule::Cooperative
            && quality.worst_optimization_tetrahedron.is_some()
        {
            return Ok(DelaunayVolumeRefinementProgress {
                refinement: DelaunayVolumeRefinement {
                    topology,
                    quality,
                    insertions,
                },
                stalled: false,
            });
        }
        let candidate =
            match select_delaunay_volume_refinement_candidate_excluding_from_admitted_input(
                current,
                options.step.candidate,
                &unusable_node_identities,
                cancellation,
                &queries,
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
        queries
            .append(std::slice::from_ref(&candidate.node), cancellation)
            .map_err(super::insertion::candidate_error)?;
        insertions.push(candidate.node);
        topology = step.topology;
        quality = step.quality;
        unusable_node_identities.clear();
        if schedule == RefinementSchedule::Cooperative {
            return Ok(DelaunayVolumeRefinementProgress {
                refinement: DelaunayVolumeRefinement {
                    topology,
                    quality,
                    insertions,
                },
                stalled: false,
            });
        }
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
            || node
                .coordinates_m
                .canonical_value()
                .iter()
                .any(|value| !value.is_finite())
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
