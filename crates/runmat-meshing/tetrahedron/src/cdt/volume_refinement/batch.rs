pub(super) mod sources;
use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    construct_candidate,
    insertion::{candidate_error, error, insertion_error, quality_error},
    CandidateQueries, CandidateWork, DelaunayVolumeRefinementInput, DelaunayVolumeRefinementStep,
    DelaunayVolumeRefinementStepError, DelaunayVolumeRefinementStepErrorKind,
    DelaunayVolumeRefinementStepOptions,
};
use crate::cdt::{
    insertion::{
        apply_prepared_delaunay_volume_node_insertion,
        prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint, PreparedInsertion,
    },
    mutation::TopologyMutation,
    volume_quality::evaluate_delaunay_volume_quality_from_admitted_input,
    DelaunayInsertionErrorKind, DelaunayVolumeNode, DelaunayVolumeQualityErrorKind,
    DelaunayVolumeRefinementCandidateErrorKind,
};

const MAXIMUM_SOURCE_CANDIDATE_ATTEMPTS: usize = 16;

pub(super) struct RefinementBatchOptions<'a> {
    pub(super) step: DelaunayVolumeRefinementStepOptions,
    pub(super) maximum_insertions: usize,
    pub(super) frontier: &'a mut sources::Frontier,
}

pub(super) struct RefinementBatch {
    pub(super) step: DelaunayVolumeRefinementStep,
    pub(super) inserted_nodes: Vec<DelaunayVolumeNode>,
}

pub(super) enum RefinementBatchOutcome {
    Admitted(Box<RefinementBatch>),
    Unavailable,
    MoreSources,
    RetrySmaller,
}

struct PreparedRefinement {
    node: DelaunayVolumeNode,
    insertion: PreparedInsertion,
}

/// Applies a deterministic independent set of refinement points in one mutable
/// transaction. Each insertion cavity is prepared against the same admitted
/// topology. Selected cavities are separated by at least one tetrahedron layer,
/// so their replacements commute; one constrained legalization pass then admits
/// the combined topology.
pub(super) fn insert_independent_refinement_batch(
    input: DelaunayVolumeRefinementInput<'_>,
    queries: &CandidateQueries,
    options: RefinementBatchOptions<'_>,
    excluded_node_identities: &mut BTreeSet<StableDigest>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<RefinementBatchOutcome, DelaunayVolumeRefinementStepError> {
    let RefinementBatchOptions {
        step: options,
        maximum_insertions,
        frontier,
    } = options;
    let span = tracing::info_span!(
        "meshing.tetrahedron.refinement_batch",
        maximum_insertions,
        tetrahedron_count = input.topology.tetrahedra.len(),
    );
    let _guard = span.enter();
    if maximum_insertions < 2
        || options.candidate.maximum_candidate_evaluations < 2
        || worst_violation(input.quality) == f64::MAX
    {
        return Ok(RefinementBatchOutcome::Unavailable);
    }
    // A batch is a bounded scheduling window, not a demand to fill every slot.
    // Remaining violating cells stay in the accepted quality inventory.
    let sources = sources::select_window(input, maximum_insertions, frontier, cancellation)?;
    let source_count = sources.len();
    let mut examined_sources = 0usize;
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<BTreeSet<_>>();
    let mut claimed_tetrahedra = BTreeSet::new();
    let mut candidate_locations = BTreeSet::new();
    let mut prepared = Vec::new();
    let indexed_topology = queries.topology(input.topology);
    let mut attempts = 0u64;
    let mut insertion_rejections = 0u64;
    let mut overlap_rejections = 0u64;
    let mut work = CandidateWork::new(options.candidate, cancellation);
    'sources: for source in sources {
        if prepared.len() >= maximum_insertions {
            break;
        }
        if !work.can_construct() {
            break;
        }
        frontier.advance(input, source);
        examined_sources += 1;
        for _ in 0..MAXIMUM_SOURCE_CANDIDATE_ATTEMPTS {
            if !work.can_construct() {
                break 'sources;
            }
            attempts += 1;
            if attempts.is_multiple_of(1024) {
                tracing::info!(
                    attempts,
                    insertion_rejections,
                    overlap_rejections,
                    selected = prepared.len(),
                    "tetrahedral refinement batch search progress"
                );
            }
            let candidate = match construct_candidate(
                input.topology,
                queries,
                source,
                input.quality_options,
                input.fixed_node_identities,
                excluded_node_identities,
                &mut work,
            ) {
                Ok(candidate) => candidate,
                Err(failure)
                    if matches!(
                        failure.kind,
                        DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure
                            | DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate
                    ) =>
                {
                    continue 'sources;
                }
                Err(failure) => return Err(candidate_error(failure)),
            };
            let Some((source_index, _)) = crate::cdt::topology::queries::find_oriented_tetrahedron(
                input.topology,
                candidate.source_node_identities,
            ) else {
                return Err(error(
                    DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
                    "refinement source tetrahedron is absent from the admitted topology",
                ));
            };
            let insertion =
                match prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint(
                    &indexed_topology,
                    candidate.node,
                    &protected_faces,
                    true,
                    Some(source_index as u32),
                    options.insertion,
                    cancellation,
                ) {
                    Ok(insertion) => insertion,
                    Err(failure)
                        if matches!(
                            failure.kind,
                            DelaunayInsertionErrorKind::InvalidNode
                                | DelaunayInsertionErrorKind::InvalidTopology
                                | DelaunayInsertionErrorKind::PointOutsideTopology
                        ) =>
                    {
                        insertion_rejections += 1;
                        if insertion_rejections <= 3 {
                            tracing::info!(reason = %failure, "tetrahedral refinement insertion rejected");
                        }
                        excluded_node_identities.insert(candidate.node.identity);
                        continue;
                    }
                    Err(failure) => return Err(insertion_error(failure)),
                };
            let prepared_node = insertion.node;
            if insertion
                .removed
                .iter()
                .any(|tetrahedron| claimed_tetrahedra.contains(tetrahedron))
            {
                overlap_rejections += 1;
                continue 'sources;
            }
            let location = candidate.node.coordinates_m.canonical_value().map(|value| {
                if value == 0.0 {
                    0
                } else {
                    value.to_bits()
                }
            });
            if !candidate_locations.insert(location) {
                excluded_node_identities.insert(candidate.node.identity);
                continue;
            }
            for tetrahedron in &insertion.removed {
                claimed_tetrahedra.insert(*tetrahedron);
                let source = &input.topology.tetrahedra[*tetrahedron as usize];
                claimed_tetrahedra.extend(source.neighbors.into_iter().flatten());
            }
            prepared.push(PreparedRefinement {
                node: prepared_node,
                insertion,
            });
            continue 'sources;
        }
    }
    if prepared.is_empty() {
        return Ok(if frontier.has_more() || examined_sources < source_count {
            RefinementBatchOutcome::MoreSources
        } else {
            RefinementBatchOutcome::Unavailable
        });
    }
    tracing::info!(
        candidate_count = prepared.len(),
        tetrahedron_count = input.topology.tetrahedra.len(),
        "tetrahedral refinement batch selected"
    );

    let mut mutation = TopologyMutation::from_canonical(input.topology.clone());
    let mut inserted_nodes = Vec::with_capacity(prepared.len());
    for candidate in prepared {
        match apply_prepared_delaunay_volume_node_insertion(
            &mut mutation,
            candidate.insertion,
            &protected_faces,
            true,
            options.insertion,
            cancellation,
        ) {
            Ok(()) => {}
            Err(failure)
                if matches!(
                    failure.kind,
                    DelaunayInsertionErrorKind::InvalidNode
                        | DelaunayInsertionErrorKind::InvalidTopology
                        | DelaunayInsertionErrorKind::PointOutsideTopology
                ) =>
            {
                return Ok(RefinementBatchOutcome::RetrySmaller);
            }
            Err(failure) => return Err(insertion_error(failure)),
        }
        inserted_nodes.push(candidate.node);
    }
    tracing::info!(
        accepted_count = inserted_nodes.len(),
        "tetrahedral refinement batch insertions completed"
    );
    if inserted_nodes.is_empty() {
        return Ok(RefinementBatchOutcome::Unavailable);
    }
    // Rejection retains input.topology; this owned candidate no longer needs
    // its mutation indexes during canonical output and quality admission.
    let topology = mutation
        .freeze(options.insertion.topology, cancellation)
        .map_err(crate::cdt::insertion::mutation_topology_error)
        .map_err(insertion_error)?;
    let quality = match evaluate_delaunay_volume_quality_from_admitted_input(
        &topology,
        input.metric_request,
        input.provenance,
        input.quality_options,
        cancellation,
    ) {
        Ok(quality) => quality,
        Err(failure)
            if matches!(
                failure.kind,
                DelaunayVolumeQualityErrorKind::NumericalFailure
                    | DelaunayVolumeQualityErrorKind::InvalidQuality
            ) =>
        {
            tracing::info!(
                accepted_count = inserted_nodes.len(),
                reason = %failure,
                "tetrahedral refinement batch rolled back after quality evaluation"
            );
            return Ok(RefinementBatchOutcome::RetrySmaller);
        }
        Err(failure) => return Err(quality_error(failure)),
    };
    let prior_worst = worst_violation(input.quality);
    let resulting_worst = worst_violation(&quality);
    if !super::admission::canonical_progress(input.quality, &quality, cancellation)?
        || creates_new_all_boundary_violation(input, &quality)
    {
        tracing::info!(
            accepted_count = inserted_nodes.len(),
            prior_worst,
            resulting_worst,
            "tetrahedral refinement batch rejected"
        );
        return Ok(RefinementBatchOutcome::RetrySmaller);
    }
    if inserted_nodes.iter().any(|node| {
        !quality
            .tetrahedra
            .iter()
            .any(|tetrahedron| tetrahedron.node_identities.contains(&node.identity))
    }) {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
            "batched refinement produced an inserted node without an incident tetrahedron",
        ));
    }
    tracing::info!(
        accepted_count = inserted_nodes.len(),
        prior_worst,
        resulting_worst,
        "tetrahedral refinement batch admitted"
    );
    Ok(RefinementBatchOutcome::Admitted(Box::new(
        RefinementBatch {
            step: DelaunayVolumeRefinementStep { topology, quality },
            inserted_nodes,
        },
    )))
}

#[cfg(test)]
pub(super) fn quality_strictly_improves(
    prior: &crate::cdt::DelaunayVolumeQuality,
    resulting: &crate::cdt::DelaunayVolumeQuality,
) -> bool {
    super::admission::compare(
        prior,
        resulting,
        crate::cdt::volume_quality::QualityObjective::Refinement,
        &runmat_meshing_core::NeverCancelled,
    )
    .unwrap()
        == std::cmp::Ordering::Less
}

fn worst_violation(quality: &crate::cdt::DelaunayVolumeQuality) -> f64 {
    quality
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.refinement_violation_ratio)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

fn creates_new_all_boundary_violation(
    input: DelaunayVolumeRefinementInput<'_>,
    quality: &crate::cdt::DelaunayVolumeQuality,
) -> bool {
    let boundary_nodes = input
        .provenance
        .facets
        .iter()
        .flat_map(|facet| facet.node_identities)
        .collect::<BTreeSet<_>>();
    let prior = input
        .quality
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.node_identities)
        .collect::<BTreeSet<_>>();
    quality.tetrahedra.iter().any(|tetrahedron| {
        tetrahedron.requires_refinement()
            && tetrahedron
                .node_identities
                .iter()
                .all(|identity| boundary_nodes.contains(identity))
            && !prior.contains(&tetrahedron.node_identities)
    })
}
