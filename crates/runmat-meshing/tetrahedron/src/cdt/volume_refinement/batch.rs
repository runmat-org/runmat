use std::{cmp::Ordering, collections::BTreeSet};

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    construct_candidate,
    insertion::{candidate_error, error, insertion_error, quality_error},
    CandidateWork, DelaunayVolumeRefinementInput, DelaunayVolumeRefinementStep,
    DelaunayVolumeRefinementStepError, DelaunayVolumeRefinementStepErrorKind,
    DelaunayVolumeRefinementStepOptions,
};
use crate::cdt::{
    insertion::{
        apply_prepared_delaunay_volume_node_insertion,
        prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint, PreparedInsertion,
    },
    mutation::TopologyMutation,
    volume_quality::{
        evaluate_delaunay_volume_quality_from_admitted_input, evaluate_tetrahedron_geometry,
    },
    DelaunayInsertionErrorKind, DelaunayVolumeNode, DelaunayVolumeQualityErrorKind,
    DelaunayVolumeRefinementCandidateErrorKind,
};

const MAXIMUM_SOURCE_CANDIDATE_ATTEMPTS: usize = 16;

pub(super) struct RefinementBatch {
    pub(super) step: DelaunayVolumeRefinementStep,
    pub(super) inserted_nodes: Vec<DelaunayVolumeNode>,
}

pub(super) enum RefinementBatchOutcome {
    Admitted(Box<RefinementBatch>),
    Unavailable,
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
    options: DelaunayVolumeRefinementStepOptions,
    excluded_node_identities: &mut BTreeSet<StableDigest>,
    maximum_insertions: usize,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<RefinementBatchOutcome, DelaunayVolumeRefinementStepError> {
    let span = tracing::info_span!(
        "meshing.tetrahedron.refinement_batch",
        maximum_insertions,
        tetrahedron_count = input.topology.tetrahedra.len(),
    );
    let _guard = span.enter();
    if maximum_insertions < 2 || worst_violation(input.quality) == f64::MAX {
        return Ok(RefinementBatchOutcome::Unavailable);
    }
    let mut sources = input
        .quality
        .tetrahedra
        .iter()
        .filter(|quality| quality.requires_refinement())
        .collect::<Vec<_>>();
    sources.sort_by(|left, right| {
        crate::cdt::volume_quality::compare_violation_priority(
            right.refinement_violation_ratio,
            crate::cdt::volume_quality::tetrahedron_geometry_key(
                input.topology,
                right.node_identities,
            ),
            right.node_identities,
            left.refinement_violation_ratio,
            crate::cdt::volume_quality::tetrahedron_geometry_key(
                input.topology,
                left.node_identities,
            ),
            left.node_identities,
        )
    });
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<BTreeSet<_>>();
    let exact_boundary_nodes = protected_faces
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let source_indices = input
        .topology
        .tetrahedra
        .iter()
        .enumerate()
        .map(|(index, tetrahedron)| {
            (
                tetrahedron
                    .vertex_indices
                    .map(|vertex| input.topology.nodes[vertex as usize].identity),
                index as u32,
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut claimed_tetrahedra = BTreeSet::new();
    let mut candidate_locations = BTreeSet::new();
    let mut prepared = Vec::new();
    let mut work = CandidateWork::new(options.candidate, cancellation);
    'sources: for source in sources {
        if prepared.len() >= maximum_insertions {
            break;
        }
        for _ in 0..MAXIMUM_SOURCE_CANDIDATE_ATTEMPTS {
            let candidate = match construct_candidate(
                input.topology,
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
            let Some(source_index) = source_indices
                .get(&candidate.source_node_identities)
                .copied()
            else {
                return Err(error(
                    DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
                    "refinement source tetrahedron is absent from the admitted topology",
                ));
            };
            let insertion =
                match prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint(
                    input.topology,
                    candidate.node,
                    &protected_faces,
                    true,
                    Some(source_index),
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
                        excluded_node_identities.insert(candidate.node.identity);
                        continue;
                    }
                    Err(failure) => return Err(insertion_error(failure)),
                };
            let prepared_node = insertion.node;
            if !prepared_cavity_meets_quality_floor(
                input.topology,
                &candidate,
                &insertion,
                &exact_boundary_nodes,
                input.quality_options,
            ) {
                excluded_node_identities.insert(candidate.node.identity);
                continue;
            }
            if insertion
                .removed
                .iter()
                .any(|tetrahedron| claimed_tetrahedra.contains(tetrahedron))
            {
                continue 'sources;
            }
            let location = candidate.node.coordinates_m.map(f64::to_bits);
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
    if prepared.len() < 2 {
        return Ok(RefinementBatchOutcome::Unavailable);
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
    if inserted_nodes.len() < 2 {
        return Ok(RefinementBatchOutcome::Unavailable);
    }
    let topology = mutation
        .snapshot(options.insertion.topology, cancellation)
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
    if !quality_strictly_improves(input.quality, &quality)
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

fn prepared_cavity_meets_quality_floor(
    topology: &crate::cdt::DelaunayVolumeTopology,
    candidate: &crate::cdt::DelaunayVolumeRefinementCandidate,
    insertion: &PreparedInsertion,
    exact_boundary_nodes: &BTreeSet<StableDigest>,
    options: crate::cdt::DelaunayVolumeQualityOptions,
) -> bool {
    insertion.boundary.iter().all(|face| {
        let points = [
            topology.nodes[face[0] as usize].coordinates_m,
            topology.nodes[face[1] as usize].coordinates_m,
            topology.nodes[face[2] as usize].coordinates_m,
            candidate.node.coordinates_m,
        ];
        let Some(geometry) = evaluate_tetrahedron_geometry(points, candidate.resolved_metric)
        else {
            return false;
        };
        let size_violation =
            geometry.maximum_metric_edge_length / options.maximum_metric_edge_length;
        let boundary_node_count = face
            .iter()
            .filter(|index| {
                exact_boundary_nodes.contains(&topology.nodes[**index as usize].identity)
            })
            .count();
        let violation = if boundary_node_count == 0 {
            size_violation
                .max(geometry.metric_radius_edge_ratio / options.maximum_radius_edge_ratio)
        } else {
            size_violation
        };
        let jacobian_violation =
            options.minimum_metric_scaled_jacobian / geometry.metric_scaled_jacobian;
        violation.is_finite()
            && jacobian_violation.is_finite()
            && crate::cdt::volume_quality::violation_priority_key(jacobian_violation)
                <= crate::cdt::volume_quality::violation_priority_key(1.0)
            && crate::cdt::volume_quality::violation_priority_key(violation)
                <= crate::cdt::volume_quality::violation_priority_key(
                    candidate.source_violation_ratio,
                )
    })
}

pub(super) fn quality_strictly_improves(
    prior: &crate::cdt::DelaunayVolumeQuality,
    resulting: &crate::cdt::DelaunayVolumeQuality,
) -> bool {
    compare_violation_spectra(resulting, prior) == Ordering::Less
}

fn compare_violation_spectra(
    left: &crate::cdt::DelaunayVolumeQuality,
    right: &crate::cdt::DelaunayVolumeQuality,
) -> Ordering {
    let spectrum = |quality: &crate::cdt::DelaunayVolumeQuality| {
        let mut values = quality
            .tetrahedra
            .iter()
            .filter(|tetrahedron| tetrahedron.requires_refinement())
            .map(|tetrahedron| tetrahedron.refinement_violation_ratio)
            .collect::<Vec<_>>();
        values.sort_by(|left, right| {
            crate::cdt::volume_quality::violation_priority_key(*right)
                .cmp(&crate::cdt::volume_quality::violation_priority_key(*left))
        });
        values
    };
    let left = spectrum(left);
    let right = spectrum(right);
    for (left, right) in left.iter().zip(&right) {
        let ordering = crate::cdt::volume_quality::violation_priority_key(*left)
            .cmp(&crate::cdt::volume_quality::violation_priority_key(*right));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
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
