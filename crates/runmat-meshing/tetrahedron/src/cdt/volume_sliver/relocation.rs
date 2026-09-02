use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{predicate::tetrahedron_volume, MeshingCancellationSignal, StableDigest};
use sha2::{Digest, Sha256};

use super::admission::{evaluate_candidate, CandidateAdmissionPolicy, CandidateTopologyContract};
use super::retriangulate::{retriangulate_deleted_vertex, retriangulate_relocated_vertex};
use super::work::OptimizationWork;
use super::{
    error, DelaunayVolumeSliverDeletion, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind,
    DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions, DelaunayVolumeSliverRelocation,
    SliverSearchPolicy, CAVITY_SMOOTHING_STEPS, MAXIMUM_CAVITY_SEARCH_DIRECTIONS,
    MAXIMUM_CAVITY_SMOOTHING_ITERATIONS,
};
use crate::cdt::{
    topology::{
        build_delaunay_volume_topology_with_regions,
        orient_materialized_volume_nodes as orient_volume_nodes,
    },
    DelaunayTetrahedronQuality, DelaunayTopologyErrorKind, DelaunayVolumeNode,
    DelaunayVolumeProvenance, DelaunayVolumeQuality, DelaunayVolumeRefinementInput,
    DelaunayVolumeTopology,
};

const RELOCATION_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/sliver-relocation/1\0";

pub(super) struct RelocationCandidate {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) quality: DelaunayVolumeQuality,
    pub(super) mutation: DelaunayVolumeSliverMutation,
}

pub(super) struct RelocationSearch {
    pub(super) candidate: Option<RelocationCandidate>,
    pub(super) rejection_counts: std::collections::BTreeMap<&'static str, u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RelocationSearchScope {
    /// Search every supported topological and geometric relocation.
    Complete,
    /// Keep the current cavity connectivity. Compound searches use this scope so a bounded
    /// lookahead cannot recursively launch another global Delaunay legalization.
    FixedStarOnly,
}

pub(super) fn relocation_candidates(
    input: DelaunayVolumeRefinementInput<'_>,
    source: &DelaunayTetrahedronQuality,
    current_spectrum: &[f64],
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    scope: RelocationSearchScope,
) -> Result<RelocationSearch, DelaunayVolumeSliverError> {
    let SliverSearchPolicy {
        options,
        refinement: refinement_policy,
        objective,
    } = policy;
    let topology = input.topology;
    let protected = protected_nodes(topology, input.provenance, input.fixed_node_identities);
    let boundary = topology
        .incidence
        .boundary_facets
        .iter()
        .flat_map(|facet| facet.vertex_indices)
        .map(|index| topology.nodes[index as usize].identity)
        .collect::<BTreeSet<_>>();
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    let mut source_nodes = source.node_identities;
    source_nodes.sort_unstable();
    let eligible_sources = source_nodes
        .into_iter()
        .filter(|identity| !protected.contains(identity) && !boundary.contains(identity))
        .collect::<Vec<_>>();
    const FRACTIONS: [f64; 4] = [1.0, 0.5, 0.25, 0.125];
    let mut rejection_counts = std::collections::BTreeMap::new();

    // A conforming interior-edge collapse is the least disruptive topological
    // repair for a free volume vertex. In particular, it completes the standard
    // split-then-collapse escape for slivers whose edge star cannot be improved
    // by a direct swap. Both endpoints must be unconstrained volume vertices;
    // the topology and PLC admission below independently enforce the link
    // condition, region separation, and preservation of every protected facet.
    for source_identity in eligible_sources
        .iter()
        .copied()
        .filter(|_| matches!(scope, RelocationSearchScope::Complete))
    {
        let source_index = topology
            .nodes
            .binary_search_by_key(&source_identity, |node| node.identity)
            .map_err(|_| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "sliver source node is absent from canonical topology",
                )
            })?;
        let mut targets = topology.incidence.vertex_stars[source_index]
            .iter()
            .flat_map(|tetrahedron| topology.tetrahedra[*tetrahedron as usize].vertex_indices)
            .map(|vertex| topology.nodes[vertex as usize].identity)
            .filter(|identity| {
                *identity != source_identity
                    && !protected.contains(identity)
                    && !boundary.contains(identity)
            })
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        for target_identity in targets {
            work.evaluate()?;
            let Some(candidate_topology) = collapse_interior_edge(
                topology,
                source_index,
                target_identity,
                options,
                cancellation,
            )?
            else {
                reject(
                    &mut rejection_counts,
                    "interior_edge_collapse_not_admissible",
                );
                continue;
            };
            let Some(quality) = evaluate_candidate(
                &candidate_topology,
                input,
                &protected_faces,
                current_spectrum,
                CandidateAdmissionPolicy {
                    topology_contract: CandidateTopologyContract::CertifiedConformingCavity,
                    refinement: refinement_policy,
                    objective,
                    options,
                },
                cancellation,
                &mut rejection_counts,
            )?
            else {
                continue;
            };
            return Ok(RelocationSearch {
                candidate: Some(RelocationCandidate {
                    topology: candidate_topology,
                    quality,
                    mutation: DelaunayVolumeSliverMutation::Deleted(DelaunayVolumeSliverDeletion {
                        source_node_identity: source_identity,
                        source_tetrahedron_node_identities: source.node_identities,
                    }),
                }),
                rejection_counts,
            });
        }
    }

    // If no adjacent volume vertex is a legal collapse target, refill the
    // complete deleted-vertex cavity before searching relocation coordinates.
    for source_identity in eligible_sources
        .iter()
        .copied()
        .filter(|_| matches!(scope, RelocationSearchScope::Complete))
    {
        let Some(source_index) = topology
            .nodes
            .binary_search_by_key(&source_identity, |node| node.identity)
            .ok()
        else {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "sliver source node is absent from canonical topology",
            ));
        };
        work.evaluate()?;
        if let Some(candidate_topology) = retriangulate_deleted_vertex(
            topology,
            source_index,
            &protected_faces,
            options.insertion,
            cancellation,
        )? {
            if let Some(quality) = evaluate_candidate(
                &candidate_topology,
                input,
                &protected_faces,
                current_spectrum,
                CandidateAdmissionPolicy {
                    topology_contract: CandidateTopologyContract::Delaunay,
                    refinement: refinement_policy,
                    objective,
                    options,
                },
                cancellation,
                &mut rejection_counts,
            )? {
                return Ok(RelocationSearch {
                    candidate: Some(RelocationCandidate {
                        topology: candidate_topology,
                        quality,
                        mutation: DelaunayVolumeSliverMutation::Deleted(
                            DelaunayVolumeSliverDeletion {
                                source_node_identity: source_identity,
                                source_tetrahedron_node_identities: source.node_identities,
                            },
                        ),
                    }),
                    rejection_counts,
                });
            }
        } else {
            reject(&mut rejection_counts, "vertex_star_could_not_be_refilled");
        }
    }
    for source_identity in eligible_sources {
        let source_index = topology
            .nodes
            .binary_search_by_key(&source_identity, |node| node.identity)
            .map_err(|_| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "sliver source node is absent from canonical topology",
                )
            })?;
        let source_coordinates = topology.nodes[source_index].coordinates_m;
        let mut candidate_coordinates = smooth_vertex_star_points(
            topology,
            source_index,
            source.resolved_metric,
            cancellation,
        )?;
        let targets = [
            sliver_height_target(
                topology,
                source,
                source_identity,
                input.quality_options.minimum_metric_scaled_jacobian,
            )?,
            Some(volume_weighted_star_centroid(topology, source_index)?),
            Some(neighbor_centroid(topology, source_index)?),
        ];
        for target in targets.into_iter().flatten() {
            for fraction in FRACTIONS {
                let coordinates = std::array::from_fn(|axis| {
                    source_coordinates[axis] + fraction * (target[axis] - source_coordinates[axis])
                });
                if !candidate_coordinates.contains(&coordinates) {
                    candidate_coordinates.push(coordinates);
                }
            }
        }
        for coordinates_m in candidate_coordinates {
            work.evaluate()?;
            let replacement_node =
                DelaunayVolumeNode::explicit(replacement_identity(source_identity), coordinates_m);
            if let Some(candidate_topology) = rebuild_with_relocated_node(
                topology,
                source_index,
                replacement_node,
                options,
                cancellation,
            )? {
                if let Some(candidate_quality) = evaluate_candidate(
                    &candidate_topology,
                    input,
                    &protected_faces,
                    current_spectrum,
                    CandidateAdmissionPolicy {
                        topology_contract: CandidateTopologyContract::CertifiedConformingCavity,
                        refinement: refinement_policy,
                        objective,
                        options,
                    },
                    cancellation,
                    &mut rejection_counts,
                )? {
                    return Ok(RelocationSearch {
                        candidate: Some(RelocationCandidate {
                            topology: candidate_topology,
                            quality: candidate_quality,
                            mutation: relocation_mutation(
                                source_identity,
                                replacement_node,
                                source.node_identities,
                            ),
                        }),
                        rejection_counts,
                    });
                }
            } else {
                reject(
                    &mut rejection_counts,
                    "fixed_star_relocation_not_admissible",
                );
            }
            if matches!(scope, RelocationSearchScope::FixedStarOnly) {
                continue;
            }
            let Some(candidate_topology) = retriangulate_relocated_vertex(
                topology,
                source_index,
                replacement_node,
                &protected_faces,
                options.insertion,
                cancellation,
            )?
            else {
                reject(
                    &mut rejection_counts,
                    "relocated_vertex_star_could_not_be_refilled",
                );
                continue;
            };
            let Some(candidate_quality) = evaluate_candidate(
                &candidate_topology,
                input,
                &protected_faces,
                current_spectrum,
                CandidateAdmissionPolicy {
                    topology_contract: CandidateTopologyContract::Delaunay,
                    refinement: refinement_policy,
                    objective,
                    options,
                },
                cancellation,
                &mut rejection_counts,
            )?
            else {
                continue;
            };
            return Ok(RelocationSearch {
                candidate: Some(RelocationCandidate {
                    topology: candidate_topology,
                    quality: candidate_quality,
                    mutation: relocation_mutation(
                        source_identity,
                        replacement_node,
                        source.node_identities,
                    ),
                }),
                rejection_counts,
            });
        }
    }
    Ok(RelocationSearch {
        candidate: None,
        rejection_counts,
    })
}

fn collapse_interior_edge(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    target_identity: StableDigest,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    let target_index = topology
        .nodes
        .binary_search_by_key(&target_identity, |node| node.identity)
        .map_err(|_| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "interior edge-collapse target is absent from canonical topology",
            )
        })?;
    if source_index == target_index {
        return Ok(None);
    }
    let nodes = topology
        .nodes
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != source_index)
        .map(|(_, node)| *node)
        .collect::<Vec<_>>();
    let target_after_removal = target_index - usize::from(target_index > source_index);
    let remap = |vertex: u32| {
        let vertex = vertex as usize;
        if vertex == source_index {
            target_after_removal as u32
        } else {
            (vertex - usize::from(vertex > source_index)) as u32
        }
    };
    let mut unique = BTreeMap::new();
    for tetrahedron in &topology.tetrahedra {
        let vertices = tetrahedron.vertex_indices.map(remap);
        let mut key = vertices;
        key.sort_unstable();
        if key.windows(2).any(|pair| pair[0] == pair[1]) {
            continue;
        }
        match unique.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((vertices, tetrahedron.region_id.clone()));
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get().1 != tetrahedron.region_id {
                    return Ok(None);
                }
            }
        }
    }
    let tetrahedra = unique.into_values().collect::<Vec<_>>();
    match build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(topology) => Ok(Some(topology)),
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayTopologyErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => Ok(None),
        },
    }
}

fn smooth_vertex_star_points(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    metric: runmat_meshing_size::metric::MetricTensor3,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<[f64; 3]>, DelaunayVolumeSliverError> {
    let star = &topology.incidence.vertex_stars[source_index];
    let mut link_faces = star
        .iter()
        .map(|tetrahedron| {
            let mut face = topology.tetrahedra[*tetrahedron as usize]
                .vertex_indices
                .into_iter()
                .filter(|vertex| *vertex as usize != source_index)
                .collect::<Vec<_>>();
            face.sort_unstable();
            <[u32; 3]>::try_from(face).expect("vertex-star tetrahedron has one source vertex")
        })
        .collect::<Vec<_>>();
    link_faces.sort_unstable();
    link_faces.dedup();
    let start = topology.nodes[source_index].coordinates_m;
    let initial_score =
        relocated_star_minimum_jacobian(topology, star, source_index, start, metric);
    let mut results = Vec::new();
    for step in CAVITY_SMOOTHING_STEPS {
        let mut point = start;
        let mut score = initial_score;
        let mut improved = false;
        for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
            if iteration.is_multiple_of(16) && cancellation.is_cancelled() {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::Cancelled,
                    "vertex-star smoothing was cancelled",
                ));
            }
            let mut best = None;
            for face in link_faces.iter().take(MAXIMUM_CAVITY_SEARCH_DIRECTIONS) {
                let centroid: [f64; 3] = std::array::from_fn(|axis| {
                    face.iter()
                        .map(|vertex| topology.nodes[*vertex as usize].coordinates_m[axis])
                        .sum::<f64>()
                        / 3.0
                });
                let candidate =
                    std::array::from_fn(|axis| point[axis] + step * (centroid[axis] - point[axis]));
                if !point_is_in_vertex_star_kernel(topology, star, source_index, candidate)? {
                    continue;
                }
                let candidate_score = relocated_star_minimum_jacobian(
                    topology,
                    star,
                    source_index,
                    candidate,
                    metric,
                );
                if candidate_score <= score {
                    continue;
                }
                if best
                    .as_ref()
                    .is_none_or(|(best_score, best_point): &(f64, [f64; 3])| {
                        candidate_score > *best_score
                            || candidate_score == *best_score
                                && candidate.map(f64::to_bits) < best_point.map(f64::to_bits)
                    })
                {
                    best = Some((candidate_score, candidate));
                }
            }
            let Some((best_score, best_point)) = best else {
                break;
            };
            score = best_score;
            point = best_point;
            improved = true;
        }
        if improved && !results.contains(&point) {
            results.push(point);
        }
    }
    let mut starts = Vec::with_capacity(results.len() + 1);
    starts.push(start);
    starts.extend(results.iter().copied());
    for start in starts {
        if let Some(point) =
            pattern_search_vertex_star(topology, star, source_index, start, metric, cancellation)?
        {
            if !results.contains(&point) {
                results.push(point);
            }
        }
    }
    Ok(results)
}

/// Deterministic derivative-free maximin smoothing over one fixed vertex star.
///
/// The scaled-Jacobian objective is nonsmooth where the identity of the worst incident
/// tetrahedron changes, so a finite-difference gradient is unreliable. A shrinking 26-direction
/// stencil instead searches the whole local kernel while evaluating only incident tetrahedra.
/// Candidate topology construction and canonical global admission remain the caller's concern.
fn pattern_search_vertex_star(
    topology: &DelaunayVolumeTopology,
    star: &[u32],
    source_index: usize,
    start: [f64; 3],
    metric: runmat_meshing_size::metric::MetricTensor3,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let source = topology.nodes[source_index].coordinates_m;
    let mut neighbors = star
        .iter()
        .flat_map(|tetrahedron| topology.tetrahedra[*tetrahedron as usize].vertex_indices)
        .filter(|vertex| *vertex as usize != source_index)
        .collect::<Vec<_>>();
    neighbors.sort_unstable();
    neighbors.dedup();
    let local_scale = neighbors
        .into_iter()
        .map(|vertex| {
            topology.nodes[vertex as usize]
                .coordinates_m
                .into_iter()
                .zip(source)
                .map(|(left, right)| (left - right) * (left - right))
                .sum::<f64>()
                .sqrt()
        })
        .filter(|distance| distance.is_finite() && *distance > 0.0)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    if local_scale == 0.0 {
        return Ok(None);
    }

    let mut point = start;
    let initial_score =
        relocated_star_minimum_jacobian(topology, star, source_index, point, metric);
    let mut score = initial_score;
    let mut radius = local_scale * 0.25;
    let minimum_radius = local_scale * 2_f64.powi(-24);
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "vertex-star pattern search was cancelled",
            ));
        }
        let mut best = None;
        for x in -1_i8..=1 {
            for y in -1_i8..=1 {
                for z in -1_i8..=1 {
                    if x == 0 && y == 0 && z == 0 {
                        continue;
                    }
                    let norm = f64::from(x * x + y * y + z * z).sqrt();
                    let direction = [
                        f64::from(x) / norm,
                        f64::from(y) / norm,
                        f64::from(z) / norm,
                    ];
                    let candidate =
                        std::array::from_fn(|axis| point[axis] + radius * direction[axis]);
                    if !point_is_in_vertex_star_kernel(topology, star, source_index, candidate)? {
                        continue;
                    }
                    let candidate_score = relocated_star_minimum_jacobian(
                        topology,
                        star,
                        source_index,
                        candidate,
                        metric,
                    );
                    if candidate_score <= score {
                        continue;
                    }
                    if best
                        .as_ref()
                        .is_none_or(|(best_score, best_point): &(f64, [f64; 3])| {
                            candidate_score > *best_score
                                || candidate_score == *best_score
                                    && candidate.map(f64::to_bits) < best_point.map(f64::to_bits)
                        })
                    {
                        best = Some((candidate_score, candidate));
                    }
                }
            }
        }
        if let Some((best_score, best_point)) = best {
            score = best_score;
            point = best_point;
        } else {
            radius *= 0.5;
            if radius < minimum_radius {
                break;
            }
        }
    }
    Ok((score > initial_score).then_some(point))
}

fn point_is_in_vertex_star_kernel(
    topology: &DelaunayVolumeTopology,
    star: &[u32],
    source_index: usize,
    point: [f64; 3],
) -> Result<bool, DelaunayVolumeSliverError> {
    let source = topology.nodes[source_index];
    let probe = DelaunayVolumeNode::explicit(replacement_identity(source.identity), point);
    for tetrahedron in star {
        let face = topology.tetrahedra[*tetrahedron as usize]
            .vertex_indices
            .into_iter()
            .filter(|vertex| *vertex as usize != source_index)
            .map(|vertex| topology.nodes[vertex as usize])
            .collect::<Vec<_>>();
        let face: [DelaunayVolumeNode; 3] = face
            .try_into()
            .expect("vertex-star tetrahedron has one source vertex");
        let expected =
            orient_volume_nodes([face[0], face[1], face[2], source]).map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("vertex-star smoothing predicate failed: {failure:?}"),
                )
            })?;
        let observed =
            orient_volume_nodes([face[0], face[1], face[2], probe]).map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("vertex-star candidate predicate failed: {failure:?}"),
                )
            })?;
        if observed != expected {
            return Ok(false);
        }
    }
    Ok(true)
}

fn relocated_star_minimum_jacobian(
    topology: &DelaunayVolumeTopology,
    star: &[u32],
    source_index: usize,
    point: [f64; 3],
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> f64 {
    star.iter()
        .filter_map(|tetrahedron| {
            let points = topology.tetrahedra[*tetrahedron as usize]
                .vertex_indices
                .map(|vertex| {
                    if vertex as usize == source_index {
                        point
                    } else {
                        topology.nodes[vertex as usize].coordinates_m
                    }
                });
            super::super::volume_quality::evaluate_tetrahedron_geometry(points, metric)
                .map(|quality| quality.metric_scaled_jacobian)
        })
        .min_by(f64::total_cmp)
        .unwrap_or(0.0)
}

fn rebuild_with_relocated_node(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    replacement: DelaunayVolumeNode,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    let mut ordered = topology
        .nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(index, node)| {
            (
                index,
                if index == source_index {
                    replacement
                } else {
                    node
                },
            )
        })
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(_, node)| node.identity);
    if ordered
        .windows(2)
        .any(|pair| pair[0].1.identity >= pair[1].1.identity)
    {
        return Ok(None);
    }
    let mut remap = vec![0_u32; ordered.len()];
    let nodes = ordered
        .into_iter()
        .enumerate()
        .map(|(canonical, (original, node))| {
            remap[original] = canonical as u32;
            node
        })
        .collect::<Vec<_>>();
    let tetrahedra = topology
        .tetrahedra
        .iter()
        .map(|tetrahedron| {
            (
                tetrahedron
                    .vertex_indices
                    .map(|vertex| remap[vertex as usize]),
                tetrahedron.region_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    match build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(topology) => Ok(Some(topology)),
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayTopologyErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => Ok(None),
        },
    }
}

fn relocation_mutation(
    source_node_identity: StableDigest,
    replacement_node: DelaunayVolumeNode,
    source_tetrahedron_node_identities: [StableDigest; 4],
) -> DelaunayVolumeSliverMutation {
    DelaunayVolumeSliverMutation::Relocated(DelaunayVolumeSliverRelocation {
        source_node_identity,
        replacement_node,
        source_tetrahedron_node_identities,
    })
}

fn reject(counts: &mut std::collections::BTreeMap<&'static str, u64>, reason: &'static str) {
    *counts.entry(reason).or_default() += 1;
}

/// Moves one eligible sliver vertex away from its opposite face far enough to make a useful
/// quality change. Fractions of this target are checked by the caller because the rest of the
/// vertex star, protected facets, and the exact Delaunay contract remain authoritative gates.
fn sliver_height_target(
    topology: &DelaunayVolumeTopology,
    source: &DelaunayTetrahedronQuality,
    source_identity: StableDigest,
    minimum_scaled_jacobian: f64,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let source_index = topology
        .nodes
        .binary_search_by_key(&source_identity, |node| node.identity)
        .map_err(|_| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "sliver source node is absent from canonical topology",
            )
        })?;
    let opposite = source
        .node_identities
        .iter()
        .copied()
        .filter(|identity| *identity != source_identity)
        .map(|identity| {
            topology
                .nodes
                .binary_search_by_key(&identity, |node| node.identity)
                .map(|index| topology.nodes[index].coordinates_m)
                .map_err(|_| {
                    error(
                        DelaunayVolumeSliverErrorKind::InvalidTopology,
                        "sliver opposite face is absent from canonical topology",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [first, second, third]: [[f64; 3]; 3] = opposite.try_into().map_err(|_| {
        error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver source identity is absent from its tetrahedron",
        )
    })?;
    let subtract = |left: [f64; 3], right: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|axis| left[axis] - right[axis])
    };
    let first_edge = subtract(second, first);
    let second_edge = subtract(third, first);
    let normal = [
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    ];
    let normal_length = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !normal_length.is_finite() || normal_length == 0.0 {
        return Ok(None);
    }
    let unit_normal = normal.map(|value| value / normal_length);
    let coordinates = topology.nodes[source_index].coordinates_m;
    let signed_height = subtract(coordinates, first)
        .into_iter()
        .zip(unit_normal)
        .map(|(value, normal)| value * normal)
        .sum::<f64>();
    if !signed_height.is_finite() || signed_height == 0.0 {
        return Ok(None);
    }
    let face_scale = [first_edge, second_edge, subtract(third, second)]
        .into_iter()
        .map(|edge| edge.iter().map(|value| value * value).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    let desired_height = (face_scale * minimum_scaled_jacobian).max(signed_height.abs() * 2.0);
    Ok(Some(std::array::from_fn(|axis| {
        coordinates[axis]
            + unit_normal[axis] * signed_height.signum() * (desired_height - signed_height.abs())
    })))
}

fn protected_nodes(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
    fixed_node_identities: &[StableDigest],
) -> BTreeSet<StableDigest> {
    let mut protected = provenance
        .nodes
        .iter()
        .map(|binding| binding.node_identity)
        .collect::<BTreeSet<_>>();
    protected.extend(
        provenance
            .segments
            .iter()
            .flat_map(|binding| binding.node_identities),
    );
    protected.extend(fixed_node_identities.iter().copied());
    protected.extend(
        provenance
            .facets
            .iter()
            .flat_map(|binding| binding.node_identities),
    );
    protected.retain(|identity| {
        topology
            .nodes
            .binary_search_by_key(identity, |node| node.identity)
            .is_ok()
    });
    protected
}

fn neighbor_centroid(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
) -> Result<[f64; 3], DelaunayVolumeSliverError> {
    let mut neighbors = topology.incidence.vertex_stars[source_index]
        .iter()
        .flat_map(|tetrahedron| topology.tetrahedra[*tetrahedron as usize].vertex_indices)
        .filter(|vertex| *vertex as usize != source_index)
        .map(|vertex| topology.nodes[vertex as usize].identity)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|identity| {
            let index = topology
                .nodes
                .binary_search_by_key(&identity, |node| node.identity)
                .map_err(|_| {
                    error(
                        DelaunayVolumeSliverErrorKind::InvalidTopology,
                        "vertex-star neighbor is absent from topology",
                    )
                })?;
            Ok(topology.nodes[index].coordinates_m)
        })
        .collect::<Result<Vec<_>, DelaunayVolumeSliverError>>()?;
    if neighbors.len() < 4 {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "interior relocation requires at least four distinct star neighbors",
        ));
    }
    let count = neighbors.len() as f64;
    let sum = neighbors.drain(..).fold([0.0; 3], |mut sum, point| {
        for axis in 0..3 {
            sum[axis] += point[axis];
        }
        sum
    });
    Ok(sum.map(|value| value / count))
}

/// Returns the volume-weighted centroid of the complete tetrahedral cavity around a vertex.
///
/// This is the conservative three-dimensional Laplacian move used by production tetrahedral
/// optimizers: weighting tetrahedron centroids by physical volume avoids bias toward a densely
/// subdivided side of the star while preserving the cavity's center of mass.
fn volume_weighted_star_centroid(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
) -> Result<[f64; 3], DelaunayVolumeSliverError> {
    let mut weighted_sum = [0.0; 3];
    let mut total_volume = 0.0;
    for tetrahedron_index in &topology.incidence.vertex_stars[source_index] {
        let points = topology.tetrahedra[*tetrahedron_index as usize]
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
        let volume = tetrahedron_volume(points);
        if !volume.is_finite() || volume <= 0.0 {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "interior relocation cavity contains a non-positive tetrahedron",
            ));
        }
        for axis in 0..3 {
            let centroid = points.iter().map(|point| point[axis]).sum::<f64>() / 4.0;
            weighted_sum[axis] += centroid * volume;
        }
        total_volume += volume;
    }
    if !total_volume.is_finite() || total_volume <= 0.0 {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "interior relocation cavity has no positive finite volume",
        ));
    }
    Ok(weighted_sum.map(|value| value / total_volume))
}

fn replacement_identity(source: StableDigest) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(RELOCATION_IDENTITY_DOMAIN);
    hasher.update(source.bytes());
    StableDigest::from_bytes(hasher.finalize().into())
}

pub(super) fn relocation_identity_is_valid(relocation: &DelaunayVolumeSliverRelocation) -> bool {
    relocation.replacement_node.identity == replacement_identity(relocation.source_node_identity)
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{NeverCancelled, StableDigest};
    use runmat_meshing_size::metric::MetricTensor3;

    use super::{
        pattern_search_vertex_star, rebuild_with_relocated_node, relocated_star_minimum_jacobian,
        volume_weighted_star_centroid,
    };
    use crate::cdt::{
        build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode,
        DelaunayVolumeSliverOptions,
    };

    #[test]
    fn fixed_star_relocation_canonically_remaps_node_indices() {
        let identity = |value| StableDigest::from_bytes([value; 32]);
        let node = |value, point| DelaunayVolumeNode::explicit(identity(value), point);
        let topology = build_delaunay_volume_topology(
            vec![
                node(1, [0.0, 0.0, 0.0]),
                node(2, [1.0, 0.0, 0.0]),
                node(3, [0.0, 1.0, 0.0]),
                node(4, [0.0, 0.0, 1.0]),
            ],
            vec![[0, 1, 2, 3]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let mut replacement_identity = [3; 32];
        replacement_identity[31] = 4;
        let replacement = DelaunayVolumeNode::explicit(
            StableDigest::from_bytes(replacement_identity),
            [0.8, 0.0, 0.0],
        );

        let rebuilt = rebuild_with_relocated_node(
            &topology,
            1,
            replacement,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap()
        .expect("same-connectivity relocation should remain admissible");

        assert_eq!(rebuilt.nodes.len(), 4);
        assert_eq!(rebuilt.nodes[2], replacement);
        assert_eq!(rebuilt.tetrahedra.len(), 1);
        assert!(rebuilt.tetrahedra[0].vertex_indices.contains(&2));
    }

    #[test]
    fn cavity_centroid_is_weighted_by_tetrahedron_volume() {
        let identity = |value| StableDigest::from_bytes([value; 32]);
        let node = |value, point| DelaunayVolumeNode::explicit(identity(value), point);
        let topology = build_delaunay_volume_topology(
            vec![
                node(1, [0.0, 0.0, 0.0]),
                node(2, [1.0, 0.0, 0.0]),
                node(3, [0.0, 1.0, 0.0]),
                node(4, [0.0, 0.0, 1.0]),
                node(5, [0.1, 0.1, 0.1]),
            ],
            vec![[4, 0, 1, 2], [4, 0, 3, 1], [4, 0, 2, 3], [4, 1, 3, 2]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();

        let centroid = volume_weighted_star_centroid(&topology, 4).unwrap();
        for coordinate in centroid {
            assert!((coordinate - 0.25).abs() <= 1.0e-14);
        }
    }

    #[test]
    fn pattern_search_improves_the_worst_incident_tetrahedron() {
        let identity = |value| StableDigest::from_bytes([value; 32]);
        let node = |value, point| DelaunayVolumeNode::explicit(identity(value), point);
        let topology = build_delaunay_volume_topology(
            vec![
                node(1, [0.0, 0.0, 0.0]),
                node(2, [1.0, 0.0, 0.0]),
                node(3, [0.0, 1.0, 0.0]),
                node(4, [0.0, 0.0, 1.0]),
                node(5, [0.02, 0.02, 0.02]),
            ],
            vec![[4, 0, 1, 2], [4, 0, 3, 1], [4, 0, 2, 3], [4, 1, 3, 2]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let source_index = 4;
        let star = &topology.incidence.vertex_stars[source_index];
        let metric = MetricTensor3::isotropic_length_m(1.0).unwrap();
        let start = topology.nodes[source_index].coordinates_m;
        let before = relocated_star_minimum_jacobian(&topology, star, source_index, start, metric);

        let point = pattern_search_vertex_star(
            &topology,
            star,
            source_index,
            start,
            metric,
            &NeverCancelled,
        )
        .unwrap()
        .expect("the off-center interior vertex should be movable");
        let after = relocated_star_minimum_jacobian(&topology, star, source_index, point, metric);

        assert!(after > before);
        assert!(after > 0.1);
    }
}
