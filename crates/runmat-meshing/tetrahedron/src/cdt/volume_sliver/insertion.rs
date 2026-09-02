use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        compose_dyadic_point, Point3Construction, PredicateSign, SpatialPredicatePoint,
    },
    MeshingCancellationSignal, StableDigest,
};
use sha2::{Digest, Sha256};

use super::admission::{
    candidate_quality_is_admissible, compare_spectra, evaluate_candidate_unranked,
    quality_spectrum, CandidateTopologyContract, QualityObjective,
};
use super::reconnection::reconnection_candidates;
use super::work::OptimizationWork;
use super::{
    error, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind, DelaunayVolumeSliverInsertion,
    DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions, SliverSearchPolicy,
    CAVITY_SMOOTHING_STEPS, MAXIMUM_CAVITY_SEARCH_DIRECTIONS, MAXIMUM_CAVITY_SMOOTHING_ITERATIONS,
};
use crate::cdt::{
    insertion::insert_delaunay_volume_node_locally_with_barriers,
    mutation::TopologyMutation,
    topology::{
        build_delaunay_volume_topology_with_regions,
        orient_materialized_volume_nodes as orient_volume_nodes,
    },
    DelaunayInsertionErrorKind, DelaunayTetrahedronQuality, DelaunayTopologyErrorKind,
    DelaunayVolumeNode, DelaunayVolumeQuality, DelaunayVolumeRefinementInput,
    DelaunayVolumeTopology,
};
use crate::reconnect::local_tetrahedron_boundary_faces;

const INSERTION_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/volume-optimization-insertion/1\0";
const KERNEL_PROBE_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/volume-kernel-probe/1\0";
const EDGE_STAR_DIRECTION_WEIGHTS: [(u64, u8); 3] = [(61, 6), (13, 4), (1, 2)];

pub(super) struct InsertionCandidate {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) quality: DelaunayVolumeQuality,
    pub(super) mutations: Vec<DelaunayVolumeSliverMutation>,
}

pub(super) struct InsertionSearch {
    pub(super) candidate: Option<InsertionCandidate>,
    pub(super) rejection_counts: BTreeMap<&'static str, u64>,
    pub(super) first_rejection_details: BTreeMap<&'static str, String>,
}

struct InsertionSeed {
    topology: DelaunayVolumeTopology,
    quality: DelaunayVolumeQuality,
    mutation: DelaunayVolumeSliverMutation,
}

pub(super) fn insertion_candidates(
    input: DelaunayVolumeRefinementInput<'_>,
    source: &DelaunayTetrahedronQuality,
    current_spectrum: &[f64],
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    remaining_lookahead_depth: u8,
) -> Result<InsertionSearch, DelaunayVolumeSliverError> {
    let SliverSearchPolicy {
        options,
        refinement: refinement_policy,
        objective,
    } = policy;
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    let protected_segments = input
        .provenance
        .segments
        .iter()
        .map(|segment| segment.node_identities)
        .collect::<BTreeSet<_>>();
    let source_vertices = source.node_identities.map(|identity| {
        input
            .topology
            .nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .map(|index| index as u32)
            .map_err(|_| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "optimization source node is absent from canonical topology",
                )
            })
    });
    let source_vertices = source_vertices
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("source tetrahedron has four nodes");
    let mut rejection_counts = BTreeMap::new();
    let mut first_rejection_details = BTreeMap::new();
    let mut best_unranked_spectrum: Option<Vec<f64>> = None;
    let mut best_candidate: Option<InsertionCandidate> = None;
    let mut lookahead_seeds = Vec::new();
    let choices = edge_star_choices(
        input.topology,
        source_vertices,
        &source.region_id,
        &protected_faces,
        &protected_segments,
        source.resolved_metric,
        cancellation,
    )?;
    for mut choice in choices {
        work.evaluate()?;
        if let InsertionStrategy::Cavity { neighbor_index } = choice.strategy {
            if let Some(coordinates_m) = smooth_visibility_cavity_point(
                input.topology,
                choice.coordinates_m,
                VisibilityContext {
                    source_vertices,
                    neighbor_index,
                    protected_faces: &protected_faces,
                    metric: source.resolved_metric,
                    minimum_improvement: source.metric_scaled_jacobian,
                    options,
                    cancellation,
                },
            )? {
                choice.coordinates_m = coordinates_m;
                choice.construction = Point3Construction::Explicit;
            }
        }
        let node = DelaunayVolumeNode {
            identity: insertion_identity(
                source.node_identities,
                choice.edge_identities,
                choice.index,
            ),
            coordinates_m: choice.coordinates_m,
            construction: choice.construction,
        };
        if input.topology.nodes.iter().any(|existing| {
            existing.identity == node.identity || existing.coordinates_m == node.coordinates_m
        }) {
            reject(
                &mut rejection_counts,
                "candidate_collides_with_existing_node",
            );
            continue;
        }
        let topology = match choice.strategy {
            InsertionStrategy::Cavity { neighbor_index } => {
                match split_two_tetrahedron_cavity(
                    input.topology,
                    node,
                    VisibilityContext {
                        source_vertices,
                        neighbor_index,
                        protected_faces: &protected_faces,
                        metric: source.resolved_metric,
                        minimum_improvement: source.metric_scaled_jacobian,
                        options,
                        cancellation,
                    },
                )? {
                    EdgeStarSplit::Topology(topology) => topology,
                    EdgeStarSplit::Rejected(detail) => {
                        reject(&mut rejection_counts, "cavity_insertion_rejected");
                        first_rejection_details
                            .entry("cavity_insertion_rejected")
                            .or_insert(detail);
                        continue;
                    }
                }
            }
            InsertionStrategy::EdgeStar => match split_complete_edge_star(
                input.topology,
                node,
                choice.edge_identities,
                options,
                cancellation,
            )? {
                EdgeStarSplit::Topology(topology) => topology,
                EdgeStarSplit::Rejected(edge_star_detail) => {
                    match insert_delaunay_volume_node_locally_with_barriers(
                        input.topology.clone(),
                        node,
                        &protected_faces,
                        options.insertion,
                        cancellation,
                    ) {
                        Ok(topology) => topology,
                        Err(failure)
                            if matches!(
                                failure.kind,
                                DelaunayInsertionErrorKind::InvalidNode
                                    | DelaunayInsertionErrorKind::InvalidTopology
                                    | DelaunayInsertionErrorKind::PointOutsideTopology
                            ) =>
                        {
                            reject(&mut rejection_counts, "edge_star_insertion_rejected");
                            first_rejection_details
                            .entry("edge_star_insertion_rejected")
                            .or_insert_with(|| {
                                format!(
                                    "edge-star split: {edge_star_detail}; constrained insertion: {failure}"
                                )
                            });
                            continue;
                        }
                        Err(failure) => return Err(insertion_error(failure)),
                    }
                }
            },
        };
        let Some(quality) = evaluate_candidate_unranked(
            &topology,
            input,
            &protected_faces,
            CandidateTopologyContract::CertifiedConformingCavity,
            options,
            cancellation,
            &mut rejection_counts,
        )?
        else {
            continue;
        };
        let candidate_spectrum = quality_spectrum(&quality, objective);
        if best_unranked_spectrum.as_ref().is_none_or(|best| {
            compare_spectra(&candidate_spectrum, best) == std::cmp::Ordering::Less
        }) {
            best_unranked_spectrum = Some(candidate_spectrum);
        }
        let insertion = DelaunayVolumeSliverMutation::Inserted(DelaunayVolumeSliverInsertion {
            inserted_node: node,
            source_tetrahedron_node_identities: source.node_identities,
            source_edge_node_identities: choice.edge_identities,
            construction_point_index: choice.index,
        });
        lookahead_seeds.push(InsertionSeed {
            topology: topology.clone(),
            quality: quality.clone(),
            mutation: insertion.clone(),
        });
        if candidate_quality_is_admissible(
            &quality,
            input,
            current_spectrum,
            refinement_policy,
            objective,
            &mut rejection_counts,
        ) {
            retain_better_candidate(
                &mut best_candidate,
                InsertionCandidate {
                    topology: topology.clone(),
                    quality: quality.clone(),
                    mutations: vec![insertion.clone()],
                },
                objective,
            );
        }
        let Some(reconnection_source) =
            super::admission::worst_quality_source(&quality, &topology, objective)
        else {
            continue;
        };
        let candidate_input = DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality: &quality,
            quality_options: input.quality_options,
            fixed_node_identities: input.fixed_node_identities,
        };
        let reconnection = reconnection_candidates(
            candidate_input,
            reconnection_source,
            current_spectrum,
            cancellation,
            work,
            SliverSearchPolicy {
                options,
                refinement: refinement_policy,
                objective,
            },
            super::reconnection::ReconnectionMode {
                allow_lookahead: false,
                maximum_candidates: Some(options.maximum_compound_reconnections_per_insertion),
            },
        )?;
        if let Some(reconnection) = reconnection.candidate {
            let mut mutations = vec![insertion];
            mutations.extend(reconnection.mutations);
            retain_better_candidate(
                &mut best_candidate,
                InsertionCandidate {
                    topology: reconnection.topology,
                    quality: reconnection.quality,
                    mutations,
                },
                objective,
            );
        }
    }
    if best_candidate.is_none() && remaining_lookahead_depth > 0 {
        const MAXIMUM_LOOKAHEAD_WIDTH: usize = 2;
        lookahead_seeds.sort_by(|left, right| {
            compare_spectra(
                &quality_spectrum(&left.quality, objective),
                &quality_spectrum(&right.quality, objective),
            )
            .then_with(|| {
                left.topology
                    .nodes
                    .last()
                    .map(|node| node.identity)
                    .cmp(&right.topology.nodes.last().map(|node| node.identity))
            })
        });
        lookahead_seeds.truncate(MAXIMUM_LOOKAHEAD_WIDTH);
        for seed in lookahead_seeds {
            let Some(next_source) =
                super::admission::worst_quality_source(&seed.quality, &seed.topology, objective)
            else {
                continue;
            };
            let seed_input = DelaunayVolumeRefinementInput {
                topology: &seed.topology,
                metric_request: input.metric_request,
                provenance: input.provenance,
                quality: &seed.quality,
                quality_options: input.quality_options,
                fixed_node_identities: input.fixed_node_identities,
            };
            let relocation = super::relocation::relocation_candidates(
                seed_input,
                next_source,
                current_spectrum,
                cancellation,
                work,
                SliverSearchPolicy {
                    options,
                    refinement: refinement_policy,
                    objective,
                },
                super::relocation::RelocationSearchScope::FixedStarOnly,
            )?;
            if let Some(relocation) = relocation.candidate {
                return Ok(InsertionSearch {
                    candidate: Some(InsertionCandidate {
                        topology: relocation.topology,
                        quality: relocation.quality,
                        mutations: vec![seed.mutation.clone(), relocation.mutation],
                    }),
                    rejection_counts,
                    first_rejection_details,
                });
            }
            let next = insertion_candidates(
                seed_input,
                next_source,
                current_spectrum,
                cancellation,
                work,
                SliverSearchPolicy {
                    options,
                    refinement: refinement_policy,
                    objective,
                },
                remaining_lookahead_depth - 1,
            )?;
            if let Some(next) = next.candidate {
                let mut mutations = vec![seed.mutation];
                mutations.extend(next.mutations);
                return Ok(InsertionSearch {
                    candidate: Some(InsertionCandidate {
                        topology: next.topology,
                        quality: next.quality,
                        mutations,
                    }),
                    rejection_counts,
                    first_rejection_details,
                });
            }
            for (reason, count) in next.rejection_counts {
                *rejection_counts.entry(reason).or_default() += count;
            }
            for (reason, detail) in next.first_rejection_details {
                first_rejection_details.entry(reason).or_insert(detail);
            }
        }
    }
    if let Some(spectrum) = best_unranked_spectrum {
        first_rejection_details.insert(
            "best_unranked_optimization_spectrum",
            format!("{:?}", spectrum.into_iter().take(8).collect::<Vec<_>>()),
        );
    }
    Ok(InsertionSearch {
        candidate: best_candidate,
        rejection_counts,
        first_rejection_details,
    })
}

fn retain_better_candidate(
    best: &mut Option<InsertionCandidate>,
    candidate: InsertionCandidate,
    objective: QualityObjective,
) {
    let replace = best.as_ref().is_none_or(|current| {
        compare_spectra(
            &quality_spectrum(&candidate.quality, objective),
            &quality_spectrum(&current.quality, objective),
        ) == std::cmp::Ordering::Less
    });
    if replace {
        *best = Some(candidate);
    }
}

enum EdgeStarSplit {
    Topology(DelaunayVolumeTopology),
    Rejected(String),
}

type VisibilityCavity = (
    BTreeSet<u32>,
    Option<runmat_geometry_core::PersistentEntityId>,
);
type VisibilityCavityResult = Result<VisibilityCavity, String>;

struct VisibilityCavityRequest<'a> {
    node: DelaunayVolumeNode,
    source_index: u32,
    neighbor_index: u32,
    protected_faces: &'a BTreeSet<[StableDigest; 3]>,
    quality_growth: Option<(runmat_meshing_size::metric::MetricTensor3, f64)>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &'a dyn MeshingCancellationSignal,
}

struct VisibilityContext<'a> {
    source_vertices: [u32; 4],
    neighbor_index: u32,
    protected_faces: &'a [[StableDigest; 3]],
    metric: runmat_meshing_size::metric::MetricTensor3,
    minimum_improvement: f64,
    options: DelaunayVolumeSliverOptions,
    cancellation: &'a dyn MeshingCancellationSignal,
}

fn grow_visibility_cavity(
    topology: &DelaunayVolumeTopology,
    request: VisibilityCavityRequest<'_>,
) -> Result<VisibilityCavityResult, DelaunayVolumeSliverError> {
    let VisibilityCavityRequest {
        node,
        source_index,
        neighbor_index,
        protected_faces,
        quality_growth,
        options,
        cancellation,
    } = request;
    let Some(source) = topology.tetrahedra.get(source_index as usize) else {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver cavity source is absent",
        ));
    };
    let Some(neighbor) = topology.tetrahedra.get(neighbor_index as usize) else {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver cavity neighbor is absent",
        ));
    };
    if source.region_id != neighbor.region_id || !source.neighbors.contains(&Some(neighbor_index)) {
        return Ok(Err(
            "sliver cavity tetrahedra are not same-region neighbors".to_owned(),
        ));
    }
    let region = source.region_id.clone();
    let mut removed = BTreeSet::from([source_index, neighbor_index]);
    loop {
        if removed.len() as u64 > options.insertion.maximum_cavity_tetrahedra {
            return Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                "sliver insertion visibility cavity exceeded its tetrahedron limit",
            ));
        }
        if cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "sliver insertion visibility cavity was cancelled",
            ));
        }
        let mut expansion = None;
        'cavity: for (tetrahedron_index, opposite, face) in
            cavity_boundary_faces(topology, &removed)
        {
            let tetrahedron = &topology.tetrahedra[tetrahedron_index as usize];
            let face_nodes = face.map(|vertex| topology.nodes[vertex as usize]);
            let interior = topology.nodes[tetrahedron.vertex_indices[opposite] as usize];
            let interior_side =
                orient_volume_nodes([face_nodes[0], face_nodes[1], face_nodes[2], interior])
                    .map_err(visibility_predicate_error)?;
            let point_side =
                orient_volume_nodes([face_nodes[0], face_nodes[1], face_nodes[2], node])
                    .map_err(visibility_predicate_error)?;
            if point_side != PredicateSign::Zero && point_side == interior_side {
                continue;
            }
            let face_identities = face.map(|vertex| topology.nodes[vertex as usize].identity);
            if protected_faces.contains(&face_identities) {
                return Ok(Err(
                    "sliver cavity visibility growth reached a protected facet".to_owned(),
                ));
            }
            let Some(adjacent) = tetrahedron.neighbors[opposite] else {
                return Ok(Err(
                    "sliver insertion point lies outside the volume boundary".to_owned(),
                ));
            };
            if topology.tetrahedra[adjacent as usize].region_id != region {
                return Ok(Err(
                    "sliver cavity visibility growth reached a region boundary".to_owned(),
                ));
            }
            expansion = Some(adjacent);
            break 'cavity;
        }
        let adjacent = match expansion {
            Some(adjacent) => adjacent,
            None => {
                let Some((metric, target)) = quality_growth else {
                    return Ok(Ok((removed, region)));
                };
                let worst = cavity_boundary_faces(topology, &removed)
                    .into_iter()
                    .filter_map(|(tetrahedron_index, opposite, face)| {
                        let points = [
                            topology.nodes[face[0] as usize].coordinates_m,
                            topology.nodes[face[1] as usize].coordinates_m,
                            topology.nodes[face[2] as usize].coordinates_m,
                            node.coordinates_m,
                        ];
                        super::super::volume_quality::evaluate_tetrahedron_geometry(points, metric)
                            .map(|quality| {
                                (
                                    quality.metric_scaled_jacobian,
                                    tetrahedron_index,
                                    opposite,
                                    face,
                                )
                            })
                    })
                    .min_by(|left, right| {
                        left.0
                            .total_cmp(&right.0)
                            .then_with(|| left.3.cmp(&right.3))
                    });
                let Some((score, tetrahedron_index, opposite, face)) = worst else {
                    return Ok(Ok((removed, region)));
                };
                if score > target {
                    return Ok(Ok((removed, region)));
                }
                let tetrahedron = &topology.tetrahedra[tetrahedron_index as usize];
                let face_identities = face.map(|vertex| topology.nodes[vertex as usize].identity);
                let Some(adjacent) = tetrahedron.neighbors[opposite] else {
                    return Ok(Ok((removed, region)));
                };
                if protected_faces.contains(&face_identities)
                    || topology.tetrahedra[adjacent as usize].region_id != region
                {
                    return Ok(Ok((removed, region)));
                }
                adjacent
            }
        };
        if !removed.insert(adjacent) {
            return Ok(Err(
                "sliver cavity visibility growth did not make progress".to_owned()
            ));
        }
    }
}

fn cavity_boundary_faces(
    topology: &DelaunayVolumeTopology,
    removed: &BTreeSet<u32>,
) -> Vec<(u32, usize, [u32; 3])> {
    let mut faces = Vec::new();
    for tetrahedron_index in removed.iter().copied() {
        let tetrahedron = &topology.tetrahedra[tetrahedron_index as usize];
        for opposite in 0..4 {
            if tetrahedron.neighbors[opposite].is_some_and(|adjacent| removed.contains(&adjacent)) {
                continue;
            }
            let mut face = [0_u32; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.vertex_indices.into_iter().enumerate() {
                if slot != opposite {
                    face[cursor] = vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            faces.push((tetrahedron_index, opposite, face));
        }
    }
    faces.sort_by_key(|(_, _, face)| *face);
    faces
}

fn visibility_predicate_error(
    failure: runmat_meshing_core::quality::predicate::SpatialPredicateError,
) -> DelaunayVolumeSliverError {
    error(
        DelaunayVolumeSliverErrorKind::InvalidTopology,
        format!("sliver cavity visibility predicate failed: {failure:?}"),
    )
}

fn smooth_visibility_cavity_point(
    topology: &DelaunayVolumeTopology,
    start: [f64; 3],
    context: VisibilityContext<'_>,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let VisibilityContext {
        source_vertices,
        neighbor_index,
        protected_faces,
        metric,
        minimum_improvement,
        options,
        cancellation,
    } = context;
    let source_index = topology
        .tetrahedra
        .iter()
        .position(|tetrahedron| {
            source_vertices
                .iter()
                .all(|vertex| tetrahedron.vertex_indices.contains(vertex))
        })
        .expect("quality source tetrahedron remains in topology") as u32;
    let protected_faces = protected_faces
        .iter()
        .map(|face| {
            let mut face = *face;
            face.sort_unstable();
            face
        })
        .collect::<BTreeSet<_>>();
    let probe = |point| DelaunayVolumeNode::explicit(kernel_probe_identity(point), point);
    let Ok((removed, _)) = grow_visibility_cavity(
        topology,
        VisibilityCavityRequest {
            node: probe(start),
            source_index,
            neighbor_index,
            protected_faces: &protected_faces,
            quality_growth: Some((metric, minimum_improvement)),
            options,
            cancellation,
        },
    )?
    else {
        return Ok(None);
    };
    let boundary = cavity_boundary_faces(topology, &removed);
    let score = |point: [f64; 3]| {
        boundary
            .iter()
            .filter_map(|(_, _, face)| {
                let points = [
                    topology.nodes[face[0] as usize].coordinates_m,
                    topology.nodes[face[1] as usize].coordinates_m,
                    topology.nodes[face[2] as usize].coordinates_m,
                    point,
                ];
                super::super::volume_quality::evaluate_tetrahedron_geometry(points, metric)
                    .map(|quality| quality.metric_scaled_jacobian)
            })
            .min_by(f64::total_cmp)
            .unwrap_or(0.0)
    };
    let in_kernel = |point: [f64; 3]| -> Result<bool, DelaunayVolumeSliverError> {
        for (tetrahedron_index, opposite, face) in &boundary {
            let tetrahedron = &topology.tetrahedra[*tetrahedron_index as usize];
            let nodes = face.map(|vertex| topology.nodes[vertex as usize]);
            let interior = topology.nodes[tetrahedron.vertex_indices[*opposite] as usize];
            let expected = orient_volume_nodes([nodes[0], nodes[1], nodes[2], interior])
                .map_err(visibility_predicate_error)?;
            let observed = orient_volume_nodes([nodes[0], nodes[1], nodes[2], probe(point)])
                .map_err(visibility_predicate_error)?;
            if observed == PredicateSign::Zero || observed != expected {
                return Ok(false);
            }
        }
        Ok(true)
    };
    let cavity_vertices = removed
        .iter()
        .flat_map(|index| topology.tetrahedra[*index as usize].vertex_indices)
        .collect::<BTreeSet<_>>();
    let local_scale = cavity_vertices
        .iter()
        .flat_map(|left| cavity_vertices.iter().map(move |right| (*left, *right)))
        .map(|(left, right)| {
            topology.nodes[left as usize]
                .coordinates_m
                .into_iter()
                .zip(topology.nodes[right as usize].coordinates_m)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        })
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    if local_scale == 0.0 {
        return Ok(None);
    }
    let mut point = start;
    let initial_score = score(point);
    let mut best_score = initial_score;
    let mut radius = local_scale * 0.25;
    let minimum_radius = local_scale * 2_f64.powi(-24);
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "sliver cavity smoothing was cancelled",
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
                    let candidate = std::array::from_fn(|axis| {
                        point[axis] + radius * f64::from([x, y, z][axis]) / norm
                    });
                    if !in_kernel(candidate)? {
                        continue;
                    }
                    let candidate_score = score(candidate);
                    if candidate_score > best_score
                        && best
                            .as_ref()
                            .is_none_or(|(score, point): &(f64, [f64; 3])| {
                                candidate_score > *score
                                    || candidate_score == *score
                                        && candidate.map(f64::to_bits) < point.map(f64::to_bits)
                            })
                    {
                        best = Some((candidate_score, candidate));
                    }
                }
            }
        }
        if let Some((score, candidate)) = best {
            best_score = score;
            point = candidate;
        } else {
            radius *= 0.5;
            if radius < minimum_radius {
                break;
            }
        }
    }
    Ok((best_score > initial_score).then_some(point))
}

fn split_two_tetrahedron_cavity(
    topology: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    context: VisibilityContext<'_>,
) -> Result<EdgeStarSplit, DelaunayVolumeSliverError> {
    let VisibilityContext {
        source_vertices,
        neighbor_index,
        protected_faces,
        metric,
        minimum_improvement,
        options,
        cancellation,
    } = context;
    let Some(source_index) = topology
        .tetrahedra
        .iter()
        .position(|tetrahedron| {
            source_vertices
                .iter()
                .all(|vertex| tetrahedron.vertex_indices.contains(vertex))
        })
        .and_then(|index| u32::try_from(index).ok())
    else {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver cavity source tetrahedron is absent",
        ));
    };
    let protected_faces = protected_faces
        .iter()
        .map(|face| {
            let mut face = *face;
            face.sort_unstable();
            face
        })
        .collect::<BTreeSet<_>>();
    let (removed, region) = match grow_visibility_cavity(
        topology,
        VisibilityCavityRequest {
            node,
            source_index,
            neighbor_index,
            protected_faces: &protected_faces,
            quality_growth: Some((metric, minimum_improvement)),
            options,
            cancellation,
        },
    )? {
        Ok(cavity) => cavity,
        Err(detail) => return Ok(EdgeStarSplit::Rejected(detail)),
    };

    let removed_tetrahedra = removed
        .iter()
        .map(|index| topology.tetrahedra[*index as usize].vertex_indices)
        .collect::<Vec<_>>();
    let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);

    let mut mutation = TopologyMutation::from_canonical(topology.clone());
    let inserted = match mutation.append_node(node) {
        Ok(inserted) => inserted,
        Err(failure) => return Ok(EdgeStarSplit::Rejected(failure.to_string())),
    };
    let created = boundary
        .into_iter()
        .map(|face| ([face[0], face[1], face[2], inserted], region.clone()))
        .collect();
    if let Err(failure) = mutation.replace_cavity(&removed, created) {
        return Ok(EdgeStarSplit::Rejected(failure.to_string()));
    }
    match mutation.freeze(options.insertion.topology, cancellation) {
        Ok(topology) => Ok(EdgeStarSplit::Topology(topology)),
        Err(failure) => match failure.kind {
            crate::cdt::mutation::TopologyMutationErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            crate::cdt::mutation::TopologyMutationErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            crate::cdt::mutation::TopologyMutationErrorKind::InvalidTopology => {
                Ok(EdgeStarSplit::Rejected(failure.to_string()))
            }
        },
    }
}

/// TetGen's `INSTAR` sliver split replaces the complete star of the selected edge rather than
/// asking an unconstrained Bowyer-Watson cavity to choose it again. This is important at a smooth
/// boundary: global Delaunay legalization can legally recreate the same flat two-facet wedge.
/// The canonical topology builder independently verifies orientation, overlap, manifoldness, and
/// incidence; the caller additionally verifies PLC conformance and strict global quality progress.
fn split_complete_edge_star(
    topology: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    edge_identities: [StableDigest; 2],
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<EdgeStarSplit, DelaunayVolumeSliverError> {
    let edge = edge_identities.map(|identity| {
        topology
            .nodes
            .binary_search_by_key(&identity, |candidate| candidate.identity)
            .map(|index| index as u32)
    });
    let Ok(edge) = edge.into_iter().collect::<Result<Vec<_>, _>>() else {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver insertion edge is absent from canonical topology",
        ));
    };
    let edge: [u32; 2] = edge.try_into().expect("sliver edge has two nodes");
    let star = topology.incidence.vertex_stars[edge[0] as usize]
        .iter()
        .copied()
        .filter(|tetrahedron| {
            topology.incidence.vertex_stars[edge[1] as usize]
                .binary_search(tetrahedron)
                .is_ok()
        })
        .collect::<Vec<_>>();
    if star.len() < 3
        || topology.incidence.boundary_facets.iter().any(|facet| {
            facet.vertex_indices.contains(&edge[0]) && facet.vertex_indices.contains(&edge[1])
        })
    {
        return Ok(EdgeStarSplit::Rejected(
            "selected edge does not have a complete interior star".to_owned(),
        ));
    }
    let region = topology.tetrahedra[star[0] as usize].region_id.clone();
    if star
        .iter()
        .any(|tetrahedron| topology.tetrahedra[*tetrahedron as usize].region_id != region)
    {
        return Ok(EdgeStarSplit::Rejected(
            "selected edge star crosses a region boundary".to_owned(),
        ));
    }
    let removed = star.iter().copied().collect::<BTreeSet<_>>();
    let mut nodes = topology.nodes.clone();
    let inserted = nodes
        .binary_search_by_key(&node.identity, |candidate| candidate.identity)
        .expect_err("candidate identity collision is rejected before edge-star splitting");
    nodes.insert(inserted, node);
    let inserted = inserted as u32;
    let remap = |vertex: u32| vertex + u32::from(vertex >= inserted);
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
        .map(|(_, tetrahedron)| {
            (
                tetrahedron.vertex_indices.map(remap),
                tetrahedron.region_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    tetrahedra.extend(
        edge_star_boundary_faces(topology, &star, edge)
            .into_iter()
            .map(|face| {
                (
                    [remap(face[0]), remap(face[1]), remap(face[2]), inserted],
                    region.clone(),
                )
            }),
    );
    match build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(topology) => Ok(EdgeStarSplit::Topology(topology)),
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayTopologyErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => Ok(EdgeStarSplit::Rejected(failure.to_string())),
        },
    }
}

#[derive(Clone, Copy)]
struct EdgeStarChoice {
    edge_identities: [StableDigest; 2],
    index: u16,
    coordinates_m: [f64; 3],
    construction: Point3Construction,
    minimum_created_jacobian: f64,
    strategy: InsertionStrategy,
}

#[derive(Clone, Copy)]
enum InsertionStrategy {
    EdgeStar,
    Cavity { neighbor_index: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeCandidatePolicy {
    ExcludedSegment,
    OffEdgeOnly,
    IncludeMidpoint,
}

fn edge_star_choices(
    topology: &DelaunayVolumeTopology,
    source: [u32; 4],
    region: &runmat_geometry_core::PersistentEntityId,
    protected_faces: &[[StableDigest; 3]],
    protected_segments: &BTreeSet<[StableDigest; 2]>,
    metric: runmat_meshing_size::metric::MetricTensor3,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<EdgeStarChoice>, DelaunayVolumeSliverError> {
    let mut choices = Vec::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            let mut edge = [source[left], source[right]];
            edge.sort_unstable();
            let edge_identities = edge.map(|vertex| topology.nodes[vertex as usize].identity);
            let star = topology.incidence.vertex_stars[edge[0] as usize]
                .iter()
                .copied()
                .filter(|tetrahedron| {
                    topology.incidence.vertex_stars[edge[1] as usize]
                        .binary_search(tetrahedron)
                        .is_ok()
                })
                .collect::<Vec<_>>();
            if star.len() < 3
                || topology.incidence.boundary_facets.iter().any(|facet| {
                    facet.vertex_indices.contains(&edge[0])
                        && facet.vertex_indices.contains(&edge[1])
                })
                || star.iter().any(|tetrahedron| {
                    topology.tetrahedra[*tetrahedron as usize]
                        .region_id
                        .as_ref()
                        != Some(region)
                })
            {
                continue;
            }
            let ring = star
                .iter()
                .flat_map(|tetrahedron| topology.tetrahedra[*tetrahedron as usize].vertex_indices)
                .filter(|vertex| !edge.contains(vertex))
                .collect::<BTreeSet<_>>();
            if ring.len() < 3 {
                continue;
            }
            let endpoint_nodes = edge.map(|vertex| topology.nodes[vertex as usize]);
            let mut edge_identities = edge_identities;
            edge_identities.sort_unstable();
            let edge_policy =
                edge_candidate_policy(edge_identities, protected_faces, protected_segments);
            if edge_policy == EdgeCandidatePolicy::ExcludedSegment {
                continue;
            }
            // A midpoint on a protected surface edge would be an embedded node unless the
            // surface facet were also subdivided. Smoothed candidates move strictly into the
            // edge-star kernel and preserve that facet, so—as in TetGen's INSTAR sliver split—
            // only true PLC segments exclude the edge from the search altogether.
            let Some((midpoint, midpoint_construction)) = compose_dyadic_point(
                endpoint_nodes.map(DelaunayVolumeNode::predicate_point),
                [1, 1],
                1,
            ) else {
                continue;
            };
            if edge_policy == EdgeCandidatePolicy::IncludeMidpoint {
                choices.push(EdgeStarChoice {
                    edge_identities,
                    index: 0,
                    coordinates_m: midpoint,
                    construction: midpoint_construction,
                    minimum_created_jacobian: split_minimum_jacobian(
                        topology, &star, edge, midpoint, metric,
                    ),
                    strategy: InsertionStrategy::EdgeStar,
                });
            }
            let boundary_faces = edge_star_boundary_faces(topology, &star, edge);
            for (direction, face) in boundary_faces
                .iter()
                .take(MAXIMUM_CAVITY_SEARCH_DIRECTIONS)
                .enumerate()
            {
                for (fraction_index, (midpoint_weight, exponent)) in
                    EDGE_STAR_DIRECTION_WEIGHTS.into_iter().enumerate()
                {
                    let Some((coordinates_m, construction)) = compose_dyadic_point(
                        [
                            SpatialPredicatePoint {
                                identity: kernel_probe_identity(midpoint),
                                coordinates: midpoint,
                                construction: midpoint_construction,
                            },
                            topology.nodes[face[0] as usize].predicate_point(),
                            topology.nodes[face[1] as usize].predicate_point(),
                            topology.nodes[face[2] as usize].predicate_point(),
                        ],
                        [midpoint_weight, 1, 1, 1],
                        exponent,
                    ) else {
                        continue;
                    };
                    if !point_is_in_edge_star_kernel(
                        topology,
                        &star,
                        edge,
                        coordinates_m,
                        construction,
                    )? {
                        continue;
                    }
                    choices.push(EdgeStarChoice {
                        edge_identities,
                        index: 2
                            + (direction * EDGE_STAR_DIRECTION_WEIGHTS.len() + fraction_index)
                                as u16,
                        coordinates_m,
                        construction,
                        minimum_created_jacobian: split_minimum_jacobian(
                            topology,
                            &star,
                            edge,
                            coordinates_m,
                            metric,
                        ),
                        strategy: InsertionStrategy::EdgeStar,
                    });
                }
            }
            let mut smoothing_starts = vec![midpoint];
            for (search_index, step) in CAVITY_SMOOTHING_STEPS.into_iter().enumerate() {
                if let Some(coordinates_m) = smooth_edge_star_point(
                    topology,
                    &star,
                    edge,
                    midpoint,
                    metric,
                    step,
                    cancellation,
                )? {
                    smoothing_starts.push(coordinates_m);
                    choices.push(EdgeStarChoice {
                        edge_identities,
                        index: 1_000 + search_index as u16,
                        coordinates_m,
                        construction: Point3Construction::Explicit,
                        minimum_created_jacobian: split_minimum_jacobian(
                            topology,
                            &star,
                            edge,
                            coordinates_m,
                            metric,
                        ),
                        strategy: InsertionStrategy::EdgeStar,
                    });
                }
            }
            smoothing_starts.sort_by_key(|point| point.map(f64::to_bits));
            smoothing_starts.dedup();
            for (search_index, start) in smoothing_starts.into_iter().enumerate() {
                if let Some(coordinates_m) =
                    pattern_search_edge_star(topology, &star, edge, start, metric, cancellation)?
                {
                    choices.push(EdgeStarChoice {
                        edge_identities,
                        index: 2_000 + search_index as u16,
                        coordinates_m,
                        construction: Point3Construction::Explicit,
                        minimum_created_jacobian: split_minimum_jacobian(
                            topology,
                            &star,
                            edge,
                            coordinates_m,
                            metric,
                        ),
                        strategy: InsertionStrategy::EdgeStar,
                    });
                }
            }
        }
    }
    let source_index = topology
        .tetrahedra
        .iter()
        .position(|tetrahedron| {
            source
                .iter()
                .all(|vertex| tetrahedron.vertex_indices.contains(vertex))
        })
        .expect("quality source tetrahedron remains in topology");
    let mut lineage_edge = [
        topology.nodes[source[0] as usize].identity,
        topology.nodes[source[1] as usize].identity,
    ];
    lineage_edge.sort_unstable();
    for (slot, neighbor) in topology.tetrahedra[source_index]
        .neighbors
        .into_iter()
        .enumerate()
    {
        let Some(neighbor_index) = neighbor else {
            continue;
        };
        let neighbor = &topology.tetrahedra[neighbor_index as usize];
        if neighbor.region_id.as_ref() != Some(region) {
            continue;
        }
        let vertices = neighbor
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
        let neighbor_centroid: [f64; 3] = std::array::from_fn(|axis| {
            vertices[0][axis]
                + (vertices[1][axis] - vertices[0][axis] + vertices[2][axis] - vertices[0][axis]
                    + vertices[3][axis]
                    - vertices[0][axis])
                    * 0.25
        });
        let source_vertices = topology.tetrahedra[source_index]
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
        let source_centroid = average(source_vertices.into_iter());
        for (blend_index, neighbor_weight) in [0.25, 0.5, 0.75].into_iter().enumerate() {
            let coordinates_m = std::array::from_fn(|axis| {
                source_centroid[axis]
                    + neighbor_weight * (neighbor_centroid[axis] - source_centroid[axis])
            });
            choices.push(EdgeStarChoice {
                edge_identities: lineage_edge,
                index: 10_000 + (slot * 3 + blend_index) as u16,
                coordinates_m,
                construction: Point3Construction::Explicit,
                minimum_created_jacobian: 0.0,
                strategy: InsertionStrategy::Cavity { neighbor_index },
            });
        }
    }
    let source_tetrahedron = &topology.tetrahedra[source_index];
    for opposite in 0..4 {
        let mut face = source_tetrahedron
            .vertex_indices
            .into_iter()
            .enumerate()
            .filter_map(|(slot, vertex)| (slot != opposite).then_some(vertex))
            .collect::<Vec<_>>();
        face.sort_unstable();
        let face: [u32; 3] = face.try_into().expect("tetrahedron face has three nodes");
        let face_identities = face.map(|vertex| topology.nodes[vertex as usize].identity);
        if !protected_faces.contains(&face_identities) {
            continue;
        }
        let Some(neighbor_index) =
            source_tetrahedron
                .neighbors
                .into_iter()
                .flatten()
                .find(|neighbor| {
                    topology.tetrahedra[*neighbor as usize].region_id.as_ref() == Some(region)
                })
        else {
            continue;
        };
        let points = face.map(|vertex| topology.nodes[vertex as usize].coordinates_m);
        let first: [f64; 3] = std::array::from_fn(|axis| points[1][axis] - points[0][axis]);
        let second: [f64; 3] = std::array::from_fn(|axis| points[2][axis] - points[0][axis]);
        let mut normal = [
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ];
        let norm = normal
            .into_iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            continue;
        }
        normal.iter_mut().for_each(|value| *value /= norm);
        let centroid = average(points.into_iter());
        let interior =
            topology.nodes[source_tetrahedron.vertex_indices[opposite] as usize].coordinates_m;
        if normal
            .into_iter()
            .enumerate()
            .map(|(axis, value)| value * (interior[axis] - centroid[axis]))
            .sum::<f64>()
            < 0.0
        {
            normal.iter_mut().for_each(|value| *value = -*value);
        }
        let shortest_edge = [(0, 1), (1, 2), (2, 0)]
            .into_iter()
            .map(|(left, right)| {
                (0..3)
                    .map(|axis| {
                        let delta = points[left][axis] - points[right][axis];
                        delta * delta
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .min_by(f64::total_cmp)
            .expect("face has edges");
        let mut lineage_edge = [face_identities[0], face_identities[1]];
        lineage_edge.sort_unstable();
        for (offset_index, factor) in [0.25, 0.5, 1.0].into_iter().enumerate() {
            let coordinates_m =
                std::array::from_fn(|axis| centroid[axis] + factor * shortest_edge * normal[axis]);
            choices.push(EdgeStarChoice {
                edge_identities: lineage_edge,
                index: 20_000 + (opposite * 3 + offset_index) as u16,
                coordinates_m,
                construction: Point3Construction::Explicit,
                minimum_created_jacobian: 0.0,
                strategy: InsertionStrategy::Cavity { neighbor_index },
            });
        }
    }
    choices.sort_by(|left, right| {
        right
            .minimum_created_jacobian
            .total_cmp(&left.minimum_created_jacobian)
            .then_with(|| {
                left.edge_identities
                    .cmp(&right.edge_identities)
                    .then_with(|| left.index.cmp(&right.index))
            })
    });
    Ok(choices)
}

fn insertion_error(failure: crate::cdt::DelaunayInsertionError) -> DelaunayVolumeSliverError {
    let kind = match failure.kind {
        DelaunayInsertionErrorKind::ResourceLimit => DelaunayVolumeSliverErrorKind::ResourceLimit,
        DelaunayInsertionErrorKind::Cancelled => DelaunayVolumeSliverErrorKind::Cancelled,
        _ => DelaunayVolumeSliverErrorKind::InvalidTopology,
    };
    error(kind, failure.to_string())
}

fn edge_candidate_policy(
    edge: [StableDigest; 2],
    protected_faces: &[[StableDigest; 3]],
    protected_segments: &BTreeSet<[StableDigest; 2]>,
) -> EdgeCandidatePolicy {
    if protected_segments.contains(&edge) {
        EdgeCandidatePolicy::ExcludedSegment
    } else if protected_faces
        .iter()
        .any(|facet| facet.contains(&edge[0]) && facet.contains(&edge[1]))
    {
        EdgeCandidatePolicy::OffEdgeOnly
    } else {
        EdgeCandidatePolicy::IncludeMidpoint
    }
}

/// Finds a point in the kernel of an interior edge-star cavity while maximizing the minimum
/// created scaled Jacobian. This is one deterministic scale of TetGen-style link-face smoothing:
/// start at the edge midpoint, move toward each cavity boundary-face centroid, and retain the
/// best strict improvement until convergence. The caller searches several bounded step scales.
fn smooth_edge_star_point(
    topology: &DelaunayVolumeTopology,
    star: &[u32],
    edge: [u32; 2],
    midpoint: [f64; 3],
    metric: runmat_meshing_size::metric::MetricTensor3,
    step: f64,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let boundary_faces = edge_star_boundary_faces(topology, star, edge);

    let mut point = midpoint;
    let mut score = split_minimum_jacobian(topology, star, edge, point, metric);
    let mut improved = false;
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(16) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "edge-star insertion smoothing was cancelled",
            ));
        }
        let mut best = None;
        for face in boundary_faces.iter().take(MAXIMUM_CAVITY_SEARCH_DIRECTIONS) {
            let centroid = average(
                face.iter()
                    .map(|vertex| topology.nodes[*vertex as usize].coordinates_m),
            );
            let candidate =
                std::array::from_fn(|axis| point[axis] + step * (centroid[axis] - point[axis]));
            if !point_is_in_edge_star_kernel(
                topology,
                star,
                edge,
                candidate,
                Point3Construction::Explicit,
            )? {
                continue;
            }
            let candidate_score = split_minimum_jacobian(topology, star, edge, candidate, metric);
            if candidate_score <= score {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(best_score, best_point): &(f64, [f64; 3])| {
                    candidate_score > *best_score
                        || candidate_score == *best_score
                            && coordinate_key(candidate) < coordinate_key(*best_point)
                })
            {
                best = Some((candidate_score, candidate));
            }
        }
        let Some((best_score, best_point)) = best else {
            break;
        };
        point = best_point;
        score = best_score;
        improved = true;
    }
    Ok(improved.then_some(point))
}

/// Deterministic derivative-free maximin search in the kernel of a complete edge star.
/// Face-centroid directions are useful warm starts, but do not span the local quality gradient
/// on curved boundaries; this bounded stencil supplies the independent directions used by
/// production sliver splitters.
fn pattern_search_edge_star(
    topology: &DelaunayVolumeTopology,
    star: &[u32],
    edge: [u32; 2],
    start: [f64; 3],
    metric: runmat_meshing_size::metric::MetricTensor3,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let local_scale = edge
        .map(|vertex| topology.nodes[vertex as usize].coordinates_m)
        .into_iter()
        .reduce(|left, right| std::array::from_fn(|axis| left[axis] - right[axis]))
        .map(|delta| {
            delta
                .into_iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .unwrap_or(0.0);
    if local_scale == 0.0 {
        return Ok(None);
    }
    let mut point = start;
    let initial_score = split_minimum_jacobian(topology, star, edge, point, metric);
    let mut score = initial_score;
    let mut radius = local_scale * 0.25;
    let minimum_radius = local_scale * 2_f64.powi(-24);
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "edge-star pattern search was cancelled",
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
                    let direction = [x, y, z].map(|value| f64::from(value) / norm);
                    let candidate =
                        std::array::from_fn(|axis| point[axis] + radius * direction[axis]);
                    if !point_is_in_edge_star_kernel(
                        topology,
                        star,
                        edge,
                        candidate,
                        Point3Construction::Explicit,
                    )? {
                        continue;
                    }
                    let candidate_score =
                        split_minimum_jacobian(topology, star, edge, candidate, metric);
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

fn edge_star_boundary_faces(
    topology: &DelaunayVolumeTopology,
    star: &[u32],
    edge: [u32; 2],
) -> Vec<[u32; 3]> {
    let mut boundary_faces = star
        .iter()
        .flat_map(|tetrahedron| {
            let opposite = topology.tetrahedra[*tetrahedron as usize]
                .vertex_indices
                .into_iter()
                .filter(|vertex| !edge.contains(vertex))
                .collect::<Vec<_>>();
            [
                [edge[0], opposite[0], opposite[1]],
                [edge[1], opposite[0], opposite[1]],
            ]
        })
        .map(|mut face| {
            face.sort_unstable();
            face
        })
        .collect::<Vec<_>>();
    boundary_faces.sort_unstable();
    boundary_faces.dedup();
    boundary_faces
}

fn point_is_in_edge_star_kernel(
    topology: &DelaunayVolumeTopology,
    star: &[u32],
    edge: [u32; 2],
    point: [f64; 3],
    construction: Point3Construction,
) -> Result<bool, DelaunayVolumeSliverError> {
    let probe = DelaunayVolumeNode {
        identity: kernel_probe_identity(point),
        coordinates_m: point,
        construction,
    };
    for tetrahedron in star {
        let opposite = topology.tetrahedra[*tetrahedron as usize]
            .vertex_indices
            .into_iter()
            .filter(|vertex| !edge.contains(vertex))
            .collect::<Vec<_>>();
        for (face_endpoint, interior_endpoint) in [(edge[0], edge[1]), (edge[1], edge[0])] {
            let face = [
                topology.nodes[face_endpoint as usize],
                topology.nodes[opposite[0] as usize],
                topology.nodes[opposite[1] as usize],
            ];
            let expected = orient_volume_nodes([
                face[0],
                face[1],
                face[2],
                topology.nodes[interior_endpoint as usize],
            ])
            .map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("edge-star kernel predicate rejected source geometry: {failure:?}"),
                )
            })?;
            let observed =
                orient_volume_nodes([face[0], face[1], face[2], probe]).map_err(|failure| {
                    error(
                        DelaunayVolumeSliverErrorKind::InvalidTopology,
                        format!(
                            "edge-star kernel predicate rejected candidate geometry: {failure:?}"
                        ),
                    )
                })?;
            if expected == PredicateSign::Zero || observed != expected {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn kernel_probe_identity(point: [f64; 3]) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(KERNEL_PROBE_IDENTITY_DOMAIN);
    for coordinate in point {
        hasher.update(coordinate.to_bits().to_be_bytes());
    }
    StableDigest::from_bytes(hasher.finalize().into())
}

fn coordinate_key(point: [f64; 3]) -> [u64; 3] {
    point.map(f64::to_bits)
}

fn split_minimum_jacobian(
    topology: &DelaunayVolumeTopology,
    star: &[u32],
    edge: [u32; 2],
    inserted: [f64; 3],
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> f64 {
    star.iter()
        .flat_map(|tetrahedron| {
            let opposite = topology.tetrahedra[*tetrahedron as usize]
                .vertex_indices
                .into_iter()
                .filter(|vertex| !edge.contains(vertex))
                .map(|vertex| topology.nodes[vertex as usize].coordinates_m)
                .collect::<Vec<_>>();
            let edge_points = edge.map(|vertex| topology.nodes[vertex as usize].coordinates_m);
            [
                [edge_points[0], inserted, opposite[0], opposite[1]],
                [inserted, edge_points[1], opposite[0], opposite[1]],
            ]
        })
        .filter_map(|points| {
            super::super::volume_quality::evaluate_tetrahedron_geometry(points, metric)
                .map(|quality| quality.metric_scaled_jacobian)
        })
        .min_by(f64::total_cmp)
        .unwrap_or(0.0)
}

fn average(points: impl Iterator<Item = [f64; 3]>) -> [f64; 3] {
    let (sum, count) = points.fold(([0.0; 3], 0_u64), |(mut sum, count), point| {
        for axis in 0..3 {
            sum[axis] += point[axis];
        }
        (sum, count + 1)
    });
    sum.map(|value| value / count as f64)
}

fn insertion_identity(
    source: [StableDigest; 4],
    edge: [StableDigest; 2],
    index: u16,
) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(INSERTION_IDENTITY_DOMAIN);
    for identity in source.into_iter().chain(edge) {
        hasher.update(identity.bytes());
    }
    hasher.update(index.to_be_bytes());
    StableDigest::from_bytes(hasher.finalize().into())
}

pub(in crate::cdt) fn insertion_identity_is_valid(
    insertion: &DelaunayVolumeSliverInsertion,
) -> bool {
    insertion.inserted_node.identity
        == insertion_identity(
            insertion.source_tetrahedron_node_identities,
            insertion.source_edge_node_identities,
            insertion.construction_point_index,
        )
}

fn reject(counts: &mut BTreeMap<&'static str, u64>, reason: &'static str) {
    *counts.entry(reason).or_default() += 1;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    use runmat_meshing_core::{NeverCancelled, StableDigest};

    use super::{
        edge_candidate_policy, split_complete_edge_star, EdgeCandidatePolicy, EdgeStarSplit,
    };
    use crate::cdt::{
        topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions,
        DelaunayVolumeNode, DelaunayVolumeSliverOptions,
    };

    #[test]
    fn protected_facets_allow_off_edge_sliver_candidates_but_segments_do_not() {
        let identity = |value| StableDigest::from_bytes([value; 32]);
        let edge = [identity(1), identity(2)];
        let facets = [[identity(1), identity(2), identity(3)]];

        assert_eq!(
            edge_candidate_policy(edge, &facets, &BTreeSet::new()),
            EdgeCandidatePolicy::OffEdgeOnly
        );
        assert_eq!(
            edge_candidate_policy(edge, &[], &BTreeSet::new()),
            EdgeCandidatePolicy::IncludeMidpoint
        );
        assert_eq!(
            edge_candidate_policy(edge, &facets, &BTreeSet::from([edge])),
            EdgeCandidatePolicy::ExcludedSegment
        );
    }

    #[test]
    fn complete_edge_star_split_preserves_the_cavity_boundary() {
        let identity = |value| StableDigest::from_bytes([value; 32]);
        let node = |value, coordinates| DelaunayVolumeNode::explicit(identity(value), coordinates);
        let nodes = vec![
            node(1, [0.0, 0.0, -1.0]),
            node(2, [0.0, 0.0, 1.0]),
            node(3, [1.0, 0.0, 0.0]),
            node(4, [-0.5, 0.866_025_403_784, 0.0]),
            node(5, [-0.5, -0.866_025_403_784, 0.0]),
        ];
        let region = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: "solid".to_owned(),
            assembly_path: Vec::new(),
        };
        let topology = build_delaunay_volume_topology_with_regions(
            nodes,
            vec![
                ([0, 1, 2, 3], Some(region.clone())),
                ([0, 1, 3, 4], Some(region.clone())),
                ([0, 1, 4, 2], Some(region)),
            ],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let mut inserted_identity = [3; 32];
        inserted_identity[31] = 4;
        let inserted = DelaunayVolumeNode::explicit(
            StableDigest::from_bytes(inserted_identity),
            [0.05, 0.03, 0.02],
        );

        let EdgeStarSplit::Topology(split) = split_complete_edge_star(
            &topology,
            inserted,
            [identity(1), identity(2)],
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap() else {
            panic!("strict interior edge-star point should be admissible");
        };

        assert_eq!(split.nodes.len(), 6);
        assert_eq!(split.tetrahedra.len(), 6);
        assert_eq!(split.incidence.boundary_facets.len(), 6);
    }
}
