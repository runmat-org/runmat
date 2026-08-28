use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use sha2::{Digest, Sha256};

use super::{
    checkpoint, error, quality_error, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind,
    DelaunayVolumeSliverOptions, DelaunayVolumeSliverRelocation,
};
use crate::cdt::{
    evaluate_delaunay_volume_quality, insertion::validate_constrained_delaunay_volume_topology,
    topology::build_delaunay_volume_topology_with_regions, validate_delaunay_volume_provenance,
    DelaunayInsertionErrorKind, DelaunayTetrahedronQuality, DelaunayTopologyErrorKind,
    DelaunayVolumeNode, DelaunayVolumeProvenance, DelaunayVolumeProvenanceErrorKind,
    DelaunayVolumeQuality, DelaunayVolumeRefinementInput, DelaunayVolumeTopology,
};

const RELOCATION_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/sliver-relocation/1\0";

pub(super) struct RelocationCandidate {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) quality: DelaunayVolumeQuality,
    pub(super) relocation: DelaunayVolumeSliverRelocation,
    spectrum: Vec<f64>,
}

pub(super) fn quality_spectrum(quality: &DelaunayVolumeQuality) -> Vec<f64> {
    let mut spectrum = quality
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.refinement_violation_ratio)
        .collect::<Vec<_>>();
    spectrum.sort_by(|left, right| right.total_cmp(left));
    spectrum
}

pub(super) fn relocation_candidates(
    input: DelaunayVolumeRefinementInput<'_>,
    source: &DelaunayTetrahedronQuality,
    current_spectrum: &[f64],
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<RelocationCandidate>, DelaunayVolumeSliverError> {
    let topology = input.topology;
    let protected = protected_nodes(topology, input.provenance);
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
    let possible_evaluations = (eligible_sources.len() as u64).saturating_mul(4);
    let mut candidates = Vec::new();
    let mut evaluations = 0u64;

    for source_identity in eligible_sources {
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
        let target = neighbor_centroid(topology, source_index)?;
        let source_coordinates = topology.nodes[source_index].coordinates_m;
        let mut fraction = 1.0;
        for _ in 0..4 {
            if evaluations >= options.maximum_candidate_evaluations_per_pass {
                break;
            }
            checkpoint(evaluations, options, cancellation)?;
            evaluations += 1;
            let coordinates_m = std::array::from_fn(|axis| {
                source_coordinates[axis] + fraction * (target[axis] - source_coordinates[axis])
            });
            fraction *= 0.5;
            let replacement_node = DelaunayVolumeNode {
                identity: replacement_identity(source_identity, coordinates_m),
                coordinates_m,
            };
            let Some(candidate_topology) = rebuild_relocated_star(
                topology,
                source_index,
                replacement_node,
                options,
                cancellation,
            )?
            else {
                continue;
            };
            match validate_constrained_delaunay_volume_topology(
                &candidate_topology,
                &protected_faces,
                options.insertion,
                cancellation,
            ) {
                Ok(()) => {}
                Err(failure) => match failure.kind {
                    DelaunayInsertionErrorKind::ResourceLimit => {
                        return Err(error(
                            DelaunayVolumeSliverErrorKind::ResourceLimit,
                            failure.to_string(),
                        ));
                    }
                    DelaunayInsertionErrorKind::Cancelled => {
                        return Err(error(
                            DelaunayVolumeSliverErrorKind::Cancelled,
                            failure.to_string(),
                        ));
                    }
                    _ => continue,
                },
            }
            match validate_delaunay_volume_provenance(
                &candidate_topology,
                input.provenance,
                input.quality_options.provenance,
                cancellation,
            ) {
                Ok(()) => {}
                Err(failure) => match failure.kind {
                    DelaunayVolumeProvenanceErrorKind::ResourceLimit => {
                        return Err(error(
                            DelaunayVolumeSliverErrorKind::ResourceLimit,
                            failure.to_string(),
                        ));
                    }
                    DelaunayVolumeProvenanceErrorKind::Cancelled => {
                        return Err(error(
                            DelaunayVolumeSliverErrorKind::Cancelled,
                            failure.to_string(),
                        ));
                    }
                    _ => continue,
                },
            }
            let candidate_quality = evaluate_delaunay_volume_quality(
                &candidate_topology,
                input.metric_request,
                input.provenance,
                input.quality_options,
                cancellation,
            )
            .map_err(quality_error)?;
            let spectrum = quality_spectrum(&candidate_quality);
            if compare_spectra(&spectrum, current_spectrum) != Ordering::Less {
                continue;
            }
            candidates.push(RelocationCandidate {
                topology: candidate_topology,
                quality: candidate_quality,
                relocation: DelaunayVolumeSliverRelocation {
                    source_node_identity: source_identity,
                    replacement_node,
                    source_tetrahedron_node_identities: source.node_identities,
                },
                spectrum,
            });
        }
        if evaluations >= options.maximum_candidate_evaluations_per_pass {
            break;
        }
    }
    candidates.sort_by(|left, right| {
        compare_spectra(&left.spectrum, &right.spectrum).then_with(|| {
            left.relocation
                .replacement_node
                .identity
                .cmp(&right.relocation.replacement_node.identity)
        })
    });
    if candidates.is_empty() && evaluations < possible_evaluations {
        return Err(error(
            DelaunayVolumeSliverErrorKind::ResourceLimit,
            format!(
                "sliver relocation exhausted {} candidate evaluations before completing {} legal evaluations",
                options.maximum_candidate_evaluations_per_pass, possible_evaluations
            ),
        ));
    }
    Ok(candidates)
}

fn protected_nodes(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
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

fn rebuild_relocated_star(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    replacement: DelaunayVolumeNode,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    if topology.nodes.iter().enumerate().any(|(index, node)| {
        index != source_index
            && (node.identity == replacement.identity
                || node.coordinates_m == replacement.coordinates_m)
    }) {
        return Ok(None);
    }
    let source_identity = topology.nodes[source_index].identity;
    let mut nodes = topology
        .nodes
        .iter()
        .copied()
        .filter(|node| node.identity != source_identity)
        .collect::<Vec<_>>();
    nodes.push(replacement);
    nodes.sort_by_key(|node| node.identity);
    let indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    let mut tetrahedra = Vec::with_capacity(topology.tetrahedra.len());
    for tetrahedron in &topology.tetrahedra {
        let identities = tetrahedron.vertex_indices.map(|vertex| {
            let identity = topology.nodes[vertex as usize].identity;
            if identity == source_identity {
                replacement.identity
            } else {
                identity
            }
        });
        let mut vertex_indices = [0; 4];
        for (slot, identity) in identities.into_iter().enumerate() {
            vertex_indices[slot] = *indices.get(&identity).ok_or_else(|| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "relocated topology remap is incomplete",
                )
            })?;
        }
        tetrahedra.push((vertex_indices, tetrahedron.region_id.clone()));
    }
    match build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(candidate) => Ok(Some(candidate)),
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

fn replacement_identity(source: StableDigest, coordinates_m: [f64; 3]) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(RELOCATION_IDENTITY_DOMAIN);
    hasher.update(source.bytes());
    for coordinate in coordinates_m {
        hasher.update(coordinate.to_bits().to_be_bytes());
    }
    StableDigest::from_bytes(hasher.finalize().into())
}

pub(super) fn relocation_identity_is_valid(relocation: &DelaunayVolumeSliverRelocation) -> bool {
    relocation.replacement_node.identity
        == replacement_identity(
            relocation.source_node_identity,
            relocation.replacement_node.coordinates_m,
        )
}

fn compare_spectra(left: &[f64], right: &[f64]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.total_cmp(right))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}
