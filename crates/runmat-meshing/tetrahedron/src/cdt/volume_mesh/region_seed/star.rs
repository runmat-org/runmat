use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    quality::predicate::tetrahedron_volume, MeshingCancellationSignal, StableDigest,
};

use crate::cdt::{
    insertion::validate_constrained_delaunay_volume_topology,
    topology::build_delaunay_volume_topology_with_regions, DelaunayInsertionErrorKind,
    DelaunayInsertionOptions, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind,
    DelaunayVolumeMeshStage, DelaunayVolumeNode, DelaunayVolumeTopology,
};

/// Cones a star-shaped region boundary to its deterministic seed. Equal volume and identical
/// region-boundary faces are required before the topology is admitted; otherwise the caller
/// falls back to protected Bowyer-Watson insertion.
pub(super) fn try_seeded_star(
    topology: &DelaunayVolumeTopology,
    region: &PersistentEntityId,
    seed: DelaunayVolumeNode,
    protected_faces: &[[StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeMeshError> {
    let boundary = region_boundary_faces(topology, region);
    let boundary_identities = boundary_face_identities(topology, &boundary);
    if boundary.is_empty() {
        return Ok(None);
    }
    let expected_volume = region_volume(topology, region);
    let mut nodes = topology.nodes.clone();
    nodes.push(seed);
    nodes.sort_by_key(|node| node.identity);
    let node_indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    let seed_index = node_indices[&seed.identity];
    let remap = |old_index: u32| node_indices[&topology.nodes[old_index as usize].identity];

    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.region_id.as_ref() != Some(region))
        .map(|tetrahedron| {
            (
                tetrahedron.vertex_indices.map(remap),
                tetrahedron.region_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    tetrahedra.extend(boundary.iter().map(|face| {
        (
            [remap(face[0]), remap(face[1]), remap(face[2]), seed_index],
            Some(region.clone()),
        )
    }));
    let candidate = match build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.topology,
        cancellation,
    ) {
        Ok(candidate) => candidate,
        Err(failure) => return topology_result(failure.kind, failure.to_string()),
    };
    if !volumes_match(expected_volume, region_volume(&candidate, region))
        || boundary_face_identities(&candidate, &region_boundary_faces(&candidate, region))
            != boundary_identities
    {
        return Ok(None);
    }
    match validate_constrained_delaunay_volume_topology(
        &candidate,
        protected_faces,
        options,
        cancellation,
    ) {
        Ok(()) => Ok(Some(candidate)),
        Err(failure)
            if matches!(
                failure.kind,
                DelaunayInsertionErrorKind::ResourceLimit | DelaunayInsertionErrorKind::Cancelled
            ) =>
        {
            insertion_result(failure.kind, failure.to_string())
        }
        Err(_) => Ok(None),
    }
}

fn boundary_face_identities(
    topology: &DelaunayVolumeTopology,
    faces: &BTreeSet<[u32; 3]>,
) -> BTreeSet<[StableDigest; 3]> {
    faces
        .iter()
        .map(|face| {
            let mut identities = face.map(|vertex| topology.nodes[vertex as usize].identity);
            identities.sort_unstable();
            identities
        })
        .collect()
}

fn region_boundary_faces(
    topology: &DelaunayVolumeTopology,
    region: &PersistentEntityId,
) -> BTreeSet<[u32; 3]> {
    let mut faces = BTreeSet::new();
    for tetrahedron in &topology.tetrahedra {
        if tetrahedron.region_id.as_ref() != Some(region) {
            continue;
        }
        for opposite in 0..4 {
            let same_region_neighbor = tetrahedron.neighbors[opposite].is_some_and(|neighbor| {
                topology.tetrahedra[neighbor as usize].region_id.as_ref() == Some(region)
            });
            if same_region_neighbor {
                continue;
            }
            let mut face = [0u32; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            faces.insert(face);
        }
    }
    faces
}

fn region_volume(topology: &DelaunayVolumeTopology, region: &PersistentEntityId) -> f64 {
    topology
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.region_id.as_ref() == Some(region))
        .map(|tetrahedron| {
            tetrahedron_volume(tetrahedron.vertex_indices.map(|vertex| {
                topology.nodes[vertex as usize]
                    .coordinates_m
                    .canonical_value()
            }))
        })
        .sum()
}

fn volumes_match(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs()
            <= 4_096.0 * f64::EPSILON * left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
}

fn topology_result(
    kind: crate::cdt::DelaunayTopologyErrorKind,
    reason: String,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeMeshError> {
    match kind {
        crate::cdt::DelaunayTopologyErrorKind::ResourceLimit => Err(super::super::error(
            DelaunayVolumeMeshStage::PointSet,
            DelaunayVolumeMeshErrorKind::ResourceLimit,
            reason,
        )),
        crate::cdt::DelaunayTopologyErrorKind::Cancelled => Err(super::super::error(
            DelaunayVolumeMeshStage::PointSet,
            DelaunayVolumeMeshErrorKind::Cancelled,
            reason,
        )),
        _ => Ok(None),
    }
}

fn insertion_result(
    kind: DelaunayInsertionErrorKind,
    reason: String,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeMeshError> {
    match kind {
        DelaunayInsertionErrorKind::ResourceLimit => Err(super::super::error(
            DelaunayVolumeMeshStage::PointSet,
            DelaunayVolumeMeshErrorKind::ResourceLimit,
            reason,
        )),
        DelaunayInsertionErrorKind::Cancelled => Err(super::super::error(
            DelaunayVolumeMeshStage::PointSet,
            DelaunayVolumeMeshErrorKind::Cancelled,
            reason,
        )),
        _ => Ok(None),
    }
}
