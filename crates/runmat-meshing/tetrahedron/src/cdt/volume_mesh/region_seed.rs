use std::collections::BTreeSet;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    quality::predicate::{orient3d, tetrahedron_volume, PredicateSign},
    MeshingCancellationSignal, StableDigest,
};
use sha2::{Digest, Sha256};

mod admission;
mod star;

use crate::cdt::{
    insertion::{
        insert_delaunay_volume_node_locally_with_barriers,
        validate_constrained_delaunay_volume_topology,
    },
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayInsertionOptions,
    DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshStage,
    DelaunayVolumeNode, DelaunayVolumeTopology,
};

const REGION_SEED_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/region-seed/1\0";

pub(super) struct SeededRegionTopology {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) node_identities: Vec<StableDigest>,
}

/// Inserts one deterministic interior node per carved region. Besides improving the initial
/// volume distribution, this breaks the co-spherical boundary degeneracy of primitives such as
/// spheres before quality refinement begins.
pub(super) fn seed_carved_regions(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &[[StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SeededRegionTopology, DelaunayVolumeMeshError> {
    let regions = topology
        .tetrahedra
        .iter()
        .filter_map(|tetrahedron| tetrahedron.region_id.clone())
        .collect::<BTreeSet<_>>();
    let mut node_identities = Vec::with_capacity(regions.len());
    for region in &regions {
        let coordinates_m = region_seed(&topology, region)?;
        if let Some(existing) = topology
            .nodes
            .iter()
            .find(|node| node.coordinates_m.canonical_value() == coordinates_m)
        {
            if existing.identity == region_seed_identity(region) {
                node_identities.push(existing.identity);
            }
            continue;
        }
        let node = DelaunayVolumeNode::explicit(region_seed_identity(region), coordinates_m);
        if let Some(starred) = star::try_seeded_star(
            &topology,
            region,
            node,
            protected_faces,
            options,
            cancellation,
        )? {
            topology = starred;
            node_identities.push(node.identity);
            continue;
        }
        let candidate = match insert_delaunay_volume_node_locally_with_barriers(
            topology.clone(),
            node,
            protected_faces,
            options,
            cancellation,
        ) {
            Ok(candidate) => candidate,
            Err(failure) if seed_is_unavailable(&failure) => continue,
            Err(failure) => return Err(region_seed_insertion_error(failure)),
        };
        let candidate =
            match admission::admit_or_repair(candidate, protected_faces, options, cancellation) {
                Ok(candidate) => candidate,
                Err(failure) if seed_is_unavailable(&failure) => continue,
                Err(failure) => return Err(region_seed_insertion_error(failure)),
            };
        topology = candidate;
        node_identities.push(node.identity);
    }
    node_identities.sort_unstable();
    Ok(SeededRegionTopology {
        topology,
        node_identities,
    })
}

fn seed_is_unavailable(failure: &DelaunayInsertionError) -> bool {
    matches!(
        failure.kind,
        DelaunayInsertionErrorKind::InvalidNode
            | DelaunayInsertionErrorKind::InvalidTopology
            | DelaunayInsertionErrorKind::PointOutsideTopology
    )
}

fn region_seed_insertion_error(failure: DelaunayInsertionError) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        DelaunayInsertionErrorKind::ResourceLimit => DelaunayVolumeMeshErrorKind::ResourceLimit,
        DelaunayInsertionErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
        DelaunayInsertionErrorKind::InvalidNode
        | DelaunayInsertionErrorKind::PointOutsideTopology => {
            DelaunayVolumeMeshErrorKind::InvalidGeometry
        }
        DelaunayInsertionErrorKind::InvalidOptions
        | DelaunayInsertionErrorKind::InvalidTopology => {
            DelaunayVolumeMeshErrorKind::InvalidTopology
        }
    };
    super::error(
        DelaunayVolumeMeshStage::PointSet,
        kind,
        format!("region seed insertion failed: {failure}"),
    )
}

fn region_seed(
    topology: &DelaunayVolumeTopology,
    region: &PersistentEntityId,
) -> Result<[f64; 3], DelaunayVolumeMeshError> {
    let tetrahedra = topology
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.region_id.as_ref() == Some(region))
        .map(|tetrahedron| {
            let points = tetrahedron.vertex_indices.map(|vertex| {
                topology.nodes[vertex as usize]
                    .coordinates_m
                    .canonical_value()
            });
            (points, tetrahedron_volume(points))
        })
        .collect::<Vec<_>>();
    if tetrahedra.is_empty() {
        return Err(super::error(
            DelaunayVolumeMeshStage::Carving,
            DelaunayVolumeMeshErrorKind::InvalidTopology,
            "carved region has no tetrahedron for interior seeding",
        ));
    }
    let weighted =
        super::centroid::canonical_weighted_average(tetrahedra.iter().map(|(points, volume)| {
            (
                super::centroid::canonical_average(*points).expect("tetrahedron has four points"),
                *volume,
            )
        }))
        .ok_or_else(|| {
            super::error(
                DelaunayVolumeMeshStage::Carving,
                DelaunayVolumeMeshErrorKind::InvalidTopology,
                "carved region has no finite positive volume for interior seeding",
            )
        })?;
    if point_is_in_region(&tetrahedra, weighted)? {
        return Ok(weighted);
    }
    let points = tetrahedra
        .iter()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("nonempty region tetrahedron inventory")
        .0;
    Ok(std::array::from_fn(|axis| {
        points[0][axis]
            + (points[1][axis] - points[0][axis] + points[2][axis] - points[0][axis]
                + points[3][axis]
                - points[0][axis])
                * 0.25
    }))
}

fn point_is_in_region(
    tetrahedra: &[([[f64; 3]; 4], f64)],
    point: [f64; 3],
) -> Result<bool, DelaunayVolumeMeshError> {
    for (points, _) in tetrahedra {
        let mut inside = true;
        for replace in 0..4 {
            let mut simplex = *points;
            simplex[replace] = point;
            if orient3d(simplex).map_err(|failure| {
                super::error(
                    DelaunayVolumeMeshStage::Carving,
                    DelaunayVolumeMeshErrorKind::InvalidGeometry,
                    format!("region seed containment predicate failed: {failure:?}"),
                )
            })? != PredicateSign::Positive
            {
                inside = false;
                break;
            }
        }
        if inside {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn region_seed_identity(region: &PersistentEntityId) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(REGION_SEED_IDENTITY_DOMAIN);
    hasher.update((region.source_topology_id.len() as u64).to_be_bytes());
    hasher.update(region.source_topology_id.as_bytes());
    hasher.update((region.assembly_path.len() as u64).to_be_bytes());
    for segment in &region.assembly_path {
        hasher.update((segment.len() as u64).to_be_bytes());
        hasher.update(segment.as_bytes());
    }
    StableDigest::from_bytes(hasher.finalize().into())
}
