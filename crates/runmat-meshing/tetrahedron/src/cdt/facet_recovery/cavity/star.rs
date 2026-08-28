use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::quality::{
    predicate::{
        orient3d, point_in_closed_triangle_surface, tetrahedron_centroid, tetrahedron_volume,
        PointInClosedSurface, PredicateSign,
    },
    tolerance::MeshingTolerance,
};

use super::{resource_or_cancelled, FacetRecoveryWork};
use crate::{
    cavity::constrained::ConstrainedCavity,
    cdt::{DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind, DelaunayVolumeTopology},
};

pub(in crate::cdt::facet_recovery) fn star_refill(
    cavity: &ConstrainedCavity,
    topology: &DelaunayVolumeTopology,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<[u32; 4]>>, DelaunayFacetRecoveryError> {
    let boundary = cavity
        .boundary_faces
        .iter()
        .map(|face| {
            face.node_ids
                .map(|node| topology.nodes[node as usize].coordinates_m)
        })
        .collect::<Vec<_>>();
    let apexes = cavity
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids)
        .collect::<BTreeSet<_>>();
    for apex in apexes {
        work.cavity_apex_attempt(constraint_index)?;
        let mut tetrahedra = cavity
            .boundary_faces
            .iter()
            .filter(|face| !face.node_ids.contains(&apex))
            .map(|face| [face.node_ids[0], face.node_ids[1], face.node_ids[2], apex])
            .collect::<Vec<_>>();
        if tetrahedra.is_empty() {
            continue;
        }
        tetrahedra
            .iter_mut()
            .for_each(|tetrahedron| tetrahedron.sort_unstable());
        tetrahedra.sort_unstable();
        if tetrahedra.windows(2).any(|pair| pair[0] == pair[1])
            || !valid_tetrahedra(&tetrahedra, topology, &boundary, cavity.target_volume_m3)
            || refill_boundary(&tetrahedra)
                != cavity
                    .boundary_faces
                    .iter()
                    .map(|face| canonical_face(face.node_ids))
                    .collect()
        {
            continue;
        }
        if tetrahedra.len() as u64 > work.options.maximum_cavity_tetrahedra {
            return Err(resource_or_cancelled(
                DelaunayFacetRecoveryErrorKind::ResourceLimit,
                constraint_index,
                "facet star refill tetrahedron limit exceeded".to_owned(),
            ));
        }
        return Ok(Some(tetrahedra));
    }
    Ok(None)
}

fn valid_tetrahedra(
    tetrahedra: &[[u32; 4]],
    topology: &DelaunayVolumeTopology,
    boundary: &[[[f64; 3]; 3]],
    target_volume: f64,
) -> bool {
    let mut volume = 0.0;
    for tetrahedron in tetrahedra {
        let points = tetrahedron.map(|node| topology.nodes[node as usize].coordinates_m);
        if !matches!(
            orient3d(points),
            Ok(PredicateSign::Positive | PredicateSign::Negative)
        ) || point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            boundary,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            return false;
        }
        volume += tetrahedron_volume(points);
    }
    let tolerance = target_volume.max(1.0e-18) * 1.0e-9;
    volume.is_finite() && (volume - target_volume).abs() <= tolerance
}

fn refill_boundary(tetrahedra: &[[u32; 4]]) -> BTreeSet<[u32; 3]> {
    let mut counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in tetrahedra {
        for opposite in 0..4 {
            let mut face = [0; 3];
            let mut cursor = 0;
            for (index, node) in tetrahedron.iter().enumerate() {
                if index != opposite {
                    face[cursor] = *node;
                    cursor += 1;
                }
            }
            *counts.entry(canonical_face(face)).or_default() += 1;
        }
    }
    if counts.values().any(|count| *count > 2) {
        return BTreeSet::new();
    }
    counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect()
}

fn canonical_face(mut face: [u32; 3]) -> [u32; 3] {
    face.sort_unstable();
    face
}
