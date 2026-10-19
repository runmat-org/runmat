use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::quality::{
    predicate::{
        orient3d, point_in_closed_triangle_surface, tetrahedron_centroid, tetrahedron_volume,
        Point3, PointInClosedSurface, PredicateSign,
    },
    tolerance::MeshingTolerance,
};

use super::ConstrainedCavity;

/// Returns the canonical star refill for one boundary apex when every generated tetrahedron is
/// inside the cavity and the refill reproduces the cavity boundary and volume exactly enough for
/// the shared meshing tolerance contract.
pub fn constrained_cavity_star_refill_for_apex(
    cavity: &ConstrainedCavity,
    apex: u32,
    mut coordinates: impl FnMut(u32) -> Option<Point3>,
) -> Option<Vec<[u32; 4]>> {
    let boundary = cavity
        .boundary_faces
        .iter()
        .map(|face| coordinates_for(face.node_ids, &mut coordinates))
        .collect::<Option<Vec<_>>>()?;
    let mut tetrahedra = cavity
        .boundary_faces
        .iter()
        .filter(|face| !face.node_ids.contains(&apex))
        .map(|face| [face.node_ids[0], face.node_ids[1], face.node_ids[2], apex])
        .collect::<Vec<_>>();
    if tetrahedra.is_empty() {
        return None;
    }
    tetrahedra
        .iter_mut()
        .for_each(|tetrahedron| tetrahedron.sort_unstable());
    tetrahedra.sort_unstable();
    if tetrahedra.windows(2).any(|pair| pair[0] == pair[1])
        || !valid_tetrahedra(
            &tetrahedra,
            &mut coordinates,
            &boundary,
            cavity.target_volume_m3,
        )
        || refill_boundary(&tetrahedra)
            != cavity
                .boundary_faces
                .iter()
                .map(|face| canonical_face(face.node_ids))
                .collect()
    {
        return None;
    }
    Some(tetrahedra)
}

pub fn constrained_cavity_boundary_node_ids(cavity: &ConstrainedCavity) -> BTreeSet<u32> {
    cavity
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids)
        .collect()
}

fn valid_tetrahedra(
    tetrahedra: &[[u32; 4]],
    coordinates: &mut impl FnMut(u32) -> Option<Point3>,
    boundary: &[[[f64; 3]; 3]],
    target_volume: f64,
) -> bool {
    let mut volume = 0.0;
    for tetrahedron in tetrahedra {
        let Some(points) = coordinates_for(*tetrahedron, &mut *coordinates) else {
            return false;
        };
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

fn coordinates_for<const N: usize>(
    node_ids: [u32; N],
    coordinates: &mut impl FnMut(u32) -> Option<Point3>,
) -> Option<[Point3; N]> {
    let mut points = [[0.0; 3]; N];
    for (slot, node_id) in node_ids.into_iter().enumerate() {
        points[slot] = coordinates(node_id)?;
    }
    Some(points)
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
