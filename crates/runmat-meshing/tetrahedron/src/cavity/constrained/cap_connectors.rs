use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, PointInClosedSurface, Triangle3,
    },
    quality::tolerance::MeshingTolerance,
};

use super::{
    raw_refill_tetrahedron_with_rejection_reason,
    topology::{sorted_face, sorted_tetrahedron_nodes, tetrahedron_faces},
    ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    MAX_CAP_SIDE_CONNECTORS_PER_CHAIN_FACE, MAX_CAP_SIDE_CONNECTOR_CHAIN_CANDIDATES,
    MAX_CAP_SIDE_CONNECTOR_CHAIN_DEPTH, MAX_CAP_SIDE_CONNECTOR_CHAIN_FACES_PER_DEPTH,
};

#[cfg(test)]
pub(super) fn append_cap_side_connector_tetrahedra(
    cap_tetrahedron_start: usize,
    cap_tetrahedron_count: usize,
    candidate_tetrahedra: &mut Vec<ConstrainedCavityRefillTetrahedron>,
    seen_tetrahedra: &mut BTreeSet<[u32; 4]>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    inserted_node_ids: &BTreeSet<u32>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> usize {
    let cap_tetrahedra = candidate_tetrahedra
        .iter()
        .skip(cap_tetrahedron_start)
        .take(cap_tetrahedron_count)
        .cloned()
        .collect::<Vec<_>>();
    let mut inserted_count = 0_usize;
    for cap_tetrahedron in cap_tetrahedra {
        for face in tetrahedron_faces(cap_tetrahedron.node_ids) {
            if !face
                .iter()
                .any(|node_id| inserted_node_ids.contains(node_id))
            {
                continue;
            }
            for node_id in node_points.keys().copied() {
                if face.contains(&node_id) {
                    continue;
                }
                let tetrahedron_node_ids = [face[0], face[1], face[2], node_id];
                if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                    continue;
                }
                let tetrahedron_points = tetrahedron_node_ids.map(|id| node_points[&id]);
                if point_in_closed_triangle_surface(
                    tetrahedron_centroid(tetrahedron_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
                {
                    continue;
                }
                let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                    tetrahedron_node_ids,
                    tetrahedron_points,
                    options,
                ) else {
                    continue;
                };
                candidate_tetrahedra.push(tetrahedron);
                inserted_count += 1;
            }
        }
    }
    inserted_count
        + append_cap_side_connector_chain_tetrahedra(
            candidate_tetrahedra,
            seen_tetrahedra,
            node_points,
            inserted_node_ids,
            boundary_triangles,
            options,
        )
}

pub(super) fn append_cap_side_connector_chain_tetrahedra(
    candidate_tetrahedra: &mut Vec<ConstrainedCavityRefillTetrahedron>,
    seen_tetrahedra: &mut BTreeSet<[u32; 4]>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    inserted_node_ids: &BTreeSet<u32>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> usize {
    let mut inserted_count = 0_usize;
    let mut processed_faces = BTreeSet::<[u32; 3]>::new();
    for _ in 0..MAX_CAP_SIDE_CONNECTOR_CHAIN_DEPTH {
        let frontier = open_inserted_node_faces(candidate_tetrahedra, inserted_node_ids)
            .into_iter()
            .filter(|face| !processed_faces.contains(face))
            .take(MAX_CAP_SIDE_CONNECTOR_CHAIN_FACES_PER_DEPTH)
            .collect::<Vec<_>>();
        if frontier.is_empty() {
            break;
        }
        let mut inserted_this_depth = 0_usize;
        for face in frontier {
            processed_faces.insert(face);
            let connectors = connector_tetrahedra_for_face(
                face,
                node_points,
                seen_tetrahedra,
                boundary_triangles,
                options,
                MAX_CAP_SIDE_CONNECTORS_PER_CHAIN_FACE,
            );
            for tetrahedron in connectors {
                if inserted_count >= MAX_CAP_SIDE_CONNECTOR_CHAIN_CANDIDATES {
                    return inserted_count;
                }
                if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron.node_ids)) {
                    continue;
                }
                candidate_tetrahedra.push(tetrahedron);
                inserted_count += 1;
                inserted_this_depth += 1;
            }
        }
        if inserted_this_depth == 0 {
            break;
        }
    }
    inserted_count
}

fn open_inserted_node_faces(
    candidate_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    inserted_node_ids: &BTreeSet<u32>,
) -> Vec<[u32; 3]> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in candidate_tetrahedra {
        for face in tetrahedron_faces(tetrahedron.node_ids).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    face_counts
        .into_iter()
        .filter_map(|(face, count)| {
            (count == 1
                && face
                    .iter()
                    .any(|node_id| inserted_node_ids.contains(node_id)))
            .then_some(face)
        })
        .collect()
}

fn connector_tetrahedra_for_face(
    face: [u32; 3],
    node_points: &BTreeMap<u32, [f64; 3]>,
    seen_tetrahedra: &BTreeSet<[u32; 4]>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
    limit: usize,
) -> Vec<ConstrainedCavityRefillTetrahedron> {
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for node_id in node_points.keys().copied() {
        if face.contains(&node_id) {
            continue;
        }
        let tetrahedron_node_ids = [face[0], face[1], face[2], node_id];
        if seen_tetrahedra.contains(&sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
            continue;
        }
        let tetrahedron_points = tetrahedron_node_ids.map(|id| node_points[&id]);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
            tetrahedron_node_ids,
            tetrahedron_points,
            options,
        ) else {
            continue;
        };
        candidates.push(tetrahedron);
    }
    candidates.sort_by(|left, right| {
        right
            .exact_scaled_jacobian
            .total_cmp(&left.exact_scaled_jacobian)
            .then_with(|| left.aspect_ratio.total_cmp(&right.aspect_ratio))
            .then_with(|| {
                sorted_tetrahedron_nodes(left.node_ids)
                    .cmp(&sorted_tetrahedron_nodes(right.node_ids))
            })
    });
    candidates.truncate(limit);
    candidates
}
