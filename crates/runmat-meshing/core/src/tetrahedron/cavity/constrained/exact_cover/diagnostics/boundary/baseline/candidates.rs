use super::*;

pub(super) struct BoundaryExactCoverCandidateSummary {
    pub(super) boundary_faces: BTreeSet<[u32; 3]>,
    pub(super) candidates: Vec<ConstrainedCavityRefillTetrahedron>,
    pub(super) solid_candidates: Vec<ConstrainedCavityRefillTetrahedron>,
    pub(super) face_candidate_counts: Vec<usize>,
    pub(super) solid_face_candidate_counts: Vec<usize>,
}

pub(super) fn boundary_exact_cover_candidate_summary(
    cavity: &ConstrainedCavity,
    node_ids: &[u32],
    boundary_node_map: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> BoundaryExactCoverCandidateSummary {
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut solid_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tetrahedron_touches_boundary(tetrahedron_node_ids, &boundary_faces) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        solid_candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    let face_candidate_counts = candidate_counts_by_boundary_face(&boundary_faces, &candidates);
    let solid_face_candidate_counts =
        candidate_counts_by_boundary_face(&boundary_faces, &solid_candidates);

    BoundaryExactCoverCandidateSummary {
        boundary_faces,
        candidates,
        solid_candidates,
        face_candidate_counts,
        solid_face_candidate_counts,
    }
}

fn tetrahedron_touches_boundary(
    tetrahedron_node_ids: [u32; 4],
    boundary_faces: &BTreeSet<[u32; 3]>,
) -> bool {
    tetrahedron_faces(tetrahedron_node_ids)
        .map(sorted_face)
        .iter()
        .any(|face| boundary_faces.contains(face))
}

fn candidate_counts_by_boundary_face(
    boundary_faces: &BTreeSet<[u32; 3]>,
    candidates: &[ConstrainedCavityRefillTetrahedron],
) -> Vec<usize> {
    boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect()
}
