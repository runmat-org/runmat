use runmat_meshing_core::quality::predicate::Triangle3;

use super::*;

pub(in super::super) fn solid_empty_boundary_faces(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Vec<[u32; 3]> {
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut solid_faces = BTreeSet::<[u32; 3]>::new();
    for first in 0..boundary_node_ids.len() {
        for second in (first + 1)..boundary_node_ids.len() {
            for third in (second + 1)..boundary_node_ids.len() {
                for fourth in (third + 1)..boundary_node_ids.len() {
                    let tetrahedron_node_ids = [
                        boundary_node_ids[first],
                        boundary_node_ids[second],
                        boundary_node_ids[third],
                        boundary_node_ids[fourth],
                    ];
                    let candidate_faces = tetrahedron_faces(tetrahedron_node_ids).map(sorted_face);
                    if !candidate_faces
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_nodes[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    )
                    .is_ok()
                    {
                        for face in candidate_faces {
                            if boundary_faces.contains(&face) {
                                solid_faces.insert(face);
                            }
                        }
                    }
                }
            }
        }
    }
    boundary_faces
        .into_iter()
        .filter(|face| !solid_faces.contains(face))
        .collect()
}
