use runmat_meshing_core::quality::predicate::Triangle3;

use super::*;

mod on_demand;
pub(in crate::cavity::constrained) use on_demand::on_demand_interior_mate_faces_for_trace;

pub(in super::super) use on_demand::exact_cover_refill_from_on_demand_interior_mates;

pub(in super::super) fn boundary_node_exact_cover_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if node_ids.len() < 4
        || node_ids.len() > MAX_BOUNDARY_EXACT_COVER_NODES
        || cavity.boundary_faces.len() > MAX_BOUNDARY_EXACT_COVER_FACES
    {
        return Ok(None);
    }
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut all_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidate_keys = BTreeSet::<[u32; 4]>::new();
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
                    let touches_boundary = tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
                    let points = tetrahedron_node_ids.map(|node_id| boundary_nodes[&node_id]);
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
                        options,
                    ) {
                        if touches_boundary
                            && candidate_keys.insert(sorted_tetrahedron_nodes(tetrahedron.node_ids))
                        {
                            candidates.push(tetrahedron.clone());
                        }
                        all_candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    if candidates.is_empty() || candidates.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
        return Ok(None);
    }
    if let Some(refill) =
        exact_cover_refill_from_candidate_tetrahedra(cavity, &candidates, options)?
    {
        return Ok(Some(refill));
    }
    exact_cover_refill_from_on_demand_interior_mates(cavity, candidates, all_candidates, options)
}

pub(in super::super) fn exact_cover_refill_from_candidate_tetrahedra(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    if candidates.is_empty() {
        return Ok(None);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, candidates, options.volume_relative_tolerance);
    let Some(selected_indices) = search.search_best() else {
        return Ok(None);
    };
    let selected_tetrahedra = selected_indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect::<Vec<_>>();
    refill_from_tetrahedra(
        cavity,
        selected_tetrahedra,
        options.volume_relative_tolerance,
    )
    .map(Some)
}
