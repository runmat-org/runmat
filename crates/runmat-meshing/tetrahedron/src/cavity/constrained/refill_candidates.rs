use std::collections::BTreeMap;

use runmat_meshing_core::{
    predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
        Triangle3,
    },
    tolerance::MeshingTolerance,
};

mod boundary_node;
mod centroid;
mod multi_interior;
pub(super) use boundary_node::{
    boundary_node_refill_candidate, boundary_node_refill_rejection_reason,
    boundary_node_refill_validation_reason,
};
pub(super) use centroid::centroid_interior_refill_candidate;
#[cfg(test)]
pub(super) use multi_interior::multi_interior_exact_cover_failure_reason;
pub(super) use multi_interior::multi_interior_node_refill_candidate;

use super::{
    cavity_boundary_node_ids, exact_cover_refill_from_candidate_tetrahedra,
    refill_tetrahedra::{
        raw_refill_tetrahedron, raw_refill_tetrahedron_with_rejection_reason,
        refill_from_tetrahedra, refill_is_better,
    },
    tetrahedralize_points, ConnectivityPoint, ConstrainedCavity, ConstrainedCavityNode,
    ConstrainedCavityRefill, ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError,
};

pub(super) fn single_tetrahedron_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if node_ids.len() != 4 {
        return Ok(None);
    }
    let points = [
        boundary_nodes[&node_ids[0]],
        boundary_nodes[&node_ids[1]],
        boundary_nodes[&node_ids[2]],
        boundary_nodes[&node_ids[3]],
    ];
    let Some(tetrahedron) = raw_refill_tetrahedron(
        [node_ids[0], node_ids[1], node_ids[2], node_ids[3]],
        points,
        options,
    ) else {
        return Ok(None);
    };
    let refill =
        refill_from_tetrahedra(cavity, vec![tetrahedron], options.volume_relative_tolerance)?;
    Ok(Some(refill))
}

pub(super) fn two_interior_node_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    interior_candidates: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let mut best = None::<ConstrainedCavityRefill>;
    let mut first_rejection = None::<&'static str>;
    for left in 0..interior_candidates.len() {
        for right in (left + 1)..interior_candidates.len() {
            let pair = [
                interior_candidates[left].clone(),
                interior_candidates[right].clone(),
            ];
            let mut points = boundary_node_ids
                .iter()
                .map(|node_id| ConnectivityPoint {
                    node_id: *node_id,
                    coordinates_m: boundary_nodes[node_id],
                    is_super: false,
                })
                .collect::<Vec<_>>();
            points.extend(pair.iter().map(|node| ConnectivityPoint {
                node_id: node.node_id,
                coordinates_m: node.coordinates_m,
                is_super: false,
            }));
            let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
            for tetrahedron in tetrahedralize_points(&points) {
                let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
                let tetrahedron_points = tetrahedron
                    .vertices
                    .map(|index| points[index].coordinates_m);
                if point_in_closed_triangle_surface(
                    tetrahedron_centroid(tetrahedron_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
                {
                    continue;
                }
                match raw_refill_tetrahedron_with_rejection_reason(
                    node_ids,
                    tetrahedron_points,
                    options,
                ) {
                    Ok(tetrahedron) => refill_tetrahedra.push(tetrahedron),
                    Err(reason) => {
                        if first_rejection.is_none() {
                            first_rejection = Some(boundary_node_refill_rejection_reason(reason));
                        }
                    }
                }
            }
            if refill_tetrahedra.is_empty() {
                if first_rejection.is_none() {
                    first_rejection = Some("two_interior_delaunay_empty");
                }
                continue;
            }
            match refill_from_tetrahedra(
                cavity,
                refill_tetrahedra.clone(),
                options.volume_relative_tolerance,
            ) {
                Ok(mut refill) => {
                    refill.inserted_nodes = pair.to_vec();
                    if best
                        .as_ref()
                        .is_none_or(|current| refill_is_better(&refill, current))
                    {
                        best = Some(refill);
                    }
                }
                Err(err) => {
                    if let Some(mut refill) = exact_cover_refill_from_candidate_tetrahedra(
                        cavity,
                        &refill_tetrahedra,
                        options,
                    )? {
                        refill.inserted_nodes = pair.to_vec();
                        if best
                            .as_ref()
                            .is_none_or(|current| refill_is_better(&refill, current))
                        {
                            best = Some(refill);
                        }
                        continue;
                    }
                    if first_rejection.is_none() {
                        first_rejection = Some(boundary_node_refill_validation_reason(&err));
                    }
                }
            }
        }
    }
    Ok(best
        .map(Ok)
        .unwrap_or_else(|| Err(first_rejection.unwrap_or("two_interior_no_candidate"))))
}
