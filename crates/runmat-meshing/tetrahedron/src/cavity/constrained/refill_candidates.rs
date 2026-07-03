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
mod two_interior;
pub(super) use boundary_node::{
    boundary_node_refill_candidate, boundary_node_refill_rejection_reason,
    boundary_node_refill_validation_reason,
};
pub(super) use centroid::centroid_interior_refill_candidate;
#[cfg(test)]
pub(super) use multi_interior::multi_interior_exact_cover_failure_reason;
pub(super) use multi_interior::multi_interior_node_refill_candidate;
pub(super) use two_interior::two_interior_node_refill_candidate;

use super::{
    cavity_boundary_node_ids,
    refill_tetrahedra::{raw_refill_tetrahedron, refill_from_tetrahedra},
    ConstrainedCavity, ConstrainedCavityRefill, ConstrainedCavityRefillOptions,
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
