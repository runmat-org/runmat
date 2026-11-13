use super::{resource_or_cancelled, FacetRecoveryWork};
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::{
    cavity::constrained::{
        constrained_cavity_boundary_node_ids, constrained_cavity_star_refill_for_apex,
        ConstrainedCavity,
    },
    cdt::{DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind},
};

pub(in crate::cdt::facet_recovery) fn star_refill(
    cavity: &ConstrainedCavity,
    topology: &impl TetrahedralTopology,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<[u32; 4]>>, DelaunayFacetRecoveryError> {
    for apex in constrained_cavity_boundary_node_ids(cavity) {
        work.cavity_apex_attempt(constraint_index)?;
        let Some(tetrahedra) = constrained_cavity_star_refill_for_apex(cavity, apex, |node| {
            topology
                .nodes()
                .get(node as usize)
                .map(|node| node.coordinates_m.canonical_value())
        }) else {
            continue;
        };
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
