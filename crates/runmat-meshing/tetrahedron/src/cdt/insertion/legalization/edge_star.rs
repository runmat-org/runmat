//! Exact geometric admission of boundary-linked edge-star proposals.
use super::*;
use crate::reconnect::BoundaryEdgeStarCandidates;

pub(super) fn replacement(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    candidates: &BoundaryEdgeStarCandidates<'_>,
    target: [u32; 2],
    work: &mut InsertionWork<'_>,
) -> Result<Option<Vec<[u32; 4]>>, DelaunayInsertionError> {
    let mut created = Vec::new();
    for cell in candidates.tetrahedra(target, &mut || work.checkpoint())? {
        work.predicate()?;
        if orient_volume_nodes(cell.map(|node| topology.nodes()[node as usize]))
            .map_err(super::super::predicate_error)?
            != PredicateSign::Zero
        {
            created.push(cell);
        }
    }
    Ok(
        (created.len() >= 2
            && local_tetrahedron_boundary_faces(&created) == *candidates.boundary())
        .then_some(created),
    )
}
