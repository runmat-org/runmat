//! Conservative whole-segment bounds; every exact crossing remains a candidate.
use super::*;

pub(super) fn candidate_cells(
    topology: &impl TetrahedralTopology,
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Vec<u32>, DelaunaySegmentRecoveryError> {
    let points = [left, right]
        .into_iter()
        .flat_map(|vertex| {
            let bounds = topology.node_bounds(vertex);
            [bounds.min_m, bounds.max_m]
        })
        .collect::<Vec<_>>();
    let bounds = runmat_meshing_core::Aabb3::from_points(&points)
        .expect("canonical finite segment endpoints");
    topology
        .prepare_tetrahedron_candidates(&mut || work.search_cell_index_step(constraint_index))?;
    topology.tetrahedron_candidates_in_bounds(bounds, &mut || {
        work.search_cell_query_step(constraint_index)
    })
}
