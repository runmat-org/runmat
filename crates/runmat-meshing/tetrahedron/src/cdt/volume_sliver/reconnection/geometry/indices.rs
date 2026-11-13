//! Order only this cavity's corners before invoking index-based flip kernels.
//! A working slot is not a canonical ordinal. Local ranks preserve the original
//! identity ordering without remapping any unrelated node or cell.
use super::*;

pub(super) fn flip<const N: usize, E>(
    topology: &(impl VolumeTopologyRead + ?Sized),
    cells: [u32; N],
    make: impl FnOnce(
        [LocalTetrahedron; N],
        &dyn Fn(u32) -> u32,
    ) -> Result<LocalTetrahedronFlipCandidate, E>,
) -> Result<LocalTetrahedronFlipCandidate, E> {
    let mut nodes = cells
        .iter()
        .flat_map(|index| topology.cell(*index).vertex_indices)
        .collect::<Vec<_>>();
    nodes.sort_unstable_by_key(|index| topology.node(*index).identity);
    nodes.dedup();
    let rank = |index| {
        nodes
            .binary_search_by_key(&topology.node(index).identity, |handle| {
                topology.node(*handle).identity
            })
            .expect("local corner belongs to cavity") as u32
    };
    let local = cells.map(|index| LocalTetrahedron {
        tetrahedron_id: index,
        node_ids: topology.cell(index).vertex_indices.map(rank),
    });
    let mut candidate = make(local, &rank)?;
    candidate
        .created_tetrahedra
        .iter_mut()
        .for_each(|vertices| {
            *vertices = vertices.map(|rank| nodes[rank as usize]);
        });
    candidate.shared_face = candidate
        .shared_face
        .map(|face| face.map(|rank| nodes[rank as usize]));
    candidate.shared_edge = candidate
        .shared_edge
        .map(|edge| edge.map(|rank| nodes[rank as usize]));
    Ok(candidate)
}
