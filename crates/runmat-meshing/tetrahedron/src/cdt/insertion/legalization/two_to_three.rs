//! Shared two-to-three cavity recipe and canonical application.
use super::*;

pub(super) fn replacement(
    topology: &DelaunayVolumeTopology,
    left_index: usize,
    right_index: usize,
    face: [u32; 3],
) -> Result<Vec<[u32; 4]>, DelaunayInsertionError> {
    let left = &topology.tetrahedra[left_index];
    let right = &topology.tetrahedra[right_index];
    if left.region_id != right.region_id {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "Delaunay legalization cannot cross a region boundary",
        ));
    }
    let left_apex = left
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !face.contains(vertex))
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "missing left apex",
            )
        })?;
    let right_apex = right
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !face.contains(vertex))
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "missing right apex",
            )
        })?;
    Ok([[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]]
        .map(|edge| [left_apex, right_apex, edge[0], edge[1]])
        .to_vec())
}

#[cfg(test)]
pub(super) fn apply_two_to_three_flip(
    topology: DelaunayVolumeTopology,
    left_index: usize,
    right_index: usize,
    face: [u32; 3],
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let created = replacement(&topology, left_index, right_index, face)?;
    apply_reconnection(topology, &[left_index, right_index], created, work)
}
