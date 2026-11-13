//! Governed cycle-state seed and cavity deltas for physical restoration.
//! Candidate hints never replace embedded-manifold or physical admission.
use super::*;
use crate::cdt::topology::ConnectivitySearchState;
pub(super) type RepairCandidate = (DelaunayVolumeTopology, ConnectivitySearchState);

pub(super) fn seed(
    topology: &DelaunayVolumeTopology,
    work: &mut InsertionWork<'_>,
) -> Result<ConnectivitySearchState, DelaunayInsertionError> {
    let mut state = ConnectivitySearchState::default();
    for cell in &topology.tetrahedra {
        work.checkpoint()?;
        state.insert(cell.vertex_indices);
    }
    Ok(state)
}

pub(super) fn replacement(
    mut state: ConnectivitySearchState,
    topology: &DelaunayVolumeTopology,
    removed: &[usize],
    created: &[[u32; 4]],
    work: &mut InsertionWork<'_>,
) -> Result<ConnectivitySearchState, DelaunayInsertionError> {
    for index in removed {
        work.checkpoint()?;
        state.remove(topology.tetrahedra[*index].vertex_indices);
    }
    for vertices in created {
        work.checkpoint()?;
        state.insert(*vertices);
    }
    Ok(state)
}

pub(super) fn apply(
    topology: DelaunayVolumeTopology,
    state: ConnectivitySearchState,
    removed: &[usize],
    created: Vec<[u32; 4]>,
    work: &mut InsertionWork<'_>,
) -> Result<RepairCandidate, DelaunayInsertionError> {
    let next = replacement(state, &topology, removed, &created, work)?;
    let topology = apply_reconnection(topology, removed, created, work)?;
    Ok((topology, next))
}

#[cfg(test)]
mod tests;
