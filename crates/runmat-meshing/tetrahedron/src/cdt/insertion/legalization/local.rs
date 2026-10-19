//! Local Delaunay repair includes both adding and removing edges and hull diagonals.
//! Neighbor-driven cavity reinsertion handles stars that simple flips cannot repair.
//! Global violation search is a separate, more expensive completeness search.
use super::*;
use crate::cdt::insertion::{resource, validate_options, DelaunayInsertionOptions};
use runmat_meshing_core::MeshingCancellationSignal;

pub(in crate::cdt) fn legalize_delaunay_volume_topology_locally_with_barriers(
    topology: DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    validate_options(options)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(resource("protected-face inventory limit exceeded"));
    }
    let protected_faces = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    let mut work = InsertionWork::new(options, cancellation);
    restore(topology, &protected_faces, true, &mut work)
}

pub(in crate::cdt::insertion) fn restore(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    // Recovery can enter here after insertion, not only through the public
    // restoration coordinator. Consume simple physical violations with the
    // shared maintained-incidence provider before the harder rotation and
    // reinsertion searches. All providers share this invocation's work budget.
    if protect_region_boundaries {
        topology = super::mutable_repair::restore(topology, protected_faces, work)?;
    }
    restore_advanced(topology, protected_faces, protect_region_boundaries, work)
}

/// The admission coordinator has already consumed incremental repair and
/// independently checked its result. Continue without rebuilding that inventory.
pub(in crate::cdt::insertion) fn restore_advanced(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut state = search_state::seed(&topology, work)?;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(state.fingerprint()) {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "local Delaunay reconnection repeated a topology",
            ));
        }
        if let Some((removed, created)) = first_illegal_boundary_edge(
            &topology,
            protected_faces,
            SphereRelationPolicy::PhysicalAdmission,
            work,
        )? {
            record_flip(work)?;
            (topology, state) = search_state::apply(topology, state, &removed, created, work)?;
            continue;
        }
        if let Some((removed, created)) = first_reducible_edge(
            &topology,
            protected_faces,
            protect_region_boundaries,
            true,
            work,
        )? {
            record_flip(work)?;
            (topology, state) = search_state::apply(topology, state, &removed, created, work)?;
            continue;
        }
        if let Some((left, right, face)) = first_illegal_face(
            &topology,
            protected_faces,
            protect_region_boundaries,
            None,
            SphereRelationPolicy::PhysicalAdmission,
            work,
        )? {
            record_flip(work)?;
            let created = two_to_three::replacement(&topology, left, right, face)?;
            (topology, state) =
                search_state::apply(topology, state, &[left, right], created, work)?;
            continue;
        }
        if let Some(repaired) =
            rotation::first_candidate(&topology, protected_faces, &visited, state, work)?
        {
            record_flip(work)?;
            (topology, state) = repaired;
            continue;
        }
        if let Some(repaired) =
            reinsertion::first_candidate(&topology, protected_faces, &visited, state, work)?
        {
            record_flip(work)?;
            (topology, state) = repaired;
            continue;
        }
        return Ok(topology);
    }
}

mod reinsertion;
mod rotation;

#[cfg(test)]
mod tests;
