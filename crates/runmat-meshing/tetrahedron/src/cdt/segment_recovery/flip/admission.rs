//! Checked recovery transaction; raw local edits never confer admission.
use super::*;

pub(in crate::cdt::segment_recovery) fn try_recover_edge_with_face_flip(
    topology: &mut TopologyMutation,
    left: StableDigest,
    right: StableDigest,
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    // Populate the original owner's persistent broad phase before branching;
    // successful canonical handoff can retain it under stable cell corners.
    topology.prepare_cell_spatial_index(&mut || work.search_cell_index_step(constraint_index))?;
    match recover_trial(
        topology,
        [left, right],
        protected_edges,
        protected_faces,
        constraint_index,
        work,
    )? {
        TrialRecovery::Accepted => return Ok(true),
        TrialRecovery::Rejected => {}
        TrialRecovery::Candidate(candidate) => {
            if let Some(admitted) = super::super::candidate::admit(
                candidate,
                [left, right],
                protected_edges,
                protected_faces,
                constraint_index,
                work,
            )? {
                *topology = retained_mutation(admitted, protected_faces, constraint_index, work)?;
                return Ok(true);
            }
        }
    }
    if protected_faces.is_empty() {
        let mut repaired = topology.cavity_trial();
        if crate::cdt::insertion::restore_admitted_mutation(
            &mut repaired,
            work.options.insertion,
            work.cancellation,
        )
        .map_err(|failure| super::super::insertion_error(failure, constraint_index))?
        {
            return if super::super::candidate::preserves_edges(
                &*repaired,
                [left, right],
                protected_edges,
                constraint_index,
                work,
            )? {
                {
                    repaired.commit();
                    Ok(true)
                }
            } else {
                Ok(false)
            };
        }
    }
    // A raw edit can create the requested edge but lose it again while
    // restoring empty spheres. That rejected transaction must not suppress
    // repair of the original neighborhood before subdivision is considered.
    let current = topology
        .snapshot(work.options.insertion.topology, work.cancellation)
        .map_err(|failure| invalid_topology(constraint_index, failure.to_string()))?;
    let repaired =
        match crate::cdt::insertion::legalize_delaunay_volume_topology_locally_with_barriers(
            current,
            protected_faces,
            work.options.insertion,
            work.cancellation,
        ) {
            Ok(repaired) => repaired,
            Err(failure) => {
                super::super::candidate::reject_or_propagate(&failure, constraint_index)?;
                return Ok(false);
            }
        };
    let admitted = super::super::candidate::admit(
        repaired,
        [left, right],
        protected_edges,
        protected_faces,
        constraint_index,
        work,
    )?;
    if let Some(admitted) = admitted {
        *topology = retained_mutation(admitted, protected_faces, constraint_index, work)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

enum TrialRecovery {
    Accepted,
    Rejected,
    Candidate(DelaunayVolumeTopology),
}

fn recover_trial(
    topology: &mut TopologyMutation,
    endpoints: [StableDigest; 2],
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<TrialRecovery, DelaunaySegmentRecoveryError> {
    let mut mutation = topology.cavity_trial();
    if !try_recover_edge_with_face_flip_mutation(
        &mut mutation,
        endpoints[0],
        endpoints[1],
        protected_edges,
        protected_faces,
        constraint_index,
        work,
    )? {
        return Ok(TrialRecovery::Rejected);
    }
    let physically_admitted = protected_faces.is_empty()
        && crate::cdt::insertion::restore_admitted_mutation(
            &mut mutation,
            work.options.insertion,
            work.cancellation,
        )
        .map_err(|failure| super::super::insertion_error(failure, constraint_index))?;
    if physically_admitted {
        if super::super::candidate::preserves_edges(
            &*mutation,
            endpoints,
            protected_edges,
            constraint_index,
            work,
        )? {
            mutation.commit();
            return Ok(TrialRecovery::Accepted);
        }
        return Ok(TrialRecovery::Rejected);
    }
    // Complete legacy admission still receives the original canonical candidate.
    // The local owner rolls back before any whole-owner adoption takes place.
    let candidate = mutation
        .snapshot(work.options.insertion.topology, work.cancellation)
        .map_err(|failure| invalid_topology(constraint_index, failure.to_string()))?;
    Ok(TrialRecovery::Candidate(candidate))
}

fn retained_mutation(
    topology: DelaunayVolumeTopology,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<TopologyMutation, DelaunaySegmentRecoveryError> {
    if protected_faces.is_empty() {
        let admitted = crate::cdt::insertion::AdmittedPhysicalTopology::verify(
            topology,
            work.options.insertion,
            work.cancellation,
        )
        .map_err(|failure| super::super::insertion_error(failure, constraint_index))?;
        Ok(TopologyMutation::from_admitted(admitted))
    } else {
        Ok(TopologyMutation::from_canonical(topology))
    }
}

#[cfg(test)]
mod tests;
