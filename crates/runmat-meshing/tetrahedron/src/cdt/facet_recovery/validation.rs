use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    construct_delaunay_facet_recovery, error, node_index, segment_error, validate_options,
    DelaunayConstraints, DelaunayFacetRecovery, DelaunayFacetRecoveryError,
    DelaunayFacetRecoveryErrorKind, DelaunayFacetRecoveryOptions, DelaunaySegmentRecovery,
    DelaunayVolumeTopology, FacetRecoveryWork,
};
use crate::cdt::{
    segment_recovery::validate_delaunay_segment_recovery_on_topology,
    validate_delaunay_segment_recovery,
};

pub fn validate_delaunay_facet_recovery(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayFacetRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayFacetRecoveryError> {
    validate_options(options)?;
    if recovery.facets.len() != constraints.facets.len() {
        return Err(error(
            DelaunayFacetRecoveryErrorKind::InvalidConstraints,
            None,
            "facet recovery evidence count does not match the constraint inventory",
        ));
    }
    if !recovery
        .segment_recovery
        .topology
        .incidence
        .regions
        .is_empty()
    {
        return Err(error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            None,
            "facet recovery prerequisite has assigned regions",
        ));
    }
    validate_inputs(
        &recovery.segment_recovery,
        constraints,
        options,
        cancellation,
    )?;
    validate_steiner_insertions(recovery, constraints, options)?;
    let mut protected_faces = recovery
        .facets
        .iter()
        .flat_map(|facet| facet.triangles.iter())
        .map(|triangle| {
            let mut identities = triangle.node_identities;
            identities.sort_unstable();
            identities
        })
        .collect::<Vec<_>>();
    protected_faces.sort_unstable();
    protected_faces.dedup();
    validate_delaunay_segment_recovery_on_topology(
        &recovery.segment_recovery,
        &recovery.topology,
        constraints,
        &protected_faces,
        options.segment_recovery,
        cancellation,
    )
    .map_err(segment_error)?;
    let expected = construct_delaunay_facet_recovery(
        recovery.segment_recovery.clone(),
        constraints,
        options,
        cancellation,
    )?;
    if recovery != &expected {
        return Err(error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            None,
            "facet recovery differs from deterministic replay of its segment-recovery prerequisite",
        ));
    }
    Ok(())
}

fn validate_steiner_insertions(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayFacetRecoveryOptions,
) -> Result<(), DelaunayFacetRecoveryError> {
    if recovery.steiner_insertions.len() as u64 > options.maximum_cavity_steiner_nodes
        || !recovery.steiner_insertions.windows(2).all(|pair| {
            (
                pair[0].constraint_index,
                pair[0].support_node_identities,
                pair[0].insertion_round,
                pair[0].candidate_rank,
                pair[0].node_identity,
            ) < (
                pair[1].constraint_index,
                pair[1].support_node_identities,
                pair[1].insertion_round,
                pair[1].candidate_rank,
                pair[1].node_identity,
            )
        })
    {
        return Err(error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            None,
            "facet Steiner insertion lineage is over budget or noncanonical",
        ));
    }
    let prerequisite_nodes = recovery
        .segment_recovery
        .topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    let introduced_nodes = recovery
        .topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .filter(|identity| !prerequisite_nodes.contains(identity))
        .collect::<BTreeSet<_>>();
    let mut evidenced_nodes = BTreeSet::new();
    for insertion in &recovery.steiner_insertions {
        if constraints
            .facets
            .get(insertion.constraint_index as usize)
            .is_none()
        {
            return Err(error(
                DelaunayFacetRecoveryErrorKind::InvalidConstraints,
                Some(insertion.constraint_index),
                "facet Steiner insertion names an unknown constraint",
            ));
        }
        let mut support = insertion.support_node_identities;
        support.sort_unstable();
        if support != insertion.support_node_identities
            || !support
                .iter()
                .all(|identity| prerequisite_nodes.contains(identity))
            || super::cavity::steiner_identity(
                support,
                insertion.insertion_round,
                insertion.candidate_rank,
            ) != insertion.node_identity
            || !evidenced_nodes.insert(insertion.node_identity)
        {
            return Err(error(
                DelaunayFacetRecoveryErrorKind::InvalidTopology,
                Some(insertion.constraint_index),
                "facet Steiner insertion lineage is inconsistent with its canonical support",
            ));
        }
    }
    if evidenced_nodes != introduced_nodes {
        return Err(error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            None,
            "facet Steiner insertion lineage does not exactly cover introduced nodes",
        ));
    }
    Ok(())
}

pub(super) fn validate_inputs(
    segment_recovery: &DelaunaySegmentRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayFacetRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayFacetRecoveryError> {
    validate_delaunay_segment_recovery(
        segment_recovery,
        constraints,
        options.segment_recovery,
        cancellation,
    )
    .map_err(segment_error)
}

pub(super) fn face_exists(
    topology: &DelaunayVolumeTopology,
    identities: [StableDigest; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let indices = identities.map(|identity| {
        node_index(topology, identity).ok_or_else(|| {
            error(
                DelaunayFacetRecoveryErrorKind::InvalidTopology,
                Some(constraint_index),
                "facet node is missing from topology",
            )
        })
    });
    let [first, second, third] = indices;
    let [first, second, third] = [first?, second?, third?];
    for tetrahedron_index in &topology.incidence.vertex_stars[first] {
        work.search_step(constraint_index)?;
        let tetrahedron = &topology.tetrahedra[*tetrahedron_index as usize];
        if tetrahedron.vertex_indices.contains(&(second as u32))
            && tetrahedron.vertex_indices.contains(&(third as u32))
        {
            return Ok(true);
        }
    }
    Ok(false)
}
