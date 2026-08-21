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
