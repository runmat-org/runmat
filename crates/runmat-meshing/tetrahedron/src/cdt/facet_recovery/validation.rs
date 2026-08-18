use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::support::facet_support;
use super::{
    error, node_index, segment_error, validate_options, DelaunayConstraints, DelaunayFacetRecovery,
    DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind, DelaunayFacetRecoveryOptions,
    DelaunaySegmentRecovery, DelaunayVolumeTopology, FacetRecoveryWork,
};
use crate::cdt::validate_delaunay_segment_recovery;

pub fn validate_delaunay_facet_recovery(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayFacetRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayFacetRecoveryError> {
    validate_options(options)?;
    validate_inputs(
        &recovery.segment_recovery,
        constraints,
        options,
        cancellation,
    )?;
    if recovery.facets.len() != constraints.facets.len() {
        return Err(error(
            DelaunayFacetRecoveryErrorKind::InvalidConstraints,
            None,
            "facet recovery evidence count does not match the constraint inventory",
        ));
    }
    let mut work = FacetRecoveryWork::new(options, cancellation);
    for (expected_index, recovered) in recovery.facets.iter().enumerate() {
        let expected = facet_support(
            &recovery.segment_recovery,
            constraints,
            expected_index as u32,
            &mut work,
        )?;
        if recovered.constraint_index != expected_index as u32 || recovered.triangles != expected {
            return Err(error(
                DelaunayFacetRecoveryErrorKind::InvalidConstraints,
                Some(expected_index as u32),
                "recovered facet does not retain its oriented constraint support",
            ));
        }
        for triangle in &recovered.triangles {
            if !face_exists(
                &recovery.segment_recovery.topology,
                triangle.node_identities,
                expected_index as u32,
                &mut work,
            )? {
                return Err(error(
                    DelaunayFacetRecoveryErrorKind::InvalidTopology,
                    Some(expected_index as u32),
                    "recovered facet support is absent from tetrahedron face incidence",
                ));
            }
        }
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
