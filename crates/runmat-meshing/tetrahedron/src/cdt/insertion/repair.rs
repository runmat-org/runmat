//! Shared physical topology admission before and after bounded local repair.
use super::*;

pub(in crate::cdt) fn restore_constrained_delaunay_topology(
    topology: DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    match validate_constrained_delaunay_volume_topology(
        &topology,
        protected_faces,
        options,
        cancellation,
    ) {
        Ok(()) => return Ok(topology),
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::InvalidTopology => {}
        Err(failure) => return Err(failure),
    }
    // Constrained recovery validates both bounded repair providers independently.
    let barriers = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    let mut work = InsertionWork::new(options, cancellation);
    let topology = legalization::mutable_repair::restore(topology, &barriers, &mut work)?;
    match validate_constrained_delaunay_volume_topology(
        &topology,
        protected_faces,
        options,
        cancellation,
    ) {
        Ok(()) => return Ok(topology),
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::InvalidTopology => {}
        Err(failure) => return Err(failure),
    }
    let repaired = legalization::local::restore_advanced(topology, &barriers, true, &mut work)?;
    let repaired = match validate_constrained_delaunay_volume_topology(
        &repaired,
        protected_faces,
        options,
        cancellation,
    ) {
        Ok(()) => return Ok(repaired),
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::InvalidTopology => {
            legalize_delaunay_volume_topology_complete(repaired, &barriers, true, &mut work)?
        }
        Err(failure) => return Err(failure),
    };
    validate_constrained_delaunay_volume_topology(
        &repaired,
        protected_faces,
        options,
        cancellation,
    )?;
    Ok(repaired)
}
