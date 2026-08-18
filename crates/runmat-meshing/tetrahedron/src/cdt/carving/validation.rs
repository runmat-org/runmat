use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    classification::classify_and_build, error, facet_error, CarvingWork, DelaunayCarving,
    DelaunayCarvingError, DelaunayCarvingErrorKind, DelaunayCarvingOptions, DelaunayConstraints,
    DelaunayFacetRecovery,
};
use crate::cdt::validate_delaunay_facet_recovery;

pub fn validate_delaunay_carving(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    carving: &DelaunayCarving,
    options: DelaunayCarvingOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayCarvingError> {
    validate_options(options)?;
    validate_inputs(recovery, constraints, options, cancellation)?;
    let mut work = CarvingWork::new(options, cancellation);
    let expected = classify_and_build(recovery, constraints, &mut work)?;
    if expected != *carving {
        return Err(error(
            DelaunayCarvingErrorKind::InvalidTopology,
            None,
            "carved topology or removal evidence differs from independent classification",
        ));
    }
    Ok(())
}

pub(super) fn validate_inputs(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayCarvingOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayCarvingError> {
    validate_delaunay_facet_recovery(recovery, constraints, options.facet_recovery, cancellation)
        .map_err(facet_error)
}

pub(super) fn validate_options(
    options: DelaunayCarvingOptions,
) -> Result<(), DelaunayCarvingError> {
    if options.maximum_flood_steps == 0 {
        return Err(error(
            DelaunayCarvingErrorKind::InvalidOptions,
            None,
            "carving flood-step limit must be nonzero",
        ));
    }
    Ok(())
}
