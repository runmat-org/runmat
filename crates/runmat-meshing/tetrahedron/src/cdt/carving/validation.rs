use std::collections::BTreeSet;

use runmat_geometry_core::PersistentEntityKind;
use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    classification::classify_and_build, error, facet_error, CarvingWork, DelaunayCarving,
    DelaunayCarvingError, DelaunayCarvingErrorKind, DelaunayCarvingOptions, DelaunayCarvingSeeds,
    DelaunayConstraints, DelaunayFacetRecovery,
};
use crate::cdt::validate_delaunay_facet_recovery;

pub fn validate_delaunay_carving(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    seeds: &DelaunayCarvingSeeds,
    carving: &DelaunayCarving,
    options: DelaunayCarvingOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayCarvingError> {
    validate_options(options)?;
    validate_inputs(recovery, constraints, seeds, options, cancellation)?;
    let mut work = CarvingWork::new(options, cancellation);
    let expected = classify_and_build(recovery, seeds, &mut work)?;
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
    seeds: &DelaunayCarvingSeeds,
    options: DelaunayCarvingOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayCarvingError> {
    validate_delaunay_facet_recovery(recovery, constraints, options.facet_recovery, cancellation)
        .map_err(facet_error)?;
    if seeds.regions.is_empty()
        || seeds.regions.len() as u64 > options.maximum_region_seeds
        || seeds.voids.len() as u64 > options.maximum_void_seeds
    {
        return Err(error(
            DelaunayCarvingErrorKind::InvalidSeeds,
            None,
            "carving requires bounded region seeds and bounded void seeds",
        ));
    }
    let mut coordinates = BTreeSet::new();
    for (index, seed) in seeds.regions.iter().enumerate() {
        seed.region_id.validate().map_err(|validation| {
            error(
                DelaunayCarvingErrorKind::InvalidSeeds,
                Some(index as u32),
                format!("region seed identity is invalid: {validation}"),
            )
        })?;
        if seed.region_id.kind != PersistentEntityKind::Region
            || index > 0 && seeds.regions[index - 1].region_id >= seed.region_id
            || !coordinates.insert(coordinate_key(seed.coordinates_m))
        {
            return Err(error(
                DelaunayCarvingErrorKind::InvalidSeeds,
                Some(index as u32),
                "region seeds require ordered unique region identities and positions",
            ));
        }
    }
    let mut previous_void = None;
    for (index, seed) in seeds.voids.iter().enumerate() {
        let key = coordinate_key(seed.coordinates_m);
        if previous_void.is_some_and(|previous| previous >= key) || !coordinates.insert(key) {
            return Err(error(
                DelaunayCarvingErrorKind::InvalidSeeds,
                Some(index as u32),
                "void seeds require ordered unique positions distinct from region seeds",
            ));
        }
        previous_void = Some(key);
    }
    if coordinates
        .iter()
        .flatten()
        .any(|bits| !f64::from_bits(*bits).is_finite())
    {
        return Err(error(
            DelaunayCarvingErrorKind::InvalidSeeds,
            None,
            "carving seed coordinates must be finite",
        ));
    }
    Ok(())
}

pub(super) fn validate_options(
    options: DelaunayCarvingOptions,
) -> Result<(), DelaunayCarvingError> {
    if options.maximum_region_seeds == 0
        || options.maximum_void_seeds == 0
        || options.maximum_location_steps == 0
        || options.maximum_flood_steps == 0
    {
        return Err(error(
            DelaunayCarvingErrorKind::InvalidOptions,
            None,
            "carving limits must be nonzero",
        ));
    }
    Ok(())
}

fn coordinate_key(coordinates: [f64; 3]) -> [u64; 3] {
    coordinates.map(|value| if value == 0.0 { 0 } else { value.to_bits() })
}
