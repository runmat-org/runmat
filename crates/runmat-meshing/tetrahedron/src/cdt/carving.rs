use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    DelaunayConstraints, DelaunayFacetRecovery, DelaunayFacetRecoveryError,
    DelaunayFacetRecoveryErrorKind, DelaunayFacetRecoveryOptions, DelaunayVolumeTopology,
};

mod classification;
mod validation;
mod work;

use classification::classify_and_build;
pub use validation::validate_delaunay_carving;
use validation::{validate_inputs, validate_options};
use work::CarvingWork;

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayRegionSeed {
    pub region_id: PersistentEntityId,
    pub coordinates_m: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaunayVoidSeed {
    pub coordinates_m: [f64; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayCarvingSeeds {
    pub regions: Vec<DelaunayRegionSeed>,
    pub voids: Vec<DelaunayVoidSeed>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayCarvingOptions {
    pub facet_recovery: DelaunayFacetRecoveryOptions,
    pub maximum_region_seeds: u64,
    pub maximum_void_seeds: u64,
    pub maximum_location_steps: u64,
    pub maximum_flood_steps: u64,
}

impl Default for DelaunayCarvingOptions {
    fn default() -> Self {
        Self {
            facet_recovery: DelaunayFacetRecoveryOptions::default(),
            maximum_region_seeds: 1_000_000,
            maximum_void_seeds: 1_000_000,
            maximum_location_steps: 1_000_000_000,
            maximum_flood_steps: 2_000_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayCarving {
    pub topology: DelaunayVolumeTopology,
    pub removed_tetrahedra: Vec<[StableDigest; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayCarvingErrorKind {
    InvalidOptions,
    InvalidConstraints,
    InvalidTopology,
    InvalidSeeds,
    AmbiguousClassification,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayCarvingError {
    pub kind: DelaunayCarvingErrorKind,
    pub seed_index: Option<u32>,
    pub reason: String,
}

impl std::fmt::Display for DelaunayCarvingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay carving {:?} at seed {:?}: {}",
            self.kind, self.seed_index, self.reason
        )
    }
}

impl std::error::Error for DelaunayCarvingError {}

pub fn carve_delaunay_volume(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    seeds: &DelaunayCarvingSeeds,
    options: DelaunayCarvingOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayCarving, DelaunayCarvingError> {
    validate_options(options)?;
    validate_inputs(recovery, constraints, seeds, options, cancellation)?;
    let mut work = CarvingWork::new(options, cancellation);
    let carving = classify_and_build(recovery, seeds, &mut work)?;
    validate_delaunay_carving(
        recovery,
        constraints,
        seeds,
        &carving,
        options,
        cancellation,
    )?;
    Ok(carving)
}

fn facet_error(error_value: DelaunayFacetRecoveryError) -> DelaunayCarvingError {
    let kind = match error_value.kind {
        DelaunayFacetRecoveryErrorKind::InvalidOptions => DelaunayCarvingErrorKind::InvalidOptions,
        DelaunayFacetRecoveryErrorKind::ResourceLimit => DelaunayCarvingErrorKind::ResourceLimit,
        DelaunayFacetRecoveryErrorKind::Cancelled => DelaunayCarvingErrorKind::Cancelled,
        DelaunayFacetRecoveryErrorKind::InvalidTopology => {
            DelaunayCarvingErrorKind::InvalidTopology
        }
        DelaunayFacetRecoveryErrorKind::InvalidConstraints
        | DelaunayFacetRecoveryErrorKind::UnsatisfiableConstraint => {
            DelaunayCarvingErrorKind::InvalidConstraints
        }
    };
    error(kind, None, error_value.to_string())
}

fn resource(seed_index: Option<u32>, reason: &'static str) -> DelaunayCarvingError {
    error(DelaunayCarvingErrorKind::ResourceLimit, seed_index, reason)
}

fn error(
    kind: DelaunayCarvingErrorKind,
    seed_index: Option<u32>,
    reason: impl Into<String>,
) -> DelaunayCarvingError {
    DelaunayCarvingError {
        kind,
        seed_index,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "carving/tests.rs"]
mod tests;
