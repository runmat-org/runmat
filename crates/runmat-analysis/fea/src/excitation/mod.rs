//! Realization of solver-neutral authored excitations into FEA-owned numerical inputs.

mod action;
mod activation;
mod coupling;
mod evidence;
mod model;
mod projection;
mod realize;
mod sample;
mod target;
mod time_grid;

pub(crate) use coupling::realize_coupling_contexts;
pub use evidence::{
    ProjectionSample, ProjectionScalar, ProjectionVector3, SpatialIntegrationRule,
    SpatialProjectionBalance, SpatialProjectionEvidence, SpatialProjectionEvidenceError,
};
#[cfg(test)]
pub(crate) use model::{ComplexScalar, RealizedExcitationPolicy};
pub(crate) use model::{
    ComplexVector3, RealizedExcitation, RealizedExcitationKind, RealizedExcitationTarget,
    RealizedExcitations,
};
pub(crate) use projection::{ScalarProjectionAccumulator, SpatialProjectionAccumulator};
pub(crate) use realize::realize_step_excitations;
pub(crate) use sample::{
    expect_scalar, expect_vector3, sample_prescribed_heat_flux, sample_prescribed_temperature,
    sample_signal, time_discontinuities, ExcitationSamplePoint,
};
pub(crate) use time_grid::ProcedureTimeGrid;

#[cfg(test)]
mod tests;
