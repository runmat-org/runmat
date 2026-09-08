//! Realization of solver-neutral authored excitations into FEA-owned numerical inputs.

mod action;
mod activation;
mod evidence;
mod model;
mod projection;
mod realize;
mod sample;
mod target;

pub use evidence::{
    ProjectionScalar, ProjectionVector3, SpatialIntegrationRule, SpatialProjectionBalance,
    SpatialProjectionEvidence,
};
#[cfg(test)]
pub(crate) use model::{ComplexScalar, RealizedExcitationPolicy};
pub(crate) use model::{
    ComplexVector3, RealizedExcitation, RealizedExcitationKind, RealizedExcitationTarget,
    RealizedExcitations,
};
pub(crate) use projection::{ScalarProjectionAccumulator, SpatialProjectionAccumulator};
pub(crate) use realize::{realize_excitations, realize_model_excitations};
pub(crate) use sample::{
    expect_scalar, sample_prescribed_heat_flux, sample_prescribed_temperature, sample_signal,
    ExcitationSamplePoint,
};

#[cfg(test)]
mod tests;
