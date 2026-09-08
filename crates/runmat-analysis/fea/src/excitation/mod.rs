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
pub(crate) use projection::SpatialProjectionAccumulator;
pub(crate) use realize::realize_model_excitations;

#[cfg(test)]
mod tests;
