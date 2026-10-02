//! Realization of solver-neutral authored excitations into FEA-owned numerical inputs.

mod action;
mod activation;
mod coupling;
mod evidence;
pub(crate) mod field;
mod kinematics;
mod model;
mod projection;
mod realize;
mod sample;
pub(crate) mod sample_identity;
mod target;
mod time_grid;
mod waveform;

pub(crate) use activation::{active_actions, ActiveAction};

pub(crate) use coupling::realize_coupling_contexts;
pub use evidence::{
    ProjectionSample, ProjectionScalar, ProjectionVector3, SpatialIntegrationRule,
    SpatialProjectionBalance, SpatialProjectionEvidence, SpatialProjectionEvidenceError,
};
pub(crate) use kinematics::{StructuralMotionError, StructuralMotionSet};
#[cfg(test)]
pub(crate) use model::RealizedExcitationPolicy;
pub(crate) use model::{
    ComplexComponent, ComplexScalar, ComplexVector3, RealizedExcitation, RealizedExcitationKind,
    RealizedExcitationTarget, RealizedExcitations,
};
pub(crate) use projection::{ScalarProjectionAccumulator, SpatialProjectionAccumulator};
pub(crate) use realize::realize_step_excitations;
pub(crate) use realize::RealizationError;
pub(crate) use sample::{
    expect_scalar, expect_vector3, sample_prescribed_heat_flux, sample_prescribed_potential,
    sample_prescribed_pressure, sample_prescribed_temperature, sample_signal, time_breakpoints,
    ExcitationSamplePoint,
};
pub(crate) use time_grid::ProcedureTimeGrid;

#[cfg(test)]
mod selection_tests;
#[cfg(test)]
mod tests;
