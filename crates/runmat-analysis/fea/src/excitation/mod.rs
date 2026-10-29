//! Realization of solver-neutral authored excitations into FEA-owned numerical inputs.

mod action;
mod activation;
mod coupling;
mod evidence;
pub(crate) mod field;
mod initial_target;
mod kinematics;
pub(crate) use initial_target::{initial_target_nodes, InitialTargetError};
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
    expect_scalar, expect_vector3, expect_vector3_or_direction, initial_signal_sample_point,
    sample_prescribed_current, sample_prescribed_heat_flux, sample_prescribed_potential,
    sample_prescribed_pressure, sample_prescribed_temperature,
    sample_prescribed_temperature_difference, sample_prescribed_velocity, sample_signal,
    time_breakpoints, time_breakpoints_profiled, ExcitationSamplePoint,
};
pub(crate) use time_grid::{ProcedureTimeGrid, ProcedureTimeSchedule, TimeScheduleError};

#[cfg(test)]
mod selection_tests;
#[cfg(test)]
mod tests;
