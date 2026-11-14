//! The source-language observation quantities: each one's `.fea` spelling and the field
//! locations each physics publishes it at (`analysis.<step>.field.<location>.<name>`).
//!
//! [`ObservationQuantity::from_source_name`] and the language's field references read this one
//! table, and so do the checker and Simulation setup's `outputs` choices
//! ([`crate::published_locations`]). The locations are the native layouts the solvers publish
//! (runmat-analysis-fea's recovered and native field inventories, and the layouts its pipelines
//! build); runmat-analysis-fea's `source_field_locations` test checks the table against them both
//! ways. A quantity with no locations
//! has no field form: it is read some other way (a reduction, a port or line parameter).
use crate::{FieldLocation as L, PhysicsFamily as P};

use super::ObservationQuantity as Q;

/// One source-language quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceQuantity {
    /// The `.fea` spelling.
    pub name: &'static str,
    pub quantity: Q,
    /// Per single physics, the locations a step of that physics publishes the quantity at as a
    /// field. A coupled family publishes what its physics publish.
    pub fields: &'static [(P, &'static [L])],
}

const fn row(name: &'static str, quantity: Q, fields: &'static [(P, &'static [L])]) -> SourceQuantity {
    SourceQuantity { name, quantity, fields }
}

/// Every source-language quantity, in the order a host lists them.
pub const SOURCE_QUANTITIES: &[SourceQuantity] = &[
    row("displacement", Q::Displacement, &[(P::Structural, &[L::Node])]),
    row("rotation", Q::Rotation, &[(P::Structural, &[L::Node])]),
    row("velocity", Q::Velocity, &[(P::Structural, &[L::Node])]),
    row("acceleration", Q::Acceleration, &[(P::Structural, &[L::Node])]),
    row("strain", Q::Strain, &[(P::Structural, &[L::Element, L::IntegrationPoint])]),
    row("stress", Q::Stress, &[(P::Structural, &[L::Element, L::IntegrationPoint, L::Node])]),
    row("von_mises_stress", Q::VonMisesStress, &[(P::Structural, &[L::Element, L::Node])]),
    row("reaction_force", Q::ReactionForce, &[(P::Structural, &[L::Node, L::Constraint, L::Connector])]),
    row("reaction_moment", Q::ReactionMoment, &[(P::Structural, &[L::Node, L::Constraint])]),
    row("strain_energy", Q::StrainEnergy, &[(P::Structural, &[L::Element, L::Global, L::Connector])]),
    row("beam_axial_force", Q::BeamAxialForce, &[(P::Structural, &[L::Element])]),
    row("beam_shear_force", Q::BeamShearForce, &[(P::Structural, &[L::Element])]),
    row("beam_torsion_moment", Q::BeamTorsionMoment, &[(P::Structural, &[L::Element])]),
    row("beam_bending_moment", Q::BeamBendingMoment, &[(P::Structural, &[L::Element])]),
    row("beam_bending_stress", Q::BeamBendingStress, &[(P::Structural, &[L::Element])]),
    row("beam_torsion_stress", Q::BeamTorsionStress, &[(P::Structural, &[L::Element])]),
    row("shell_membrane_force", Q::ShellMembraneForce, &[(P::Structural, &[L::Element])]),
    row("shell_bending_moment", Q::ShellBendingMoment, &[(P::Structural, &[L::Element])]),
    row("shell_transverse_shear", Q::ShellTransverseShear, &[(P::Structural, &[L::Element])]),
    row("shell_von_mises", Q::ShellVonMises, &[(P::Structural, &[L::Element])]),
    row("temperature", Q::Temperature, &[(P::Thermal, &[L::Node])]),
    row("temperature_gradient", Q::TemperatureGradient, &[(P::Thermal, &[L::Element])]),
    row("heat_flux", Q::HeatFlux, &[(P::Thermal, &[L::Element, L::Interface])]),
    row("heat_rate", Q::HeatRate, &[]),
    row("thermal_energy_storage_rate_density", Q::ThermalEnergyStorageRateDensity, &[(P::Thermal, &[L::Element])]),
    row("electric_potential", Q::ElectricPotential, &[(P::Electromagnetic, &[L::Node])]),
    row("electric_field", Q::ElectricField, &[(P::Electromagnetic, &[L::Edge, L::Element, L::IntegrationPoint])]),
    row("electric_displacement", Q::ElectricDisplacement, &[(P::Electromagnetic, &[L::Element, L::IntegrationPoint])]),
    row("electric_energy_density", Q::ElectricEnergyDensity, &[(P::Electromagnetic, &[L::Element])]),
    row("joule_power_density", Q::JoulePowerDensity, &[(P::Electromagnetic, &[L::Edge, L::Element, L::IntegrationPoint])]),
    row("electric_current", Q::ElectricCurrent, &[]),
    row("electric_current_density", Q::ElectricCurrentDensity, &[(P::Electromagnetic, &[L::Edge, L::Element, L::IntegrationPoint])]),
    row("electric_charge", Q::ElectricCharge, &[]),
    row("electric_edge_circulation", Q::ElectricEdgeCirculation, &[(P::Electromagnetic, &[L::Edge])]),
    row("magnetic_edge_circulation", Q::MagneticEdgeCirculation, &[(P::Electromagnetic, &[L::Edge])]),
    row("magnetic_vector_potential", Q::MagneticVectorPotential, &[(P::Electromagnetic, &[L::IntegrationPoint])]),
    row("magnetic_flux_density", Q::MagneticFluxDensity, &[(P::Electromagnetic, &[L::Element, L::IntegrationPoint])]),
    row("magnetic_field_intensity", Q::MagneticFieldIntensity, &[(P::Electromagnetic, &[L::Element, L::IntegrationPoint])]),
    row("acoustic_pressure", Q::AcousticPressure, &[(P::Acoustic, &[L::Node, L::Mode])]),
    row("acoustic_particle_velocity", Q::AcousticParticleVelocity, &[(P::Acoustic, &[L::Node, L::Element])]),
    row("acoustic_intensity", Q::AcousticIntensity, &[(P::Acoustic, &[L::Element])]),
    row("acoustic_port_intensity", Q::AcousticPortIntensity, &[(P::Acoustic, &[L::Element])]),
    row("sound_pressure_level", Q::SoundPressureLevel, &[]),
    row("transmission_loss", Q::TransmissionLoss, &[]),
    row("fluid_pressure", Q::FluidPressure, &[(P::Fluid, &[L::Node, L::Cell, L::Face, L::Interface])]),
    row("fluid_velocity", Q::FluidVelocity, &[(P::Fluid, &[L::Node, L::Cell])]),
    row("vorticity", Q::Vorticity, &[(P::Fluid, &[L::Cell])]),
    row("mass_flow_rate", Q::MassFlowRate, &[]),
    row("volume_flow_rate", Q::VolumeFlowRate, &[(P::Fluid, &[L::Face])]),
    row("wall_shear_stress", Q::WallShearStress, &[(P::Fluid, &[L::Face])]),
    row("modal_frequency", Q::ModalFrequency, &[(P::Structural, &[L::Mode])]),
    row("modal_participation", Q::ModalParticipation, &[(P::Structural, &[L::Mode])]),
    row("eigenvalue", Q::Eigenvalue, &[(P::Structural, &[L::Mode])]),
    row("residual", Q::Residual, &[(P::Structural, &[L::Global, L::Mode]), (P::Thermal, &[L::Global]), (P::Electromagnetic, &[L::Global]), (P::Fluid, &[L::Global])]),
    row("balance", Q::Balance, &[]),
    row("constraint_violation", Q::ConstraintViolation, &[]),
    row("contact_gap", Q::ContactGap, &[(P::Structural, &[L::Interface])]),
    row("contact_pressure", Q::ContactPressure, &[(P::Structural, &[L::Interface])]),
];
