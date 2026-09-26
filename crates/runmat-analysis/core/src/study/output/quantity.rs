use runmat_types::PhysicalDimension;
use serde::{Deserialize, Serialize};

use super::ObservationValueShape;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationQuantity {
    Displacement,
    Rotation,
    Velocity,
    Acceleration,
    Strain,
    Stress,
    VonMisesStress,
    ReactionForce,
    ReactionMoment,
    StrainEnergy,
    BeamAxialForce,
    BeamShearForce,
    BeamTorsionMoment,
    BeamBendingMoment,
    BeamBendingStress,
    BeamTorsionStress,
    ShellMembraneForce,
    ShellBendingMoment,
    ShellTransverseShear,
    ShellVonMises,
    Temperature,
    TemperatureGradient,
    HeatFlux,
    HeatRate,
    /// Volume-averaged thermal energy accumulation rate, in W/m³.
    ThermalEnergyStorageRateDensity,
    ElectricPotential,
    ElectricField,
    ElectricCurrent,
    ElectricCurrentDensity,
    ElectricCharge,
    MagneticVectorPotential,
    MagneticFluxDensity,
    MagneticFieldIntensity,
    AcousticPressure,
    AcousticParticleVelocity,
    SoundPressureLevel,
    FluidPressure,
    FluidVelocity,
    Vorticity,
    MassFlowRate,
    WallShearStress,
    ModalFrequency,
    ModalParticipation,
    Eigenvalue,
    Residual,
    Balance,
    ConstraintViolation,
    ContactGap,
    ContactPressure,
}

impl ObservationQuantity {
    pub fn physical_dimension(self) -> PhysicalDimension {
        use PhysicalDimension as D;
        match self {
            Self::Displacement | Self::ContactGap => D::LENGTH,
            Self::Rotation
            | Self::Strain
            | Self::ModalParticipation
            | Self::Eigenvalue
            | Self::Residual
            | Self::Balance
            | Self::ConstraintViolation => D::DIMENSIONLESS,
            Self::Velocity | Self::AcousticParticleVelocity | Self::FluidVelocity => D::VELOCITY,
            Self::Acceleration => D::ACCELERATION,
            Self::Stress
            | Self::VonMisesStress
            | Self::BeamBendingStress
            | Self::BeamTorsionStress
            | Self::AcousticPressure
            | Self::FluidPressure
            | Self::WallShearStress
            | Self::ContactPressure => D::PRESSURE,
            Self::ReactionForce | Self::BeamAxialForce | Self::BeamShearForce => D::FORCE,
            Self::ReactionMoment | Self::BeamTorsionMoment | Self::BeamBendingMoment => D::MOMENT,
            Self::StrainEnergy => D::ENERGY,
            Self::ShellMembraneForce | Self::ShellTransverseShear => {
                D::FORCE.divide(D::LENGTH).expect("static dimension")
            }
            Self::ShellBendingMoment => D::MOMENT.divide(D::LENGTH).expect("static dimension"),
            Self::ShellVonMises => D::FORCE.divide(D::LENGTH).expect("static dimension"),
            Self::Temperature => D::TEMPERATURE,
            Self::TemperatureGradient => {
                D::TEMPERATURE.divide(D::LENGTH).expect("static dimension")
            }
            Self::HeatFlux => D::HEAT_FLUX,
            Self::HeatRate => D::POWER,
            Self::ThermalEnergyStorageRateDensity => {
                D::POWER.divide(D::VOLUME).expect("static dimension")
            }
            Self::ElectricPotential => D::ELECTRIC_POTENTIAL,
            Self::ElectricField => D::ELECTRIC_POTENTIAL
                .divide(D::LENGTH)
                .expect("static dimension"),
            Self::ElectricCurrent => D::ELECTRIC_CURRENT,
            Self::ElectricCurrentDensity => D::ELECTRIC_CURRENT
                .divide(D::AREA)
                .expect("static dimension"),
            Self::ElectricCharge => D::ELECTRIC_CURRENT
                .multiply(D::TIME)
                .expect("static dimension"),
            Self::MagneticVectorPotential => D::MAGNETIC_FLUX_DENSITY
                .multiply(D::LENGTH)
                .expect("static dimension"),
            Self::MagneticFluxDensity => D::MAGNETIC_FLUX_DENSITY,
            Self::MagneticFieldIntensity => D::ELECTRIC_CURRENT
                .divide(D::LENGTH)
                .expect("static dimension"),
            Self::SoundPressureLevel => D::DIMENSIONLESS,
            Self::Vorticity | Self::ModalFrequency => D::FREQUENCY,
            Self::MassFlowRate => D::MASS.divide(D::TIME).expect("static dimension"),
        }
    }

    /// Resolve the stable source-language spelling of an observation quantity.
    ///
    /// Keeping this closed mapping beside the canonical enum prevents frontends from growing
    /// independent string-dispatch tables for the same result quantities.
    pub fn from_source_name(name: &str) -> Option<Self> {
        Some(match name {
            "displacement" => Self::Displacement,
            "rotation" => Self::Rotation,
            "velocity" => Self::Velocity,
            "acceleration" => Self::Acceleration,
            "strain" => Self::Strain,
            "stress" => Self::Stress,
            "von_mises_stress" => Self::VonMisesStress,
            "reaction_force" => Self::ReactionForce,
            "reaction_moment" => Self::ReactionMoment,
            "strain_energy" => Self::StrainEnergy,
            "beam_axial_force" => Self::BeamAxialForce,
            "beam_shear_force" => Self::BeamShearForce,
            "beam_torsion_moment" => Self::BeamTorsionMoment,
            "beam_bending_moment" => Self::BeamBendingMoment,
            "beam_bending_stress" => Self::BeamBendingStress,
            "beam_torsion_stress" => Self::BeamTorsionStress,
            "shell_membrane_force" => Self::ShellMembraneForce,
            "shell_bending_moment" => Self::ShellBendingMoment,
            "shell_transverse_shear" => Self::ShellTransverseShear,
            "shell_von_mises" => Self::ShellVonMises,
            "temperature" => Self::Temperature,
            "temperature_gradient" => Self::TemperatureGradient,
            "heat_flux" => Self::HeatFlux,
            "heat_rate" => Self::HeatRate,
            "thermal_energy_storage_rate_density" => Self::ThermalEnergyStorageRateDensity,
            "electric_potential" => Self::ElectricPotential,
            "electric_field" => Self::ElectricField,
            "electric_current" => Self::ElectricCurrent,
            "electric_current_density" => Self::ElectricCurrentDensity,
            "electric_charge" => Self::ElectricCharge,
            "magnetic_vector_potential" => Self::MagneticVectorPotential,
            "magnetic_flux_density" => Self::MagneticFluxDensity,
            "magnetic_field_intensity" => Self::MagneticFieldIntensity,
            "acoustic_pressure" => Self::AcousticPressure,
            "acoustic_particle_velocity" => Self::AcousticParticleVelocity,
            "sound_pressure_level" => Self::SoundPressureLevel,
            "fluid_pressure" => Self::FluidPressure,
            "fluid_velocity" => Self::FluidVelocity,
            "vorticity" => Self::Vorticity,
            "mass_flow_rate" => Self::MassFlowRate,
            "wall_shear_stress" => Self::WallShearStress,
            "modal_frequency" => Self::ModalFrequency,
            "modal_participation" => Self::ModalParticipation,
            "eigenvalue" => Self::Eigenvalue,
            "residual" => Self::Residual,
            "balance" => Self::Balance,
            "constraint_violation" => Self::ConstraintViolation,
            "contact_gap" => Self::ContactGap,
            "contact_pressure" => Self::ContactPressure,
            _ => return None,
        })
    }

    /// Canonical component shape of this physical quantity.
    ///
    /// This is the shared authority used by authored-output validation, physics recovery, and
    /// observation evaluation. Storage may use an equivalent explicit tensor extent, but it must
    /// not redefine the quantity's semantics.
    pub fn canonical_value_shape(self) -> ObservationValueShape {
        match self {
            Self::Displacement
            | Self::Rotation
            | Self::Velocity
            | Self::Acceleration
            | Self::ReactionForce
            | Self::ReactionMoment
            | Self::TemperatureGradient
            | Self::HeatFlux
            | Self::ElectricField
            | Self::ElectricCurrentDensity
            | Self::MagneticVectorPotential
            | Self::MagneticFluxDensity
            | Self::MagneticFieldIntensity
            | Self::AcousticParticleVelocity
            | Self::FluidVelocity
            | Self::Vorticity
            | Self::WallShearStress => ObservationValueShape::Vector3,
            Self::Strain | Self::Stress => ObservationValueShape::SymmetricTensor3,
            Self::BeamShearForce
            | Self::BeamBendingMoment
            | Self::BeamBendingStress
            | Self::ShellTransverseShear => ObservationValueShape::Tensor { extents: vec![2] },
            Self::ShellMembraneForce | Self::ShellBendingMoment => {
                ObservationValueShape::Tensor { extents: vec![3] }
            }
            _ => ObservationValueShape::Scalar,
        }
    }
}

#[cfg(test)]
mod thermal_storage_quantity_tests {
    use super::*;

    #[test]
    fn thermal_storage_density_has_scalar_power_per_volume_semantics() {
        let quantity = ObservationQuantity::ThermalEnergyStorageRateDensity;
        assert_eq!(
            quantity.canonical_value_shape(),
            ObservationValueShape::Scalar
        );
        assert_eq!(
            quantity.physical_dimension(),
            PhysicalDimension::POWER
                .divide(PhysicalDimension::VOLUME)
                .unwrap()
        );
        assert_ne!(
            quantity.physical_dimension(),
            ObservationQuantity::HeatRate.physical_dimension()
        );
        // The shared integral reduction multiplies by the target's measure: a volume
        // integral must recover power, rather than power times volume.
        assert_eq!(
            quantity
                .physical_dimension()
                .multiply(PhysicalDimension::VOLUME),
            Some(PhysicalDimension::POWER)
        );
        assert_eq!(
            ObservationQuantity::from_source_name("thermal_energy_storage_rate_density"),
            Some(quantity)
        );
        let encoded = serde_json::to_string(&quantity).unwrap();
        assert_eq!(encoded, "\"thermal_energy_storage_rate_density\"");
        assert_eq!(
            serde_json::from_str::<ObservationQuantity>(&encoded).unwrap(),
            quantity
        );
    }
}
