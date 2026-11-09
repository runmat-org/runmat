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
    /// Physical volume-averaged dielectric displacement, C/m².
    ElectricDisplacement,
    /// Physical volume average of electrostatic energy, J/m³ (not energy of mean E).
    ElectricEnergyDensity,
    /// Physical volume average of conduction heating, W/m³.
    JoulePowerDensity,
    ElectricCurrent,
    ElectricCurrentDensity,
    ElectricCharge,
    /// Oriented electric edge coefficient ∫ E·dl, V; not scalar electric potential.
    ElectricEdgeCirculation,
    /// Oriented edge coefficient ∫ A·dl, Wb; not the vector potential A in Wb/m.
    MagneticEdgeCirculation,
    MagneticVectorPotential,
    MagneticFluxDensity,
    MagneticFieldIntensity,
    AcousticPressure,
    AcousticParticleVelocity,
    /// Mean active acoustic intensity ½Re(p v*), W/m².
    AcousticIntensity,
    /// Plane-wave power density crossing an acoustic port face, W/m²: the authored incident
    /// wave entering through an incident-pressure face, or the outgoing wave leaving through an
    /// anechoic (radiation) face. Zero away from port faces.
    AcousticPortIntensity,
    SoundPressureLevel,
    /// Transmission loss between an incident and a transmitted port, dB.
    TransmissionLoss,
    FluidPressure,
    FluidVelocity,
    Vorticity,
    MassFlowRate,
    /// Signed owner-oriented volume transport, m³/s; not mass flow or velocity.
    VolumeFlowRate,
    WallShearStress,
    ModalFrequency,
    ModalParticipation,
    Eigenvalue,
    Residual,
    Balance,
    ConstraintViolation,
    ContactGap,
    ContactPressure,
    // Result-only quantities: published by solvers, not authored as observations.
    /// Force carried by a connector element, N.
    ConnectorForce,
    /// Interface traction vector, Pa.
    Traction,
    /// Volumetric strain-energy density, J/m³.
    StrainEnergyDensity,
    KineticEnergy,
    ExternalWork,
    SupportWork,
    DampingDissipation,
    /// Total acoustic energy in the domain, J.
    AcousticEnergy,
    /// Energy absorbed by impedance boundaries, J.
    AbsorbedEnergy,
    AngularVelocity,
    AngularAcceleration,
    PlasticStrain,
    EquivalentPlasticStrain,
    /// Modal or effective mass, kg.
    ModalMass,
    /// Generalized modal stiffness, N/m.
    ModalStiffness,
    /// Effective-mass fraction of the total free mass.
    MassRatio,
    /// Mass-orthogonality check matrix entries.
    MassOrthogonality,
    /// Relative separation between neighbouring modal frequencies.
    FrequencySeparation,
    /// Load (buckling or continuation) factor.
    LoadFactor,
    /// Scale applied to an assembled equation block.
    EquationScale,
    ReynoldsNumber,
    /// Temperature difference across a thermal interface, K.
    TemperatureJump,
    /// Number of coupling iterations taken.
    IterationCount,
    /// A sampled time coordinate, s.
    Time,
    /// A sampled frequency coordinate, Hz.
    Frequency,
}

impl ObservationQuantity {
    pub fn physical_dimension(self) -> PhysicalDimension {
        use PhysicalDimension as D;
        match self {
            Self::Displacement | Self::ContactGap => D::LENGTH,
            Self::Rotation
            | Self::Strain
            | Self::PlasticStrain
            | Self::EquivalentPlasticStrain
            | Self::MassRatio
            | Self::MassOrthogonality
            | Self::FrequencySeparation
            | Self::LoadFactor
            | Self::EquationScale
            | Self::ReynoldsNumber
            | Self::IterationCount
            | Self::ModalParticipation
            | Self::Eigenvalue
            | Self::Residual
            | Self::Balance
            | Self::ConstraintViolation => D::DIMENSIONLESS,
            Self::Velocity | Self::AcousticParticleVelocity | Self::FluidVelocity => D::VELOCITY,
            Self::AcousticIntensity | Self::AcousticPortIntensity => {
                D::POWER.divide(D::AREA).expect("static dimension")
            }
            Self::Acceleration => D::ACCELERATION,
            Self::Stress
            | Self::VonMisesStress
            | Self::BeamBendingStress
            | Self::BeamTorsionStress
            | Self::AcousticPressure
            | Self::FluidPressure
            | Self::WallShearStress
            | Self::Traction
            | Self::ContactPressure => D::PRESSURE,
            Self::ReactionForce
            | Self::ConnectorForce
            | Self::BeamAxialForce
            | Self::BeamShearForce => D::FORCE,
            Self::KineticEnergy
            | Self::ExternalWork
            | Self::SupportWork
            | Self::DampingDissipation
            | Self::AcousticEnergy
            | Self::AbsorbedEnergy => D::ENERGY,
            Self::StrainEnergyDensity => D::ENERGY.divide(D::VOLUME).expect("static dimension"),
            Self::AngularVelocity | Self::Frequency => D::FREQUENCY,
            Self::AngularAcceleration => D::FREQUENCY.divide(D::TIME).expect("static dimension"),
            Self::ModalMass => D::MASS,
            Self::ModalStiffness => D::FORCE.divide(D::LENGTH).expect("static dimension"),
            Self::TemperatureJump => D::TEMPERATURE,
            Self::Time => D::TIME,
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
            Self::ElectricDisplacement => D::ELECTRIC_CURRENT
                .multiply(D::TIME)
                .expect("static dimension")
                .divide(D::AREA)
                .expect("static dimension"),
            Self::ElectricEnergyDensity => D::ENERGY.divide(D::VOLUME).expect("static dimension"),
            Self::JoulePowerDensity => D::POWER.divide(D::VOLUME).expect("static dimension"),
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
            Self::MagneticEdgeCirculation => D::MAGNETIC_FLUX_DENSITY
                .multiply(D::AREA)
                .expect("static dimension"),
            Self::ElectricEdgeCirculation => D::ELECTRIC_POTENTIAL,
            Self::MagneticVectorPotential => D::MAGNETIC_FLUX_DENSITY
                .multiply(D::LENGTH)
                .expect("static dimension"),
            Self::MagneticFluxDensity => D::MAGNETIC_FLUX_DENSITY,
            Self::MagneticFieldIntensity => D::ELECTRIC_CURRENT
                .divide(D::LENGTH)
                .expect("static dimension"),
            Self::SoundPressureLevel | Self::TransmissionLoss => D::DIMENSIONLESS,
            Self::Vorticity | Self::ModalFrequency => D::FREQUENCY,
            Self::MassFlowRate => D::MASS.divide(D::TIME).expect("static dimension"),
            Self::VolumeFlowRate => D::VOLUME.divide(D::TIME).expect("static dimension"),
        }
    }

    /// Physical descriptor for a concrete recovered field. Generic diagnostics
    /// require producer/basis metadata and cannot be declared dimensionless here.
    pub fn quantity_descriptor(self) -> Option<runmat_types::physical::DynamicDimension> {
        use runmat_types::physical::{DynamicDimension as D, QuantityKind as K};
        let known = match self {
            Self::Displacement | Self::ContactGap => Some(K::Displacement),
            Self::Rotation => Some(K::Rotation),
            Self::Velocity | Self::AcousticParticleVelocity | Self::FluidVelocity => {
                Some(K::Velocity)
            }
            Self::Acceleration => Some(K::Acceleration),
            Self::Stress
            | Self::VonMisesStress
            | Self::BeamBendingStress
            | Self::BeamTorsionStress
            | Self::AcousticPressure
            | Self::FluidPressure
            | Self::WallShearStress
            | Self::ContactPressure => Some(K::Pressure),
            Self::ReactionForce | Self::BeamAxialForce | Self::BeamShearForce => Some(K::Force),
            Self::ReactionMoment | Self::BeamTorsionMoment | Self::BeamBendingMoment => {
                Some(K::Moment)
            }
            Self::StrainEnergy => Some(K::Energy),
            Self::Temperature => Some(K::Temperature),
            Self::HeatFlux => Some(K::HeatFlux),
            Self::HeatRate => Some(K::Power),
            Self::ThermalEnergyStorageRateDensity | Self::JoulePowerDensity => {
                Some(K::HeatGenerationDensity)
            }
            Self::ElectricPotential | Self::ElectricEdgeCirculation => Some(K::ElectricPotential),
            Self::ElectricField => Some(K::ElectricField),
            Self::ElectricCurrent => Some(K::ElectricCurrent),
            Self::ElectricCurrentDensity => Some(K::ElectricCurrentDensity),
            Self::ElectricCharge => Some(K::ElectricCharge),
            Self::MagneticVectorPotential => Some(K::MagneticVectorPotential),
            Self::MagneticFluxDensity => Some(K::MagneticFluxDensity),
            Self::Vorticity => Some(K::AngularVelocity),
            Self::ModalFrequency => Some(K::Frequency),
            Self::MassFlowRate => Some(K::MassFlowRate),
            Self::VolumeFlowRate => Some(K::VolumeFlowRate),
            Self::ModalParticipation
            | Self::Eigenvalue
            | Self::Residual
            | Self::Balance
            | Self::ConstraintViolation => return None,
            _ => None,
        };
        Some(match known {
            Some(kind) => D::quantity_type(kind).with_semantic_kind(Some(kind)).ok()?,
            None => self.physical_dimension().into(),
        })
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
            "electric_displacement" => Self::ElectricDisplacement,
            "electric_energy_density" => Self::ElectricEnergyDensity,
            "joule_power_density" => Self::JoulePowerDensity,
            "electric_current" => Self::ElectricCurrent,
            "electric_current_density" => Self::ElectricCurrentDensity,
            "electric_charge" => Self::ElectricCharge,
            "electric_edge_circulation" => Self::ElectricEdgeCirculation,
            "magnetic_edge_circulation" => Self::MagneticEdgeCirculation,
            "magnetic_vector_potential" => Self::MagneticVectorPotential,
            "magnetic_flux_density" => Self::MagneticFluxDensity,
            "magnetic_field_intensity" => Self::MagneticFieldIntensity,
            "acoustic_pressure" => Self::AcousticPressure,
            "acoustic_particle_velocity" => Self::AcousticParticleVelocity,
            "acoustic_intensity" => Self::AcousticIntensity,
            "acoustic_port_intensity" => Self::AcousticPortIntensity,
            "sound_pressure_level" => Self::SoundPressureLevel,
            "transmission_loss" => Self::TransmissionLoss,
            "fluid_pressure" => Self::FluidPressure,
            "fluid_velocity" => Self::FluidVelocity,
            "vorticity" => Self::Vorticity,
            "mass_flow_rate" => Self::MassFlowRate,
            "volume_flow_rate" => Self::VolumeFlowRate,
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
            | Self::ElectricDisplacement
            | Self::ElectricCurrentDensity
            | Self::MagneticVectorPotential
            | Self::MagneticFluxDensity
            | Self::MagneticFieldIntensity
            | Self::AcousticParticleVelocity
            | Self::AcousticIntensity
            | Self::FluidVelocity
            | Self::Vorticity
            | Self::ConnectorForce
            | Self::Traction
            | Self::AngularVelocity
            | Self::AngularAcceleration
            | Self::WallShearStress => ObservationValueShape::Vector3,
            Self::Strain | Self::Stress | Self::PlasticStrain => {
                ObservationValueShape::SymmetricTensor3
            }
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
mod magnetic_native_quantity_tests {
    use super::*;
    #[test]
    fn electric_circulation_is_scalar_voltage_not_magnetic_flux_or_potential_field() {
        let quantity = ObservationQuantity::from_source_name("electric_edge_circulation").unwrap();
        assert_eq!(quantity, ObservationQuantity::ElectricEdgeCirculation);
        assert_eq!(
            quantity.canonical_value_shape(),
            ObservationValueShape::Scalar
        );
        assert_eq!(
            quantity.physical_dimension(),
            PhysicalDimension::ELECTRIC_POTENTIAL
        );
        assert_ne!(quantity, ObservationQuantity::ElectricPotential);
        assert_ne!(
            quantity.physical_dimension(),
            ObservationQuantity::MagneticEdgeCirculation.physical_dimension()
        );
        assert_eq!(
            serde_json::from_str::<ObservationQuantity>(&serde_json::to_string(&quantity).unwrap())
                .unwrap(),
            quantity
        );
    }
    #[test]
    fn circulation_is_scalar_flux_not_vector_potential() {
        let circulation =
            ObservationQuantity::from_source_name("magnetic_edge_circulation").unwrap();
        assert_eq!(circulation, ObservationQuantity::MagneticEdgeCirculation);
        assert_eq!(
            circulation.canonical_value_shape(),
            ObservationValueShape::Scalar
        );
        assert_eq!(
            circulation.physical_dimension(),
            PhysicalDimension::MAGNETIC_FLUX_DENSITY
                .multiply(PhysicalDimension::AREA)
                .unwrap()
        );
        assert_ne!(
            circulation.physical_dimension(),
            ObservationQuantity::MagneticVectorPotential.physical_dimension()
        );
    }
}

#[cfg(test)]
mod electrical_native_quantity_tests {
    use super::*;

    #[test]
    fn native_electrical_volume_means_have_honest_closed_quantity_semantics() {
        let charge_per_area = PhysicalDimension::ELECTRIC_CURRENT
            .multiply(PhysicalDimension::TIME)
            .unwrap()
            .divide(PhysicalDimension::AREA)
            .unwrap();
        for (name, quantity, dimension, shape) in [
            (
                "electric_displacement",
                ObservationQuantity::ElectricDisplacement,
                charge_per_area,
                ObservationValueShape::Vector3,
            ),
            (
                "electric_energy_density",
                ObservationQuantity::ElectricEnergyDensity,
                PhysicalDimension::ENERGY
                    .divide(PhysicalDimension::VOLUME)
                    .unwrap(),
                ObservationValueShape::Scalar,
            ),
            (
                "joule_power_density",
                ObservationQuantity::JoulePowerDensity,
                PhysicalDimension::POWER
                    .divide(PhysicalDimension::VOLUME)
                    .unwrap(),
                ObservationValueShape::Scalar,
            ),
        ] {
            assert_eq!(ObservationQuantity::from_source_name(name), Some(quantity));
            assert_eq!(quantity.physical_dimension(), dimension);
            assert_eq!(quantity.canonical_value_shape(), shape);
            let bytes = serde_json::to_vec(&quantity).unwrap();
            assert_eq!(
                serde_json::from_slice::<ObservationQuantity>(&bytes).unwrap(),
                quantity
            );
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
