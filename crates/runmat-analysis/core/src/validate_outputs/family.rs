//! Which observation quantities a step's physics family produces, and at which field locations it
//! publishes them.
use crate::{
    AnalysisProcedure, CoupledPhysicsFamily, FieldLocation, ObservationQuantity, PhysicsFamily,
    ProcedureStep,
};

pub fn quantity_supported(step: &ProcedureStep, quantity: ObservationQuantity) -> bool {
    family_supports_quantity(step.family, step.procedure, quantity)
}

/// Whether a step of `family` running `procedure` produces `quantity`: the same rule
/// [`quantity_supported`] applies to a lowered step, for a host that knows only the step's
/// physics and procedure (a study being set up, before its `.fea` exists).
pub fn family_supports_quantity(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    quantity: ObservationQuantity,
) -> bool {
    // A `line_parameters` step publishes exactly the line parameters of its cross-section.
    let line = quantity.line_parameter_scope().is_some();
    if procedure == AnalysisProcedure::LineParameters || line {
        return procedure == AnalysisProcedure::LineParameters
            && family == PhysicsFamily::Electromagnetic
            && line;
    }
    (match family {
        PhysicsFamily::Structural => is_structural(quantity),
        PhysicsFamily::Thermal => is_thermal(quantity),
        PhysicsFamily::Electromagnetic => is_electromagnetic(quantity),
        PhysicsFamily::Acoustic => is_acoustic(quantity),
        PhysicsFamily::Fluid => is_fluid(quantity),
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical) => {
            is_structural(quantity) || is_thermal(quantity)
        }
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal) => {
            is_electromagnetic(quantity) || is_thermal(quantity)
        }
        PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer) => {
            is_fluid(quantity) || is_thermal(quantity)
        }
        PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction) => {
            is_fluid(quantity) || is_structural(quantity)
        }
    }) || is_universal(quantity)
}

/// The field locations a step of `family` publishes `quantity` at
/// (`analysis.<step>.field.<location>.<quantity>`), read from the source-name table
/// ([`crate::SOURCE_QUANTITIES`]). A coupled family publishes what its physics
/// publish, plus the rows keyed to the coupled family itself. Empty when the family publishes no field of the quantity. The checker refuses any
/// other location, and Simulation setup offers only these.
pub fn published_locations(
    family: PhysicsFamily,
    quantity: ObservationQuantity,
) -> Vec<FieldLocation> {
    use PhysicsFamily as F;
    let physics: &[PhysicsFamily] = match family {
        F::Coupled(CoupledPhysicsFamily::Thermomechanical) => &[F::Structural, F::Thermal],
        F::Coupled(CoupledPhysicsFamily::Electrothermal) => &[F::Electromagnetic, F::Thermal],
        F::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer) => &[F::Fluid, F::Thermal],
        F::Coupled(CoupledPhysicsFamily::FluidStructureInteraction) => &[F::Fluid, F::Structural],
        _ => std::slice::from_ref(&family),
    };
    let mut locations = Vec::new();
    for row in crate::SOURCE_QUANTITIES
        .iter()
        .filter(|row| row.quantity == quantity)
    {
        for (owner, published) in row.fields {
            if *owner == family || physics.contains(owner) {
                for location in *published {
                    if !locations.contains(location) {
                        locations.push(*location);
                    }
                }
            }
        }
    }
    locations
}

fn is_structural(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::Displacement
            | ObservationQuantity::Rotation
            | ObservationQuantity::Velocity
            | ObservationQuantity::Acceleration
            | ObservationQuantity::Strain
            | ObservationQuantity::Stress
            | ObservationQuantity::VonMisesStress
            | ObservationQuantity::ReactionForce
            | ObservationQuantity::ReactionMoment
            | ObservationQuantity::StrainEnergy
            | ObservationQuantity::BeamAxialForce
            | ObservationQuantity::BeamShearForce
            | ObservationQuantity::BeamTorsionMoment
            | ObservationQuantity::BeamBendingMoment
            | ObservationQuantity::BeamBendingStress
            | ObservationQuantity::BeamTorsionStress
            | ObservationQuantity::ShellMembraneForce
            | ObservationQuantity::ShellBendingMoment
            | ObservationQuantity::ShellTransverseShear
            | ObservationQuantity::ShellVonMises
            | ObservationQuantity::ContactGap
            | ObservationQuantity::ContactPressure
            | ObservationQuantity::ModalFrequency
            | ObservationQuantity::ModalParticipation
            | ObservationQuantity::Eigenvalue
    )
}

fn is_thermal(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::Temperature
            | ObservationQuantity::TemperatureGradient
            | ObservationQuantity::HeatFlux
            | ObservationQuantity::HeatRate
            | ObservationQuantity::ThermalEnergyStorageRateDensity
    )
}

fn is_electromagnetic(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::ElectricPotential
            | ObservationQuantity::ElectricField
            | ObservationQuantity::ElectricDisplacement
            | ObservationQuantity::ElectricEnergyDensity
            | ObservationQuantity::JoulePowerDensity
            | ObservationQuantity::ElectricCurrent
            | ObservationQuantity::ElectricCurrentDensity
            | ObservationQuantity::ElectricCharge
            | ObservationQuantity::ElectricEdgeCirculation
            | ObservationQuantity::MagneticEdgeCirculation
            | ObservationQuantity::MagneticVectorPotential
            | ObservationQuantity::MagneticFluxDensity
            | ObservationQuantity::MagneticFieldIntensity
            | ObservationQuantity::HeatRate
    )
}

fn is_acoustic(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::AcousticPressure
            | ObservationQuantity::AcousticParticleVelocity
            | ObservationQuantity::AcousticIntensity
            | ObservationQuantity::AcousticPortIntensity
            | ObservationQuantity::SoundPressureLevel
            | ObservationQuantity::TransmissionLoss
    )
}

fn is_fluid(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::FluidPressure
            | ObservationQuantity::FluidVelocity
            | ObservationQuantity::Vorticity
            | ObservationQuantity::MassFlowRate
            | ObservationQuantity::VolumeFlowRate
            | ObservationQuantity::WallShearStress
            | ObservationQuantity::HeatRate
    )
}

fn is_universal(quantity: ObservationQuantity) -> bool {
    matches!(
        quantity,
        ObservationQuantity::Residual
            | ObservationQuantity::Balance
            | ObservationQuantity::ConstraintViolation
    )
}

#[cfg(test)]
mod tests {
    use crate::{ObservationQuantity, ProcedureStep};

    use super::{published_locations, quantity_supported};

    #[test]
    fn published_locations_follow_the_family() {
        use crate::{
            CoupledPhysicsFamily as C, FieldLocation as L, ObservationQuantity as Q,
            PhysicsFamily as F,
        };
        assert_eq!(
            published_locations(F::Structural, Q::Displacement),
            [L::Node]
        );
        assert_eq!(
            published_locations(F::Structural, Q::VonMisesStress),
            [L::Element, L::Node]
        );
        assert_eq!(
            published_locations(F::Fluid, Q::FluidVelocity),
            [L::Node, L::Cell]
        );
        assert!(published_locations(F::Thermal, Q::Displacement).is_empty());
        let coupled = F::Coupled(C::Thermomechanical);
        assert_eq!(
            published_locations(coupled, Q::Temperature),
            [L::Node, L::Element, L::IntegrationPoint]
        );
        assert_eq!(published_locations(F::Thermal, Q::Temperature), [L::Node]);
        assert_eq!(
            published_locations(F::Coupled(C::Electrothermal), Q::Temperature),
            [L::Node]
        );
        assert_eq!(published_locations(coupled, Q::Displacement), [L::Node]);
        assert!(published_locations(F::Structural, Q::HeatRate).is_empty());
    }

    #[test]
    fn quantity_support_follows_physics_family() {
        let structural = ProcedureStep::structural_static("structural");
        assert!(quantity_supported(&structural, ObservationQuantity::Stress));
        for quantity in [
            ObservationQuantity::VonMisesStress,
            ObservationQuantity::BeamAxialForce,
            ObservationQuantity::BeamShearForce,
            ObservationQuantity::BeamTorsionMoment,
            ObservationQuantity::BeamBendingMoment,
            ObservationQuantity::BeamBendingStress,
            ObservationQuantity::BeamTorsionStress,
            ObservationQuantity::ShellMembraneForce,
            ObservationQuantity::ShellBendingMoment,
            ObservationQuantity::ShellTransverseShear,
            ObservationQuantity::ShellVonMises,
        ] {
            assert!(quantity_supported(&structural, quantity));
        }
        assert!(!quantity_supported(
            &structural,
            ObservationQuantity::Temperature
        ));
        assert!(quantity_supported(
            &structural,
            ObservationQuantity::Residual
        ));

        let thermal = ProcedureStep::thermal_transient("thermal");
        assert!(quantity_supported(&thermal, ObservationQuantity::HeatFlux));
        assert!(quantity_supported(
            &thermal,
            ObservationQuantity::ThermalEnergyStorageRateDensity
        ));
        assert!(!quantity_supported(
            &structural,
            ObservationQuantity::ThermalEnergyStorageRateDensity
        ));
        for family in [
            crate::PhysicsFamily::Electromagnetic,
            crate::PhysicsFamily::Acoustic,
            crate::PhysicsFamily::Fluid,
            crate::PhysicsFamily::Coupled(crate::CoupledPhysicsFamily::FluidStructureInteraction),
        ] {
            let mut nonthermal = structural.clone();
            nonthermal.family = family;
            assert!(!quantity_supported(
                &nonthermal,
                ObservationQuantity::ThermalEnergyStorageRateDensity
            ));
        }
        assert!(!quantity_supported(
            &thermal,
            ObservationQuantity::Displacement
        ));
    }
}
