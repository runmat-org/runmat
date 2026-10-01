//! Closed observation field locations, dimensions and shapes.
use crate::{FeaType, PhysicalDimension};
use runmat_analysis_core::{FieldLocation, ObservationQuantity};

pub(crate) fn field_location(name: &str) -> Option<FieldLocation> {
    Some(match name {
        "node" => FieldLocation::Node,
        "cell" => FieldLocation::Cell,
        "element" => FieldLocation::Element,
        "integration_point" => FieldLocation::IntegrationPoint,
        "face" => FieldLocation::Face,
        "edge" => FieldLocation::Edge,
        "interface" => FieldLocation::Interface,
        "connector" => FieldLocation::Connector,
        "constraint" => FieldLocation::Constraint,
        "reference_point" => FieldLocation::ReferencePoint,
        "mode" => FieldLocation::Mode,
        "global" => FieldLocation::Global,
        _ => return None,
    })
}

pub(crate) fn field_reference_type(name: &str) -> Option<(ObservationQuantity, FeaType)> {
    use ObservationQuantity as Q;
    let quantity = ObservationQuantity::from_source_name(name)?;
    let (dimension, components) = match quantity {
        Q::Displacement => (PhysicalDimension::LENGTH, 3),
        Q::ContactGap => (PhysicalDimension::LENGTH, 1),
        Q::Rotation => (PhysicalDimension::DIMENSIONLESS, 3),
        Q::Velocity | Q::AcousticParticleVelocity | Q::FluidVelocity => {
            (PhysicalDimension::VELOCITY, 3)
        }
        Q::Acceleration => (PhysicalDimension::ACCELERATION, 3),
        Q::Strain => (PhysicalDimension::DIMENSIONLESS, 6),
        Q::Stress => (PhysicalDimension::PRESSURE, 6),
        Q::VonMisesStress
        | Q::BeamTorsionStress
        | Q::ShellVonMises
        | Q::ContactPressure
        | Q::FluidPressure
        | Q::AcousticPressure => (PhysicalDimension::PRESSURE, 1),
        Q::BeamBendingStress => (PhysicalDimension::PRESSURE, 2),
        Q::WallShearStress => (PhysicalDimension::PRESSURE, 3),
        Q::ReactionForce => (PhysicalDimension::FORCE, 3),
        Q::BeamAxialForce => (PhysicalDimension::FORCE, 1),
        Q::BeamShearForce => (PhysicalDimension::FORCE, 2),
        Q::ReactionMoment => (PhysicalDimension::MOMENT, 3),
        Q::BeamTorsionMoment => (PhysicalDimension::MOMENT, 1),
        Q::BeamBendingMoment => (PhysicalDimension::MOMENT, 2),
        Q::StrainEnergy => (PhysicalDimension::ENERGY, 1),
        Q::ShellMembraneForce => (
            PhysicalDimension::FORCE.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::ShellBendingMoment => (
            PhysicalDimension::MOMENT.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::ShellTransverseShear => (
            PhysicalDimension::FORCE.divide(PhysicalDimension::LENGTH)?,
            2,
        ),
        Q::Temperature => (PhysicalDimension::TEMPERATURE, 1),
        Q::TemperatureGradient => (
            PhysicalDimension::TEMPERATURE.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::HeatFlux => (PhysicalDimension::HEAT_FLUX, 3),
        Q::HeatRate => (PhysicalDimension::POWER, 1),
        Q::ThermalEnergyStorageRateDensity => (
            PhysicalDimension::POWER.divide(PhysicalDimension::VOLUME)?,
            1,
        ),
        Q::ElectricPotential => (PhysicalDimension::ELECTRIC_POTENTIAL, 1),
        Q::ElectricField => (
            PhysicalDimension::ELECTRIC_POTENTIAL.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::ElectricDisplacement => (quantity.physical_dimension(), 3),
        Q::ElectricEnergyDensity | Q::JoulePowerDensity => (quantity.physical_dimension(), 1),
        Q::ElectricCurrent => (PhysicalDimension::ELECTRIC_CURRENT, 1),
        Q::ElectricCurrentDensity => (
            PhysicalDimension::ELECTRIC_CURRENT.divide(PhysicalDimension::AREA)?,
            3,
        ),
        Q::ElectricCharge => (
            PhysicalDimension::ELECTRIC_CURRENT.multiply(PhysicalDimension::TIME)?,
            1,
        ),
        Q::MagneticEdgeCirculation => (quantity.physical_dimension(), 1),
        Q::MagneticVectorPotential => (
            PhysicalDimension::MAGNETIC_FLUX_DENSITY.multiply(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::MagneticFluxDensity => (PhysicalDimension::MAGNETIC_FLUX_DENSITY, 3),
        Q::MagneticFieldIntensity => (
            PhysicalDimension::ELECTRIC_CURRENT.divide(PhysicalDimension::LENGTH)?,
            3,
        ),
        Q::SoundPressureLevel
        | Q::ModalParticipation
        | Q::Eigenvalue
        | Q::Residual
        | Q::Balance
        | Q::ConstraintViolation => (PhysicalDimension::DIMENSIONLESS, 1),
        Q::Vorticity => (PhysicalDimension::FREQUENCY, 3),
        Q::ModalFrequency => (PhysicalDimension::FREQUENCY, 1),
        Q::MassFlowRate => (PhysicalDimension::MASS.divide(PhysicalDimension::TIME)?, 1),
    };
    let scalar = FeaType::Quantity(dimension);
    Some((
        quantity,
        if components == 1 {
            scalar
        } else {
            FeaType::Vector {
                element: Box::new(scalar),
                length: Some(components),
            }
        },
    ))
}

#[cfg(test)]
mod quantity_tests {
    use super::*;

    #[test]
    fn electromagnetic_density_references_use_domain_dimensions_and_shapes() {
        for (name, components) in [
            ("electric_displacement", 3),
            ("electric_energy_density", 1),
            ("joule_power_density", 1),
            ("magnetic_edge_circulation", 1),
            ("magnetic_vector_potential", 3),
            ("magnetic_flux_density", 3),
            ("magnetic_field_intensity", 3),
        ] {
            let quantity = ObservationQuantity::from_source_name(name).unwrap();
            let scalar = FeaType::Quantity(quantity.physical_dimension());
            let expected = if components == 1 {
                scalar
            } else {
                FeaType::Vector {
                    element: Box::new(scalar),
                    length: Some(components),
                }
            };
            assert_eq!(field_reference_type(name), Some((quantity, expected)));
        }
    }

    #[test]
    fn thermal_storage_density_reference_is_scalar_power_per_volume() {
        assert_eq!(
            field_reference_type("thermal_energy_storage_rate_density"),
            Some((
                ObservationQuantity::ThermalEnergyStorageRateDensity,
                FeaType::Quantity(
                    PhysicalDimension::POWER
                        .divide(PhysicalDimension::VOLUME)
                        .unwrap()
                ),
            ))
        );
    }
}
