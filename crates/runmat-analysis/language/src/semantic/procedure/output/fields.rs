//! Closed observation field locations, dimensions and shapes.
use crate::FeaType;
#[cfg(test)]
use crate::PhysicalDimension;
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
    let quantity = ObservationQuantity::from_source_name(name)?;
    let dimension = quantity.quantity_descriptor()?;
    let components = quantity.canonical_value_shape().component_count()?;
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
    use runmat_types::physical::{DynamicDimension, QuantityKind};

    #[test]
    fn volume_flow_is_scalar_volume_per_time_not_mass_flow() {
        let (_, volume) = field_reference_type("volume_flow_rate").unwrap();
        let (_, mass) = field_reference_type("mass_flow_rate").unwrap();
        assert_eq!(
            volume,
            FeaType::Quantity(
                DynamicDimension::try_new(
                    PhysicalDimension::VOLUME
                        .divide(PhysicalDimension::TIME)
                        .unwrap(),
                    Some(QuantityKind::VolumeFlowRate),
                )
                .unwrap()
            )
        );
        assert_ne!(volume, mass);
    }

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
            let scalar = FeaType::Quantity(quantity.quantity_descriptor().unwrap());
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
                    DynamicDimension::try_new(
                        PhysicalDimension::POWER
                            .divide(PhysicalDimension::VOLUME)
                            .unwrap(),
                        Some(QuantityKind::HeatGenerationDensity),
                    )
                    .unwrap()
                ),
            ))
        );
    }
}
