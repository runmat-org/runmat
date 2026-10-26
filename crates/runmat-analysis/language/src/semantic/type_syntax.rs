use super::{FeaTopologyRole, FeaType, PhysicalDimension};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeSyntaxError {
    Empty,
    Unknown(String),
    InvalidParameterization(String),
}

impl fmt::Display for TypeSyntaxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("type is empty"),
            Self::Unknown(value) => write!(formatter, "unknown FEA type `{value}`"),
            Self::InvalidParameterization(value) => {
                write!(formatter, "invalid parameterized FEA type `{value}`")
            }
        }
    }
}

pub fn parse_type_syntax(source: &str) -> Result<FeaType, TypeSyntaxError> {
    let compact = source
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect::<String>();
    if compact.is_empty() {
        return Err(TypeSyntaxError::Empty);
    }
    if let Some(role) = compact
        .strip_prefix("region<")
        .and_then(|value| value.strip_suffix('>'))
    {
        return Ok(FeaType::Region(match role {
            "vertex" => FeaTopologyRole::Vertex,
            "edge" => FeaTopologyRole::Edge,
            "face" => FeaTopologyRole::Face,
            "solid" => FeaTopologyRole::Solid,
            "cell" => FeaTopologyRole::Cell,
            _ => return Err(TypeSyntaxError::InvalidParameterization(compact)),
        }));
    }
    if compact.contains(['<', '>']) {
        return Err(TypeSyntaxError::InvalidParameterization(compact));
    }
    Ok(match compact.as_str() {
        "boolean" => FeaType::Boolean,
        "integer" => FeaType::Integer,
        "real" => FeaType::Real,
        "complex" => FeaType::Complex,
        "string" => FeaType::String,
        "dimensionless" => FeaType::Quantity((PhysicalDimension::DIMENSIONLESS).into()),
        "length" => FeaType::Quantity((PhysicalDimension::LENGTH).into()),
        "area" => FeaType::Quantity((PhysicalDimension::AREA).into()),
        "volume" => FeaType::Quantity((PhysicalDimension::VOLUME).into()),
        "time" => FeaType::Quantity((PhysicalDimension::TIME).into()),
        "frequency" => FeaType::Quantity((PhysicalDimension::FREQUENCY).into()),
        "mass" => FeaType::Quantity((PhysicalDimension::MASS).into()),
        "mass_density" => FeaType::Quantity((PhysicalDimension::DENSITY).into()),
        "temperature" => known_quantity(runmat_types::physical::QuantityKind::Temperature),
        "velocity" => FeaType::Quantity((PhysicalDimension::VELOCITY).into()),
        "acceleration" => FeaType::Quantity((PhysicalDimension::ACCELERATION).into()),
        "force" => FeaType::Quantity((PhysicalDimension::FORCE).into()),
        "moment" => known_quantity(runmat_types::physical::QuantityKind::Moment),
        "pressure" | "stress" => FeaType::Quantity((PhysicalDimension::PRESSURE).into()),
        "energy" => known_quantity(runmat_types::physical::QuantityKind::Energy),
        "power" => FeaType::Quantity((PhysicalDimension::POWER).into()),
        "heat_flux" => FeaType::Quantity((PhysicalDimension::HEAT_FLUX).into()),
        "thermal_conductivity" => {
            FeaType::Quantity((PhysicalDimension::THERMAL_CONDUCTIVITY).into())
        }
        "heat_transfer_coefficient" => {
            FeaType::Quantity(PhysicalDimension::HEAT_TRANSFER_COEFFICIENT.into())
        }
        "electric_potential" => FeaType::Quantity((PhysicalDimension::ELECTRIC_POTENTIAL).into()),
        "electric_current" => FeaType::Quantity((PhysicalDimension::ELECTRIC_CURRENT).into()),
        "magnetic_flux_density" => {
            FeaType::Quantity((PhysicalDimension::MAGNETIC_FLUX_DENSITY).into())
        }
        "angle" => FeaType::Quantity(runmat_types::physical::DynamicDimension::quantity_type(
            runmat_types::physical::QuantityKind::Angle,
        )),
        "displacement" => known_quantity(runmat_types::physical::QuantityKind::Displacement),
        "rotation" => known_quantity(runmat_types::physical::QuantityKind::Rotation),
        "angular_velocity" => known_quantity(runmat_types::physical::QuantityKind::AngularVelocity),
        "angular_acceleration" => {
            known_quantity(runmat_types::physical::QuantityKind::AngularAcceleration)
        }
        "temperature_difference" => {
            known_quantity(runmat_types::physical::QuantityKind::TemperatureDifference)
        }
        "body_force_density" => {
            known_quantity(runmat_types::physical::QuantityKind::BodyForceDensity)
        }
        "heat_generation_density" => {
            known_quantity(runmat_types::physical::QuantityKind::HeatGenerationDensity)
        }
        "electric_charge" => known_quantity(runmat_types::physical::QuantityKind::ElectricCharge),
        "electric_current_density" => {
            known_quantity(runmat_types::physical::QuantityKind::ElectricCurrentDensity)
        }
        "electric_charge_density" => {
            known_quantity(runmat_types::physical::QuantityKind::ElectricChargeDensity)
        }
        "magnetic_vector_potential" => {
            known_quantity(runmat_types::physical::QuantityKind::MagneticVectorPotential)
        }
        "mass_flow_rate" => known_quantity(runmat_types::physical::QuantityKind::MassFlowRate),
        "volume_flow_rate" => known_quantity(runmat_types::physical::QuantityKind::VolumeFlowRate),
        "material" => FeaType::Declaration(crate::FeaSymbolKind::Material),
        "geometry" => FeaType::Declaration(crate::FeaSymbolKind::Geometry),
        "mesh" => FeaType::Declaration(crate::FeaSymbolKind::Mesh),
        "model" => FeaType::Declaration(crate::FeaSymbolKind::Model),
        "study" => FeaType::Declaration(crate::FeaSymbolKind::Study),
        "execution" => FeaType::Declaration(crate::FeaSymbolKind::Execution),
        _ => return Err(TypeSyntaxError::Unknown(compact)),
    })
}

fn known_quantity(kind: runmat_types::physical::QuantityKind) -> FeaType {
    FeaType::Quantity(
        runmat_types::physical::DynamicDimension::try_new(kind.dimension(), Some(kind))
            .expect("catalog kind"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_physical_and_topology_types() {
        assert_eq!(
            parse_type_syntax("region < solid >").unwrap(),
            FeaType::Region(FeaTopologyRole::Solid)
        );
        assert_eq!(
            parse_type_syntax("force").unwrap(),
            FeaType::Quantity((PhysicalDimension::FORCE).into())
        );
    }

    #[test]
    fn rejects_open_ended_parameterized_types() {
        assert!(matches!(
            parse_type_syntax("region<anything>"),
            Err(TypeSyntaxError::InvalidParameterization(_))
        ));
    }

    #[test]
    fn rejects_retired_type_aliases() {
        for retired in ["bool", "scalar", "density"] {
            assert!(matches!(
                parse_type_syntax(retired),
                Err(TypeSyntaxError::Unknown(value)) if value == retired
            ));
        }
    }
}
