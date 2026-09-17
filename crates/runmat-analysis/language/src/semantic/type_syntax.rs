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
        "boolean" | "bool" => FeaType::Boolean,
        "integer" => FeaType::Integer,
        "real" | "scalar" => FeaType::Real,
        "complex" => FeaType::Complex,
        "string" => FeaType::String,
        "dimensionless" => FeaType::Quantity(PhysicalDimension::DIMENSIONLESS),
        "length" => FeaType::Quantity(PhysicalDimension::LENGTH),
        "area" => FeaType::Quantity(PhysicalDimension::AREA),
        "volume" => FeaType::Quantity(PhysicalDimension::VOLUME),
        "time" => FeaType::Quantity(PhysicalDimension::TIME),
        "frequency" => FeaType::Quantity(PhysicalDimension::FREQUENCY),
        "mass" => FeaType::Quantity(PhysicalDimension::MASS),
        "density" | "mass_density" => FeaType::Quantity(PhysicalDimension::DENSITY),
        "temperature" => FeaType::Quantity(PhysicalDimension::TEMPERATURE),
        "velocity" => FeaType::Quantity(PhysicalDimension::VELOCITY),
        "acceleration" => FeaType::Quantity(PhysicalDimension::ACCELERATION),
        "force" => FeaType::Quantity(PhysicalDimension::FORCE),
        "moment" => FeaType::Quantity(PhysicalDimension::MOMENT),
        "pressure" | "stress" => FeaType::Quantity(PhysicalDimension::PRESSURE),
        "energy" => FeaType::Quantity(PhysicalDimension::ENERGY),
        "power" => FeaType::Quantity(PhysicalDimension::POWER),
        "heat_flux" => FeaType::Quantity(PhysicalDimension::HEAT_FLUX),
        "thermal_conductivity" => FeaType::Quantity(PhysicalDimension::THERMAL_CONDUCTIVITY),
        "electric_potential" => FeaType::Quantity(PhysicalDimension::ELECTRIC_POTENTIAL),
        "electric_current" => FeaType::Quantity(PhysicalDimension::ELECTRIC_CURRENT),
        "magnetic_flux_density" => FeaType::Quantity(PhysicalDimension::MAGNETIC_FLUX_DENSITY),
        "material" => FeaType::Declaration(crate::FeaSymbolKind::Material),
        "geometry" => FeaType::Declaration(crate::FeaSymbolKind::Geometry),
        "mesh" => FeaType::Declaration(crate::FeaSymbolKind::Mesh),
        "model" => FeaType::Declaration(crate::FeaSymbolKind::Model),
        "study" => FeaType::Declaration(crate::FeaSymbolKind::Study),
        "execution" => FeaType::Declaration(crate::FeaSymbolKind::Execution),
        _ => return Err(TypeSyntaxError::Unknown(compact)),
    })
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
            FeaType::Quantity(PhysicalDimension::FORCE)
        );
    }

    #[test]
    fn rejects_open_ended_parameterized_types() {
        assert!(matches!(
            parse_type_syntax("region<anything>"),
            Err(TypeSyntaxError::InvalidParameterization(_))
        ));
    }
}
