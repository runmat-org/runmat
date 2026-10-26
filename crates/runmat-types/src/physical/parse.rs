use super::UnitDefinition;
use crate::PhysicalDimension;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitExpressionError {
    Empty,
    UnknownUnit(String),
    ExpectedUnit,
    ExpectedExponent,
    InvalidExponent(String),
    AffineComposition,
    ExponentOverflow,
    TrailingOperator,
    InvalidScale,
}

impl fmt::Display for UnitExpressionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("unit expression is empty"),
            Self::UnknownUnit(unit) => write!(formatter, "unknown unit `{unit}`"),
            Self::ExpectedUnit => formatter.write_str("expected a unit symbol"),
            Self::ExpectedExponent => formatter.write_str("expected an integer unit exponent"),
            Self::InvalidExponent(value) => write!(formatter, "invalid unit exponent `{value}`"),
            Self::AffineComposition => {
                formatter.write_str("affine units cannot be multiplied, divided, or exponentiated")
            }
            Self::ExponentOverflow => formatter.write_str("unit exponent exceeds supported bounds"),
            Self::TrailingOperator => formatter.write_str("unit expression ends with an operator"),
            Self::InvalidScale => {
                formatter.write_str("unit conversion scale must be finite and positive")
            }
        }
    }
}

pub fn parse_unit_expression(source: &str) -> Result<UnitDefinition, UnitExpressionError> {
    let normalized = source.replace(['µ', 'μ'], "u");
    let compact = normalized
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect::<String>();
    if compact.is_empty() {
        return Err(UnitExpressionError::Empty);
    }
    let bytes = compact.as_bytes();
    let mut cursor = 0usize;
    let mut divide = false;
    let mut dimension = PhysicalDimension::DIMENSIONLESS;
    let mut scale = 1.0f64;
    let mut affine = false;
    let mut angle_power = 0_i8;
    let mut terms = 0usize;
    while cursor < bytes.len() {
        let start = cursor;
        while cursor < bytes.len() && is_unit_character(bytes[cursor]) {
            cursor += 1;
        }
        if cursor == start {
            return Err(UnitExpressionError::ExpectedUnit);
        }
        let symbol = &compact[start..cursor];
        let unit =
            unit(symbol).ok_or_else(|| UnitExpressionError::UnknownUnit(symbol.to_string()))?;
        let mut exponent = 1i8;
        if bytes.get(cursor) == Some(&b'^') {
            cursor += 1;
            let exponent_start = cursor;
            if matches!(bytes.get(cursor), Some(b'+') | Some(b'-')) {
                cursor += 1;
            }
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor == exponent_start
                || (cursor == exponent_start + 1 && matches!(bytes[exponent_start], b'+' | b'-'))
            {
                return Err(UnitExpressionError::ExpectedExponent);
            }
            let value = &compact[exponent_start..cursor];
            exponent = value
                .parse::<i8>()
                .map_err(|_| UnitExpressionError::InvalidExponent(value.to_string()))?;
        }
        if divide {
            exponent = exponent
                .checked_neg()
                .ok_or(UnitExpressionError::ExponentOverflow)?;
        }
        if unit.offset() != 0.0 && (terms != 0 || exponent != 1 || cursor < bytes.len()) {
            return Err(UnitExpressionError::AffineComposition);
        }
        dimension = dimension
            .multiply(
                unit.dimension
                    .power(exponent)
                    .ok_or(UnitExpressionError::ExponentOverflow)?,
            )
            .ok_or(UnitExpressionError::ExponentOverflow)?;
        angle_power = angle_power
            .checked_add(
                unit.angle_power()
                    .checked_mul(exponent)
                    .ok_or(UnitExpressionError::ExponentOverflow)?,
            )
            .ok_or(UnitExpressionError::ExponentOverflow)?;
        scale *= unit.scale().powi(i32::from(exponent));
        affine |= unit.offset() != 0.0;
        terms += 1;
        if cursor == bytes.len() {
            break;
        }
        match bytes[cursor] {
            b'*' => divide = false,
            b'/' => divide = true,
            _ => return Err(UnitExpressionError::ExpectedUnit),
        }
        cursor += 1;
        if cursor == bytes.len() {
            return Err(UnitExpressionError::TrailingOperator);
        }
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(UnitExpressionError::InvalidScale);
    }
    Ok(UnitDefinition {
        dimension,
        angle_power,
        scale_bits: scale.to_bits(),
        offset_bits: if affine {
            unit(&compact).map_or(0.0_f64, |unit| unit.offset())
        } else {
            0.0
        }
        .to_bits(),
    })
}

fn is_unit_character(value: u8) -> bool {
    value.is_ascii_alphabetic() || value == b'_'
}

fn unit(symbol: &str) -> Option<UnitDefinition> {
    let (dimension, scale, offset) = match symbol {
        "m" => (PhysicalDimension::LENGTH, 1.0, 0.0),
        "mm" => (PhysicalDimension::LENGTH, 1e-3, 0.0),
        "um" => (PhysicalDimension::LENGTH, 1e-6, 0.0),
        "nm" => (PhysicalDimension::LENGTH, 1e-9, 0.0),
        "cm" => (PhysicalDimension::LENGTH, 1e-2, 0.0),
        "km" => (PhysicalDimension::LENGTH, 1e3, 0.0),
        "in" | "inch" => (PhysicalDimension::LENGTH, 0.0254, 0.0),
        "ft" => (PhysicalDimension::LENGTH, 0.3048, 0.0),
        "s" => (PhysicalDimension::TIME, 1.0, 0.0),
        "ms" => (PhysicalDimension::TIME, 1e-3, 0.0),
        "us" => (PhysicalDimension::TIME, 1e-6, 0.0),
        "ns" => (PhysicalDimension::TIME, 1e-9, 0.0),
        "Hz" => (PhysicalDimension::FREQUENCY, 1.0, 0.0),
        "kHz" => (PhysicalDimension::FREQUENCY, 1e3, 0.0),
        "MHz" => (PhysicalDimension::FREQUENCY, 1e6, 0.0),
        "GHz" => (PhysicalDimension::FREQUENCY, 1e9, 0.0),
        "kg" => (PhysicalDimension::MASS, 1.0, 0.0),
        "gram" => (PhysicalDimension::MASS, 1e-3, 0.0),
        "g" => (PhysicalDimension::ACCELERATION, 9.806_65, 0.0),
        "N" => (PhysicalDimension::FORCE, 1.0, 0.0),
        "kN" => (PhysicalDimension::FORCE, 1e3, 0.0),
        "Pa" => (PhysicalDimension::PRESSURE, 1.0, 0.0),
        "kPa" => (PhysicalDimension::PRESSURE, 1e3, 0.0),
        "MPa" => (PhysicalDimension::PRESSURE, 1e6, 0.0),
        "GPa" => (PhysicalDimension::PRESSURE, 1e9, 0.0),
        "J" => (PhysicalDimension::ENERGY, 1.0, 0.0),
        "W" => (PhysicalDimension::POWER, 1.0, 0.0),
        "mW" => (PhysicalDimension::POWER, 1e-3, 0.0),
        "kW" => (PhysicalDimension::POWER, 1e3, 0.0),
        "MW" => (PhysicalDimension::POWER, 1e6, 0.0),
        "K" => (PhysicalDimension::TEMPERATURE, 1.0, 0.0),
        "degC" => (PhysicalDimension::TEMPERATURE, 1.0, 273.15),
        "delta_degC" => (PhysicalDimension::TEMPERATURE, 1.0, 0.0),
        "A" => (PhysicalDimension::ELECTRIC_CURRENT, 1.0, 0.0),
        "mA" => (PhysicalDimension::ELECTRIC_CURRENT, 1e-3, 0.0),
        "kA" => (PhysicalDimension::ELECTRIC_CURRENT, 1e3, 0.0),
        "C" => (PhysicalDimension::new([0, 0, 1, 1, 0, 0, 0, 0]), 1.0, 0.0),
        "V" => (PhysicalDimension::ELECTRIC_POTENTIAL, 1.0, 0.0),
        "mV" => (PhysicalDimension::ELECTRIC_POTENTIAL, 1e-3, 0.0),
        "kV" => (PhysicalDimension::ELECTRIC_POTENTIAL, 1e3, 0.0),
        "MV" => (PhysicalDimension::ELECTRIC_POTENTIAL, 1e6, 0.0),
        "T" => (PhysicalDimension::MAGNETIC_FLUX_DENSITY, 1.0, 0.0),
        "mT" => (PhysicalDimension::MAGNETIC_FLUX_DENSITY, 1e-3, 0.0),
        "rad" => (PhysicalDimension::DIMENSIONLESS, 1.0, 0.0),
        "deg" => (
            PhysicalDimension::DIMENSIONLESS,
            std::f64::consts::PI / 180.0,
            0.0,
        ),
        "B" => (PhysicalDimension::INFORMATION, 1.0, 0.0),
        "KiB" => (PhysicalDimension::INFORMATION, 1024.0, 0.0),
        "MiB" => (PhysicalDimension::INFORMATION, 1024.0 * 1024.0, 0.0),
        "GiB" => (
            PhysicalDimension::INFORMATION,
            1024.0 * 1024.0 * 1024.0,
            0.0,
        ),
        _ => return None,
    };
    Some(UnitDefinition {
        dimension,
        angle_power: if matches!(symbol, "rad" | "deg") {
            1
        } else {
            0
        },
        scale_bits: f64::to_bits(scale),
        offset_bits: f64::to_bits(offset),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // witness: FEA-PUBLIC-LANGUAGE-001/unit_prefixes @ shared_infrastructure/mesh_independent/source_check; coverage: partial; boundary: kV/mm, kA, mT and MHz parse with the correct scale and dimension; kW, MW, mA, MN and mN are not asserted here.
    #[test]
    fn electrical_prefixes_scale_to_si() {
        let field = parse_unit_expression("kV/mm").unwrap();
        assert_eq!(
            field.dimension,
            PhysicalDimension::ELECTRIC_POTENTIAL
                .divide(PhysicalDimension::LENGTH)
                .unwrap()
        );
        assert_eq!(field.scale(), 1e6);
        assert_eq!(parse_unit_expression("kA").unwrap().scale(), 1e3);
        assert_eq!(parse_unit_expression("mT").unwrap().scale(), 1e-3);
        assert_eq!(parse_unit_expression("MHz").unwrap().scale(), 1e6);
    }

    #[test]
    fn parses_composed_si_and_scaled_units() {
        let density = parse_unit_expression("kg / m^3").unwrap();
        assert_eq!(density.dimension, PhysicalDimension::DENSITY);
        assert_eq!(density.scale(), 1.0);
        let pressure = parse_unit_expression("GPa").unwrap();
        assert_eq!(pressure.dimension, PhysicalDimension::PRESSURE);
        assert_eq!(pressure.scale(), 1e9);
    }

    #[test]
    fn distinguishes_acceleration_and_information_units() {
        assert_eq!(
            parse_unit_expression("g").unwrap().dimension,
            PhysicalDimension::ACCELERATION
        );
        assert_eq!(
            parse_unit_expression("GiB").unwrap().dimension,
            PhysicalDimension::INFORMATION
        );
    }

    #[test]
    fn rejects_affine_units_inside_compound_expressions() {
        assert_eq!(
            parse_unit_expression("degC/s"),
            Err(UnitExpressionError::AffineComposition)
        );
    }
}
