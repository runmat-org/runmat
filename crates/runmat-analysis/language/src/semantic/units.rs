use super::{ParsedUnit, PhysicalDimension};
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
        }
    }
}

pub fn parse_unit_expression(source: &str) -> Result<ParsedUnit, UnitExpressionError> {
    let compact = source
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
    Ok(ParsedUnit {
        dimension,
        scale_bits: scale.to_bits(),
        offset_bits: if affine {
            unit(&compact).map_or(0.0, ParsedUnit::offset)
        } else {
            0.0
        }
        .to_bits(),
    })
}

/// What a dimension measures and its coherent SI unit, for diagnostics that name the expected unit.
pub(crate) fn dimension_requirement(dimension: PhysicalDimension) -> Option<(&'static str, &'static str)> {
    let force_per = |denominator: PhysicalDimension| PhysicalDimension::FORCE.divide(denominator);
    let power_per = |denominator: PhysicalDimension| PhysicalDimension::POWER.divide(denominator);
    let current_time = PhysicalDimension::ELECTRIC_CURRENT.multiply(PhysicalDimension::TIME)?;
    let table = [
        (Some(PhysicalDimension::DIMENSIONLESS), "dimensionless", "1"),
        (Some(PhysicalDimension::LENGTH), "length", "m"),
        (Some(PhysicalDimension::AREA), "area", "m^2"),
        (Some(PhysicalDimension::VOLUME), "volume", "m^3"),
        (Some(PhysicalDimension::TIME), "time", "s"),
        (Some(PhysicalDimension::FREQUENCY), "frequency", "Hz"),
        (Some(PhysicalDimension::MASS), "mass", "kg"),
        (Some(PhysicalDimension::DENSITY), "mass per volume", "kg/m^3"),
        (Some(PhysicalDimension::TEMPERATURE), "temperature", "K"),
        (Some(PhysicalDimension::VELOCITY), "length per time", "m/s"),
        (Some(PhysicalDimension::ACCELERATION), "length per time squared", "m/s^2"),
        (Some(PhysicalDimension::FORCE), "force", "N"),
        (Some(PhysicalDimension::MOMENT), "force times length", "N*m"),
        (Some(PhysicalDimension::PRESSURE), "force per area", "Pa"),
        (force_per(PhysicalDimension::VOLUME), "force per volume", "N/m^3"),
        (force_per(PhysicalDimension::LENGTH), "force per length", "N/m"),
        (Some(PhysicalDimension::POWER), "power", "W"),
        (Some(PhysicalDimension::HEAT_FLUX), "power per area", "W/m^2"),
        (power_per(PhysicalDimension::VOLUME), "power per volume", "W/m^3"),
        (Some(PhysicalDimension::THERMAL_CONDUCTIVITY), "thermal conductivity", "W/(m*K)"),
        (Some(PhysicalDimension::SPECIFIC_HEAT), "specific heat", "J/(kg*K)"),
        (Some(PhysicalDimension::ELECTRIC_CURRENT), "electric current", "A"),
        (Some(PhysicalDimension::ELECTRIC_POTENTIAL), "electric potential", "V"),
        (Some(current_time), "electric charge", "C"),
        (
            PhysicalDimension::ELECTRIC_CURRENT.divide(PhysicalDimension::AREA),
            "current per area",
            "A/m^2",
        ),
        (current_time.divide(PhysicalDimension::VOLUME), "charge per volume", "C/m^3"),
        (Some(PhysicalDimension::MAGNETIC_FLUX_DENSITY), "magnetic flux density", "T"),
        (
            PhysicalDimension::MAGNETIC_FLUX_DENSITY.multiply(PhysicalDimension::LENGTH),
            "flux density times length",
            "T*m",
        ),
        (
            PhysicalDimension::VOLUME.divide(PhysicalDimension::TIME),
            "volume per time",
            "m^3/s",
        ),
        (
            PhysicalDimension::MASS.divide(PhysicalDimension::TIME),
            "mass per time",
            "kg/s",
        ),
    ];
    table
        .into_iter()
        .find(|(candidate, _, _)| *candidate == Some(dimension))
        .map(|(_, description, unit)| (description, unit))
}

/// When `value` is a quantity whose authored unit measures something other than `expected`,
/// returns the message (`<subject> must be <what>, found <unit>`) and the caret label
/// (`expected <SI unit>`) for a diagnostic located at the value.
pub(crate) fn unit_mismatch(
    subject: &str,
    value: &crate::CheckedExpression,
    expected: PhysicalDimension,
) -> Option<(String, String)> {
    let crate::CheckedExpressionKind::Quantity { unit, .. } = &value.kind else {
        return None;
    };
    let found = match &value.value_type {
        crate::FeaType::Quantity(found) => *found,
        crate::FeaType::Vector { element, .. } => match element.as_ref() {
            crate::FeaType::Quantity(found) => *found,
            _ => return None,
        },
        _ => return None,
    };
    if found == expected {
        return None;
    }
    let (description, expected_unit) =
        dimension_requirement(expected).unwrap_or(("a different physical quantity", "SI units"));
    Some((
        format!("{} must be {description}, found {unit}", subject.replace('_', " ")),
        format!("expected {expected_unit}"),
    ))
}

fn is_unit_character(value: u8) -> bool {
    value.is_ascii_alphabetic() || value == b'_'
}

fn unit(symbol: &str) -> Option<ParsedUnit> {
    let (dimension, scale, offset) = match symbol {
        "m" => (PhysicalDimension::LENGTH, 1.0, 0.0),
        "mm" => (PhysicalDimension::LENGTH, 1e-3, 0.0),
        "cm" => (PhysicalDimension::LENGTH, 1e-2, 0.0),
        "km" => (PhysicalDimension::LENGTH, 1e3, 0.0),
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
    Some(ParsedUnit {
        dimension,
        scale_bits: f64::to_bits(scale),
        offset_bits: f64::to_bits(offset),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // witness: FEA-PUBLIC-LANGUAGE-001/unit_prefixes @
    //   shared_infrastructure/mesh_independent/source_check; coverage: partial;
    //   boundary: kV/mm, kA, mT and MHz parse with the correct scale and dimension; kW, MW, mA, MN and
    //   mN are not asserted here.
    #[test]
    fn electrical_prefixes_scale_to_si() {
        let field = parse_unit_expression("kV/mm").unwrap();
        assert_eq!(
            field.dimension,
            PhysicalDimension::ELECTRIC_POTENTIAL.divide(PhysicalDimension::LENGTH).unwrap()
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
    fn names_expected_si_units_for_dimensions() {
        assert_eq!(
            dimension_requirement(PhysicalDimension::PRESSURE),
            Some(("force per area", "Pa"))
        );
        assert_eq!(
            dimension_requirement(parse_unit_expression("W/m^3").unwrap().dimension),
            Some(("power per volume", "W/m^3"))
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
