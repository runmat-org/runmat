//! Unit expressions use the same physical authority as geometry and solvers.
pub use runmat_types::physical::{parse_unit_expression, UnitExpressionError};
use runmat_types::PhysicalDimension;

/// What a dimension measures and its coherent SI unit, for diagnostics that name the expected unit.
pub(crate) fn dimension_requirement(
    dimension: PhysicalDimension,
) -> Option<(&'static str, &'static str)> {
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
        (
            Some(PhysicalDimension::DENSITY),
            "mass per volume",
            "kg/m^3",
        ),
        (Some(PhysicalDimension::TEMPERATURE), "temperature", "K"),
        (Some(PhysicalDimension::VELOCITY), "length per time", "m/s"),
        (
            Some(PhysicalDimension::ACCELERATION),
            "length per time squared",
            "m/s^2",
        ),
        (Some(PhysicalDimension::FORCE), "force", "N"),
        (Some(PhysicalDimension::MOMENT), "force times length", "N*m"),
        (Some(PhysicalDimension::PRESSURE), "force per area", "Pa"),
        (
            force_per(PhysicalDimension::VOLUME),
            "force per volume",
            "N/m^3",
        ),
        (
            force_per(PhysicalDimension::LENGTH),
            "force per length",
            "N/m",
        ),
        (Some(PhysicalDimension::POWER), "power", "W"),
        (
            Some(PhysicalDimension::HEAT_FLUX),
            "power per area",
            "W/m^2",
        ),
        (
            power_per(PhysicalDimension::VOLUME),
            "power per volume",
            "W/m^3",
        ),
        (
            Some(PhysicalDimension::THERMAL_CONDUCTIVITY),
            "thermal conductivity",
            "W/(m*K)",
        ),
        (
            Some(PhysicalDimension::SPECIFIC_HEAT),
            "specific heat",
            "J/(kg*K)",
        ),
        (
            Some(PhysicalDimension::ELECTRIC_CURRENT),
            "electric current",
            "A",
        ),
        (
            Some(PhysicalDimension::ELECTRIC_POTENTIAL),
            "electric potential",
            "V",
        ),
        (Some(current_time), "electric charge", "C"),
        (
            PhysicalDimension::ELECTRIC_CURRENT.divide(PhysicalDimension::AREA),
            "current per area",
            "A/m^2",
        ),
        (
            current_time.divide(PhysicalDimension::VOLUME),
            "charge per volume",
            "C/m^3",
        ),
        (
            Some(PhysicalDimension::MAGNETIC_FLUX_DENSITY),
            "magnetic flux density",
            "T",
        ),
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

/// A quantity type as an author reads it: the semantic kind when known (`temperature difference`),
/// else the dimension's description (`power per volume`), with its SI unit when the dimension
/// has one (`temperature difference (K)`). Never a Rust debug rendering.
pub(crate) fn describe_quantity_type(quantity: runmat_types::physical::DynamicDimension) -> String {
    use runmat_types::physical::QuantitySpec;
    let requirement = dimension_requirement(quantity.physical_dimension());
    // QuantityKind variant names are the authored type names in CamelCase.
    let kind = quantity.semantic_kind().map(|kind| {
        let mut words = String::new();
        for character in format!("{kind:?}").chars() {
            if character.is_ascii_uppercase() && !words.is_empty() {
                words.push(' ');
            }
            words.push(character.to_ascii_lowercase());
        }
        words
    });
    let name = kind
        .or_else(|| requirement.map(|(description, _)| description.to_string()))
        .unwrap_or_else(|| "physical quantity".to_string());
    match requirement {
        Some((_, unit)) => format!("{name} ({unit})"),
        None => name,
    }
}

/// A checked value type as an author reads it, for type-mismatch diagnostics.
pub(crate) fn describe_fea_type(value_type: &crate::FeaType) -> String {
    use crate::FeaType;
    match value_type {
        FeaType::Quantity(quantity) => describe_quantity_type(*quantity),
        FeaType::Vector { element, length } => match length {
            Some(length) => format!("{length}-component vector of {}", describe_fea_type(element)),
            None => format!("vector of {}", describe_fea_type(element)),
        },
        FeaType::Integer => "integer".into(),
        FeaType::Real => "real number".into(),
        FeaType::Boolean => "boolean".into(),
        FeaType::String => "string".into(),
        FeaType::Error => "an invalid value".into(),
        other => type_word(other),
    }
}

fn type_word(value_type: &crate::FeaType) -> String {
    // Remaining variants are declaration and structural types; their variant name is the
    // author-facing word (`Material`, `Region`, ...), never a nested debug structure.
    let debug = format!("{value_type:?}");
    let word = debug.split(['(', ' ', '{']).next().unwrap_or("value");
    word.to_ascii_lowercase()
}

/// When `value` is a quantity whose authored unit measures something other than `expected`,
/// returns the message (`<subject> must be <what>, found <unit>`) and the caret label
/// (`expected <SI unit>`) for a diagnostic located at the value.
pub(crate) fn unit_mismatch(
    subject: &str,
    value: &crate::CheckedExpression,
    expected: PhysicalDimension,
) -> Option<(String, String)> {
    let crate::CheckedExpressionKind::Quantity {
        source_unit_symbol: unit,
        ..
    } = &value.kind
    else {
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
    if found.dimension() == expected {
        return None;
    }
    let (description, expected_unit) =
        dimension_requirement(expected).unwrap_or(("a different physical quantity", "SI units"));
    // A total power (W) on a heat-generation region is planned: thermal.heat-generation.total-power.
    let pending = (subject == "heat_generation_density"
        && found.dimension() == PhysicalDimension::POWER)
        .then_some(super::pending::HEAT_GENERATION_TOTAL_POWER);
    Some((
        super::pending::with_pending(
            format!(
                "{} must be {description}, found {unit}",
                subject.replace('_', " ")
            ),
            pending,
        ),
        format!("expected {expected_unit}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_expected_si_units_for_dimensions() {
        assert_eq!(
            dimension_requirement(PhysicalDimension::PRESSURE),
            Some(("force per area", "Pa"))
        );
        assert_eq!(
            dimension_requirement(parse_unit_expression("W/m^3").unwrap().dimension()),
            Some(("power per volume", "W/m^3"))
        );
    }
}
