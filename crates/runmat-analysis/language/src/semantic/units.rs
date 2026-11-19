//! Unit expressions use the same physical authority as geometry and solvers.
use runmat_types::physical::{dimension_requirement, unit_mismatch_message, FoundUnit};
pub use runmat_types::physical::{parse_unit_expression, UnitExpressionError};
use runmat_types::PhysicalDimension;

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
            Some(length) => format!(
                "{length}-component vector of {}",
                describe_fea_type(element)
            ),
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

/// When `value` is a quantity whose authored unit measures something other than `expected`, or a
/// plain number where `expected` has a dimension, returns the shared message
/// (`<subject> needs a unit of <what> such as <SI unit>; <unit> measures <what>`) and the caret
/// label (`expected <SI unit>`) for a diagnostic located at the value.
pub(crate) fn unit_mismatch(
    subject: &str,
    value: &crate::CheckedExpression,
    expected: PhysicalDimension,
) -> Option<(String, String)> {
    let expected_dimension =
        runmat_types::physical::DynamicDimension::try_new(expected, None).ok()?;
    if is_plain_number(value) && expected != PhysicalDimension::DIMENSIONLESS {
        let text = unit_mismatch_message(subject, expected_dimension, FoundUnit::Missing);
        return Some((text.message, text.label));
    }
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
    // The shared formatter (runmat-types::physical::unit_message) writes the one wrong-unit
    // message every design language reports.
    let text = unit_mismatch_message(
        subject,
        expected_dimension,
        FoundUnit::Authored(unit, found),
    );
    Some((text.message, text.label))
}

/// Whether `value` is written without a unit: a number, a negated number, or a list of them
/// (the load form `[0, 0, -10]`).
fn is_plain_number(value: &crate::CheckedExpression) -> bool {
    match &value.kind {
        crate::CheckedExpressionKind::Number(_) => true,
        crate::CheckedExpressionKind::Unary { operand, .. } => is_plain_number(operand),
        crate::CheckedExpressionKind::List(items) => {
            !items.is_empty() && items.iter().all(is_plain_number)
        }
        _ => false,
    }
}

/// Whether `value` is a quantity (or quantity vector) of exactly `dimension`.
pub(crate) fn is_dimension(value: &crate::CheckedExpression, dimension: PhysicalDimension) -> bool {
    match &value.value_type {
        crate::FeaType::Quantity(found) => found.dimension() == dimension,
        _ => false,
    }
}
