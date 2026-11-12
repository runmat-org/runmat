//! What a dimension measures, and the one wrong-unit message every design language reports.
//!
//! `.fea`, `.part`, `.assembly`, `.drawing`, `.circuit` and `.circuit.layout` all check authored
//! units against this module. The kind table names what a dimension measures (`force per area`)
//! with its coherent SI unit (`Pa`) and the units an editor offers as quick fixes; the formatter
//! writes the message and caret label from it:
//!
//! ```text
//! youngs modulus needs a unit of force per area such as Pa; mm measures length
//!                                                             ^^^^^ expected Pa
//! ```
use super::DynamicDimension;
use crate::PhysicalDimension;

/// One row of the kind table: what a dimension measures, its coherent SI unit and the units an
/// editor offers when a value of this kind is written in the wrong unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitKind {
    pub dimension: PhysicalDimension,
    /// The nominal angle power: 1 for an angle (`rad`), 0 for every non-angular kind.
    pub angle_power: i8,
    /// What the dimension measures, as a message names it (`force per area`).
    pub kind: &'static str,
    /// The coherent SI unit (`Pa`).
    pub si_example: &'static str,
    /// Units a quick fix offers, most common first. Each parses with
    /// [`super::parse_unit_expression`] to this row's dimension.
    pub suggestions: &'static [&'static str],
}

const fn row(
    exponents: [i8; 8],
    angle_power: i8,
    kind: &'static str,
    si_example: &'static str,
    suggestions: &'static [&'static str],
) -> UnitKind {
    UnitKind {
        dimension: PhysicalDimension::new(exponents),
        angle_power,
        kind,
        si_example,
        suggestions,
    }
}

/// The kind table. The first row of a dimension names it, so `[2, 1, -2]` reads as
/// `force times length` (moment and energy share it) and `[0, 0, -1]` as `frequency`.
pub const UNIT_KINDS: &[UnitKind] = &[
    row([0; 8], 0, "dimensionless", "1", &[]),
    row([1, 0, 0, 0, 0, 0, 0, 0], 0, "length", "m", &["mm", "cm", "m", "in"]),
    row([2, 0, 0, 0, 0, 0, 0, 0], 0, "area", "m^2", &["mm^2", "cm^2", "m^2"]),
    row([3, 0, 0, 0, 0, 0, 0, 0], 0, "volume", "m^3", &["mm^3", "cm^3", "m^3"]),
    row([0, 0, 1, 0, 0, 0, 0, 0], 0, "time", "s", &["s", "ms", "us"]),
    row([0, 0, -1, 0, 0, 0, 0, 0], 0, "frequency", "Hz", &["Hz", "kHz", "MHz"]),
    row([0, 1, 0, 0, 0, 0, 0, 0], 0, "mass", "kg", &["kg", "gram"]),
    row([-3, 1, 0, 0, 0, 0, 0, 0], 0, "mass per volume", "kg/m^3", &["kg/m^3"]),
    row([0, 0, 0, 0, 1, 0, 0, 0], 0, "temperature", "K", &["K", "degC"]),
    row([1, 0, -1, 0, 0, 0, 0, 0], 0, "length per time", "m/s", &["m/s", "mm/s"]),
    row([1, 0, -2, 0, 0, 0, 0, 0], 0, "length per time squared", "m/s^2", &["m/s^2", "g"]),
    row([1, 1, -2, 0, 0, 0, 0, 0], 0, "force", "N", &["N", "kN"]),
    row([2, 1, -2, 0, 0, 0, 0, 0], 0, "force times length", "N*m", &["N*m", "N*mm"]),
    row([-1, 1, -2, 0, 0, 0, 0, 0], 0, "force per area", "Pa", &["Pa", "kPa", "MPa", "GPa"]),
    row([-2, 1, -2, 0, 0, 0, 0, 0], 0, "force per volume", "N/m^3", &["N/m^3"]),
    row([0, 1, -2, 0, 0, 0, 0, 0], 0, "force per length", "N/m", &["N/m", "N/mm"]),
    row([2, 1, -3, 0, 0, 0, 0, 0], 0, "power", "W", &["W", "mW", "kW"]),
    row([0, 1, -3, 0, 0, 0, 0, 0], 0, "power per area", "W/m^2", &["W/m^2"]),
    row([-1, 1, -3, 0, 0, 0, 0, 0], 0, "power per volume", "W/m^3", &["W/m^3"]),
    row([1, 1, -3, 0, -1, 0, 0, 0], 0, "thermal conductivity", "W/(m*K)", &["W/m/K"]),
    row([2, 0, -2, 0, -1, 0, 0, 0], 0, "specific heat", "J/(kg*K)", &["J/kg/K"]),
    row([0, 0, 0, 1, 0, 0, 0, 0], 0, "electric current", "A", &["A", "mA"]),
    row([2, 1, -3, -1, 0, 0, 0, 0], 0, "electric potential", "V", &["V", "mV", "kV"]),
    row([0, 0, 1, 1, 0, 0, 0, 0], 0, "electric charge", "C", &["C"]),
    row([-2, 0, 0, 1, 0, 0, 0, 0], 0, "current per area", "A/m^2", &["A/m^2"]),
    row([-3, 0, 1, 1, 0, 0, 0, 0], 0, "charge per volume", "C/m^3", &["C/m^3"]),
    row([0, 1, -2, -1, 0, 0, 0, 0], 0, "magnetic flux density", "T", &["T", "mT"]),
    row([1, 1, -2, -1, 0, 0, 0, 0], 0, "flux density times length", "T*m", &["T*m"]),
    row([3, 0, -1, 0, 0, 0, 0, 0], 0, "volume per time", "m^3/s", &["m^3/s"]),
    row([0, 1, -1, 0, 0, 0, 0, 0], 0, "mass per time", "kg/s", &["kg/s"]),
    row([2, 1, -3, -2, 0, 0, 0, 0], 0, "electric resistance", "ohm", &["ohm", "kohm", "Mohm"]),
    row([-2, -1, 3, 2, 0, 0, 0, 0], 0, "electric conductance", "S", &["mS", "S"]),
    row([-2, -1, 4, 2, 0, 0, 0, 0], 0, "capacitance", "F", &["pF", "nF", "uF"]),
    row([2, 1, -2, -2, 0, 0, 0, 0], 0, "inductance", "H", &["nH", "uH", "mH"]),
    row([0; 8], 1, "angle", "rad", &["deg", "rad"]),
    row([0, 0, -1, 0, 0, 0, 0, 0], 1, "angle per time", "rad/s", &["rad/s", "deg/s", "rpm"]),
    row([2, 1, -2, 0, 0, 0, 0, 0], -1, "moment per angle", "N*m/rad", &["N*m/rad", "N*m/deg"]),
    row(
        [2, 1, -1, 0, 0, 0, 0, 0],
        -1,
        "moment times time per angle",
        "N*m*s/rad",
        &["N*m*s/rad"],
    ),
];

/// The kind table's row for a dimension and nominal angle power.
pub fn unit_kind(dimension: PhysicalDimension, angle_power: i8) -> Option<&'static UnitKind> {
    UNIT_KINDS
        .iter()
        .find(|row| row.dimension == dimension && row.angle_power == angle_power)
}

/// What a non-angular dimension measures and its coherent SI unit (`force per area`, `Pa`).
pub fn dimension_requirement(
    dimension: PhysicalDimension,
) -> Option<(&'static str, &'static str)> {
    unit_kind(dimension, 0).map(|row| (row.kind, row.si_example))
}

/// What an authored value carries where a unit of one kind is required.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FoundUnit<'a> {
    /// A unit as written (`mm`), with the dimension it measures.
    Authored(&'a str, DynamicDimension),
    /// A plain number: the value has no unit.
    Missing,
    /// A computed value (an expression or a reference) with no single authored unit.
    Computed(DynamicDimension),
}

/// A wrong-unit diagnostic's text: the message and the caret label under the value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitMismatchMessage {
    pub message: String,
    pub label: String,
}

/// The one wrong-unit message:
/// `<subject> needs a unit of <kind> such as <SI example>; <authored unit> measures <found kind>`,
/// with the caret label `expected <SI example>`.
///
/// `subject` is the field name; underscores read as spaces (`youngs_modulus` → `youngs modulus`).
/// The authored unit is trimmed. An expected dimension outside the kind table reads as `another
/// physical quantity` with `SI units`; a found one as `a different physical quantity`. A plain
/// number reads `a plain number has no unit`, and a computed value `the value measures <kind>`.
pub fn unit_mismatch_message(
    subject: &str,
    expected: DynamicDimension,
    found: FoundUnit<'_>,
) -> UnitMismatchMessage {
    let (kind, si_example) = unit_kind(expected.dimension(), expected.angle_power())
        .map(|row| (row.kind, row.si_example))
        .unwrap_or(("another physical quantity", "SI units"));
    let found_kind = |dimension: DynamicDimension| {
        unit_kind(dimension.dimension(), dimension.angle_power())
            .map(|row| row.kind)
            .unwrap_or("a different physical quantity")
    };
    let found = match found {
        FoundUnit::Authored(unit, dimension) => {
            format!("{} measures {}", unit.trim(), found_kind(dimension))
        }
        FoundUnit::Missing => "a plain number has no unit".to_string(),
        FoundUnit::Computed(dimension) => format!("the value measures {}", found_kind(dimension)),
    };
    UnitMismatchMessage {
        message: format!(
            "{} needs a unit of {kind} such as {si_example}; {found}",
            subject.trim().replace('_', " ")
        ),
        label: format!("expected {si_example}"),
    }
}

/// The units a quick fix offers for a value that needs `expected`: the kind table's suggestions,
/// or its SI example when the row lists none. Empty for a dimension outside the table.
pub fn suggested_units(expected: DynamicDimension) -> Vec<&'static str> {
    match unit_kind(expected.dimension(), expected.angle_power()) {
        Some(row) if !row.suggestions.is_empty() => row.suggestions.to_vec(),
        Some(row) if row.si_example != "1" => vec![row.si_example],
        _ => Vec::new(),
    }
}

/// The SI example a shared wrong-unit message names (`such as <SI example>;`), or `None` for any
/// other message. The inverse of [`unit_mismatch_message`], so an editor offers the same quick
/// fixes for every language's diagnostic.
pub fn expected_si_example(message: &str) -> Option<&str> {
    let (_, rest) = message.split_once(" needs a unit of ")?;
    let (_, rest) = rest.split_once(" such as ")?;
    let (example, _) = rest.split_once("; ")?;
    Some(example)
}

/// Quick fixes for a value written `value_text` (`10 kg`, `5`) where a unit of the kind whose SI
/// example is `si_example` is required: the value's own number with each unit of that kind
/// ([`suggested_units`]), as replacement texts (`10 mm`, `10 cm`). The number stays exactly as
/// written; a value that isn't a number literal gets no fixes.
pub fn unit_fixes(value_text: &str, si_example: &str) -> Vec<String> {
    let text = value_text.trim();
    let number_end = text
        .char_indices()
        .find(|&(at, c)| {
            !(c.is_ascii_digit()
                || c == '.'
                || ((c == '-' || c == '+')
                    && (at == 0 || matches!(text[..at].chars().last(), Some('e' | 'E'))))
                || ((c == 'e' || c == 'E')
                    && text[at + 1..]
                        .chars()
                        .next()
                        .is_some_and(|n| n.is_ascii_digit() || n == '-' || n == '+')))
        })
        .map_or(text.len(), |(at, _)| at);
    let number = &text[..number_end];
    if number.parse::<f64>().is_err() {
        return Vec::new();
    }
    let Ok(expected) = super::parse_unit_expression(si_example) else {
        return Vec::new();
    };
    let written = text[number_end..].trim();
    suggested_units(DynamicDimension::from_unit(expected))
        .into_iter()
        .filter(|unit| *unit != written)
        .map(|unit| format!("{number} {unit}"))
        .collect()
}

/// A value as every editor shows it in hover and properties: the number as authored in `unit`,
/// with at most nine significant digits, then the unit (`0.25 in`, `12 mm`). A plain number
/// (`unit` empty or `1`) shows alone.
pub fn quantity_text(value: f64, unit: &str) -> String {
    let number = if value == 0.0 {
        "0".to_string()
    } else if value.abs() < 1e-4 || value.abs() >= 1e9 {
        format!("{value:e}")
    } else {
        let digits = (8 - value.abs().log10().floor() as i32).clamp(0, 12) as usize;
        let fixed = format!("{value:.digits$}");
        let trimmed = if fixed.contains('.') {
            fixed.trim_end_matches('0').trim_end_matches('.')
        } else {
            fixed.as_str()
        };
        trimmed.to_string()
    };
    match unit.trim() {
        "" | "1" => number,
        unit => format!("{number} {unit}"),
    }
}

/// A computed value (canonical SI) as editors show it when no authored unit applies: in its
/// kind's first suggested unit (`mm` for a length, `deg` for an angle), else its SI example.
pub fn display_quantity(si: f64, dimension: DynamicDimension) -> String {
    let unit = unit_kind(dimension.dimension(), dimension.angle_power()).and_then(|row| {
        row.suggestions
            .first()
            .copied()
            .or(Some(row.si_example).filter(|unit| *unit != "1"))
    });
    match unit.and_then(|unit| Some((unit, super::parse_unit_expression(unit).ok()?))) {
        Some((unit, definition)) => match definition.from_canonical(si) {
            Ok(value) => quantity_text(value, unit),
            Err(_) => quantity_text(si, ""),
        },
        None => quantity_text(si, ""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::parse_unit_expression;

    fn of(unit: &str) -> DynamicDimension {
        DynamicDimension::from_unit(parse_unit_expression(unit).unwrap())
    }

    #[test]
    fn names_expected_si_units_for_dimensions() {
        assert_eq!(
            dimension_requirement(PhysicalDimension::PRESSURE),
            Some(("force per area", "Pa"))
        );
        assert_eq!(
            dimension_requirement(of("W/m^3").dimension()),
            Some(("power per volume", "W/m^3"))
        );
    }

    #[test]
    fn every_suggestion_parses_to_its_row() {
        for row in UNIT_KINDS {
            for unit in row.suggestions {
                let parsed = of(unit);
                assert_eq!(
                    (parsed.dimension(), parsed.angle_power()),
                    (row.dimension, row.angle_power),
                    "`{unit}` for {}",
                    row.kind
                );
            }
        }
    }

    #[test]
    fn writes_the_one_message() {
        let length = of("m");
        let cases = [
            (
                FoundUnit::Authored(" kg ", of("kg")),
                "distance needs a unit of length such as m; kg measures mass",
            ),
            (
                FoundUnit::Authored("deg", of("deg")),
                "distance needs a unit of length such as m; deg measures angle",
            ),
            (
                FoundUnit::Missing,
                "distance needs a unit of length such as m; a plain number has no unit",
            ),
            (
                FoundUnit::Computed(of("N")),
                "distance needs a unit of length such as m; the value measures force",
            ),
        ];
        for (found, message) in cases {
            let text = unit_mismatch_message("distance", length, found);
            assert_eq!(text.message, message);
            assert_eq!(text.label, "expected m");
        }
        let unknown = DynamicDimension::from_unit(parse_unit_expression("m^4/kg").unwrap());
        assert_eq!(
            unit_mismatch_message("youngs_modulus", unknown, FoundUnit::Authored("mm", of("mm")))
                .message,
            "youngs modulus needs a unit of another physical quantity such as SI units; mm measures length"
        );
        assert_eq!(
            unit_mismatch_message("force", of("N"), FoundUnit::Authored("m^4", of("m^4"))).message,
            "force needs a unit of force such as N; m^4 measures a different physical quantity"
        );
    }

    #[test]
    fn suggests_units_of_the_expected_kind() {
        assert_eq!(suggested_units(of("m")), ["mm", "cm", "m", "in"]);
        assert_eq!(suggested_units(of("deg")), ["deg", "rad"]);
        let plain = DynamicDimension::quantity_type(crate::physical::QuantityKind::Dimensionless);
        assert!(suggested_units(plain).is_empty());
    }

    #[test]
    fn shows_quantities_in_the_authored_or_display_unit() {
        assert_eq!(quantity_text(0.25, "in"), "0.25 in");
        assert_eq!(quantity_text(6.35, " mm "), "6.35 mm");
        assert_eq!(quantity_text(3.0, ""), "3");
        assert_eq!(quantity_text(1.0 / 3.0, "mm"), "0.333333333 mm");
        assert_eq!(display_quantity(0.00635, of("m")), "6.35 mm");
        assert_eq!(display_quantity(std::f64::consts::FRAC_PI_2, of("deg")), "90 deg");
        assert_eq!(display_quantity(5.0, of("N")), "5 N");
    }

    #[test]
    fn quick_fixes_keep_the_number_and_offer_units_of_the_kind() {
        let message = unit_mismatch_message("distance", of("m"), FoundUnit::Authored("kg", of("kg")));
        let example = expected_si_example(&message.message).unwrap();
        assert_eq!(example, "m");
        assert_eq!(unit_fixes("10 kg", example), ["10 mm", "10 cm", "10 m", "10 in"]);
        assert_eq!(unit_fixes("-2.5e-3", example)[0], "-2.5e-3 mm");
        assert_eq!(unit_fixes("10 mm", example), ["10 cm", "10 m", "10 in"]);
        assert!(unit_fixes("input.width", example).is_empty());
        assert!(expected_si_example("unknown unit `furlong`").is_none());
    }
}
