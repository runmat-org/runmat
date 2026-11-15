//! How a study shows one of its field outputs: the unit its legend is written in, a fixed colour
//! range and the significant digits of the legend's numbers. The study's `display` statement
//! sets these; without one, the unit is the one the study writes that quantity in elsewhere.
use serde::{Deserialize, Serialize};

/// The display of one field output. The compiler records one for each field output that has a
/// `display` statement or whose quantity the study writes in a unit other than SI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredOutputDisplay {
    /// The field output shown (`output <name> = analysis.<step>.field.…`).
    pub output_id: String,
    /// The unit symbol the legend is written in, as the source spells it (`MPa`, `degC`).
    /// `None` keeps the field's SI unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Where `unit` comes from.
    pub unit_source: DisplayUnitSource,
    /// A fixed colour range in canonical SI values. Values outside it take the end colours.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<DisplayRange>,
    /// Significant digits of the legend's numbers, 1 to 6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<u8>,
}

/// Where a display's unit comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayUnitSource {
    /// The `unit` member of the output's `display` statement.
    Display,
    /// The unit the study writes this quantity in: an assessment bound, a load, a condition or
    /// an initial value of the same physical dimension.
    Study,
    /// No authored unit: the field's SI unit.
    Si,
}

/// A fixed colour range, canonical SI values with `minimum < maximum`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisplayRange {
    pub minimum: f64,
    pub maximum: f64,
}

/// The fewest and most significant digits a display may ask for.
pub const DISPLAY_PRECISION_DIGITS: std::ops::RangeInclusive<u8> = 1..=6;

impl AuthoredOutputDisplay {
    /// The display of `output_id` among `displays`, if the study records one.
    pub fn of<'a>(displays: &'a [Self], output_id: &str) -> Option<&'a Self> {
        displays
            .iter()
            .find(|display| display.output_id == output_id)
    }
}
