//! How a study shows one of its field outputs: the unit its legend is written in, a fixed colour
//! range, the significant digits of the legend's numbers, the solved bodies it leaves out or
//! draws as grey context, and a fixed deformation scale. The study's `display` statement sets
//! these; without one, the unit is the one the study writes that quantity in elsewhere.
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
    /// Solid regions of the study geometry the view leaves out, by region name (the CAD name).
    /// They are not drawn and do not count toward the legend's extremes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hide: Vec<String>,
    /// Solid regions drawn as translucent grey context, by region name: no field colour, and
    /// they do not count toward the legend's extremes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<String>,
    /// A fixed deformation scale (a positive dimensionless factor on the displacement), so two
    /// runs exaggerate their shapes equally. `None` lets each view choose its own scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deformation: Option<f64>,
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

    /// A display that only states `output_id`'s unit and where it comes from.
    pub fn unit_only(output_id: impl Into<String>, unit_source: DisplayUnitSource) -> Self {
        Self {
            output_id: output_id.into(),
            unit: None,
            unit_source,
            range: None,
            precision: None,
            hide: Vec::new(),
            context: Vec::new(),
            deformation: None,
        }
    }

    /// Whether the view of this output draws every solved body in field colour.
    pub fn shows_every_region(&self) -> bool {
        self.hide.is_empty() && self.context.is_empty()
    }
}
