//! The top assembly's option and effectivity axes as Outline rows (`option:<name>`): the values
//! each axis takes, the value the assembly shows, and the effectivity range its values other than
//! the first are limited to (the rule `<name>_effectivity`).

use crate::assembly::model::AssemblyDocument;
use crate::assembly::options::{AxisDomain, AxisValue, Condition};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptionRow {
    pub name: String,
    /// `option` (named values), `yes_no`, `unit` (serial or MSN), `lot`, `integer` or `date`.
    pub kind: String,
    /// The values as written in the command: `a, b`, or `1-99` for a range.
    pub values: String,
    /// The value the assembly shows (the axis's default), when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shown: Option<String>,
    /// `MSN 1-99`: the range of the unit or lot axis this option's other values apply in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effectivity: Option<String>,
}

/// The rule that limits option `axis` to an effectivity range: `<axis>_effectivity`.
pub fn effectivity_rule(axis: &str) -> String {
    format!("{axis}_effectivity")
}

/// `MSN 1-99` for the effectivity rule of `axis`, when it has one in the command's form.
fn effectivity_text(doc: &AssemblyDocument, axis: &str) -> Option<String> {
    let options = doc.options.as_ref()?;
    let rule = options
        .rules
        .iter()
        .find(|r| r.name == effectivity_rule(axis))?;
    let Condition::Any { of } = &rule.condition else {
        return None;
    };
    of.iter().find_map(|c| match c {
        Condition::Range {
            axis,
            min: Some(AxisValue::Int(a)),
            max: Some(AxisValue::Int(b)),
        } => Some(format!("{} {a}-{b}", axis.to_uppercase())),
        _ => None,
    })
}

/// One row per axis of the top assembly's options block, in document order.
pub fn option_rows(top: Option<&AssemblyDocument>) -> Vec<OptionRow> {
    let Some(doc) = top else {
        return Vec::new();
    };
    let Some(options) = &doc.options else {
        return Vec::new();
    };
    options
        .axes
        .iter()
        .map(|a| {
            let (kind, values) = match &a.domain {
                AxisDomain::Enum { values } => ("option", values.join(", ")),
                AxisDomain::Bool => ("yes_no", "true, false".to_owned()),
                AxisDomain::Unit { min, max } => ("unit", format!("{min}-{max}")),
                AxisDomain::Lot { min, max } => ("lot", format!("{min}-{max}")),
                AxisDomain::Integer { min, max } => ("integer", format!("{min}-{max}")),
                AxisDomain::Date { min, max } => ("date", format!("{min} to {max}")),
            };
            OptionRow {
                name: a.name.clone(),
                kind: kind.to_owned(),
                values,
                shown: a.default.as_ref().map(ToString::to_string),
                effectivity: effectivity_text(doc, &a.name),
            }
        })
        .collect()
}
