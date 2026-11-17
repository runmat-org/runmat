//! The assembly's options block: `assembly.options-effectivity` adds or changes an option axis
//! and the serial range its values apply in, `assembly.options` picks the option values and
//! serial number the assembly shows, and Delete on an `option:<name>` row removes an axis.
//!
//! - An option's values are named (`a, b`; the first is standard). Values written as a range
//!   (`1-99`) make an integer axis.
//! - An effectivity (`MSN 1-99`, `LOT 3-5`) declares or widens the unit (or lot) axis it names
//!   and writes the rule `<option>_effectivity`: the option's values other than the first apply
//!   only inside that range.
//! - The shown values are each axis's default, so the assembly compiles in that configuration of
//!   its options until another is picked.

use super::model;
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::assembly::option_rows::effectivity_rule;
use runmat_cad::assembly::options::{
    AxisDecl, AxisDomain, AxisValue, Condition, OptionRule, OptionsBlock,
};
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

fn text<'a>(values: &'a Values, id: &str) -> &'a str {
    values.get(id).and_then(Value::as_str).unwrap_or("").trim()
}

fn valid(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `1-99` (or `1–99`, `1..99`) as an inclusive range; a single number as itself.
fn range(text: &str) -> Option<(i64, i64)> {
    let t = text.trim();
    for sep in ["..", "-", "–", " to "] {
        if let Some((a, b)) = t.split_once(sep) {
            let (a, b) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
            return (a <= b).then_some((a, b));
        }
    }
    let n = t.parse().ok()?;
    Some((n, n))
}

/// `MSN 1-99` as the effectivity axis's name (`msn`), whether it is a lot axis, and the range.
fn effectivity(text: &str) -> Result<(String, bool, (i64, i64)), EditFailure> {
    let bad = || {
        model(format!(
            "write the effectivity as a serial axis and a range, such as `MSN 1-99`, not `{text}`"
        ))
    };
    let (word, rest) = text
        .trim()
        .split_once(char::is_whitespace)
        .ok_or_else(bad)?;
    let axis = word.to_lowercase();
    if !valid(&axis) {
        return Err(bad());
    }
    let lot = axis == "lot";
    Ok((axis, lot, range(rest).ok_or_else(bad)?))
}

fn block(document: &mut AssemblyDocument) -> &mut OptionsBlock {
    document.options.get_or_insert_with(OptionsBlock::default)
}

/// Declares the unit (or lot) axis `axis`, or widens it to take `lo..hi`.
fn serial_axis(options: &mut OptionsBlock, axis: &str, lot: bool, (lo, hi): (i64, i64)) {
    match options.axes.iter_mut().find(|a| a.name == axis) {
        Some(AxisDecl {
            domain: AxisDomain::Unit { min, max } | AxisDomain::Lot { min, max },
            ..
        }) => {
            *min = (*min).min(lo);
            *max = (*max).max(hi);
        }
        Some(other) => {
            other.domain = if lot {
                AxisDomain::Lot { min: lo, max: hi }
            } else {
                AxisDomain::Unit { min: lo, max: hi }
            };
            other.default = Some(AxisValue::Int(lo));
        }
        None => options.axes.push(AxisDecl {
            name: axis.to_owned(),
            domain: if lot {
                AxisDomain::Lot { min: lo, max: hi }
            } else {
                AxisDomain::Unit { min: lo, max: hi }
            },
            default: Some(AxisValue::Int(lo)),
        }),
    }
}

/// `assembly.options-effectivity`: adds the option or changes the one its name names.
pub(super) fn define(
    document: &mut AssemblyDocument,
    values: &Values,
) -> Result<String, EditFailure> {
    let name = text(values, "option");
    if !valid(name) {
        return Err(model(format!(
            "`{name}` isn't an option name: use letters, digits and `_`, starting with a letter"
        )));
    }
    let written = text(values, "values");
    let (domain, first) = if let Some((lo, hi)) = written
        .contains(['-', '–', '.'])
        .then(|| range(written))
        .flatten()
    {
        (AxisDomain::Integer { min: lo, max: hi }, AxisValue::Int(lo))
    } else {
        let list: Vec<String> = written
            .split(',')
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty())
            .collect();
        if list.is_empty() {
            return Err(model("list the option's values, such as `a, b`"));
        }
        if let Some(bad) = list.iter().find(|v| !valid(v)) {
            return Err(model(format!(
                "`{bad}` isn't a value name: use letters, digits and `_`, starting with a letter"
            )));
        }
        if list.iter().enumerate().any(|(i, v)| list[..i].contains(v)) {
            return Err(model("each value is listed once"));
        }
        let first = AxisValue::Enum(list[0].clone());
        (AxisDomain::Enum { values: list }, first)
    };
    let eff = match text(values, "effectivity") {
        "" => None,
        t => Some(effectivity(t)?),
    };
    if eff.as_ref().is_some_and(|(axis, _, _)| axis == name) {
        return Err(model("an option can't be its own effectivity"));
    }
    let options = block(document);
    let kept = |d: &Option<AxisValue>, domain: &AxisDomain| match (d, domain) {
        (Some(AxisValue::Enum(v)), AxisDomain::Enum { values }) if values.contains(v) => d.clone(),
        (Some(AxisValue::Int(v)), AxisDomain::Integer { min, max }) if v >= min && v <= max => {
            d.clone()
        }
        _ => None,
    };
    let changed = match options.axes.iter_mut().find(|a| a.name == name) {
        Some(axis) => {
            let default = kept(&axis.default, &domain).unwrap_or_else(|| first.clone());
            axis.domain = domain;
            axis.default = Some(default);
            true
        }
        None => {
            options.axes.push(AxisDecl {
                name: name.to_owned(),
                domain,
                default: Some(first.clone()),
            });
            false
        }
    };
    let rule = effectivity_rule(name);
    options.rules.retain(|r| r.name != rule);
    let mut summary = format!(
        "{} option {name}",
        if changed { "Changed" } else { "Added" }
    );
    if let Some((axis, lot, span)) = eff {
        serial_axis(options, &axis, lot, span);
        options.rules.push(OptionRule {
            name: rule,
            condition: Condition::requires(
                Condition::negate(Condition::is(name, first)),
                Condition::range(&axis, Some(span.0), Some(span.1)),
            ),
        });
        summary.push_str(&format!(
            ", its other values on {} {}-{}",
            axis.to_uppercase(),
            span.0,
            span.1
        ));
    }
    Ok(summary)
}

/// The value `text` names on `axis`, checked against its domain.
fn value_on(axis: &AxisDecl, text: &str) -> Result<AxisValue, EditFailure> {
    let out_of = |what: String| model(format!("`{}` takes {what}, not `{text}`", axis.name));
    match &axis.domain {
        AxisDomain::Enum { values } => values
            .iter()
            .find(|v| v.as_str() == text)
            .map(|v| AxisValue::Enum(v.clone()))
            .ok_or_else(|| out_of(values.join(", "))),
        AxisDomain::Bool => match text {
            "true" | "yes" => Ok(AxisValue::Bool(true)),
            "false" | "no" => Ok(AxisValue::Bool(false)),
            _ => Err(out_of("true or false".into())),
        },
        AxisDomain::Unit { min, max }
        | AxisDomain::Lot { min, max }
        | AxisDomain::Integer { min, max } => text
            .parse::<i64>()
            .ok()
            .filter(|n| n >= min && n <= max)
            .map(AxisValue::Int)
            .ok_or_else(|| out_of(format!("{min} to {max}"))),
        AxisDomain::Date { min, max } => {
            runmat_cad::assembly::options::CivilDate::try_from(text.to_owned())
                .ok()
                .filter(|d| d >= min && d <= max)
                .map(AxisValue::Date)
                .ok_or_else(|| out_of(format!("a date from {min} to {max}")))
        }
    }
}

/// `assembly.options`: the option values and serial number the assembly shows.
pub(super) fn show(
    document: &mut AssemblyDocument,
    values: &Values,
) -> Result<String, EditFailure> {
    let Some(options) = document.options.as_mut().filter(|o| !o.axes.is_empty()) else {
        return Err(model(
            "the assembly has no options yet: add one with Options and effectivity",
        ));
    };
    let mut picked: Vec<(String, String)> = Vec::new();
    for pair in text(values, "options")
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        let (axis, value) = pair
            .split_once('=')
            .ok_or_else(|| model(format!("write each option as `name=value`, not `{pair}`")))?;
        picked.push((axis.trim().to_owned(), value.trim().to_owned()));
    }
    let eff = text(values, "effectivity");
    if !eff.is_empty() {
        let (word, n) = eff
            .split_once(char::is_whitespace)
            .ok_or_else(|| model(format!("write the effectivity as `MSN 42`, not `{eff}`")))?;
        picked.push((word.to_lowercase(), n.trim().to_owned()));
    }
    if picked.is_empty() {
        return Err(model("pick an option value or a serial number to show"));
    }
    let mut shown = Vec::new();
    for (axis, value) in picked {
        let decl = options
            .axes
            .iter_mut()
            .find(|a| a.name == axis)
            .ok_or_else(|| model(format!("the assembly has no option `{axis}`")))?;
        decl.default = Some(value_on(decl, &value)?);
        shown.push(format!("{axis} = {value}"));
    }
    Ok(format!("Showing {}", shown.join(", ")))
}

pub(super) fn selected(request: &super::EditRequest) -> impl Iterator<Item = &str> {
    request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("option:"))
}

/// Delete on `option:<name>` rows: the axis, its effectivity rule and its values in the named
/// configurations. Refused while a component, rule or change group reads it.
pub(super) fn delete(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let names: Vec<String> = selected(request).map(str::to_owned).collect();
    for name in &names {
        let reads = |c: &Condition| {
            let mut axes = Vec::new();
            c.axes(&mut axes);
            axes.contains(name)
        };
        if let Some(o) = document
            .occurrences
            .iter()
            .find(|o| o.when.as_ref().is_some_and(reads))
        {
            return Err(model(format!(
                "`{}` is used when `{name}` has a value; change its condition first",
                o.name
            )));
        }
        let options = document
            .options
            .as_mut()
            .filter(|o| o.axes.iter().any(|a| &a.name == name))
            .ok_or_else(|| model(format!("the assembly has no option `{name}`")))?;
        let own = effectivity_rule(name);
        if let Some(r) = options
            .rules
            .iter()
            .find(|r| r.name != own && reads(&r.condition))
        {
            return Err(model(format!(
                "the rule `{}` reads `{name}`; delete or change it first",
                r.name
            )));
        }
        if let Some(g) = options.groups.iter().find(|g| reads(&g.effectivity)) {
            return Err(model(format!(
                "the change group `{}` reads `{name}`; delete or change it first",
                g.name
            )));
        }
        options.axes.retain(|a| &a.name != name);
        options.rules.retain(|r| r.name != own);
        for c in options.configurations.iter_mut() {
            c.values.remove(name);
        }
    }
    if document.options.as_ref().is_some_and(|o| {
        o.axes.is_empty()
            && o.rules.is_empty()
            && o.groups.is_empty()
            && o.configurations.is_empty()
    }) {
        document.options = None;
    }
    Ok(format!("Deleted option {}", names.join(", ")))
}
