//! `assembly.configurations`: the assembly's named configurations and display states, and Delete
//! on their Outline rows (`configuration:<name>`, `display_state:<name>`).
//!
//! The form's fields: `kind` (`configuration` or `display_state`), `name` (an existing one's name
//! changes it in place), and for a configuration `values`, its input values as written
//! (`width = 40 mm, depth = 20 mm`; each must be one of the assembly's inputs), and `suppress`,
//! the components it leaves out; for a display state `hidden`, the components the view hides.

use super::publishes::valid_name;
use super::{model, parse_pick, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::model::{AssemblyConfiguration, AssemblyDisplayState, AssemblyDocument};
use runmat_cad::ids::{ConfigurationName, InputName};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::syntax::parse_assembly;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

fn text<'a>(values: &'a Map<String, Value>, key: &str) -> &'a str {
    values.get(key).and_then(Value::as_str).unwrap_or("").trim()
}

/// The component names `field`'s picks name (whole top-level components only).
fn components(
    compiled: &Compiled,
    values: &Map<String, Value>,
    field: &str,
) -> Result<BTreeSet<String>, EditFailure> {
    let mut out = BTreeSet::new();
    for pick in picks_of(values, field) {
        match parse_pick(&pick) {
            Some(p) => {
                out.insert(compiled.occurrence_name(&p.uuid)?);
            }
            None => return Err(model(format!("`{pick}` isn't a component"))),
        }
    }
    Ok(out)
}

/// `values` (`width = 40 mm, depth = 20 mm`) as input values, read by the assembly parser.
fn input_values(
    document: &AssemblyDocument,
    values: &str,
) -> Result<BTreeMap<InputName, ValueExpr>, EditFailure> {
    if values.is_empty() {
        return Ok(BTreeMap::new());
    }
    let source = format!(
        "runmat 0.7\nstd 1\n\nassembly values {{\n    configuration c = {{ overrides = {{ {values} }} }}\n}}\n"
    );
    let parsed = parse_assembly("/values.assembly", &source);
    let read = parsed
        .document
        .filter(|_| parsed.diagnostics.is_empty())
        .and_then(|d| d.configurations.into_iter().next())
        .ok_or_else(|| {
            model(format!(
                "`{values}` isn't a list of input values: write them as `width = 40 mm, depth = 20 mm`"
            ))
        })?;
    for name in read.overrides.keys() {
        if !document.inputs.iter().any(|i| &i.name == name) {
            let known: Vec<&str> = document.inputs.iter().map(|i| i.name.as_str()).collect();
            return Err(model(if known.is_empty() {
                format!("the assembly has no input `{name}`; add it with Inputs first")
            } else {
                format!(
                    "the assembly has no input `{name}`; it has {}",
                    known.join(", ")
                )
            }));
        }
    }
    Ok(read.overrides)
}

/// `assembly.configurations`: adds the configuration or display state, or changes the one
/// `name` names.
pub(super) fn apply(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    values: &Map<String, Value>,
) -> Result<String, EditFailure> {
    let name = text(values, "name").to_owned();
    if !valid_name(&name) {
        return Err(model(format!(
            "`{name}` isn't a name: use letters, digits and _, starting with a letter"
        )));
    }
    if name == "default" {
        return Err(model(
            "`default` is the assembly as written; give the configuration another name",
        ));
    }
    match text(values, "kind") {
        "display_state" => {
            let hidden = components(compiled, values, "hidden")?;
            if hidden.is_empty() {
                return Err(model("pick the components the display state hides"));
            }
            if let Some(existing) = document.display_states.iter_mut().find(|d| d.name == name) {
                existing.hidden = hidden;
                return Ok(format!("Changed display state {name}"));
            }
            document.display_states.push(AssemblyDisplayState {
                name: name.clone(),
                hidden,
            });
            Ok(format!("Added display state {name}"))
        }
        "configuration" | "" => {
            let overrides = input_values(document, text(values, "values"))?;
            let suppress = components(compiled, values, "suppress")?;
            let key = ConfigurationName::new(&name);
            if let Some(existing) = document.configurations.iter_mut().find(|c| c.name == key) {
                existing.overrides = overrides;
                existing.suppress = suppress;
                return Ok(format!("Changed configuration {name}"));
            }
            document.configurations.push(AssemblyConfiguration {
                name: key,
                overrides,
                suppress,
            });
            Ok(format!("Added configuration {name}"))
        }
        other => Err(model(format!(
            "`{other}` isn't a kind: add a configuration or a display state"
        ))),
    }
}

/// The selected configuration and display-state rows.
pub(super) fn selected(request: &super::EditRequest) -> impl Iterator<Item = &str> {
    request.selection.iter().filter_map(|id| {
        id.strip_prefix("configuration:")
            .or_else(|| id.strip_prefix("display_state:"))
            .map(|_| id.as_str())
    })
}

/// `assembly.delete` on selected configuration and display-state rows.
pub(super) fn delete(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let mut names = Vec::new();
    for id in selected(request) {
        if let Some(name) = id.strip_prefix("configuration:") {
            let at = document
                .configurations
                .iter()
                .position(|c| c.name.as_str() == name)
                .ok_or_else(|| model(format!("the assembly has no configuration `{name}`")))?;
            document.configurations.remove(at);
            names.push(name.to_owned());
        } else if let Some(name) = id.strip_prefix("display_state:") {
            let at = document
                .display_states
                .iter()
                .position(|d| d.name == name)
                .ok_or_else(|| model(format!("the assembly has no display state `{name}`")))?;
            document.display_states.remove(at);
            names.push(name.to_owned());
        }
    }
    Ok(format!("Deleted {}", names.join(", ")))
}
