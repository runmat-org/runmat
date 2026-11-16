//! `assembly.publish`: what this assembly exposes to the assemblies that use it, by a stable
//! name (`publish <name> = { target = … }`), and Delete on its Outline row (`publish:<name>`).
//!
//! The form's fields: `name` (an existing publish's name changes it in place) and `item`, the
//! pick it publishes: one of the assembly's connectors (`connector:<name>`, a frame a parent mates
//! to) or a component (`occurrence:<uuid>`, which a parent reads as that component's frame).

use super::{model, parse_pick, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::model::{AssemblyDocument, AssemblyPublish, AssemblyPublishTarget, RelPath};
use runmat_cad::ids::PublishName;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};

pub(super) fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `assembly.publish`: adds the publish, or changes the one `name` names.
pub(super) fn apply(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    values: &Map<String, Value>,
) -> Result<String, EditFailure> {
    let name = values
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_owned();
    if !valid_name(&name) {
        return Err(model(format!(
            "`{name}` isn't a publish name: use letters, digits and _, starting with a letter"
        )));
    }
    let pick = picks_of(values, "item")
        .into_iter()
        .next()
        .ok_or_else(|| model("pick a mate connector or a component to publish"))?;
    let target = if let Some(connector) = pick.strip_prefix("connector:") {
        if document.connector(connector).is_none() {
            return Err(model(format!("the assembly has no connector `{connector}`")));
        }
        AssemblyPublishTarget::Connector {
            connector: connector.to_owned(),
        }
    } else {
        match parse_pick(&pick) {
            Some(p) if p.entity.is_none() => {
                let occurrence = compiled.occurrence_name(&p.uuid)?;
                AssemblyPublishTarget::Occurrence {
                    path: RelPath::new(&[occurrence.as_str()]),
                }
            }
            _ => {
                return Err(model(
                    "publish a mate connector or a whole component; for a face, add a mate connector on it first",
                ))
            }
        }
    };
    if let Some(existing) = document.publishes.iter_mut().find(|p| p.name.as_str() == name) {
        existing.target = target;
        return Ok(format!("Changed publish {name}"));
    }
    document.publishes.push(AssemblyPublish {
        name: PublishName::new(name.clone()),
        target,
    });
    Ok(format!("Published {name}"))
}

/// The selected publish rows (`publish:<name>`).
pub(super) fn selected(request: &super::EditRequest) -> impl Iterator<Item = &str> {
    request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("publish:"))
}

/// `assembly.delete` on selected publish rows.
pub(super) fn delete(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let names: Vec<String> = selected(request).map(str::to_owned).collect();
    for name in &names {
        let at = document
            .publishes
            .iter()
            .position(|p| p.name.as_str() == name)
            .ok_or_else(|| model(format!("the assembly publishes no `{name}`")))?;
        document.publishes.remove(at);
    }
    Ok(format!("Deleted {}", names.join(", ")))
}
