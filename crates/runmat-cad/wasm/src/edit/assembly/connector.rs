//! `assembly.mate-connector`: a named mate connector on a component's face, edge, vertex or
//! datum, or on a point or line of the assembly's sketch, and Delete on its Outline row
//! (`connector:<name>`).
//!
//! The form's fields: `name` (empty: the next free `<component>_c<n>`; an existing connector's
//! name changes that connector in place), `on` (the pick; editing keeps what the connector
//! stands on when it is left empty), `flip` (Z the other way), `x`, `y`, `z` (an offset along
//! the axes of what it stands on, before the flip) and `angle` (a turn about that Z). A pick is published from its part,
//! as a mate's is ([`super::anchors`]), so the connector follows the face when the part changes.

use super::anchors::{Anchor, Publisher};
use super::mate::{connector as make_connector, field, quantity, resolve};
use super::{model, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::model::{AssemblyDocument, OffsetExpr, RotationExpr};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

fn given(values: &Values, id: &str) -> bool {
    !matches!(values.get(id), None | Some(Value::Null))
        && values.get(id).and_then(Value::as_str) != Some("")
}

fn is_zero(e: &ValueExpr) -> bool {
    matches!(e, ValueExpr::Quantity { value, .. } if *value == 0.0)
}

/// The offset the form's `x`, `y`, `z` and `angle` write; `None` when they are all zero.
fn offset(values: &Values) -> Result<Option<OffsetExpr>, EditFailure> {
    let along = |id: &str| -> Result<ValueExpr, EditFailure> {
        if given(values, id) {
            field(values, id, "mm")
        } else {
            Ok(quantity(0.0, "mm"))
        }
    };
    let translation = [along("x")?, along("y")?, along("z")?];
    let angle = if given(values, "angle") {
        Some(field(values, "angle", "deg")?).filter(|a| !is_zero(a))
    } else {
        None
    };
    if translation.iter().all(is_zero) && angle.is_none() {
        return Ok(None);
    }
    Ok(Some(OffsetExpr {
        translation,
        rotation: match angle {
            Some(angle) => RotationExpr::AxisAngle {
                axis: [0.0, 0.0, 1.0],
                angle,
            },
            None => RotationExpr::Identity,
        },
    }))
}

fn flag(values: &Values, id: &str) -> bool {
    match values.get(id) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s == "true",
        _ => false,
    }
}

fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `assembly.mate-connector`: adds the connector, or changes the one `name` names.
pub(super) fn apply(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Values,
) -> Result<String, EditFailure> {
    let name = values
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_owned();
    if !name.is_empty() && !valid_name(&name) {
        return Err(model(format!(
            "`{name}` isn't a connector name: use letters, digits and _, starting with a letter"
        )));
    }
    let existing = document.connectors.iter().position(|c| c.name == name);
    let flip = flag(values, "flip");
    let offset = offset(values)?;
    // The pick, as the new connector a mate would make on it.
    let made = match picks_of(values, "on").into_iter().next() {
        Some(id) => {
            let pick = resolve(document, compiled, &id, "the connector")?;
            if matches!(pick.anchor, Anchor::Connector(_)) {
                return Err(model(
                    "pick a face, edge, vertex or datum of a component, or a sketch point or line, for the connector",
                ));
            }
            let made = make_connector(document, publisher, &pick, flip)?;
            let at = document
                .connectors
                .iter()
                .position(|c| c.name == made)
                .expect("the connector just made");
            Some(document.connectors.remove(at))
        }
        None => None,
    };
    match (existing, made) {
        (Some(at), made) => {
            let c = &mut document.connectors[at];
            if let Some(made) = made {
                c.occurrence = made.occurrence;
                c.anchor = made.anchor;
                c.at = made.at;
                c.z = made.z;
                c.x = made.x;
            }
            c.flip = flip;
            c.offset = offset;
            Ok(format!("Changed mate connector {name}"))
        }
        (None, Some(mut made)) => {
            if !name.is_empty() {
                made.name = name;
            }
            made.flip = flip;
            made.offset = offset;
            let added = made.name.clone();
            document.connectors.push(made);
            Ok(format!("Added mate connector {added}"))
        }
        (None, None) => Err(model(
            "pick a face, edge, vertex or datum of a component, or a sketch point or line, for the connector",
        )),
    }
}

/// The selected connector rows (`connector:<name>`).
pub(super) fn selected(request: &super::EditRequest) -> impl Iterator<Item = &str> {
    request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("connector:"))
}

/// `assembly.delete` on selected connector rows. A connector a mate or a publish uses stays,
/// and the answer names what uses it.
pub(super) fn delete(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let names: Vec<String> = selected(request).map(str::to_owned).collect();
    for name in &names {
        let at = document
            .connectors
            .iter()
            .position(|c| &c.name == name)
            .ok_or_else(|| model(format!("the assembly has no connector `{name}`")))?;
        let mates: Vec<&str> = document
            .mates
            .iter()
            .filter(|m| m.connectors().contains(&name.as_str()))
            .map(|m| m.name.as_str())
            .collect();
        if !mates.is_empty() {
            return Err(model(format!(
                "mate {} uses connector `{name}`: delete the mate first",
                mates.join(", ")
            )));
        }
        if document.publishes.iter().any(|p| {
            matches!(&p.target, runmat_cad::assembly::model::AssemblyPublishTarget::Connector { connector } if connector == name)
        }) {
            return Err(model(format!(
                "the assembly publishes connector `{name}`: delete the publish first"
            )));
        }
        document.connectors.remove(at);
    }
    Ok(format!("Deleted {}", names.join(", ")))
}
