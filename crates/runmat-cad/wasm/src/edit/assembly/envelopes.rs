//! `assembly.envelope`: a box of space the assembly reserves (`envelope <name> = { min, max,
//! keep_out, owner }`), fitted round picked components with a margin, and Delete on its
//! `envelope:<name>` row.
//!
//! The box is in the assembly's coordinates: the world box of the picked components' drawn
//! geometry, grown by the margin on every side. An existing name changes that envelope in place;
//! with no components picked it keeps its box. Interference and clearance checks list components
//! inside a keep-out envelope, except its owner.

use super::publishes::valid_name;
use super::{model, parse_pick, Compiled};
use super::mate::field;
use crate::features::picks_of;
use runmat_cad::assembly::model::{AssemblyDocument, EnvelopeDecl};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

fn mm(metres: f64) -> ValueExpr {
    ValueExpr::qty((metres * 1e6).round() / 1e3, "mm")
}

/// The world box (m) of the drawn component `uuid`: its prototype's local box, moved by its
/// placement.
fn component_box(compiled: &Compiled, uuid: &str) -> Option<([f64; 3], [f64; 3])> {
    let (prototype, t) = compiled.placed(uuid)?;
    let p = &compiled.display()["prototypes"][prototype];
    let bounds = if p["shape"] == "box" {
        &p["bounds"]
    } else {
        &p["part"]["bounds"]
    };
    let corner = |k: &str| -> Option<[f64; 3]> {
        let a = bounds[k].as_array()?;
        Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?])
    };
    let (lo, hi) = (corner("min")?, corner("max")?);
    if t.len() < 12 {
        return None;
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for k in 0..8 {
        let p = [0, 1, 2].map(|a| if k & (1 << a) == 0 { lo[a] } else { hi[a] });
        for r in 0..3 {
            let v = t[r * 3] * p[0] + t[r * 3 + 1] * p[1] + t[r * 3 + 2] * p[2] + t[9 + r];
            min[r] = min[r].min(v);
            max[r] = max[r].max(v);
        }
    }
    Some((min, max))
}

/// `assembly.envelope`: adds the envelope, or changes the one `name` names.
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
            "`{name}` isn't a name: use letters, digits and `_`, starting with a letter"
        )));
    }
    let margin = match values.get("margin") {
        None | Some(Value::Null) => 0.0,
        Some(Value::String(s)) if s.trim().is_empty() => 0.0,
        _ => {
            let v = field(values, "margin", "mm")?;
            runmat_cad::units::length_mm(&v)
                .ok_or_else(|| model("the margin must be a length"))?
                / 1e3
        }
    };
    if margin < 0.0 {
        return Err(model("the margin can't be negative"));
    }
    let mut fitted: Option<([f64; 3], [f64; 3])> = None;
    for pick in picks_of(values, "around") {
        let uuid = parse_pick(&pick)
            .map(|p| p.uuid)
            .ok_or_else(|| model(format!("`{pick}` isn't a component")))?;
        let (lo, hi) = component_box(compiled, &uuid).ok_or_else(|| {
            model(format!(
                "{} draws nothing to fit an envelope round",
                compiled.occurrence_name(&uuid).unwrap_or(pick.clone())
            ))
        })?;
        fitted = Some(match fitted {
            None => (lo, hi),
            Some((a, b)) => (
                [0, 1, 2].map(|k| a[k].min(lo[k])),
                [0, 1, 2].map(|k| b[k].max(hi[k])),
            ),
        });
    }
    let owner = match picks_of(values, "owner").first() {
        None => None,
        Some(pick) => {
            let uuid = parse_pick(pick)
                .map(|p| p.uuid)
                .ok_or_else(|| model(format!("`{pick}` isn't a component")))?;
            Some(compiled.occurrence_name(&uuid)?)
        }
    };
    let keep_out = values
        .get("keep_out")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let at = document.envelopes.iter().position(|e| e.name == name);
    let (min, max) = match (fitted, at) {
        (Some((lo, hi)), _) => (
            lo.map(|v| mm(v - margin)),
            hi.map(|v| mm(v + margin)),
        ),
        (None, Some(i)) => (
            document.envelopes[i].min.clone(),
            document.envelopes[i].max.clone(),
        ),
        (None, None) => return Err(model("pick the components the envelope fits round")),
    };
    let envelope = EnvelopeDecl {
        name: name.clone(),
        owner,
        min,
        max,
        keep_out,
    };
    match at {
        Some(i) => {
            document.envelopes[i] = envelope;
            Ok(format!("Changed envelope {name}"))
        }
        None => {
            document.envelopes.push(envelope);
            Ok(format!("Added envelope {name}"))
        }
    }
}

/// The selected `envelope:<name>` rows.
pub(super) fn selected(request: &super::EditRequest) -> impl Iterator<Item = &str> {
    request
        .selection
        .iter()
        .filter_map(|id| id.strip_prefix("envelope:"))
}

/// `assembly.delete` on selected envelope rows.
pub(super) fn delete(
    document: &mut AssemblyDocument,
    request: &super::EditRequest,
) -> Result<String, EditFailure> {
    let mut names = Vec::new();
    for name in selected(request) {
        let at = document
            .envelopes
            .iter()
            .position(|e| e.name == name)
            .ok_or_else(|| model(format!("the assembly has no envelope `{name}`")))?;
        document.envelopes.remove(at);
        names.push(name.to_owned());
    }
    Ok(format!("Deleted envelope {}", names.join(", ")))
}
