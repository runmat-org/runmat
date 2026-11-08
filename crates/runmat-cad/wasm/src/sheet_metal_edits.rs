//! Sheet Metal commands that edit an existing sheet, as `sheet_metal.*` features:
//!
//! - `sheet-metal.miter-flange` → `sheet_metal.miter_flange(profile, edges, gap)`: the profile
//!   sketch's section swept along a chain of the sheet's edges, mitred at the corners;
//! - `sheet-metal.break-corner` → `sheet_metal.break_corner(edges, kind)`: the corner edges of a
//!   flat chamfered or rounded;
//! - `sheet-metal.unfold` → `sheet_metal.unfold(fixed_face, bends)`: the picked bends (all,
//!   when none are picked) flattened about the fixed face;
//! - `sheet-metal.fold` → `sheet_metal.fold(bends)`: the bends an earlier unfold flattened,
//!   folded back (all, when none are picked);
//! - `sheet-metal.lofted-bend` → `sheet_metal.lofted_bend(profiles, facets, rules)`: a sheet of
//!   the form's thickness between two open profiles, each arc formed in `bends` bends;
//! - `sheet-metal.tab-and-slot` → `sheet_metal.tab_and_slot(edge, slot_face, count, length,
//!   clearance)`: tabs on one sheet's edge and matching slots in the face of the sheet they
//!   reach into.

use crate::features::{picks_of, value};
use crate::picks::Picks;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{json, Map, Value};

type Values = Map<String, Value>;

fn model(message: impl Into<String>) -> EditFailure {
    EditFailure::Model(message.into())
}

/// A whole number field of at least 1.
fn whole(values: &Values, id: &str, what: &str) -> Result<u64, EditFailure> {
    let n = match values.get(id) {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
    .ok_or_else(|| model(format!("`{id}` needs a value")))?;
    if n < 1.0 || n.fract() != 0.0 {
        return Err(model(format!("{what} is a whole number of at least 1")));
    }
    Ok(n as u64)
}

fn some(values: &Values, id: &str, what: &str) -> Result<Vec<String>, EditFailure> {
    let all = picks_of(values, id);
    if all.is_empty() {
        Err(model(format!("pick {what}")))
    } else {
        Ok(all)
    }
}

/// The `sheet_metal.*` feature a command adds, with the stem its name takes; `None` for a
/// command this module doesn't lower.
pub fn sheet_metal_kind(command: &str, values: &Values, picks: &Picks) -> Result<Option<(&'static str, Value)>, EditFailure> {
    let sm = |kind: &str| json!({ "feature": "sheet_metal", "sheet_metal": kind });
    Ok(Some(match command {
        "sheet-metal.miter-flange" => {
            let mut k = sm("miter_flange");
            k["profile"] = json!({ "sketch": picks.sketch(&some(values, "profile", "the profile sketch")?[0])? });
            k["edges"] = json!(picks.references(&some(values, "edges", "the edges to run along")?)?);
            k["gap"] = value(values, "gap", "mm")?;
            ("miter", k)
        }
        "sheet-metal.break-corner" => {
            let mut k = sm("break_corner");
            k["edges"] = json!(picks.references(&some(values, "edges", "the corner edges")?)?);
            k["kind"] = match values.get("break").and_then(Value::as_str).unwrap_or("chamfer") {
                "chamfer" => json!({ "break": "chamfer", "distance": value(values, "distance", "mm")? }),
                "fillet" => json!({ "break": "fillet", "radius": value(values, "distance", "mm")? }),
                other => return Err(model(format!("a corner breaks with a chamfer or a fillet, not `{other}`"))),
            };
            ("corner", k)
        }
        "sheet-metal.unfold" => {
            let mut k = sm("unfold");
            k["fixed_face"] = picks.reference(&some(values, "fixed", "the face that stays fixed")?[0])?;
            k["bends"] = json!(picks.references(&picks_of(values, "bends"))?);
            ("unfold", k)
        }
        "sheet-metal.fold" => {
            let mut k = sm("fold");
            k["bends"] = json!(picks.references(&picks_of(values, "bends"))?);
            ("fold", k)
        }
        "sheet-metal.lofted-bend" => {
            let mut k = sm("lofted_bend");
            let first = picks.sketch(&some(values, "profile1", "the first profile")?[0])?;
            let second = picks.sketch(&some(values, "profile2", "the second profile")?[0])?;
            k["profiles"] = json!([{ "sketch": first }, { "sketch": second }]);
            k["facets"] = json!(whole(values, "bends", "the bends per arc")?);
            k["rules"] = json!({
                "thickness": value(values, "thickness", "mm")?,
                "bend_radius": { "form": "quantity", "value": 0.0, "unit": "mm" },
                "rule": { "rule": "k_factor", "k": { "form": "number", "value": 0.44 } },
            });
            ("loft", k)
        }
        "sheet-metal.tab-and-slot" => {
            let mut k = sm("tab_and_slot");
            k["edge"] = picks.reference(&some(values, "edge", "the edge the tabs grow from")?[0])?;
            k["slot_face"] = picks.reference(&some(values, "face", "the face the slots go through")?[0])?;
            k["count"] = json!({ "form": "number", "value": whole(values, "count", "the tab count")? });
            k["length"] = value(values, "length", "mm")?;
            k["clearance"] = value(values, "clearance", "mm")?;
            ("tabs", k)
        }
        _ => return Ok(None),
    }))
}
