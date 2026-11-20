//! Annotation views: a saved view direction each (the assembly's [`ModelView`] form), grouping
//! the annotations filed under it. Showing one turns the model to it and draws only its
//! annotations.

use super::{Edit, ANNOTATION};
use crate::drawing_edits::{text, Values};
use crate::edit::model;
use crate::features::picks_of;
use runmat_cad::assembly::views::ModelView;
use runmat_cad::model::PartDocument;
use runmat_cad::pmi::PmiSet;
use runmat_cad::syntax::edit::{edit_part, EditFailure};
use serde_json::Value;

/// The view direction (from the model toward the eye) and up of a standard orientation, in the
/// part's axes (Top is the xy plane, Front the zx plane, Right the yz plane).
pub fn orientation(word: &str) -> Option<([f64; 3], [f64; 3])> {
    let z = [0.0, 0.0, 1.0];
    Some(match word {
        "front" => ([0.0, -1.0, 0.0], z),
        "back" => ([0.0, 1.0, 0.0], z),
        "right" => ([1.0, 0.0, 0.0], z),
        "left" => ([-1.0, 0.0, 0.0], z),
        "top" => ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        "bottom" => ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
        "isometric" => ([1.0, -1.0, 1.0], z),
        _ => return None,
    })
}

fn vector(v: &Value) -> Option<[f64; 3]> {
    let f: Vec<f64> = match v {
        Value::Array(a) => a.iter().filter_map(Value::as_f64).collect(),
        Value::String(t) => t.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        _ => return None,
    };
    (f.len() == 3 && f.iter().all(|x| x.is_finite()) && f.iter().any(|x| x.abs() > 1e-9))
        .then(|| [f[0], f[1], f[2]])
}

/// The canvas camera the Desktop sends with a "Current view": `camera_from` and `camera_up`
/// (`x, y, z`), or `camera = { from, up }`.
fn camera(values: &Values) -> Option<([f64; 3], [f64; 3])> {
    let object = values.get("camera");
    let from = values
        .get("camera_from")
        .and_then(vector)
        .or_else(|| object.and_then(|c| vector(&c["from"])))?;
    let up = values
        .get("camera_up")
        .and_then(vector)
        .or_else(|| object.and_then(|c| vector(&c["up"])))
        .unwrap_or([0.0, 0.0, 1.0]);
    Some((from, up))
}

fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `annotate.annotation-view`: a new view (`name`) from a standard orientation or the canvas's
/// current camera (`camera = { from, up }`, which the Desktop sends), filing the picked
/// annotations under it, or every annotation not in a view yet when none is picked.
pub fn run(path: &str, text_: &str, _doc: &PartDocument, values: &Values) -> Edit {
    let name = text(values, "name", "")
        .trim()
        .replace([' ', '-'], "_")
        .to_lowercase();
    if !valid_name(&name) {
        return Err(model(
            "name the view with letters, digits and underscores, starting with a letter",
        ));
    }
    let orientation_word = text(values, "orientation", "current");
    let (from, up) = match orientation_word {
        "current" => camera(values).ok_or_else(|| {
            model("the current view comes from the canvas; outside it, choose Front, Top, Right or another orientation")
        })?,
        word => orientation(word)
            .ok_or_else(|| model(format!("there is no `{word}` orientation")))?,
    };
    let picked: Vec<String> = picks_of(values, "annotations")
        .iter()
        .map(|p| p.strip_prefix(ANNOTATION).unwrap_or(p).to_owned())
        .collect();
    let mut filed = 0;
    let change = edit_part(text_, path, |doc| {
        if doc.pmi.views.iter().any(|v| v.name == name) {
            return Err(model(format!(
                "the part has a view `{name}` already; pick another name"
            )));
        }
        for id in &picked {
            if !doc.pmi.annotations.iter().any(|a| &a.id == id) {
                return Err(model(format!("the part has no annotation `{id}`")));
            }
        }
        doc.pmi.views.push(ModelView {
            name: name.clone(),
            from,
            up,
            default: false,
        });
        for a in doc.pmi.annotations.iter_mut() {
            let take = if picked.is_empty() {
                a.view.is_none()
            } else {
                picked.contains(&a.id)
            };
            if take {
                a.view = Some(name.clone());
                filed += 1;
            }
        }
        Ok(())
    })?;
    Ok((
        change,
        format!("Added view {name} with {filed} annotations"),
    ))
}

/// Deletes view `name`; its annotations stay, in no view.
pub fn remove(pmi: &mut PmiSet, name: &str) -> Result<(), EditFailure> {
    let before = pmi.views.len();
    pmi.views.retain(|v| v.name != name);
    if pmi.views.len() == before {
        return Err(model(format!("the part has no annotation view `{name}`")));
    }
    for a in pmi.annotations.iter_mut() {
        if a.view.as_deref() == Some(name) {
            a.view = None;
        }
    }
    Ok(())
}

/// Properties on a selected view: its `orientation` (a standard one, or `current` with the
/// canvas camera) or `default` (the view the part opens on).
pub fn set_property(path: &str, text_: &str, name: &str, values: &Values) -> Edit {
    let change = edit_part(text_, path, |doc| {
        if !doc.pmi.views.iter().any(|v| v.name == name) {
            return Err(model(format!("the part has no annotation view `{name}`")));
        }
        if let Some(on) = values.get("default").and_then(Value::as_bool) {
            for v in doc.pmi.views.iter_mut() {
                v.default = on && v.name == name;
            }
            return Ok(());
        }
        let word = text(values, "orientation", "");
        let (from, up) = if word == "current" {
            camera(values).ok_or_else(|| model("the current view comes from the canvas"))?
        } else {
            orientation(word).ok_or_else(|| model(format!("there is no `{word}` orientation")))?
        };
        let v = doc
            .pmi
            .views
            .iter_mut()
            .find(|v| v.name == name)
            .expect("checked");
        v.from = from;
        v.up = up;
        Ok(())
    })?;
    Ok((change, format!("Set view {name}")))
}
