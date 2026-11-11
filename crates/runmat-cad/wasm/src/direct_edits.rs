//! The Direct tab's commands as features (CAD-20): each edit is written to the part's text as a
//! `direct.*` feature whose faces are referenced by their canonical names, so the edit is
//! history like any other feature and regenerates with the part.
//!
//! - `direct.move-face`: along the faces' normals, by a translation along an origin axis, or
//!   by a rotation about an axis or edge → `direct.move_faces(faces, motion)`;
//! - `direct.offset-face`, `direct.pull` → `direct.move_faces(faces, motion = offset(distance))`;
//! - `direct.replace-face` → `direct.replace_faces(faces, with)`;
//! - `direct.delete-face` → `direct.delete_faces(faces, heal)`;
//! - `direct.move-body` → `direct.move_bodies(bodies, motion)`.

use crate::edit::model;
use crate::features::{picks_of, value};
use crate::picks::Picks;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{json, Map, Value};

type Values = Map<String, Value>;

fn some(values: &Values, id: &str, what: &str) -> Result<Vec<String>, EditFailure> {
    let all = picks_of(values, id);
    if all.is_empty() {
        Err(model(format!("pick {what}")))
    } else {
        Ok(all)
    }
}

fn one(values: &Values, id: &str, what: &str) -> Result<String, EditFailure> {
    Ok(some(values, id, what)?.remove(0))
}

fn text<'a>(values: &'a Values, id: &str, default: &'a str) -> &'a str {
    values.get(id).and_then(Value::as_str).unwrap_or(default)
}

fn zero() -> Value {
    json!({ "form": "quantity", "value": 0, "unit": "mm" })
}

/// A translation by `distance` along the origin axis `axis` lowers to (dx, dy, dz).
fn along_origin_axis(axis: &Value, distance: Value) -> Option<Value> {
    let name = axis.get("datum")?.get("axis")?.as_str()?;
    let at = ["x", "y", "z"].iter().position(|a| *a == name)?;
    let mut d = [zero(), zero(), zero()];
    d[at] = distance;
    let [dx, dy, dz] = d;
    Some(json!({ "motion": "translate", "dx": dx, "dy": dy, "dz": dz }))
}

/// The `direct.*` feature a Direct command adds, with the stem its name takes; `None` for a
/// command that isn't a direct edit.
pub fn direct_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    let faces = || picks.references(&some(values, "faces", "the faces")?);
    let offset = || -> Result<Value, EditFailure> {
        Ok(json!({ "motion": "offset", "distance": value(values, "distance", "mm")? }))
    };
    let kind = match command {
        "direct.move-face" => {
            let motion = match text(values, "motion", "normal") {
                "rotate" => json!({
                    "motion": "rotate",
                    "axis": picks.axis(&one(values, "axis", "the axis to rotate about")?)?,
                    "angle": value(values, "angle", "deg")?,
                }),
                "translate" => {
                    let axis = picks.axis(&one(values, "direction", "the direction")?)?;
                    along_origin_axis(&axis, value(values, "distance", "mm")?).ok_or_else(|| {
                        model("translate faces along an origin axis (X, Y or Z), or move them along their normals")
                    })?
                }
                _ => offset()?,
            };
            (
                "move",
                json!({ "feature": "direct", "direct": "move_faces", "faces": faces()?, "motion": motion }),
            )
        }
        "direct.offset-face" | "direct.pull" => (
            if command == "direct.pull" {
                "pull"
            } else {
                "offset"
            },
            json!({ "feature": "direct", "direct": "move_faces", "faces": faces()?, "motion": offset()? }),
        ),
        "direct.replace-face" => (
            "replace",
            json!({
                "feature": "direct",
                "direct": "replace_faces",
                "faces": picks.references(&some(values, "faces", "the faces to replace")?)?,
                "with": picks.reference(&one(values, "with", "the replacement face")?)?,
            }),
        ),
        "direct.delete-face" => {
            let heal = match text(values, "heal", "heal") {
                "leave" => false,
                "patch" => {
                    return Err(model("patching the opening isn't in the part model yet; heal it from the neighbours or leave it open"))
                }
                _ => true,
            };
            (
                "delete_face",
                json!({ "feature": "direct", "direct": "delete_faces", "faces": faces()?, "heal": heal }),
            )
        }
        "direct.move-body" => {
            let bodies = picks.references(&some(values, "bodies", "the bodies to move")?)?;
            let turned = values
                .get("angle")
                .and_then(Value::as_f64)
                .is_some_and(|a| a != 0.0);
            let motion = if turned {
                json!({
                    "motion": "rotate",
                    "axis": picks.axis(&one(values, "axis", "the axis to rotate about")?)?,
                    "angle": value(values, "angle", "deg")?,
                })
            } else {
                json!({ "motion": "translate", "dx": value(values, "dx", "mm")?, "dy": value(values, "dy", "mm")?, "dz": value(values, "dz", "mm")? })
            };
            (
                "move_body",
                json!({ "feature": "direct", "direct": "move_bodies", "bodies": bodies, "motion": motion }),
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(kind))
}

#[cfg(test)]
mod tests {
    use crate::runtime::Files;
    use serde_json::{json, Value};

    /// A 40 x 20 x 10 mm plate with a 4 mm hole through it.
    const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature h = hole(face = feature.body.face.end, at = { a = point(30 mm, 10 mm, 10 mm) }, diameter = 4 mm, through_all)\n}\n";
    const END: &str = "feature.body.face.end";

    /// The part text after the command, and the compile's diagnostics.
    fn added(command: &str, values: Value) -> (String, Vec<String>) {
        let request = json!({ "command": command, "values": values }).to_string();
        let out: Value = serde_json::from_str(&crate::edit::edit_json(
            "/p/plate.part",
            PLATE,
            &Files::new(),
            &request,
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{command}: {out}");
        let text = out["files"][0]["text"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| panic!("{out}"));
        let compiled: Value = serde_json::from_str(&crate::runtime::compile_json(
            "/p/plate.part",
            &text,
            &Files::new(),
        ))
        .unwrap();
        let errors = compiled["diagnostics"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|d| d["severity"] == "error")
            .map(|d| d["message"].as_str().unwrap_or("").to_owned())
            .collect();
        (text, errors)
    }

    #[test]
    fn offset_and_move_face_are_written_as_features_by_canonical_name() {
        // The plate's right side, clear of the hole, 2 mm out.
        let (t, errors) = added(
            "direct.offset-face",
            json!({ "faces": ["feature.body.face.side(s1.l2)"], "distance": 2 }),
        );
        assert!(t.contains("feature offset1 = direct.move_faces("), "{t}");
        assert!(t.contains("faces = [feature.body.face.side(s1.l2)]"), "{t}");
        assert!(t.contains("motion = offset(distance = 2 mm)"), "{t}");
        assert!(errors.is_empty(), "{errors:?}\n{t}");
        let (t, _) = added(
            "direct.move-face",
            json!({ "faces": [END], "motion": "translate", "direction": ["origin.z"], "distance": 3 }),
        );
        assert!(
            t.contains("motion = translate(dx = 0 mm, dy = 0 mm, dz = 3 mm)"),
            "{t}"
        );
    }

    #[test]
    fn delete_and_replace_face_name_their_faces() {
        // Deleting the hole's wall heals the plate over it (a defeature).
        let (t, errors) = added(
            "direct.delete-face",
            json!({ "faces": ["feature.h.face.wall(a)"], "heal": "heal" }),
        );
        assert!(
            t.contains("feature delete_face1 = direct.delete_faces("),
            "{t}"
        );
        assert!(t.contains("faces = [feature.h.face.wall(a)]"), "{t}");
        assert!(errors.is_empty(), "{errors:?}\n{t}");
        let (t, _) = added(
            "direct.replace-face",
            json!({ "faces": [END], "with": ["feature.body.face.start"] }),
        );
        assert!(
            t.contains("feature replace1 = direct.replace_faces("),
            "{t}"
        );
        assert!(t.contains("with = feature.body.face.start"), "{t}");
    }

    #[test]
    fn edits_the_model_cant_make_say_why() {
        let request = json!({ "command": "direct.delete-face", "values": { "faces": [END], "heal": "patch" } }).to_string();
        let out: Value = serde_json::from_str(&crate::edit::edit_json(
            "/p/plate.part",
            PLATE,
            &Files::new(),
            &request,
        ))
        .unwrap();
        assert_eq!(out["ok"], false, "{out}");
        assert!(
            out["message"].as_str().unwrap().contains("heal it"),
            "{out}"
        );
    }
}
