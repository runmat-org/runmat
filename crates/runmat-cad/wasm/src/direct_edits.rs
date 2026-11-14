//! The Direct tab's commands as features (CAD-20): each edit is written to the part's text as a
//! `direct.*` feature whose faces are referenced by their canonical names, so the edit is
//! history like any other feature and regenerates with the part.
//!
//! - `direct.move-face`: each face along its own normal (offset), the faces as one rigid set
//!   along a direction (an edge or axis along itself, a plane or planar face along its normal),
//!   or about an axis → `direct.move_faces(faces, motion)`;
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

fn flag(values: &Values, id: &str) -> bool {
    values.get(id).and_then(Value::as_bool).unwrap_or(false)
}

/// `-v`: a literal with its sign flipped, else the negated expression.
fn negate(v: Value) -> Value {
    match v.get("value").and_then(Value::as_f64) {
        Some(n) if v["form"] == "quantity" || v["form"] == "number" => {
            let mut out = v.clone();
            out["value"] = json!(-n);
            out
        }
        _ => json!({ "form": "neg", "operand": v }),
    }
}

/// Move face's live relation: Fixed keeps each picked face where it is; a pair relation keeps
/// two picked faces to each other, or one picked face to the first moved face.
fn keep(values: &Values, picks: &Picks, moved: &[Value]) -> Result<Value, EditFailure> {
    let relation = text(values, "keep", "none");
    if relation == "none" {
        return Ok(json!([]));
    }
    let faces = picks.references(&some(values, "keep_faces", "the faces to keep")?)?;
    Ok(match relation {
        "fixed" => json!(faces.into_iter().map(|face| json!({ "keep": "fixed", "face": face })).collect::<Vec<_>>()),
        "parallel" | "perpendicular" | "coplanar" | "concentric" | "tangent" => {
            let (a, b) = match faces.as_slice() {
                [b] => (moved.first().cloned().unwrap_or(Value::Null), b.clone()),
                [a, b] => (a.clone(), b.clone()),
                _ => {
                    return Err(model(format!(
                        "keep {relation} takes one face (kept to the first moved face) or two faces"
                    )))
                }
            };
            json!([{ "keep": relation, "a": a, "b": b }])
        }
        other => return Err(model(format!("there is no `{other}` relation to keep"))),
    })
}

/// The `direct.*` feature a Direct command adds, with the stem its name takes; `None` for a
/// command that isn't a direct edit.
pub fn direct_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    // Direct edits move faces: an edge or vertex pick is refused, naming it.
    let faces = || -> Result<Vec<Value>, EditFailure> {
        let refs = picks.references(&some(values, "faces", "the faces")?)?;
        for (pick, r) in picks_of(values, "faces").iter().zip(&refs) {
            if crate::picks::topo(r) != "face" {
                return Err(model(format!(
                    "`{pick}` isn't a face; pick the faces to move"
                )));
            }
        }
        Ok(refs)
    };
    let offset = || -> Result<Value, EditFailure> {
        Ok(json!({ "motion": "offset", "distance": value(values, "distance", "mm")? }))
    };
    let kind = match command {
        "direct.move-face" => {
            let motion = match text(values, "motion", "normal") {
                "rotate" => {
                    let angle = value(values, "angle", "deg")?;
                    json!({
                        "motion": "rotate",
                        "axis": picks.axis(&one(values, "axis", "the axis to rotate about")?)?,
                        "angle": if flag(values, "reverse") { negate(angle) } else { angle },
                    })
                }
                // A plane or planar face moves the faces along its normal; an edge, axis or
                // sketch line along itself.
                "translate" => json!({
                    "motion": "along",
                    "direction": picks.direction(&one(values, "direction", "the direction to move along")?)?,
                    "distance": value(values, "distance", "mm")?,
                    "reverse": flag(values, "reverse"),
                }),
                _ => offset()?,
            };
            let faces = faces()?;
            let keep = keep(values, picks, &faces)?;
            (
                "move",
                json!({ "feature": "direct", "direct": "move_faces", "faces": faces, "motion": motion, "keep": keep }),
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

    /// The same plate without the hole: faces move on it in the test kernel.
    fn plain() -> String {
        PLATE
            .lines()
            .filter(|l| !l.contains("feature h ="))
            .map(|l| format!("{l}\n"))
            .collect()
    }

    /// The part text after the command, and the compile's diagnostics.
    fn added(command: &str, values: Value) -> (String, Vec<String>) {
        added_to(PLATE, command, values)
    }

    /// [`added`] on the plain plate.
    fn moved(command: &str, values: Value) -> (String, Vec<String>) {
        added_to(&plain(), command, values)
    }

    fn added_to(base: &str, command: &str, values: Value) -> (String, Vec<String>) {
        let request = json!({ "command": command, "values": values }).to_string();
        let out: Value = serde_json::from_str(&crate::edit::edit_json(
            "/p/plate.part",
            base,
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
        let (t, errors) = moved(
            "direct.move-face",
            json!({ "faces": [END], "motion": "translate", "direction": ["origin.z"], "distance": 3 }),
        );
        assert!(
            t.contains("motion = along(direction = origin.z, distance = 3 mm)"),
            "{t}"
        );
        assert!(errors.is_empty(), "{errors:?}\n{t}");
    }

    /// The plate's top (z) after a Move face on its end face.
    fn top_after(values: Value) -> f64 {
        let (t, errors) = moved("direct.move-face", values);
        assert!(errors.is_empty(), "{errors:?}\n{t}");
        let compiled: Value = serde_json::from_str(&crate::runtime::compile_json(
            "/p/plate.part",
            &t,
            &Files::new(),
        ))
        .unwrap();
        compiled["display"]["data"]["bounds"]["max"][2]
            .as_f64()
            .unwrap()
            * 1e3
    }

    /// Translate takes a direction from a plane (its normal), a planar face (its outward normal),
    /// an edge or an axis (along it), and Reverse flips it: the owner's "Direction: Top" case.
    #[test]
    fn move_face_translates_along_a_planes_normal_an_edge_or_an_axis() {
        let along = |direction: &str, reverse: bool| {
            json!({ "faces": [END], "motion": "translate", "direction": [direction], "distance": 3, "reverse": reverse })
        };
        // The Top plane, as the Outline names it, is the XY plane: its normal is +Z.
        let (t, _) = moved("direct.move-face", along("origin:top", false));
        assert!(
            t.contains("direction = { along = normal, plane = origin.xy }"),
            "{t}"
        );
        assert!((top_after(along("origin:top", false)) - 13.0).abs() < 1e-6);
        assert!((top_after(along("origin:top", true)) - 7.0).abs() < 1e-6);
        // The end face itself: its outward normal is +Z.
        assert!((top_after(along(END, false)) - 13.0).abs() < 1e-6);
        // A vertical edge of the plate, and the Z axis.
        let edge = "edges_between(feature.body.face.side(s1.l1), feature.body.face.side(s1.l2))";
        let z = top_after(along(edge, false));
        assert!((z - 13.0).abs() < 1e-6 || (z - 7.0).abs() < 1e-6, "{z}");
        assert!((top_after(along("origin:z", false)) - 13.0).abs() < 1e-6);
        // A vertex gives no direction, and says what does.
        let request = json!({ "command": "direct.move-face", "values": along("vertex_of(feature.body.face.end, feature.body.face.side(s1.l1), feature.body.face.side(s1.l2))", false) }).to_string();
        let out: Value = serde_json::from_str(&crate::edit::edit_json(
            "/p/plate.part",
            PLATE,
            &Files::new(),
            &request,
        ))
        .unwrap();
        assert_eq!(out["ok"], false, "{out}");
        assert!(
            out["message"].as_str().unwrap().contains("planar face"),
            "{out}"
        );
    }

    /// Live relations are Move face's Keep option: Fixed faces, or a face kept parallel to
    /// the moved face. An edit that breaks one fails, naming it.
    #[test]
    fn move_face_keeps_its_live_relations() {
        let (t, errors) = moved(
            "direct.move-face",
            json!({ "faces": [END], "distance": 2, "keep": "parallel", "keep_faces": ["feature.body.face.start"] }),
        );
        assert!(t.contains("keep = [{ keep = parallel, a = feature.body.face.end, b = feature.body.face.start }]"), "{t}");
        assert!(errors.is_empty(), "{errors:?}\n{t}");
        let (t, errors) = moved(
            "direct.move-face",
            json!({ "faces": [END], "distance": 2, "keep": "fixed", "keep_faces": [END] }),
        );
        assert!(t.contains("{ keep = fixed, face = feature.body.face.end }"), "{t}");
        assert!(!errors.is_empty(), "moving a Fixed face must fail\n{t}");
    }

    /// An upright face moved along the Top plane's normal would slide in its own plane: the
    /// regeneration says so and names what to pick, rather than failing in the kernel.
    #[test]
    fn a_face_moved_along_its_own_plane_says_which_way_to_pick() {
        let (t, errors) = moved(
            "direct.move-face",
            json!({ "faces": ["feature.body.face.side(s1.l2)"], "motion": "translate", "direction": ["origin:top"], "distance": 5 }),
        );
        assert!(errors.iter().any(|e| e.contains("own plane")), "{errors:?}\n{t}");
        // Across the face (the Right plane's normal, +X) it moves.
        let (t, errors) = moved(
            "direct.move-face",
            json!({ "faces": ["feature.body.face.side(s1.l2)"], "motion": "translate", "direction": ["origin:right"], "distance": 5 }),
        );
        assert!(errors.is_empty(), "{errors:?}\n{t}");
    }

    /// Offset is Move face's default motion, and Rotate's Reverse turns the other way.
    #[test]
    fn move_face_offsets_by_default_and_reverse_turns_a_rotation_back() {
        let (t, _) = moved(
            "direct.move-face",
            json!({ "faces": [END], "distance": 2 }),
        );
        assert!(t.contains("motion = offset(distance = 2 mm)"), "{t}");
        let (t, _) = moved(
            "direct.move-face",
            json!({ "faces": ["feature.body.face.side(s1.l2)"], "motion": "rotate", "axis": ["edges_between(feature.body.face.side(s1.l2), feature.body.face.start)"], "angle": 5, "reverse": true }),
        );
        assert!(t.contains("angle = -5 deg"), "{t}");
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
