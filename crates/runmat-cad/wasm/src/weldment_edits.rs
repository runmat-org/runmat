//! The Weldments tab's Structural member command as a `weldment.structural_member` feature:
//! the form's type and size text (`40x40x3`) become the profile's sizes, the picked segments
//! (sketch lines, or a whole sketch for every line in it) one member each, grouped by sketch.
//!
//! Sizes read in millimetres, or in inches for the ANSI standard, in the order the profile
//! tables list them:
//!
//! | Type | Size |
//! |---|---|
//! | Rectangular tube | width × height × wall |
//! | Square tube | side × wall |
//! | Pipe | outside diameter × wall |
//! | Angle | leg × leg × thickness, or leg × thickness for an equal angle |
//! | Channel, I-beam | depth × flange width × web × flange thickness |
//! | Flat bar | width × thickness |

use crate::edit::model;
use crate::features::{picks_of, value};
use crate::picks::Picks;
use runmat_cad::model::{FeatureKind, SketchGeometry};
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{json, Map, Value};

type Values = Map<String, Value>;

/// `40x40x3`, `40 × 40 × 3 mm`, `2 x 0.25 in` → the numbers.
pub fn size_numbers(size: &str) -> Result<Vec<f64>, String> {
    let cleaned = size
        .trim()
        .trim_end_matches("mm")
        .trim_end_matches("in")
        .trim();
    let parts: Vec<&str> = cleaned.split(['x', 'X', '×', '*']).map(str::trim).collect();
    parts
        .iter()
        .map(|p| p.parse::<f64>().ok().filter(|v| *v > 0.0))
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| {
            format!("`{size}` isn't a size; write the dimensions separated by x, such as 40x40x3")
        })
}

/// The profile's model JSON from its type and size.
fn profile(kind: &str, size: &str, unit: &str) -> Result<Value, EditFailure> {
    let n = size_numbers(size).map_err(model)?;
    let len = |v: f64| json!({ "form": "quantity", "value": v, "unit": unit });
    let want = |count: &[usize], form: &str| -> Result<(), EditFailure> {
        if count.contains(&n.len()) {
            Ok(())
        } else {
            Err(model(format!(
                "a {} size is {form}; `{size}` has {} number{}",
                kind.replace('_', " "),
                n.len(),
                if n.len() == 1 { "" } else { "s" }
            )))
        }
    };
    Ok(match kind {
        "rect_tube" => {
            want(&[3], "width x height x wall")?;
            json!({ "profile": "rect_tube", "width": len(n[0]), "height": len(n[1]), "wall": len(n[2]) })
        }
        "square_tube" => {
            want(&[2, 3], "side x wall")?;
            let wall = n[n.len() - 1];
            json!({ "profile": "rect_tube", "width": len(n[0]), "height": len(n[if n.len() == 3 { 1 } else { 0 }]), "wall": len(wall) })
        }
        "pipe" => {
            want(&[2], "outside diameter x wall")?;
            json!({ "profile": "pipe", "diameter": len(n[0]), "wall": len(n[1]) })
        }
        "angle" => {
            want(&[2, 3], "leg x leg x thickness")?;
            let (a, b) = if n.len() == 3 {
                (n[0], n[1])
            } else {
                (n[0], n[0])
            };
            json!({ "profile": "angle", "a": len(a), "b": len(b), "thickness": len(n[n.len() - 1]) })
        }
        "channel" | "i_beam" => {
            want(&[4], "depth x flange x web x flange thickness")?;
            json!({ "profile": kind, "depth": len(n[0]), "flange": len(n[1]), "web": len(n[2]), "flange_thickness": len(n[3]) })
        }
        "flat_bar" => {
            want(&[2], "width x thickness")?;
            json!({ "profile": "flat_bar", "width": len(n[0]), "thickness": len(n[1]) })
        }
        "round_bar" => {
            want(&[1], "a diameter")?;
            json!({ "profile": "round_bar", "diameter": len(n[0]) })
        }
        other => return Err(model(format!("`{other}` isn't a profile type"))),
    })
}

/// The picked segments by sketch, in pick order: a line pick (`sketch.layout.bottom`) adds its
/// line; a sketch pick adds every line in it that isn't construction.
fn segments(values: &Values, picks: &Picks) -> Result<Vec<(String, Vec<String>)>, EditFailure> {
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    let mut add =
        |sketch: String, entity: String| match groups.iter_mut().find(|(s, _)| *s == sketch) {
            Some((_, list)) if !list.contains(&entity) => list.push(entity),
            Some(_) => {}
            None => groups.push((sketch, vec![entity])),
        };
    let all = picks_of(values, "segments");
    if all.is_empty() {
        return Err(model("pick the path segments: sketch lines, or a sketch"));
    }
    for pick in all {
        if let Ok((sketch, entity)) = picks.sketch_entity(&pick) {
            add(sketch, entity);
            continue;
        }
        let sketch = picks.sketch(&pick)?;
        let lines: Vec<String> = picks
            .document
            .features
            .iter()
            .find(|f| f.id.0 == sketch)
            .and_then(|f| match &f.kind {
                FeatureKind::Sketch(s) => Some(
                    s.entities
                        .iter()
                        .filter(|e| {
                            !e.construction && matches!(e.geometry, SketchGeometry::Line { .. })
                        })
                        .map(|e| e.id.0.clone())
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        if lines.is_empty() {
            return Err(model(format!(
                "sketch `{sketch}` has no lines to place members on"
            )));
        }
        for line in lines {
            add(sketch.clone(), line);
        }
    }
    Ok(groups)
}

fn one(values: &Values, id: &str, what: &str) -> Result<String, EditFailure> {
    picks_of(values, id)
        .into_iter()
        .next()
        .ok_or_else(|| model(format!("pick {what}")))
}

/// End cap, gusset, weld bead and trim/extend: the joints placed on members already in the part.
fn joint_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    let kind = match command {
        "weldment.end-cap" => (
            "cap",
            json!({
                "feature": "weldment", "weldment": "end_cap",
                "end_face": picks.reference(&one(values, "face", "the member's end face")?)?,
                "thickness": value(values, "thickness", "mm")?,
                "inset": if values.contains_key("inset") { value(values, "inset", "mm")? } else { Value::Null },
            }),
        ),
        "weldment.gusset" => {
            let faces = picks_of(values, "faces");
            if faces.len() != 2 {
                return Err(model(format!(
                    "a gusset joins two faces that meet at a corner; {} picked",
                    faces.len()
                )));
            }
            (
                "gusset",
                json!({
                    "feature": "weldment", "weldment": "gusset",
                    "faces": picks.references(&faces)?,
                    "leg": value(values, "leg", "mm")?,
                    "thickness": value(values, "thickness", "mm")?,
                }),
            )
        }
        "weldment.weld-bead" => {
            let edges = picks_of(values, "edges");
            if edges.is_empty() {
                return Err(model("pick the edges the weld runs along"));
            }
            (
                "weld",
                json!({
                    "feature": "weldment", "weldment": "weld_bead",
                    "edges": picks.references(&edges)?,
                    "kind": { "weld": "fillet", "leg": value(values, "leg", "mm")? },
                }),
            )
        }
        "weldment.trim-extend" => {
            // A face picked on a member stands for its body (each member is its own body).
            let pick = one(values, "member", "the member to trim or extend")?;
            let member = match picks.body_of_face(&pick) {
                Some(body) => body,
                None => picks.reference(&crate::surface_edits::body_of(&pick))?,
            };
            (
                "trim",
                json!({
                    "feature": "weldment", "weldment": "trim_extend",
                    "members": [member],
                    "to": picks.reference(&one(values, "to", "the face or plane to trim to")?)?,
                }),
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(kind))
}

/// A Weldments command as its feature's model, or `None` for other commands.
pub fn weldment_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    if command != "weldment.structural-member" {
        return joint_kind(command, values, picks);
    }
    if values
        .get("mirror_profile")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(model(
            "a mirrored profile isn't in the member model yet; rotate the profile 180° instead",
        ));
    }
    let unit = if values.get("standard").and_then(Value::as_str) == Some("ansi") {
        "in"
    } else {
        "mm"
    };
    let kind = values
        .get("profile")
        .and_then(Value::as_str)
        .unwrap_or("rect_tube");
    let size = values
        .get("size")
        .and_then(Value::as_str)
        .unwrap_or("40x40x3");
    let profile = profile(kind, size, unit)?;
    let corner = values
        .get("corner")
        .and_then(Value::as_str)
        .unwrap_or("miter");
    let rotation = if values.contains_key("rotation") {
        value(values, "rotation", "deg")?
    } else {
        json!({ "form": "quantity", "value": 0.0, "unit": "deg" })
    };
    // The grade the cut list prints and weighs each member by.
    let material = values
        .get("material")
        .and_then(Value::as_str)
        .unwrap_or("s235");
    let groups: Vec<Value> = segments(values, picks)?
        .into_iter()
        .map(|(sketch, segments)| json!({ "sketch": sketch, "segments": segments, "profile": profile, "material": material, "corner": corner, "rotation": rotation }))
        .collect();
    Ok(Some((
        "members",
        json!({ "feature": "weldment", "weldment": "structural_member", "groups": groups }),
    )))
}

#[cfg(test)]
mod tests {
    use crate::runtime::Files;
    use serde_json::{json, Value};

    const FRAME: &str = "runmat 0.7\nstd 1\n\npart frame {\n    sketch layout on origin.xy {\n        bottom = line(start = [0 mm, 0 mm], end = [200 mm, 0 mm])\n        right = line(start = [200 mm, 0 mm], end = [200 mm, 100 mm])\n        top = line(start = [200 mm, 100 mm], end = [0 mm, 100 mm])\n        left = line(start = [0 mm, 100 mm], end = [0 mm, 0 mm])\n    }\n}\n";

    fn run(values: Value) -> Value {
        let request =
            json!({ "command": "weldment.structural-member", "values": values }).to_string();
        serde_json::from_str(&crate::edit::edit_json(
            "/p/frame.part",
            FRAME,
            &Files::new(),
            &request,
        ))
        .unwrap()
    }

    #[test]
    fn the_size_text_builds_the_profile_and_each_segment_is_a_member() {
        let out = run(
            json!({ "profile": "rect_tube", "size": "20x30x2", "segments": ["sketch.layout.bottom", "sketch.layout.top"] }),
        );
        assert_eq!(out["ok"], true, "{out}");
        let text = out["files"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("rect_tube(width = 20 mm, height = 30 mm, wall = 2 mm)"),
            "{text}"
        );
        assert!(text.contains("segments = [bottom, top]"), "{text}");
        // The grade defaults to structural steel; the cut list weighs the members by it.
        assert!(text.contains("material = s235"), "{text}");
        let out = run(
            json!({ "profile": "square_tube", "size": "20x20x1.5", "material": "al_6061", "segments": ["sketch:layout"] }),
        );
        let text = out["files"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("{out}"));
        assert!(text.contains("material = al_6061"), "{text}");
        // A sketch pick takes every line; an angle with two numbers is an equal angle, in inches for ANSI.
        let out = run(
            json!({ "profile": "angle", "standard": "ansi", "size": "2 x 0.25", "segments": ["sketch:layout"] }),
        );
        let text = out["files"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("{out}"));
        assert!(
            text.contains("angle(a = 2 in, b = 2 in, thickness = 0.25 in)"),
            "{text}"
        );
        assert!(
            text.contains("segments = [bottom, right, top, left]"),
            "{text}"
        );
        // A size with the wrong count says what the type takes.
        let out = run(
            json!({ "profile": "channel", "size": "40x20", "segments": ["sketch.layout.bottom"] }),
        );
        assert_eq!(out["ok"], false);
        assert!(
            out["message"]
                .as_str()
                .unwrap()
                .contains("depth x flange x web x flange thickness"),
            "{out}"
        );
    }

    /// End cap, gusset, weld bead and trim/extend write their features from the canvas's picks.
    #[test]
    fn the_joint_commands_write_their_features() {
        let edit = |command: &str, values: Value| -> String {
            let request = json!({ "command": command, "values": values }).to_string();
            let out: Value = serde_json::from_str(&crate::edit::edit_json(
                "/p/frame.part",
                FRAME,
                &Files::new(),
                &request,
            ))
            .unwrap();
            assert_eq!(out["ok"], true, "{command}: {out}");
            out["files"][0]["text"].as_str().unwrap().to_owned()
        };
        let t = edit(
            "weldment.end-cap",
            json!({ "face": ["feature.members.face.end"], "thickness": 3, "inset": 1 }),
        );
        assert!(
            t.contains("feature cap1 = weldment.end_cap(")
                && t.contains("thickness = 3 mm")
                && t.contains("inset = 1 mm"),
            "{t}"
        );
        let t = edit(
            "weldment.weld-bead",
            json!({ "edges": ["edges_between(feature.members.face.end, feature.members.face.start)"], "leg": 4 }),
        );
        assert!(
            t.contains("feature weld1 = weldment.weld_bead(") && t.contains("leg = 4 mm"),
            "{t}"
        );
        let request = json!({ "command": "weldment.gusset", "values": { "faces": ["feature.members.face.end"], "leg": 50, "thickness": 5 } }).to_string();
        let out: Value = serde_json::from_str(&crate::edit::edit_json(
            "/p/frame.part",
            FRAME,
            &Files::new(),
            &request,
        ))
        .unwrap();
        assert_eq!(out["ok"], false, "a gusset needs two faces: {out}");
    }
}
