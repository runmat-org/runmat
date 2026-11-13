//! The Mold tab's Core/cavity as `mold.core_cavity(part, parting, margin)`: a block
//! `margin` round the picked part, less the part, split by the parting plane into the core
//! (below the plane) and the cavity (above it). A face picked for the part stands for its body.

use crate::features::{picks_of, value};
use crate::picks::Picks;
use crate::surface_edits::body_of;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{json, Map, Value};

type Values = Map<String, Value>;

fn one(values: &Values, id: &str, what: &str) -> Result<String, EditFailure> {
    picks_of(values, id)
        .into_iter()
        .next()
        .ok_or_else(|| crate::edit::model(format!("pick {what}")))
}

/// The `mold.*` feature a Mold command adds, with the stem its name takes; `None` for a command
/// this module doesn't lower.
pub fn mold_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    Ok(match command {
        "mold.core-cavity" => Some((
            "tool",
            json!({
                "feature": "mold",
                "mold": "core_cavity",
                "part": picks.reference(&body_of(&one(values, "part", "the molded part")?))?,
                "parting": picks.plane(&one(values, "parting", "the parting plane")?)?,
                "margin": value(values, "margin", "mm")?,
            }),
        )),
        // Shrink: the part grows by the factor (1.005 is 0.5 %), the same on every axis or per axis.
        "mold.scale" => {
            let uniform = values
                .get("uniform")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let [x, y, z] = if uniform {
                let f = allowance(values, "factor")?;
                [f.clone(), f.clone(), f]
            } else {
                [
                    allowance(values, "fx")?,
                    allowance(values, "fy")?,
                    allowance(values, "fz")?,
                ]
            };
            Some((
                "shrink",
                json!({ "feature": "mold", "mold": "shrink", "bodies": every_body(picks)?, "x": x, "y": y, "z": z }),
            ))
        }
        // Scale body: the picked bodies by one factor about the origin.
        "part.body-scale" => {
            let bodies: Vec<String> = picks_of(values, "bodies")
                .iter()
                .map(|p| body_of(p))
                .collect();
            if bodies.is_empty() {
                return Err(crate::edit::model("pick the bodies to scale"));
            }
            let f = allowance(values, "factor")?;
            Some((
                "scale",
                json!({ "feature": "mold", "mold": "shrink", "bodies": picks.references(&bodies)?, "x": f.clone(), "y": f.clone(), "z": f }),
            ))
        }
        // Parting line: found by draft analysis of the part for the pull direction.
        "mold.parting-line" => Some((
            "parting",
            json!({
                "feature": "mold",
                "mold": "parting_line",
                "body": first_body(picks)?,
                "pull": picks.axis(&one(values, "direction", "the pull direction")?)?,
                "min_draft": value(values, "draft", "deg")?,
            }),
        )),
        // Undercuts: the part's faces that can't release along the pull.
        "mold.undercut" => Some((
            "undercuts",
            json!({
                "feature": "mold",
                "mold": "undercuts",
                "body": first_body(picks)?,
                "pull": picks.axis(&one(values, "direction", "the pull direction")?)?,
            }),
        )),
        _ => None,
    })
}

/// A scale factor (1.005) as the allowance the model keeps (0.005).
fn allowance(values: &Values, id: &str) -> Result<Value, EditFailure> {
    let f = match values.get(id) {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        Some(Value::Object(o)) => o.get("value").and_then(Value::as_f64),
        _ => None,
    }
    .ok_or_else(|| crate::edit::model(format!("`{id}` needs a factor, such as 1.005")))?;
    if f <= 0.0 {
        return Err(crate::edit::model(format!("`{id}` must be above 0")));
    }
    Ok(json!({ "form": "number", "value": ((f - 1.0) * 1e12).round() / 1e12 }))
}

fn every_body(picks: &Picks) -> Result<Vec<Value>, EditFailure> {
    let all: Vec<Value> = picks
        .bodies()
        .iter()
        .filter_map(|b| b["reference"].as_str())
        .filter_map(|r| crate::picks::lower_text(r).ok())
        .collect();
    if all.is_empty() {
        return Err(crate::edit::model("the part has no body to scale yet"));
    }
    Ok(all)
}

fn first_body(picks: &Picks) -> Result<Value, EditFailure> {
    picks
        .bodies()
        .iter()
        .filter_map(|b| b["reference"].as_str())
        .find_map(|r| crate::picks::lower_text(r).ok())
        .ok_or_else(|| crate::edit::model("the part has no body yet"))
}

#[cfg(test)]
mod tests {
    use crate::edit::edit_json;
    use crate::runtime::{compile_json, Files};
    use serde_json::{json, Value};

    const BOSS: &str = "runmat 0.7\nstd 1\n\npart boss {\n    sketch s1 on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 10 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 8 mm)\n    feature mid = plane.offset(base = origin.xy, distance = 4 mm)\n}\n";

    #[test]
    fn core_cavity_takes_the_part_and_parting_plane_from_the_canvas() {
        let request = json!({ "command": "mold.core-cavity", "values": { "part": ["feature.body.face.end"], "parting": ["feature:mid"], "margin": 6 } });
        let out: Value = serde_json::from_str(&edit_json(
            "/p/boss.part",
            BOSS,
            &Files::new(),
            &request.to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{out}");
        let t = out["files"][0]["text"].as_str().unwrap();
        assert!(t.contains("feature tool1 = mold.core_cavity(part = feature.body.body, parting = plane.mid, margin = 6 mm)"), "{t}");
        let compiled: Value =
            serde_json::from_str(&compile_json("/p/boss.part", t, &Files::new())).unwrap();
        let errors: Vec<&Value> = compiled["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| d["severity"] == "error")
            .collect();
        assert!(errors.is_empty(), "{errors:?}\n{t}");
        assert!(
            compiled["display"]["data"]["bodies"]
                .as_array()
                .unwrap()
                .len()
                >= 3,
            "the part, the core and the cavity"
        );
    }

    /// Runs `command` on the boss and compiles the result: the new line, and no error.
    fn added(command: &str, values: Value) -> String {
        let request = json!({ "command": command, "values": values });
        let out: Value = serde_json::from_str(&edit_json(
            "/p/boss.part",
            BOSS,
            &Files::new(),
            &request.to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{command}: {out}");
        let t = out["files"][0]["text"].as_str().unwrap().to_owned();
        let compiled: Value =
            serde_json::from_str(&compile_json("/p/boss.part", &t, &Files::new())).unwrap();
        let errors: Vec<&Value> = compiled["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| d["severity"] == "error")
            .collect();
        assert!(errors.is_empty(), "{command}: {errors:?}\n{t}");
        t
    }

    #[test]
    fn scale_grows_every_body_by_the_shrink_factor() {
        let t = added("mold.scale", json!({ "factor": 1.005, "uniform": true }));
        assert!(t.contains("feature shrink1 = mold.shrink(bodies = [feature.body.body], x = 0.005, y = 0.005, z = 0.005)"), "{t}");
        let t = added(
            "mold.scale",
            json!({ "uniform": false, "fx": 1.01, "fy": 1.02, "fz": 1.03 }),
        );
        assert!(t.contains("mold.shrink(") && t.contains("z = 0.03"), "{t}");
    }

    #[test]
    fn scale_body_scales_the_picked_body_about_the_origin() {
        let t = added(
            "part.body-scale",
            json!({ "bodies": ["feature.body.face.end"], "factor": 2 }),
        );
        assert!(
            t.contains(
                "feature scale1 = mold.shrink(bodies = [feature.body.body], x = 1, y = 1, z = 1)"
            ),
            "{t}"
        );
    }

    #[test]
    fn parting_line_and_undercuts_take_the_pull_direction() {
        let t = added(
            "mold.parting-line",
            json!({ "direction": ["origin:z"], "draft": 1 }),
        );
        assert!(t.contains("feature parting1 = mold.parting_line(body = feature.body.body, pull = origin.z, min_draft = 1 deg)"), "{t}");
        let t = added("mold.undercut", json!({ "direction": ["origin:z"] }));
        assert!(
            t.contains(
                "feature undercuts1 = mold.undercuts(body = feature.body.body, pull = origin.z)"
            ),
            "{t}"
        );
    }

    #[test]
    fn an_empty_form_says_what_to_pick() {
        for (command, want) in [
            ("mold.parting-line", "pull direction"),
            ("part.body-scale", "bodies"),
        ] {
            let request = json!({ "command": command, "values": {} });
            let out: Value = serde_json::from_str(&edit_json(
                "/p/boss.part",
                BOSS,
                &Files::new(),
                &request.to_string(),
            ))
            .unwrap();
            assert_eq!(out["ok"], false);
            assert!(
                out["message"].as_str().unwrap_or("").contains(want),
                "{command}: {out}"
            );
        }
    }
}
