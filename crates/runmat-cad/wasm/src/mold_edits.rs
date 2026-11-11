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
        _ => None,
    })
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
}
