//! Panel fields the ribbon audit found without an input (RIBBONAUDIT, 0.7.2): Hole's Up to
//! face, Fillet's face sets, Base flange's allowance and deduction, Draft's direction of pull,
//! and the Loft and Sweep options the part model doesn't keep yet, which are refused by name.

use crate::edit::edit_json;
use crate::runtime::{compile_json, Files};
use serde_json::{json, Value};

/// A 40 x 20 plate, 10 mm thick, and a second sketch on the YZ plane.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    sketch s2 on origin.yz {\n        c = circle(center = [10 mm, 5 mm], radius = 2 mm)\n    }\n}\n";

/// The same outline as a sketch only, for a base flange.
const SKETCH: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n}\n";

const END: &str = "feature.body.face.end";
const START: &str = "feature.body.face.start";
const SIDE1: &str = "feature.body.face.side(s1.l1)";
const SIDE2: &str = "feature.body.face.side(s1.l2)";
const SIDE3: &str = "feature.body.face.side(s1.l3)";

fn edit_in(text: &str, command: &str, values: Value) -> Value {
    let request = json!({ "command": command, "values": values });
    serde_json::from_str(&edit_json("/p/plate.part", text, &Files::new(), &request.to_string()))
        .unwrap()
}

fn written_in(text: &str, command: &str, values: Value) -> String {
    let out = edit_in(text, command, values);
    assert_eq!(out["ok"], true, "{command}: {out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

fn errors(text: &str) -> Vec<String> {
    let compiled: Value =
        serde_json::from_str(&compile_json("/p/plate.part", text, &Files::new())).unwrap();
    compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .map(|d| d["message"].as_str().unwrap_or("").to_owned())
        .collect()
}

fn refused(command: &str, values: Value) -> String {
    let out = edit_in(PLATE, command, values);
    assert_eq!(out["ok"], false, "{command}: {out}");
    out["message"].as_str().unwrap_or("").to_owned()
}

#[test]
fn a_hole_up_to_a_face_takes_the_face_from_its_own_field() {
    let t = written_in(
        PLATE,
        "part.hole",
        json!({ "type": "simple", "diameter": 4, "face": [END], "positions": ["point(30 mm, 10 mm, 10 mm)"], "end": "up_to_face", "up_to": [START] }),
    );
    assert!(t.contains("up_to_face(") && t.contains(START), "{t}");
    assert!(errors(&t).is_empty(), "{:?}\n{t}", errors(&t));
}

#[test]
fn face_fillets_and_full_rounds_take_face_sets() {
    let t = written_in(
        PLATE,
        "part.fillet",
        json!({ "type": "face", "face_set1": [END], "face_set2": [SIDE2], "radius": 2 }),
    );
    assert!(t.contains("face_fillet(") && t.contains(END) && t.contains(SIDE2), "{t}");
    let t = written_in(
        PLATE,
        "part.fillet",
        json!({ "type": "full_round", "side_set1": [SIDE1], "center_set": [END], "side_set2": [SIDE3] }),
    );
    assert!(t.contains("full_round(") && t.contains("center = ["), "{t}");
    // With neither the sets nor the combined pick, the refusal names the sets.
    assert!(refused("part.fillet", json!({ "type": "face", "radius": 2 })).contains("Face set 1"));
}

#[test]
fn base_flange_takes_a_bend_allowance_or_deduction_value() {
    for (rule, field) in [("allowance", "allowance"), ("deduction", "deduction")] {
        let t = written_in(
            SKETCH,
            "sheet-metal.base-flange",
            json!({ "profile": ["sketch:s1"], "thickness": 1.5, "radius": 1, "bend_rule": rule, field: 2.5 }),
        );
        assert!(t.contains(&format!("{rule}(value = 2.5 mm)")), "{t}");
        assert!(errors(&t).is_empty(), "{:?}\n{t}", errors(&t));
    }
}

#[test]
fn a_parting_line_draft_pulls_along_the_picked_planes_normal() {
    let t = written_in(
        PLATE,
        "part.draft",
        json!({ "type": "parting_line", "faces": [SIDE1], "parting": [format!("edges_between({SIDE1}, {START})")], "pull": ["origin:right"], "angle": 2 }),
    );
    assert!(t.contains("neutral = origin.yz"), "{t}");
}

#[test]
fn loft_and_sweep_options_the_model_does_not_keep_are_refused_by_name() {
    let profiles = json!(["sketch:s1", "sketch:s2"]);
    for values in [
        json!({ "profiles": profiles, "start_tangency": "direction", "start_direction": ["origin:top"] }),
        json!({ "profiles": profiles, "closed": true }),
        json!({ "profiles": profiles, "thin": true }),
    ] {
        assert!(refused("part.loft", values).contains("isn't in the part model yet"));
    }
    for values in [
        json!({ "profile": ["sketch:s2"], "path": ["sketch:s1"], "twist": "turns", "twist_turns": 1 }),
        json!({ "profile": ["sketch:s2"], "path": ["sketch:s1"], "guides": [SIDE1] }),
        json!({ "profile": ["sketch:s2"], "path": ["sketch:s1"], "end_tangency": "normal" }),
    ] {
        assert!(refused("part.sweep", values).contains("isn't in the part model yet"));
    }
}
