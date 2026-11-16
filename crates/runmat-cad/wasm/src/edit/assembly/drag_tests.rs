//! Dragging through the freedom mates leave (`motion.drag`, and Move component's Free drag): an
//! arm on a revolute hinge, grabbed at its box centre and pulled across, swings about the hinge
//! and stays on it; a fixed component and a zero pull are refused.
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body }\n}\n";

/// A base and an arm on a revolute hinge about Z at the base's top corner. The arm's origin is
/// on the hinge axis, so only a pull on a point away from the axis can turn it.
const HINGE: &str = "runmat 0.7\nstd 1\n\nassembly hinge {\n    use part block as base\n    use part block as arm {\n        fixed = false\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 15 mm] } }\n    }\n    connector pivot = { occurrence = [{ name = base }], anchor = { anchor = explicit, offset = { translation = [0 mm, 0 mm, 15 mm] } } }\n    connector root = { occurrence = [{ name = arm }], anchor = { anchor = origin } }\n    mate hinge = { kind = { mate = revolute }, a = pivot, b = root }\n}\n";

const PATH: &str = "/p/hinge.assembly";

fn files() -> Files {
    [("/p/block.part", BLOCK)]
        .map(|(p, t)| (p.to_owned(), t.to_owned()))
        .into()
}

fn compiled(text: &str) -> (Value, Value) {
    let (index, display, problems) =
        compile_assembly(PATH, text, &files()).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    (index, display)
}

fn uuid(text: &str, path: &str) -> String {
    let (index, _) = compiled(text);
    index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == path)
        .unwrap_or_else(|| panic!("{index}"))["uuid"]
        .as_str()
        .unwrap()
        .to_owned()
}

/// Component `path`'s world transform in the compiled display (rotation rows, then metres).
fn transform(text: &str, path: &str) -> Vec<f64> {
    let id = uuid(text, path);
    let (_, display) = compiled(text);
    let data = if display["occurrences"].is_array() {
        &display
    } else {
        &display["data"]
    };
    data["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["uuid"] == id.as_str())
        .unwrap_or_else(|| panic!("{display}"))["transform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn edit(text: &str, request: Value) -> Value {
    serde_json::from_str(&edit_json(PATH, text, &files(), &request.to_string())).unwrap()
}

fn run(text: &str, request: Value) -> String {
    let out = edit(text, request);
    assert_eq!(out["ok"], true, "{out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

/// The arm turned a quarter turn about the hinge (Z through the origin), still on it.
fn assert_quarter_turn(text: &str) {
    let t = transform(text, "arm");
    // x axis (first column) now along +Y.
    assert!(t[0].abs() < 1e-3 && (t[3] - 1.0).abs() < 1e-3, "{t:?}\n{text}");
    assert!((t[8] - 1.0).abs() < 1e-6, "Z stays the hinge axis: {t:?}");
    assert!(t[9].abs() < 1e-7 && t[10].abs() < 1e-7, "origin on the axis: {t:?}");
    assert!((t[11] - 0.015).abs() < 1e-7, "the arm stays on the base: {t:?}");
}

#[test]
fn a_pull_across_the_hinge_swings_the_arm_about_it() {
    // The arm's centre (10, 10) mm sits 45° round the hinge; pulled 20 mm in −X to (−10, 10),
    // the nearest point the hinge allows is 135° round: a quarter turn.
    let arm = uuid(HINGE, "arm");
    let text = run(
        HINGE,
        json!({ "command": "motion.drag", "values": {
            "component": [format!("occurrence:{arm}")], "dx": -20, "dy": 0, "dz": 0, "collisions": false } }),
    );
    assert_quarter_turn(&text);
}

#[test]
fn free_drag_in_move_component_drags_through_the_mates() {
    let arm = uuid(HINGE, "arm");
    let text = run(
        HINGE,
        json!({ "command": "assembly.move", "values": {
            "mode": "drag", "components": [format!("occurrence:{arm}")], "dx": -20, "dy": 0, "dz": 0 } }),
    );
    assert_quarter_turn(&text);
    // A pull along the hinge axis has nowhere to go.
    let out = edit(
        HINGE,
        json!({ "command": "assembly.move", "values": {
            "mode": "drag", "components": [format!("occurrence:{arm}")], "dz": 10 } }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(out.to_string().contains("can't move that way"), "{out}");
}

#[test]
fn a_fixed_component_or_no_pull_is_refused() {
    let base = uuid(HINGE, "base");
    let out = edit(
        HINGE,
        json!({ "command": "motion.drag", "values": { "component": [format!("occurrence:{base}")], "dx": 5 } }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(out.to_string().contains("is fixed"), "{out}");
    let arm = uuid(HINGE, "arm");
    let out = edit(
        HINGE,
        json!({ "command": "motion.drag", "values": { "component": [format!("occurrence:{arm}")] } }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(out.to_string().contains("drag the handle"), "{out}");
}
