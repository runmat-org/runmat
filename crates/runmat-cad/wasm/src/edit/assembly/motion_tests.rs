//! Motion studies from the canvas: a study added on a hinge, solved frame by frame in the
//! compile's `motion` rows (poses, drive values, the tracked component's speed), changed from
//! its row, refused on a mate that can't be driven, and deleted.
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body }\n}\n";

/// A base and an arm on a revolute hinge at the base's top corner, free to turn about Z.
const HINGE: &str = "runmat 0.7\nstd 1\n\nassembly hinge {\n    use part block as base\n    use part block as arm {\n        fixed = false\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 15 mm] } }\n    }\n    connector pivot = { occurrence = [{ name = base }], anchor = { anchor = explicit, offset = { translation = [0 mm, 0 mm, 15 mm] } } }\n    connector root = { occurrence = [{ name = arm }], anchor = { anchor = origin } }\n    mate hinge = { kind = { mate = revolute }, a = pivot, b = root }\n}\n";

const PATH: &str = "/p/hinge.assembly";

fn files() -> Files {
    [("/p/block.part", BLOCK)]
        .map(|(p, t)| (p.to_owned(), t.to_owned()))
        .into()
}

fn index(text: &str) -> Value {
    let (index, _, problems) =
        compile_assembly(PATH, text, &files()).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    index
}

fn edit(text: &str, request: Value) -> Value {
    serde_json::from_str(&edit_json(PATH, text, &files(), &request.to_string())).unwrap()
}

fn run(text: &str, request: Value) -> String {
    let out = edit(text, request);
    assert_eq!(out["ok"], true, "{out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn an_assembly_without_motion_studies_projects_none() {
    assert_eq!(index(HINGE)["motion"], json!([]));
}

#[test]
fn a_study_drives_the_hinge_and_compiles_solved_frames() {
    let text = run(
        HINGE,
        json!({ "command": "motion.animation", "values": {
            "name": "Open close", "mate": "mate:/hinge", "quantity": "angle",
            "from_angle": 0, "to_angle": 90, "duration": 2, "profile": "linear" } }),
    );
    assert!(text.contains("motion open_close"), "{text}");
    let i = index(&text);
    let study = &i["motion"][0];
    assert_eq!(study["name"], "open_close", "{study}");
    assert_eq!(study["problems"], json!([]), "{study}");
    assert!(study["stopped"].is_null(), "{study}");
    // 2 s at 12 frames a second, both ends included.
    let times = numbers(&study["times"]);
    assert_eq!(times.len(), 25);
    let values = numbers(&study["drives"][0]["values"]);
    assert_eq!(values.first(), Some(&0.0));
    assert!((values[24] - 90.0).abs() < 1e-9, "{values:?}");
    assert_eq!(study["drives"][0]["unit"], "deg");
    // Only the arm moves, and each frame is its solved pose: a quarter turn about Z at the end.
    let frames = study["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 25);
    assert_eq!(frames[0].as_array().unwrap().len(), 1, "{}", frames[0]);
    let mid = numbers(&frames[12][0]["transform"]);
    let (c, s) = (45f64.to_radians().cos(), 45f64.to_radians().sin());
    assert!((mid[0] - c).abs() < 1e-6 && (mid[3] - s).abs() < 1e-6, "{mid:?}");
    let last = numbers(&frames[24][0]["transform"]);
    assert!(last[0].abs() < 1e-6 && (last[3] - 1.0).abs() < 1e-6, "{last:?}");
    assert!((last[11] - 0.015).abs() < 1e-9, "the arm stays on the base: {last:?}");
    // The arm's box centre (10, 10) mm from the axis turns at 45 deg/s: 14.14 mm × 0.785 rad/s.
    let plot = &study["plot"];
    assert_eq!(plot["occurrence"], "arm");
    let speed = numbers(&plot["speed"]);
    let expected = 200f64.sqrt() * 45f64.to_radians();
    for v in &speed {
        assert!((v - expected).abs() / expected < 0.01, "{speed:?} vs {expected}");
    }
    let w = numbers(&plot["angular_speed"]);
    assert!((w[12] - 45.0).abs() < 0.01, "{w:?}");
    // Centripetal acceleration ω²r inside the run.
    let a = numbers(&plot["acceleration"]);
    let ar = 200f64.sqrt() * 45f64.to_radians().powi(2);
    assert!((a[12] - ar).abs() / ar < 0.01, "{a:?} vs {ar}");
}

#[test]
fn a_study_changes_from_its_row_and_is_deleted() {
    let text = run(
        HINGE,
        json!({ "command": "motion.animation", "values": {
            "name": "swing", "mate": "mate:/hinge", "from_angle": 0, "to_angle": 90, "duration": 2 } }),
    );
    // From its row: no mate pick (it keeps the hinge), a smaller swing, smooth start and stop.
    let text = run(
        &text,
        json!({ "command": "motion.animation", "selection": ["motion:swing"], "values": {
            "name": "swing", "quantity": "angle", "from_angle": 0, "to_angle": 45, "duration": 3,
            "profile": "cycloidal" } }),
    );
    assert!(text.contains("law = cycloidal"), "{text}");
    let i = index(&text);
    let study = &i["motion"][0];
    assert_eq!(study["profile"], "cycloidal");
    let values = numbers(&study["drives"][0]["values"]);
    assert_eq!(values.len(), 37);
    assert!((values[36] - 45.0).abs() < 1e-9);
    // Cycloidal: at rest at both ends.
    let speed = numbers(&study["plot"]["speed"]);
    assert!(speed[0] < 0.05 * speed[18], "{speed:?}");
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "selection": ["motion:swing"] }),
    );
    assert!(!text.contains("motion swing"), "{text}");
    assert_eq!(index(&text)["motion"], json!([]));
}

#[test]
fn a_mate_without_a_value_is_refused_as_a_drive() {
    let fastened = HINGE.replace("mate = revolute", "mate = fasten");
    let out = edit(
        &fastened,
        json!({ "command": "motion.animation", "values": {
            "name": "bad", "mate": "mate:/hinge", "from_angle": 0, "to_angle": 90, "duration": 2 } }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(out.to_string().contains("can't be driven"), "{out}");
}

#[test]
fn a_drive_of_the_wrong_kind_is_a_study_problem() {
    // A distance drive on a revolute mate is refused by the study, not silently solved.
    let text = HINGE.replace(
        "a = pivot, b = root }\n",
        "a = pivot, b = root }\n    motion wrong = { duration = 1 s, drives = [{ mate = hinge, from = 0 mm, to = 5 mm }] }\n",
    );
    let i = index(&text);
    let problems = i["motion"][0]["problems"].to_string();
    assert!(problems.contains("driven by angle"), "{problems}");
}

#[test]
fn a_mate_position_is_saved_moved_to_and_deleted() {
    let held = HINGE.replace(
        "mate = revolute }",
        "mate = revolute, angle = { target = exact, value = 30 deg } }",
    );
    // A mate with no value has nothing to save.
    let out = edit(
        HINGE,
        json!({ "command": "motion.mate-controller", "values": { "name": "open", "action": "save", "mates": ["mate:/hinge"] } }),
    );
    assert_eq!(out["ok"], false, "{out}");
    let text = run(
        &held,
        json!({ "command": "motion.mate-controller", "values": { "name": "open", "action": "save", "mates": ["mate:/hinge"] } }),
    );
    assert!(text.contains("position open"), "{text}");
    let i = index(&text);
    assert_eq!(i["positions"][0]["name"], "open");
    assert_eq!(i["positions"][0]["values"][0]["mate"], "hinge");
    assert_eq!(i["positions"][0]["values"][0]["value"], "30 deg");
    // Turned to 0 deg, then moved back to the position: the hinge holds 30 deg again.
    let closed = text.replacen("value = 30 deg } }", "value = 0 deg } }", 1);
    assert!(closed.contains("value = 0 deg"), "{closed}");
    let moved = run(
        &closed,
        json!({ "command": "motion.mate-controller", "selection": ["position:open"], "values": { "action": "move" } }),
    );
    assert!(
        moved.contains("angle = { target = exact, value = 30 deg }"),
        "{moved}"
    );
    let gone = run(
        &moved,
        json!({ "command": "assembly.delete", "selection": ["position:open"] }),
    );
    assert!(!gone.contains("position open"), "{gone}");
    assert_eq!(index(&gone)["positions"], json!([]));
}
