//! Collision detection over a motion study on OCCT through `runmat_cad_call`: an arm on a hinge
//! swings a quarter turn into a fixed stop. Stop reports the first solved frame where they meet;
//! Highlight lists their contact from then to the end of the study. A drag of the arm toward the
//! stop with Stop at collisions on ends at the last clear pose.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body }\n}\n";

/// The arm turns about Z at the base's corner; the stop stands 5 mm beyond the arm's start, in
/// its path.
const HINGE: &str = "runmat 0.7\nstd 1\n\nassembly hinge {\n    use part block as base\n    use part block as arm {\n        fixed = false\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 15 mm] } }\n    }\n    use part block as stop {\n        placement = { placement = offset, offset = { translation = [-25 mm, 5 mm, 15 mm] } }\n    }\n    connector pivot = { occurrence = [{ name = base }], anchor = { anchor = explicit, offset = { translation = [0 mm, 0 mm, 15 mm] } } }\n    connector root = { occurrence = [{ name = arm }], anchor = { anchor = origin } }\n    mate hinge = { kind = { mate = revolute }, a = pivot, b = root }\n    motion swing = { duration = 2 s, drives = [{ mate = hinge, from = 0 deg, to = 90 deg }] }\n}\n";

fn collision(mode: &str) -> Value {
    let files = json!({ "/p/block.part": BLOCK }).to_string();
    let request = json!({ "command": "motion.collision", "values": { "mode": mode } }).to_string();
    let out = call("analyze", &["/p/hinge.assembly", HINGE, &files, &request]);
    assert_eq!(out["ok"], true, "{out}");
    out
}

#[test]
fn stop_reports_the_first_frame_where_the_arm_meets_the_stop() {
    let out = collision("stop");
    let stopped = &out["data"]["stopped"];
    assert_eq!(stopped["a"], "arm", "{out}");
    assert_eq!(stopped["b"], "stop", "{out}");
    let t = stopped["time"].as_f64().unwrap();
    // The arm's far corner (28.3 mm out) reaches the stop's near edge (5 mm past the axis)
    // well before the quarter turn ends, and not at the start.
    assert!(t > 0.0 && t < 2.0, "{out}");
    assert!(
        out["summary"].as_str().unwrap().contains("stops at"),
        "{out}"
    );
    assert_eq!(out["rows"].as_array().unwrap().len(), 1, "{out}");
}

#[test]
fn highlight_lists_the_contact_to_the_end_of_the_study() {
    let out = collision("highlight");
    assert!(out["data"]["stopped"].is_null(), "{out}");
    let contacts = out["data"]["contacts"].as_array().unwrap();
    assert_eq!(contacts.len(), 1, "{out}");
    let c = &contacts[0];
    assert!(c["from"].as_f64().unwrap() > 0.0, "{out}");
    assert!((c["to"].as_f64().unwrap() - 2.0).abs() < 1e-9, "{out}");
    // At the end the arm (x −20…0, y 0…20) and the stop (x −25…−5, y 5…25) share 15 × 15 × 15 mm.
    assert!(
        (c["volume"].as_f64().unwrap() - 3.375e-6).abs() < 1e-8,
        "{out}"
    );
    assert_eq!(out["data"]["frames"], 25, "{out}");
}

fn arm_uuid(files: &str) -> String {
    let out = call("compile", &["/p/hinge.assembly", HINGE, files]);
    let rows = out["projection"]["data"]["occurrences"]
        .as_array()
        .unwrap_or_else(|| panic!("{out}"));
    rows.iter().find(|r| r["path"] == "arm").unwrap()["uuid"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn a_drag_into_the_stop_stops_at_the_last_clear_pose() {
    let files = json!({ "/p/block.part": BLOCK }).to_string();
    let arm = arm_uuid(&files);
    // Pulled toward a quarter turn, the arm would swing through the stop.
    let request = json!({ "command": "motion.drag", "values": {
        "component": [format!("occurrence:{arm}")], "dx": -20, "collisions": true } })
    .to_string();
    let out = call("edit", &["/p/hinge.assembly", HINGE, &files, &request]);
    assert_eq!(out["ok"], true, "{out}");
    assert!(
        out.to_string().contains("stopped where arm meets stop"),
        "{out}"
    );
    // Without the check the same pull goes the whole quarter turn.
    let free = json!({ "command": "motion.drag", "values": {
        "component": [format!("occurrence:{arm}")], "dx": -20, "collisions": false } })
    .to_string();
    let out = call("edit", &["/p/hinge.assembly", HINGE, &files, &free]);
    assert_eq!(out["ok"], true, "{out}");
    assert!(!out.to_string().contains("stopped"), "{out}");
}
