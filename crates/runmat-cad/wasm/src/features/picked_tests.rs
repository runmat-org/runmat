//! Picked references from the canvas, edited and regenerated: fillets and chamfers on faces and
//! features, axes from sketch lines and faces, holes at clicked points, shells with walls of
//! their own thickness, hole series, and the picks a command refuses by name.

use crate::edit::edit_json;
use crate::runtime::{compile_json, Files};
use serde_json::{json, Value};

/// A 40 x 20 plate, 10 mm thick, and a construction line in the YZ plane, 10 mm off the origin.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    sketch s2 on origin.yz {\n        ax = line(start = [10 mm, 0 mm], end = [10 mm, 10 mm], construction = true)\n    }\n}\n";

const END: &str = "feature.body.face.end";

fn edit(command: &str, values: Value) -> Value {
    let request = json!({ "command": command, "values": values });
    serde_json::from_str(&edit_json("/p/plate.part", PLATE, &Files::new(), &request.to_string()))
        .unwrap()
}

/// The edited text and its regenerated display, with no error.
fn added(command: &str, values: Value) -> (String, Value) {
    let out = edit(command, values);
    assert_eq!(out["ok"], true, "{command}: {out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    let compiled: Value =
        serde_json::from_str(&compile_json("/p/plate.part", &text, &Files::new())).unwrap();
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{command}: {errors:?}\n{text}");
    (text, compiled["display"]["data"].clone())
}

fn faces(display: &Value) -> usize {
    display["bodies"].as_array().unwrap().iter().map(|b| b["faces"].as_array().unwrap().len()).sum()
}

fn refused(command: &str, values: Value) -> String {
    let out = edit(command, values);
    assert_eq!(out["ok"], false, "{command}: {out}");
    out.to_string()
}

#[test]
fn a_chamfer_on_a_face_bevels_its_boundary_edges() {
    let (t, d) = added("part.chamfer", json!({ "type": "equal", "edges": [END], "distance": 1 }));
    assert!(t.contains("edges = [feature.body.face.end]"), "{t}");
    // The four edges of the top face each gain a bevel face.
    assert_eq!(faces(&d), 10, "{t}");
}

#[test]
fn a_fillet_on_a_face_rounds_its_boundary_edges() {
    let (t, d) = added("part.fillet", json!({ "type": "constant", "edges": [END], "radius": 1 }));
    assert!(t.contains("edges = [feature.body.face.end]"), "{t}");
    assert!(faces(&d) >= 10, "{t}");
}

#[test]
fn a_chamfer_on_a_feature_bevels_its_edges() {
    let (t, d) = added("part.chamfer", json!({ "type": "equal", "edges": ["feature:body"], "distance": 1 }));
    assert!(t.contains("feature.body.body"), "{t}");
    // Every edge of the plate the feature made gains a bevel face.
    assert!(faces(&d) >= 18, "{t}");
}

#[test]
fn a_circular_pattern_turns_about_a_sketch_centerline() {
    let (t, d) = added(
        "part.pattern",
        json!({ "layout": "circular", "seeds": ["feature:body"], "axis": ["sketch.s2.ax"], "instances": 2, "angle": 360 }),
    );
    assert!(t.contains("along = sketch_line"), "{t}");
    // The plate (x 0..40) turned half a turn about a line in the YZ plane lands at x -40..0.
    let min = d["bounds"]["min"][0].as_f64().unwrap() * 1e3;
    assert!((min + 40.0).abs() < 1e-6, "{min}\n{t}");
}

#[test]
fn a_face_or_sketch_line_axis_is_written_and_a_vertex_axis_is_refused_by_name() {
    let (t, _) = added(
        "part.pattern",
        json!({ "layout": "circular", "seeds": ["feature:body"], "axis": ["origin:z"], "instances": 2, "angle": 360 }),
    );
    assert!(t.contains("origin.z"), "{t}");
    let v = "vertex_of(feature.body.face.side(s1.l1), feature.body.face.side(s1.l2), feature.body.face.end)";
    let m = refused(
        "part.pattern",
        json!({ "layout": "circular", "seeds": ["feature:body"], "axis": [v], "instances": 2, "angle": 360 }),
    );
    assert!(m.contains("which gives no axis") && m.contains("vertex_of"), "{m}");
}

#[test]
fn a_hole_goes_where_the_face_was_clicked() {
    let (t, d) = added(
        "part.hole",
        json!({ "type": "simple", "face": [END], "positions": ["point(31.2 mm, 7.5 mm, 10 mm)"], "diameter": 3, "end": "through_all" }),
    );
    assert!(t.contains("point(31.2 mm, 7.5 mm, 10 mm)"), "{t}");
    assert!(faces(&d) > 6, "{t}");
}

#[test]
fn a_shell_wall_takes_its_own_thickness_and_a_hole_series_is_written() {
    let (t, _) = added(
        "part.shell",
        json!({ "open_faces": [END], "thickness": 1, "thick_faces": ["feature.body.face.side(s1.l1)"], "face_thickness": 3 }),
    );
    assert!(t.contains("faces = [") && t.contains("3 mm"), "{t}");
    // The series flag is written, so regeneration answers for it: one plate is one body.
    let out = edit(
        "part.hole",
        json!({ "type": "simple", "face": [END], "positions": ["point(20 mm, 10 mm, 10 mm)"], "diameter": 3, "end": "through_all", "series": true }),
    );
    let t = out["files"][0]["text"].as_str().unwrap();
    assert!(t.contains("series"), "{t}");
    let compiled = compile_json("/p/plate.part", t, &Files::new());
    assert!(compiled.contains("needs the hole axis to cross at least two bodies"), "{compiled}");
}

#[test]
fn a_pull_on_an_edge_is_refused_by_name() {
    let edge = "edges_between(feature.body.face.side(s1.l1), feature.body.face.end)";
    let m = refused("direct.pull", json!({ "faces": [edge], "distance": 1 }));
    assert!(m.contains("isn't a face"), "{m}");
}

#[test]
fn a_revolve_turns_about_a_sketch_centerline_on_the_test_kernel() {
    // A 10 x 10 square 10 to 20 mm from the centerline, revolved a half turn into a new body:
    // it sweeps to the far side of the line, 20 mm past it.
    let ring = PLATE.replace(
        "        ax = line(",
        "        q1 = line(start = [20 mm, 0 mm], end = [30 mm, 0 mm])\n        q2 = line(start = [30 mm, 0 mm], end = [30 mm, 10 mm])\n        q3 = line(start = [30 mm, 10 mm], end = [20 mm, 10 mm])\n        q4 = line(start = [20 mm, 10 mm], end = [20 mm, 0 mm])\n        ax = line(",
    );
    let request = json!({ "command": "part.revolve", "values": { "profile": ["sketch:s2"], "axis": ["sketch.s2.ax"], "op": "new", "type": "one", "angle": 180 } });
    let out: Value = serde_json::from_str(&edit_json("/p/plate.part", &ring, &Files::new(), &request.to_string())).unwrap();
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap();
    assert!(text.contains("along = sketch_line"), "{text}");
    let compiled: Value = serde_json::from_str(&compile_json("/p/plate.part", text, &Files::new())).unwrap();
    let errors: Vec<&Value> = compiled["diagnostics"].as_array().unwrap().iter().filter(|d| d["severity"] == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    assert_eq!(compiled["display"]["data"]["bodies"].as_array().unwrap().len(), 2, "{text}");
}
