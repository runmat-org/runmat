//! Pattern seeds, table and variable layouts, skipped instances and body operations, from the
//! form to the file and back through regeneration (test kernel).

use crate::edit::edit_json;
use crate::runtime::{compile_json, Files};
use serde_json::{json, Value};

/// A 40 x 20 x 10 mm plate (body) with a 10 x 10 x 5 mm boss on top, and a separate 5 mm cube
/// (`cube`, a new body) off to the side.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    input pd : length = 2 mm\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    sketch s2 on feature.body.face.end {\n        m1 = line(start = [5 mm, 5 mm], end = [10 mm, 5 mm])\n        m2 = line(start = [10 mm, 5 mm], end = [10 mm, 10 mm])\n        m3 = line(start = [10 mm, 10 mm], end = [5 mm, 10 mm])\n        m4 = line(start = [5 mm, 10 mm], end = [5 mm, 5 mm])\n    }\n    feature boss = extrude(profile = sketch.s2, distance = input.pd, mode = add)\n    sketch s3 on origin.xy {\n        c1 = line(start = [60 mm, 0 mm], end = [65 mm, 0 mm])\n        c2 = line(start = [65 mm, 0 mm], end = [65 mm, 5 mm])\n        c3 = line(start = [65 mm, 5 mm], end = [60 mm, 5 mm])\n        c4 = line(start = [60 mm, 5 mm], end = [60 mm, 0 mm])\n    }\n    feature cube = extrude(profile = sketch.s3, distance = 5 mm, mode = new)\n}\n";

fn edit(command: &str, values: Value) -> Value {
    let request = json!({ "command": command, "values": values });
    serde_json::from_str(&edit_json("/p/plate.part", PLATE, &Files::new(), &request.to_string()))
        .unwrap()
}

fn written(command: &str, values: Value) -> String {
    let out = edit(command, values);
    assert_eq!(out["ok"], true, "{command}: {out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

/// The edited text and its bodies, with no error.
fn added(command: &str, values: Value) -> (String, usize) {
    let text = written(command, values);
    let compiled: Value =
        serde_json::from_str(&compile_json("/p/plate.part", &text, &Files::new())).unwrap();
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{command}: {errors:?}\n{text}");
    let bodies = compiled["display"]["data"]["bodies"].as_array().unwrap().len();
    (text, bodies)
}

fn refused(command: &str, values: Value) -> String {
    let out = edit(command, values);
    assert_eq!(out["ok"], false, "{command}: {out}");
    out.to_string()
}

#[test]
fn a_table_pattern_writes_its_rows_and_skips_one() {
    let (text, _) = added(
        "part.pattern",
        json!({ "layout": "table", "seeds": ["feature:boss"], "table": "b: 20, 0, 0\nc: 0, 5, 0; d: 20, 5, 0", "skip": [2] }),
    );
    assert!(text.contains("layout = table("), "{text}");
    assert!(text.contains("key = b") && text.contains("key = d"), "{text}");
    assert!(text.contains("instance = row") && text.contains("key = c }"), "{text}");
}

#[test]
fn a_variable_pattern_writes_values_per_instance() {
    let (text, _) = added(
        "part.pattern",
        json!({ "layout": "variable", "seeds": ["feature:boss"], "table": "b: 20, 0, 0, pd = 4 mm\nc: 20, 5, 0, pd = 1" }),
    );
    assert!(text.contains("layout = variable("), "{text}");
    assert!(text.contains("values = { pd = 4 mm }") && text.contains("values = { pd = 1 mm }"), "{text}");
    // A table row with values, and a row without an offset, are refused by name.
    assert!(refused("part.pattern", json!({ "layout": "table", "seeds": ["feature:boss"], "table": "b: 1, 0, 0, pd = 4 mm" })).contains("offsets only"));
    assert!(refused("part.pattern", json!({ "layout": "table", "seeds": ["feature:boss"], "table": "b: 1, 0" })).contains("needs an offset"));
}

#[test]
fn skipped_linear_and_grid_instances_take_their_keys() {
    let text = written(
        "part.pattern",
        json!({ "layout": "linear", "seeds": ["feature:boss"], "direction": "origin.x", "spacing": 8, "instances": 4, "skip": [2] }),
    );
    assert!(text.contains("skip = [{ instance = linear, step = 2 }]"), "{text}");
    let text = written(
        "part.pattern",
        json!({ "layout": "linear", "seeds": ["feature:boss"], "direction": "origin.x", "spacing": 8, "instances": 3, "second_direction": true, "direction2": "origin.y", "spacing2": 6, "instances2": 2, "skip": [3] }),
    );
    assert!(text.contains("instance = grid, first = 1, second = 1"), "{text}");
    assert!(refused("part.pattern", json!({ "layout": "sketch", "seeds": ["feature:boss"], "sketch": "sketch:s3", "skip": [1] })).contains("source editor"));
}

#[test]
fn face_seeds_are_written_as_faces() {
    let text = written(
        "part.pattern",
        json!({ "layout": "linear", "seeds": ["feature.boss.face.end"], "direction": "origin.x", "spacing": 20, "instances": 2 }),
    );
    assert!(text.contains("faces = [feature.boss.face.end]"), "{text}");
    let text = written("part.mirror", json!({ "plane": "origin.yz", "seeds": ["feature.boss.face.end"] }));
    assert!(text.contains("faces = [feature.boss.face.end]"), "{text}");
    assert!(refused("part.pattern", json!({ "layout": "linear", "seeds": ["feature:boss", "feature.boss.face.end"], "direction": "origin.x", "spacing": 20, "instances": 2 })).contains("one kind"));
}

#[test]
fn bodies_copy_rotate_and_indent() {
    // Two copies of the cube, 10 mm apart: four bodies in all.
    let (text, bodies) = added(
        "part.body-move",
        json!({ "bodies": ["feature.cube.body"], "dx": 10, "dy": 0, "dz": 0, "copy": true, "copies": 2 }),
    );
    assert!(text.contains("body_op.copy("), "{text}");
    assert_eq!(bodies, 4, "{text}");
    // A quarter turn of the cube about the z axis.
    let (text, bodies) = added(
        "part.body-move",
        json!({ "bodies": ["feature.cube.body"], "axis": "origin.z", "angle": 90 }),
    );
    assert!(text.contains("body_op.rotate(") && text.contains("angle = 90 deg"), "{text}");
    assert_eq!(bodies, 2, "{text}");
    assert!(refused("part.body-move", json!({ "bodies": ["feature.cube.body"], "axis": "origin.z", "angle": 90, "dx": 3 })).contains("two steps"));
    // The cube pressed into the plate with no clearance: the cube is kept.
    let (text, bodies) = added(
        "part.body-indent",
        json!({ "target": "feature.body.body", "tool": "feature.cube.body", "clearance": 0 }),
    );
    assert!(text.contains("body_op.indent("), "{text}");
    assert_eq!(bodies, 2, "{text}");
}

#[test]
fn a_split_takes_a_body_as_its_tool() {
    let text = written("part.split", json!({ "body": "feature.body.body", "tool": "feature.cube.body" }));
    assert!(text.contains("split(body = feature.body.body, tool = feature.cube.body)"), "{text}");
}
