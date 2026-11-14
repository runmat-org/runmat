//! Sketches in an assembly: started on an origin plane or a component's face, drawn with the
//! part's sketch tools, and following the component when it moves.
use super::tests::{compile, files, run, top, RIG};
use crate::edit::edit_json;
use serde_json::{json, Value};

fn sketch(text: &str, plane: &str) -> String {
    run(
        text,
        json!({ "command": "assembly.sketch", "values": { "plane": [plane] } }),
    )
}

fn tool(text: &str, command: &str, values: Value) -> String {
    run(text, json!({ "command": command, "values": values }))
}

fn frame_origin(display: &Value, id: &str) -> [f64; 3] {
    let s = display["sketches"]
        .as_array()
        .unwrap_or_else(|| panic!("no sketches: {display}"))
        .iter()
        .find(|s| s["id"] == id)
        .unwrap_or_else(|| panic!("no sketch {id}"));
    let o = &s["frame"]["origin"];
    [0, 1, 2].map(|i| o[i].as_f64().unwrap())
}

#[test]
fn a_sketch_on_an_origin_plane_takes_lines_and_rectangles() {
    let text = sketch(RIG, "origin:top");
    assert!(text.contains("sketch sketch1 on origin.xy {"), "{text}");
    let text = tool(
        &text,
        "sketch.line",
        json!({ "sketch": "sketch1", "start": { "x": 0, "y": 0 }, "end": { "x": 40, "y": 0 } }),
    );
    let text = tool(
        &text,
        "sketch.rectangle",
        json!({ "sketch": "sketch1", "type": "corner", "first": { "x": 5, "y": 5 }, "second": { "x": 15, "y": 12 } }),
    );
    assert!(
        text.contains("l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])"),
        "{text}"
    );
    let c = compile(&text);
    let drawn = &c.display["sketches"][0];
    assert_eq!(drawn["id"], "sketch1");
    assert_eq!(drawn["entities"].as_array().unwrap().len(), 5, "{drawn}");
    let row = &c.index["sketches"][0];
    assert_eq!(row["id"], "sketch1");
    assert_eq!(row["plane"], "origin.xy");
}

#[test]
fn a_sketch_on_a_component_face_follows_the_component() {
    let c = compile(RIG);
    let text = sketch(RIG, &c.face("block", top));
    assert!(
        text.contains("sketch sketch1 on plane.block_c1 {"),
        "{text}"
    );
    let text = tool(
        &text,
        "sketch.line",
        json!({ "sketch": "sketch1", "start": { "x": 0, "y": 0 }, "end": { "x": 10, "y": 0 } }),
    );
    // The block sits at z = 40 mm and is 15 mm tall: its top is at 55 mm.
    let before = frame_origin(&compile(&text).display, "sketch1");
    assert!((before[2] - 0.055).abs() < 1e-6, "{before:?}");
    let moved = text.replace("[30 mm, 10 mm, 40 mm]", "[30 mm, 10 mm, 70 mm]");
    assert_ne!(moved, text);
    let after = frame_origin(&compile(&moved).display, "sketch1");
    assert!((after[2] - 0.085).abs() < 1e-6, "{after:?}");
    assert!((after[0] - before[0]).abs() < 1e-9 && (after[1] - before[1]).abs() < 1e-9);
}

#[test]
fn sketch_tools_need_an_open_sketch_of_the_assembly() {
    let out: Value = serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        RIG,
        &files(),
        &json!({ "command": "sketch.line", "values": { "sketch": "nope", "start": { "x": 0, "y": 0 }, "end": { "x": 1, "y": 0 } } }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false);
    assert!(
        out["message"]
            .as_str()
            .unwrap()
            .contains("no sketch `nope`"),
        "{out}"
    );
}

#[test]
fn an_assembly_sketch_whose_connector_is_gone_says_so() {
    let c = compile(RIG);
    let text = sketch(RIG, &c.face("block", top));
    let broken = text.replace(
        "sketch sketch1 on plane.block_c1",
        "sketch sketch1 on plane.missing",
    );
    let (index, _, problems) =
        crate::runtime::compile_assembly("/p/rig.assembly", &broken, &files()).unwrap();
    assert!(
        problems
            .iter()
            .any(|d| d.message.contains("connector `missing`")),
        "{problems:?}"
    );
    assert_eq!(index["sketches"][0]["status"], "failed");
}

#[test]
fn a_selected_sketch_suppresses_and_deletes() {
    let text = sketch(RIG, "origin.xy");
    let suppressed = run(
        &text,
        json!({ "command": "assembly.suppress", "selection": ["feature:sketch1"] }),
    );
    assert!(suppressed.contains("suppressed = true"), "{suppressed}");
    assert_eq!(
        compile(&suppressed).index["sketches"][0]["status"],
        "suppressed"
    );
    let deleted = run(
        &text,
        json!({ "command": "assembly.delete", "selection": ["feature:sketch1"] }),
    );
    assert!(!deleted.contains("sketch sketch1"), "{deleted}");
}
