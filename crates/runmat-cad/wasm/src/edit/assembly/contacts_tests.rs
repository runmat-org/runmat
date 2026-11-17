//! Contact regions: the block's bottom on the plate's top found as one bonded region, listed
//! as a row, made free to slide from its row, not found twice, and deleted.
use super::frames::{Frame, FrameKind};
use super::tests::{compile, files, run, top, RIG};
use crate::edit::edit_json;
use serde_json::{json, Value};

fn bottom(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[2] < -0.99
}

fn try_run(text: &str, request: Value) -> Value {
    serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        text,
        &files(),
        &request.to_string(),
    ))
    .unwrap()
}

#[test]
fn touching_faces_become_one_contact_region() {
    let c = compile(RIG);
    // Nothing touches yet: the block floats 30 mm above the plate.
    let out = try_run(RIG, json!({ "command": "assembly.contacts", "values": {} }));
    assert_eq!(out["ok"], false, "{out}");
    let text = run(
        RIG,
        json!({ "command": "assembly.mate", "values": {
            "type": "coincident", "first": [c.face("plate", top)], "second": [c.face("block", bottom)]
        } }),
    );
    let text = run(
        &text,
        json!({ "command": "assembly.contacts", "values": { "tolerance": 0.05, "bonded": true } }),
    );
    assert!(text.contains("contact contact1 = {"), "{text}");
    // Bonded is the default, so it is left out of the text.
    assert!(!text.contains("bonded = false"), "{text}");
    let after = compile(&text);
    let rows = after.index["contacts"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    let pair = [
        rows[0]["a"].as_str().unwrap(),
        rows[0]["b"].as_str().unwrap(),
    ];
    assert!(
        pair.contains(&"plate") && pair.contains(&"block"),
        "{rows:?}"
    );
    assert!(rows[0]["problem"].is_null(), "{rows:?}");
    // Run again: the pair is named already.
    let again = try_run(
        &text,
        json!({ "command": "assembly.contacts", "values": {} }),
    );
    assert_eq!(again["ok"], false, "{again}");
    // From its row: free to slide.
    let text = run(
        &text,
        json!({ "command": "assembly.contacts", "values": { "bonded": false }, "selection": ["contact:contact1"] }),
    );
    assert!(text.contains("bonded = false"), "{text}");
    // Delete from its row.
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "values": {}, "selection": ["contact:contact1"] }),
    );
    assert!(!text.contains("contact contact1"), "{text}");
}
