//! Release and Update pins: the block's file released as R1, the live file then made taller, the
//! block pinned to R1 drawing at the released height, then set back to follow the live file.
use super::tests::{files, RIG};
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const STORE: &str = "/p/.runmat/releases/block.part.json";

fn edit(text: &str, files: &Files, request: Value) -> Value {
    serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        text,
        files,
        &request.to_string(),
    ))
    .unwrap()
}

fn uuid(text: &str, files: &Files, path: &str) -> String {
    let (index, _, _) = compile_assembly("/p/rig.assembly", text, files).unwrap();
    let row = index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == path)
        .cloned()
        .unwrap();
    format!("occurrence:{}", row["uuid"].as_str().unwrap())
}

/// The first `bounds` under `v`.
fn bounds(v: &Value) -> Option<&Value> {
    match v {
        Value::Object(m) => m.get("bounds").or_else(|| m.values().find_map(bounds)),
        Value::Array(a) => a.iter().find_map(bounds),
        _ => None,
    }
}

/// The block's drawn height in millimeters: its prototype's z extent.
fn block_height(text: &str, files: &Files) -> f64 {
    let (_, display, problems) = compile_assembly("/p/rig.assembly", text, files).unwrap();
    let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    let proto = display["prototypes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| {
            p["file"]
                .as_str()
                .is_some_and(|f| f.ends_with("block.part"))
        })
        .unwrap();
    let b = bounds(proto).unwrap();
    (b["max"][2].as_f64().unwrap() - b["min"][2].as_f64().unwrap()) * 1000.0
}

#[test]
fn a_component_pinned_to_its_release_keeps_the_released_geometry() {
    let mut files = files();
    let block = uuid(RIG, &files, "block");
    // Nothing released yet: Latest released has nothing to pin to.
    let out = edit(
        RIG,
        &files,
        json!({ "command": "assembly.update-pins", "values": { "to": "released", "components": [block] } }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out["message"]
            .as_str()
            .unwrap()
            .contains("block.part has no released revision"),
        "{out}"
    );
    // Release the block's file from its row.
    let out = edit(
        RIG,
        &files,
        json!({ "command": "assembly.release", "values": {}, "selection": [block] }),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["summary"], "Released block.part as R1", "{out}");
    let store = out["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["path"] == STORE)
        .unwrap_or_else(|| panic!("no store written: {out}"));
    files.insert(STORE.into(), store["text"].as_str().unwrap().into());
    // Unchanged since R1: a second release is refused.
    let again = edit(
        RIG,
        &files,
        json!({ "command": "assembly.release", "values": {}, "selection": [block] }),
    );
    assert_eq!(again["ok"], false, "{again}");
    // The live block grows from 15 mm to 40 mm.
    let live = files["/p/block.part"].replace("distance = 15 mm", "distance = 40 mm");
    files.insert("/p/block.part".into(), live);
    assert!((block_height(RIG, &files) - 40.0).abs() < 1e-2);
    // Pinned to its latest release, it draws at the released 15 mm.
    let out = edit(
        RIG,
        &files,
        json!({ "command": "assembly.update-pins", "values": { "to": "released", "components": [block] } }),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["summary"], "Pinned block to R1", "{out}");
    let pinned = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(pinned.contains("@ \"R1\""), "{pinned}");
    assert!((block_height(&pinned, &files) - 15.0).abs() < 1e-2);
    // Back to the live file, with no components named: every component follows the latest.
    let out = edit(
        &pinned,
        &files,
        json!({ "command": "assembly.update-pins", "values": { "to": "latest" } }),
    );
    assert_eq!(out["ok"], true, "{out}");
    let live = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(!live.contains('@'), "{live}");
    assert!((block_height(&live, &files) - 40.0).abs() < 1e-2);
    // A pin to a revision the store doesn't hold is reported, not drawn from the live file.
    let missing = pinned.replace("@ \"R1\"", "@ \"R7\"");
    let (_, _, problems) = compile_assembly("/p/rig.assembly", &missing, &files).unwrap();
    assert!(
        problems.iter().any(|d| d.message.contains("R7")),
        "{problems:?}"
    );
}

#[test]
fn release_with_nothing_selected_releases_the_assembly() {
    let files = files();
    let out = edit(
        RIG,
        &files,
        json!({ "command": "assembly.release", "values": {} }),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["summary"], "Released rig.assembly as R1", "{out}");
    assert!(out["files"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f["path"] == "/p/.runmat/releases/rig.assembly.json"));
}

#[test]
fn a_part_releases_from_its_own_manage_tab() {
    let files = files();
    let out: Value = serde_json::from_str(&edit_json(
        "/p/block.part",
        &files["/p/block.part"],
        &files,
        &json!({ "command": "manage.release", "values": {} }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["summary"], "Released block.part as R1", "{out}");
    assert_eq!(out["files"][0]["path"], STORE, "{out}");
}

#[test]
fn properties_make_a_component_a_reference_and_back() {
    let files = files();
    let block = uuid(RIG, &files, "block");
    let out = edit(
        RIG,
        &files,
        json!({ "command": "assembly.set-property", "values": { "reference": true }, "selection": [block] }),
    );
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(text.contains("reference = true"), "{text}");
    let block = uuid(&text, &files, "block");
    let out = edit(
        &text,
        &files,
        json!({ "command": "assembly.set-property", "values": { "reference": false }, "selection": [block] }),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert!(
        !out["files"][0]["text"]
            .as_str()
            .unwrap()
            .contains("reference"),
        "{out}"
    );
}

#[test]
fn an_emptied_components_field_updates_every_component_whatever_is_selected() {
    let mut files = files();
    let block = uuid(RIG, &files, "block");
    for part in ["block", "plate", "pin"] {
        let path = format!("/p/{part}.part");
        let ((store, _, change), _) =
            crate::edit::release::release(&path, &files[&path], &files).unwrap();
        files.insert(store, change.source);
    }
    let out = edit(
        RIG,
        &files,
        json!({ "command": "assembly.update-pins", "values": { "to": "released", "components": [] }, "selection": [block] }),
    );
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap();
    assert_eq!(text.matches("@ \"R1\"").count(), 4, "{text}");
}
