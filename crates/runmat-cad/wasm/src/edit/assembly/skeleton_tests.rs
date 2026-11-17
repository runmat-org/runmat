//! Skeleton: a new part created beside the assembly and placed as a reference component, left
//! out of the parts list and mass properties; an existing part placed as the skeleton.
use super::tests::{compile, files, RIG};
use crate::edit::edit_json;
use serde_json::{json, Value};

fn skeleton(text: &str, name: &str) -> Value {
    serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        text,
        &files(),
        &json!({ "command": "assembly.skeleton", "values": { "name": name } }).to_string(),
    ))
    .unwrap()
}

#[test]
fn a_skeleton_is_a_new_reference_part_left_out_of_the_parts_list() {
    let out = skeleton(RIG, "layout");
    assert_eq!(out["ok"], true, "{out}");
    let created = out["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["path"] == "/p/layout.part")
        .unwrap_or_else(|| panic!("{out}"));
    assert!(created["text"].as_str().unwrap().contains("part layout"), "{created}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(text.contains("use part layout as layout {") && text.contains("reference = true"), "{text}");
    // An existing part placed as the skeleton: plate, as plate2 (plate is taken).
    let out = skeleton(RIG, "plate");
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["files"].as_array().unwrap().len(), 1, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(text.contains("use part plate as plate2 {"), "{text}");
    let c = compile(&text);
    let row = c.index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == "plate2")
        .unwrap()
        .clone();
    assert_eq!(row["reference"], true, "{row}");
    // The parts list counts one plate, not two.
    let plates: u64 = c.index["bom"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["item"]["definition"].to_string().contains("plate.part"))
        .map(|r| r["quantity"].as_u64().unwrap())
        .sum();
    assert_eq!(plates, 1, "{}", c.index["bom"]);
    let mut all = files();
    all.insert("/p/rig.assembly".into(), text.clone());
    let report = crate::mass::assembly_mass("/p/rig.assembly", &text, &all, None).unwrap();
    assert!(report.parts.iter().all(|p| p.path != "plate2"), "{report:?}");
}
