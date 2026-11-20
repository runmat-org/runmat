//! Configurations and display states: added from the form, listed in the index, compiled in,
//! changed by name, deleted from their rows.
use super::tests::{compile, files, run, RIG};
use crate::edit::edit_json;
use crate::runtime::compile_assembly_known;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// RIG with the block's height an input, `lift` (40 mm).
fn rig() -> String {
    RIG.replace(
        "assembly rig {\n",
        "assembly rig {\n    input lift : length = 40 mm\n",
    )
    .replace("[30 mm, 10 mm, 40 mm]", "[30 mm, 10 mm, input.lift]")
}

fn configure(text: &str, values: Value) -> String {
    run(
        text,
        json!({ "command": "assembly.configurations", "values": values }),
    )
}

fn row<'a>(index: &'a Value, name: &str) -> &'a Value {
    index["configurations"]
        .as_array()
        .unwrap_or_else(|| panic!("no configurations: {index}"))
        .iter()
        .find(|r| r["name"] == name)
        .unwrap_or_else(|| panic!("no row {name}: {}", index["configurations"]))
}

fn block_z(display: &Value, index: &Value) -> f64 {
    let uuid = index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == "block")
        .unwrap()["uuid"]
        .clone();
    display["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["uuid"] == uuid)
        .unwrap()["transform"][11]
        .as_f64()
        .unwrap()
}

#[test]
fn a_configuration_sets_inputs_and_suppresses_components_when_it_is_active() {
    let text = rig();
    let c = compile(&text);
    let pin2 = format!("occurrence:{}", c.uuid("pin2"));
    let text = configure(
        &text,
        json!({ "kind": "configuration", "name": "raised", "values": "lift = 60 mm", "suppress": [pin2] }),
    );
    assert!(text.contains("configuration raised = {"), "{text}");
    let c = compile(&text);
    let r = row(&c.index, "raised");
    assert_eq!(r["kind"], "configuration");
    assert_eq!(r["values"], "lift = 60 mm", "{r}");
    assert_eq!(r["components"], json!(["pin2"]));
    assert_eq!(r["items"], json!([pin2]));
    assert_eq!(r["form"]["suppress"], json!([pin2]));
    assert!(r.get("active").is_none(), "{r}");
    assert!((block_z(&c.display, &c.index) - 0.040).abs() < 1e-6);
    // Active: the block rises to 60 mm and pin2 drops out.
    let (index, display, problems) = compile_assembly_known(
        "/p/rig.assembly",
        &text,
        &files(),
        Some("raised"),
        &BTreeSet::new(),
    )
    .unwrap();
    assert!(
        problems.iter().all(|d| d.severity != "error"),
        "{problems:?}"
    );
    assert_eq!(index["configuration"], "raised");
    assert_eq!(row(&index, "raised")["active"], true);
    let z = block_z(&display, &index);
    assert!(
        (z - 0.060).abs() < 1e-6,
        "z = {z}\n{text}\n{}",
        index["diagnostics"]
    );
    assert!(
        index["suppressed"].to_string().contains("pin2"),
        "{}",
        index["suppressed"]
    );
}

#[test]
fn the_same_name_changes_it_in_place_and_unknown_inputs_are_refused() {
    let text = configure(
        &rig(),
        json!({ "kind": "configuration", "name": "raised", "values": "lift = 60 mm" }),
    );
    let text = configure(
        &text,
        json!({ "kind": "configuration", "name": "raised", "values": "lift = 80 mm" }),
    );
    assert_eq!(text.matches("configuration raised").count(), 1, "{text}");
    assert_eq!(
        row(&compile(&text).index, "raised")["values"],
        "lift = 80 mm"
    );
    let out: Value = serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        &text,
        &files(),
        &json!({ "command": "assembly.configurations", "values": { "kind": "configuration", "name": "wide", "values": "width = 9 mm" } }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false);
    assert!(
        out["message"]
            .as_str()
            .unwrap()
            .contains("has no input `width`; it has lift"),
        "{out}"
    );
}

#[test]
fn a_display_state_lists_what_it_hides_and_rows_delete() {
    let text = rig();
    let pin = format!("occurrence:{}", compile(&text).uuid("pin"));
    let text = configure(
        &text,
        json!({ "kind": "display_state", "name": "no_pins", "hidden": [pin] }),
    );
    let line = text
        .lines()
        .find(|l| l.contains("display_state no_pins = {"))
        .unwrap_or_else(|| panic!("{text}"));
    assert!(
        line.contains("hidden = [") && line.contains("pin"),
        "{text}"
    );
    let text = configure(
        &text,
        json!({ "kind": "configuration", "name": "raised", "values": "lift = 60 mm" }),
    );
    let c = compile(&text);
    let r = row(&c.index, "no_pins");
    assert_eq!(r["kind"], "display_state");
    assert_eq!(r["items"], json!([pin]));
    assert_eq!(r["form"]["hidden"], json!([pin]));
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "selection": ["display_state:no_pins", "configuration:raised"] }),
    );
    assert!(
        !text.contains("display_state") && !text.contains("configuration raised"),
        "{text}"
    );
}
