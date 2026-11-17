//! Options and effectivity: an option with a serial range, listed as rows; the shown values
//! picked and checked; deleting an axis a rule reads refused, then both deleted.
use super::tests::{compile, files, run, RIG};
use crate::edit::edit_json;
use serde_json::{json, Value};

fn try_run(text: &str, request: Value) -> Value {
    serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        text,
        &files(),
        &request.to_string(),
    ))
    .unwrap()
}

fn row<'a>(index: &'a Value, name: &str) -> &'a Value {
    index["options"]
        .as_array()
        .unwrap_or_else(|| panic!("no options: {index}"))
        .iter()
        .find(|r| r["name"] == name)
        .unwrap_or_else(|| panic!("no option {name}: {}", index["options"]))
}

#[test]
fn an_option_with_a_serial_range_is_shown_and_deleted() {
    let text = run(
        RIG,
        json!({ "command": "assembly.options-effectivity", "values": {
            "option": "engine", "values": "a, b", "effectivity": "MSN 1-99"
        } }),
    );
    assert!(text.contains("options = {"), "{text}");
    let c = compile(&text);
    let engine = row(&c.index, "engine");
    assert_eq!(engine["values"], "a, b");
    assert_eq!(engine["shown"], "a");
    assert_eq!(engine["effectivity"], "MSN 1-99");
    let msn = row(&c.index, "msn");
    assert_eq!(msn["kind"], "unit");
    assert_eq!(msn["values"], "1-99");
    // Show engine b on MSN 42; MSN 120 is out of range.
    let text = run(
        &text,
        json!({ "command": "assembly.options", "values": { "options": "engine=b", "effectivity": "MSN 42" } }),
    );
    let c = compile(&text);
    assert_eq!(row(&c.index, "engine")["shown"], "b");
    assert_eq!(row(&c.index, "msn")["shown"], "42");
    assert_eq!(c.index["variant"]["engine"], "b", "{}", c.index["variant"]);
    let out = try_run(
        &text,
        json!({ "command": "assembly.options", "values": { "effectivity": "MSN 120" } }),
    );
    assert_eq!(out["ok"], false, "{out}");
    // Change: a third value; the shown one stays.
    let text = run(
        &text,
        json!({ "command": "assembly.options-effectivity", "values": {
            "option": "engine", "values": "a, b, c", "effectivity": "MSN 1-99"
        } }),
    );
    assert_eq!(row(&compile(&text).index, "engine")["shown"], "b");
    // msn is read by engine's effectivity rule: refused until engine goes.
    let out = try_run(
        &text,
        json!({ "command": "assembly.delete", "values": {}, "selection": ["option:msn"] }),
    );
    assert_eq!(out["ok"], false, "{out}");
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "values": {}, "selection": ["option:engine"] }),
    );
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "values": {}, "selection": ["option:msn"] }),
    );
    assert!(!text.contains("options"), "{text}");
}
