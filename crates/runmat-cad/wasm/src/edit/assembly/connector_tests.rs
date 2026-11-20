//! Mate connectors: added on a picked face, offset and turned, renamed by name, mated, deleted.
use super::tests::{compile, run, top, RIG};
use crate::edit::edit_json;
use serde_json::{json, Value};

fn connector(text: &str, values: Value) -> String {
    run(
        text,
        json!({ "command": "assembly.mate-connector", "values": values }),
    )
}

/// The index row of connector `name`.
fn row<'a>(c: &'a super::tests::Compiled, name: &str) -> &'a Value {
    c.index["connectors"]
        .as_array()
        .unwrap_or_else(|| panic!("no connectors: {}", c.index))
        .iter()
        .find(|r| r["name"] == name)
        .unwrap_or_else(|| panic!("no connector {name}: {}", c.index["connectors"]))
}

#[test]
fn a_connector_on_a_face_is_published_from_its_part_and_placed_on_it() {
    let c = compile(RIG);
    let face = c.face("block", top);
    let text = connector(RIG, json!({ "name": "mount", "on": [face] }));
    assert!(
        text.contains(
            "connector mount = { occurrence = [{ name = block }], anchor = { anchor = published"
        ),
        "{text}"
    );
    let c = compile(&text);
    let r = row(&c, "mount");
    assert_eq!(r["occurrence"].as_str(), Some(c.uuid("block").as_str()));
    assert!(r["on"].as_str().unwrap().starts_with("published "), "{r}");
    // The block's top: 40 mm up plus its 15 mm height, Z up.
    let frame = &r["frame"];
    assert!((frame[0][2].as_f64().unwrap() - 0.055).abs() < 1e-6, "{r}");
    assert!((frame[2][2].as_f64().unwrap() - 1.0).abs() < 1e-6, "{r}");
    assert_eq!(r["values"]["name"], "mount");
}

#[test]
fn editing_by_name_offsets_turns_and_flips_it_in_place() {
    let face = compile(RIG).face("block", top);
    let text = connector(RIG, json!({ "name": "mount", "on": [face] }));
    let text = connector(
        &text,
        json!({ "name": "mount", "z": 5, "angle": 90, "flip": true }),
    );
    assert_eq!(text.matches("connector mount").count(), 1, "{text}");
    assert!(text.contains("flip = true"), "{text}");
    let c = compile(&text);
    let r = row(&c, "mount");
    // Flipped: Z down. The offset is along the face's frame (out of the block), before the flip.
    assert!(
        (r["frame"][2][2].as_f64().unwrap() + 1.0).abs() < 1e-6,
        "{r}"
    );
    assert!(
        (r["frame"][0][2].as_f64().unwrap() - 0.060).abs() < 1e-6,
        "{r}"
    );
    assert_eq!(r["values"]["flip"], true);
    assert_eq!(r["values"]["z"]["text"], "5 mm", "{r}");
    assert_eq!(r["values"]["angle"]["text"], "90 deg", "{r}");
}

#[test]
fn an_unnamed_connector_takes_the_next_free_name_and_mates_use_it() {
    let c = compile(RIG);
    let text = connector(RIG, json!({ "on": [c.face("block", top)] }));
    assert!(text.contains("connector block_c1 = "), "{text}");
    let text = connector(&text, json!({ "on": [c.face("plate", top)] }));
    assert!(text.contains("connector plate_c1 = "), "{text}");
    let text = run(
        &text,
        json!({ "command": "assembly.mate", "values": { "type": "coincident", "first": ["connector:plate_c1"], "second": ["connector:block_c1"] } }),
    );
    let c = compile(&text);
    assert_eq!(
        row(&c, "block_c1")["mates"].as_array().unwrap().len(),
        1,
        "{text}"
    );
    // A connector a mate uses can't be deleted.
    let out: Value = serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        &text,
        &super::tests::files(),
        &json!({ "command": "assembly.delete", "selection": ["connector:block_c1"] }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false, "{out}");
    assert!(out.to_string().contains("delete the mate first"), "{out}");
}

#[test]
fn delete_removes_an_unused_connector() {
    let face = compile(RIG).face("block", top);
    let text = connector(RIG, json!({ "name": "mount", "on": [face] }));
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "selection": ["connector:mount"] }),
    );
    assert!(!text.contains("connector mount"), "{text}");
}

#[test]
fn a_new_connector_needs_a_pick() {
    let out: Value = serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        RIG,
        &super::tests::files(),
        &json!({ "command": "assembly.mate-connector", "values": { "name": "free" } }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false, "{out}");
}

#[test]
fn publish_names_a_connector_or_a_component_and_changes_in_place() {
    let c = compile(RIG);
    let face = c.face("block", top);
    let text = connector(RIG, json!({ "name": "mount", "on": [face] }));
    let text = run(
        &text,
        json!({ "command": "assembly.publish", "values": { "name": "seat", "item": ["connector:mount"] } }),
    );
    assert!(
        text.contains("publish seat = { target = { publishes = connector, connector = mount } }"),
        "{text}"
    );
    let index = compile(&text).index;
    assert_eq!(index["publishes"][0]["item"], "connector:mount", "{index}");
    // The same name on a component changes it in place.
    let block = format!("occurrence:{}", c.uuid("block"));
    let text = run(
        &text,
        json!({ "command": "assembly.publish", "values": { "name": "seat", "item": [block] } }),
    );
    assert_eq!(text.matches("publish seat").count(), 1, "{text}");
    assert!(text.contains("publishes = occurrence"), "{text}");
    // The connector can go now that nothing publishes it; the publish goes by its row.
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "selection": ["connector:mount"] }),
    );
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "selection": ["publish:seat"] }),
    );
    assert!(!text.contains("publish seat"), "{text}");
}

#[test]
fn a_face_is_published_through_a_connector() {
    let c = compile(RIG);
    let out: Value = serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        RIG,
        &super::tests::files(),
        &json!({ "command": "assembly.publish", "values": { "name": "x", "item": [c.face("block", top)] } }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out.to_string().contains("add a mate connector on it first"),
        "{out}"
    );
}
