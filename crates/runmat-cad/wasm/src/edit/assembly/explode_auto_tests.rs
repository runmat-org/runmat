//! Auto-explode: steps along the mates, outermost first, that compile into an exploded view.
use super::tests::{bottom, compile, mate, run, top, RIG};
use serde_json::{json, Value};

fn z_of(view: &Value, path: &str) -> f64 {
    view["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["path"] == path)
        .unwrap_or_else(|| panic!("{path} doesn't move in {view}"))["transform"][11]
        .as_f64()
        .unwrap()
}

#[test]
fn auto_explode_pulls_a_stack_apart_along_its_mates_outermost_first() {
    // The block sits on the plate, and the pin stands on the block: plate → block → pin.
    let c = compile(RIG);
    let text = mate(
        RIG,
        "coincident",
        c.face("plate", top),
        c.face("block", bottom),
        json!({}),
    );
    let c = compile(&text);
    let text = mate(
        &text,
        "coincident",
        c.face("block", top),
        c.face("pin", bottom),
        json!({}),
    );
    let assembled = compile(&text);
    let block_z = assembled.transform("block")[11];
    let pin_z = assembled.transform("pin")[11];

    let text = run(
        &text,
        json!({ "command": "explode.auto", "values": { "spacing": 20, "lines": true } }),
    );
    assert!(text.contains("exploded auto_explode"), "{text}");
    let after = compile(&text);
    let view = &after.index["exploded"][0];
    // The pin (two mates out) first, then the block, which takes the pin with it.
    assert_eq!(view["steps"][0]["id"], "pin_out", "{view}");
    assert_eq!(view["steps"][0]["occurrences"], json!(["pin"]));
    assert_eq!(view["steps"][1]["id"], "block_out");
    assert_eq!(view["steps"][1]["occurrences"], json!(["block", "pin"]));
    // Both leave along +Z (the planar mates' normal, away from what they sit on), each just
    // clear of the one below by the spacing: the block 20 mm, the pin 20 mm more.
    assert!(
        (z_of(view, "block") - (block_z + 0.020)).abs() < 1e-6,
        "{view}"
    );
    assert!((z_of(view, "pin") - (pin_z + 0.040)).abs() < 1e-6, "{view}");
    // The plate stays; pin2 has no mate and stays.
    assert!(view["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .all(|o| o["path"] != "plate" && o["path"] != "pin2"));
    assert_eq!(view["lines"].as_array().map(Vec::len), Some(2), "{view}");

    // The steps are ordinary steps: a second run makes a second view beside the first.
    let again = run(
        &text,
        json!({ "command": "explode.auto", "values": { "spacing": 5, "lines": false } }),
    );
    assert!(again.contains("exploded auto_explode_2"), "{again}");

    // Delete: a step (and the pin's line, which no step left moves), then the view.
    let fewer = run(
        &text,
        json!({ "command": "assembly.delete", "selection": ["explode_step:auto_explode/pin_out"] }),
    );
    let view = &compile(&fewer).index["exploded"][0];
    assert_eq!(view["steps"].as_array().map(Vec::len), Some(1), "{view}");
    assert_eq!(
        view["lines"].as_array().map(Vec::len),
        Some(2),
        "the block step still moves the pin: {view}"
    );
    let none = run(
        &fewer,
        json!({ "command": "assembly.delete", "selection": ["exploded:auto_explode"] }),
    );
    assert!(!none.contains("exploded auto_explode"), "{none}");
    assert_eq!(compile(&none).index["exploded"], json!([]));
}

#[test]
fn auto_explode_needs_mates() {
    let out: Value = serde_json::from_str(&crate::edit::edit_json(
        "/p/rig.assembly",
        RIG,
        &super::tests::files(),
        &json!({ "command": "explode.auto", "values": {} }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out["message"]
            .as_str()
            .unwrap_or("")
            .contains("no two components are mated"),
        "{out}"
    );
}
