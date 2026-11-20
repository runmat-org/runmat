//! Envelopes: fitted round picked components with a margin, listed and drawn, changed by name
//! (keeping their box), refused without a box or a name, deleted from their rows.
use super::tests::{compile, files, run, RIG};
use crate::edit::edit_json;
use serde_json::{json, Value};

fn envelope(text: &str, values: Value) -> String {
    run(
        text,
        json!({ "command": "assembly.envelope", "values": values }),
    )
}

fn row(index: &Value, name: &str) -> Value {
    index["envelopes"]
        .as_array()
        .unwrap_or_else(|| panic!("no envelopes: {index}"))
        .iter()
        .find(|r| r["name"] == name)
        .unwrap_or_else(|| panic!("no envelope {name}: {}", index["envelopes"]))
        .clone()
}

fn mm(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| (x.as_f64().unwrap() * 1e3 * 100.0).round() / 100.0)
        .collect()
}

#[test]
fn an_envelope_fits_round_components_with_its_margin_and_is_drawn() {
    let c = compile(RIG);
    let block = format!("occurrence:{}", c.uuid("block"));
    let text = envelope(
        RIG,
        json!({ "name": "service", "around": [block.clone()], "margin": 5.0, "keep_out": true, "owner": block }),
    );
    let line = text
        .lines()
        .find(|l| l.contains("envelope service = {"))
        .unwrap_or_else(|| panic!("{text}"));
    assert!(
        line.contains("owner = block") && line.contains("keep_out = true"),
        "{line}"
    );
    let c = compile(&text);
    let r = row(&c.index, "service");
    // The block (20 x 20 x 15 mm at 30, 10, 40 mm) grown by 5 mm on every side.
    assert_eq!(mm(&r["min"]), vec![25.0, 5.0, 35.0], "{r}");
    assert_eq!(mm(&r["max"]), vec![55.0, 35.0, 60.0], "{r}");
    assert_eq!(r["owner"], "block");
    assert!(r.get("problem").is_none(), "{r}");
    let drawn = c.display["references"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["id"] == "envelope:service")
        .unwrap_or_else(|| panic!("not drawn: {}", c.display["references"]));
    assert_eq!(drawn["lines"].as_array().unwrap().len(), 6);
    // Changed by name with no components: the box stays, the rest changes.
    let text = envelope(&text, json!({ "name": "service", "keep_out": false }));
    let c = compile(&text);
    let r = row(&c.index, "service");
    assert_eq!(mm(&r["min"]), vec![25.0, 5.0, 35.0], "{r}");
    assert_eq!(r["keep_out"], false);
    assert!(r.get("owner").is_none(), "{r}");
    assert_eq!(text.matches("envelope service").count(), 1, "{text}");
    // Delete on its row.
    let text = run(
        &text,
        json!({ "command": "assembly.delete", "selection": ["envelope:service"] }),
    );
    assert!(!text.contains("envelope"), "{text}");
}

#[test]
fn an_envelope_needs_a_name_and_a_box() {
    for values in [
        json!({ "name": "1bad", "around": [] }),
        json!({ "name": "fresh" }),
    ] {
        let out: Value = serde_json::from_str(&edit_json(
            "/p/rig.assembly",
            RIG,
            &files(),
            &json!({ "command": "assembly.envelope", "values": values }).to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], false, "{values}: {out}");
    }
}
