//! Assembly features: a cut, holes and a weld bead through components, from an assembly sketch.
use super::tests::{compile, run, RIG};
use serde_json::{json, Value};

/// RIG with sketch `s1` on the assembly's XY plane: a 10 × 10 mm square over the plate where
/// the block stands (x 35–45, y 15–25), and a point at (70, 40) for a hole.
fn rig_with_sketch() -> String {
    let text = run(
        RIG,
        json!({ "command": "assembly.sketch", "values": { "plane": ["origin:top"] } }),
    );
    let text = run(
        &text,
        json!({ "command": "sketch.rectangle", "values": { "sketch": "sketch1", "type": "corner", "first": { "x": 35, "y": 15 }, "second": { "x": 45, "y": 25 } } }),
    );
    run(
        &text,
        json!({ "command": "sketch.point", "values": { "sketch": "sketch1", "at": { "x": 70, "y": 40 } } }),
    )
}

fn feature(text: &str, values: Value) -> String {
    run(text, json!({ "command": "assembly.feature", "values": values }))
}

/// The face count of component `name`'s first body, as the display draws it.
fn faces(c: &super::tests::Compiled, name: &str) -> usize {
    let uuid = c.uuid(name);
    let occurrence = c.display["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["uuid"] == uuid.as_str())
        .unwrap();
    let p = occurrence["prototype"].as_u64().unwrap() as usize;
    let bodies = c.display["prototypes"][p]["part"]["bodies"].as_array().unwrap();
    bodies
        .iter()
        .map(|b| b["faces"].as_array().map_or(0, Vec::len))
        .sum()
}

fn bodies(c: &super::tests::Compiled, name: &str) -> usize {
    let uuid = c.uuid(name);
    let occurrence = c.display["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["uuid"] == uuid.as_str())
        .unwrap();
    let p = occurrence["prototype"].as_u64().unwrap() as usize;
    c.display["prototypes"][p]["part"]["bodies"].as_array().unwrap().len()
}

#[test]
fn a_cut_through_all_both_ways_cuts_the_components_it_names_only() {
    let text = rig_with_sketch();
    let before = compile(&text);
    let (plate, block) = (faces(&before, "plate"), faces(&before, "block"));
    let c = compile(&text);
    let text = feature(
        &text,
        json!({ "type": "cut", "profile": ["feature:sketch1"], "end": "through_all_both", "components": [format!("occurrence:{}", c.uuid("plate"))] }),
    );
    assert!(
        text.contains("operation cut1 = cut(profile = sketch.sketch1, extent = through_all, both, components = [plate])"),
        "{text}"
    );
    let after = compile(&text);
    assert_eq!(faces(&after, "plate"), plate + 4, "the square pocket's four walls");
    assert_eq!(faces(&after, "block"), block, "the block isn't named");
    let row = &after.index["operations"][0];
    assert_eq!(row["name"], "cut1");
    assert_eq!(row["kind"], "cut");
    assert_eq!(row["values"]["end"], "through_all_both");
    assert!(row.get("problem").is_none(), "{row}");

    // Edit: every component, then the block is cut too.
    let text = feature(
        &text,
        json!({ "name": "cut1", "type": "cut", "profile": ["feature:sketch1"], "end": "through_all_both" }),
    );
    assert!(text.contains("operation cut1 = cut(profile = sketch.sketch1, extent = through_all, both)"), "{text}");
    assert_eq!(faces(&compile(&text), "block"), block + 4);

    // Suppress, then delete, from the row.
    let text = run(&text, json!({ "command": "assembly.suppress", "selection": ["operation:cut1"] }));
    assert!(text.contains("both, suppressed)"), "{text}");
    assert_eq!(faces(&compile(&text), "plate"), plate);
    let text = run(&text, json!({ "command": "assembly.delete", "selection": ["operation:cut1"] }));
    assert!(!text.contains("operation"), "{text}");
}

#[test]
fn a_blind_cut_reaches_only_its_depth() {
    let text = rig_with_sketch();
    let block = faces(&compile(&text), "block");
    // From z = 0 up 5 mm: into the plate, not up to the block at z = 40 mm.
    let text = feature(
        &text,
        json!({ "type": "cut", "profile": ["feature:sketch1"], "end": "blind", "distance": 5 }),
    );
    assert!(text.contains("extent = blind(distance = 5 mm)"), "{text}");
    let c = compile(&text);
    assert_eq!(faces(&c, "block"), block);
    // Reversed, it cuts downwards out of the plate's bottom only (below z = 0 there's nothing).
    let reversed = feature(
        &text,
        json!({ "name": "cut1", "type": "cut", "profile": ["feature:sketch1"], "end": "blind", "distance": 5, "reverse": true }),
    );
    assert!(reversed.contains("reverse"), "{reversed}");
}

#[test]
fn a_hole_drills_at_each_point_with_its_fasteners_clearance_and_counterbore() {
    let text = rig_with_sketch();
    let plate = faces(&compile(&text), "plate");
    let text = feature(
        &text,
        json!({ "type": "hole", "profile": ["feature:sketch1"], "end": "through_all", "size": "M6", "hole_type": "counterbore" }),
    );
    assert!(text.contains("operation hole1 = hole("), "{text}");
    assert!(text.contains("diameter = 6.6 mm"), "M6 normal clearance: {text}");
    assert!(text.contains("counterbore = {"), "{text}");
    let c = compile(&text);
    assert!(faces(&c, "plate") > plate, "the hole's walls");
    let row = &c.index["operations"][0];
    assert_eq!(row["values"]["hole_type"], "counterbore");
    assert_eq!(row["values"]["size"], "M6");
}

#[test]
fn a_weld_bead_follows_the_sketch_and_is_drawn_with_the_first_component() {
    let text = run(
        RIG,
        json!({ "command": "assembly.sketch", "values": { "plane": ["origin:top"] } }),
    );
    let text = run(
        &text,
        json!({ "command": "sketch.line", "values": { "sketch": "sketch1", "start": { "x": 0, "y": 0 }, "end": { "x": 40, "y": 0 } } }),
    );
    let c = compile(&text);
    let plate = bodies(&c, "plate");
    let text = feature(
        &text,
        json!({ "type": "weld", "profile": ["feature:sketch1"], "weld_size": 4, "components": [format!("occurrence:{}", c.uuid("plate"))] }),
    );
    assert!(
        text.contains("operation weld1 = weld(path = sketch.sketch1, size = 4 mm, components = [plate])"),
        "{text}"
    );
    assert_eq!(bodies(&compile(&text), "plate"), plate + 1);
}

#[test]
fn a_feature_on_an_open_profile_says_so() {
    let text = run(
        RIG,
        json!({ "command": "assembly.sketch", "values": { "plane": ["origin:top"] } }),
    );
    let text = run(
        &text,
        json!({ "command": "sketch.line", "values": { "sketch": "sketch1", "start": { "x": 0, "y": 0 }, "end": { "x": 40, "y": 0 } } }),
    );
    let text = feature(&text, json!({ "type": "cut", "profile": ["feature:sketch1"] }));
    let (index, _, problems) =
        crate::runtime::compile_assembly("/p/rig.assembly", &text, &super::tests::files()).unwrap();
    assert!(index["operations"][0]["problem"].as_str().unwrap().contains("closed profile"));
    assert!(
        problems.iter().any(|d| d.message.contains("assembly feature `cut1`")),
        "{problems:?}"
    );
}

#[test]
fn the_feature_needs_a_sketch() {
    let out: Value = serde_json::from_str(&crate::edit::edit_json(
        "/p/rig.assembly",
        RIG,
        &super::tests::files(),
        &json!({ "command": "assembly.feature", "values": { "type": "cut" } }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false, "{out}");
    assert!(out.to_string().contains("pick the sketch"), "{out}");
}
