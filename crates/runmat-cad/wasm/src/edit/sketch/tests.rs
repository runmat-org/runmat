use crate::runtime::compile_part;
use serde_json::{json, Value};

const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 5 mm)\n}\n";

fn run(text: &str, command: &str, values: Value) -> Value {
    let request = json!({ "command": command, "values": values }).to_string();
    serde_json::from_str(&crate::edit::edit_json(
        "/p/plate.part",
        text,
        &Default::default(),
        &request,
    ))
    .unwrap()
}

fn applied(text: &str, command: &str, values: Value) -> String {
    let out = run(text, command, values);
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    compile_part("/p/plate.part", &text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    text
}

/// The lines of `after` that `before` doesn't have.
fn added(before: &str, after: &str) -> Vec<String> {
    after
        .lines()
        .filter(|l| !before.lines().any(|b| b == *l))
        .map(str::to_owned)
        .collect()
}

#[test]
fn a_sketch_starts_on_a_picked_plane_or_face() {
    let on_top = applied(PLATE, "part.sketch", json!({ "plane": "origin:front" }));
    assert_eq!(
        added(PLATE, &on_top),
        vec!["    sketch sketch1 on origin.zx {"]
    );
    let on_face = applied(
        PLATE,
        "part.sketch",
        json!({ "plane": ["feature.body.face.end"] }),
    );
    assert!(
        added(PLATE, &on_face)[0].starts_with("    sketch sketch1 on feature.body.face.end"),
        "{on_face}"
    );
}

#[test]
fn drawing_tools_write_entities_into_the_open_sketch() {
    // A line from l2's end (snapped) along x: one statement and its horizontal relation.
    let line = applied(
        PLATE,
        "sketch.line",
        json!({ "sketch": "s1", "start": { "x": 40, "y": 20, "of": "l2.end" }, "end": { "x": 60, "y": 20 }, "align": "horizontal" }),
    );
    assert_eq!(
        added(PLATE, &line),
        vec![
            "        l5 = line(start = l2.end, end = [60 mm, 20 mm])",
            "        constrain horizontal(l5)"
        ]
    );
    // A rectangle: four lines sharing corners, two horizontal and two vertical.
    let rect = applied(
        PLATE,
        "sketch.rectangle",
        json!({ "sketch": "s1", "type": "corner", "first": { "x": 5, "y": 5 }, "second": { "x": 15, "y": 12 } }),
    );
    let new = added(PLATE, &rect);
    assert_eq!(
        new.iter().filter(|l| l.contains("= line(")).count(),
        4,
        "{new:?}"
    );
    assert!(
        new.contains(&"        l6 = line(start = l5.end, end = [15 mm, 12 mm])".to_owned()),
        "{new:?}"
    );
    assert_eq!(
        new.iter()
            .filter(|l| l.contains("constrain horizontal"))
            .count(),
        2
    );
    assert_eq!(
        new.iter()
            .filter(|l| l.contains("constrain vertical"))
            .count(),
        2
    );
    // A circle, a slot (two lines, two arcs, tangent and equal), a point.
    let circle = applied(
        PLATE,
        "sketch.circle",
        json!({ "sketch": "s1", "type": "center", "center": { "x": 30, "y": 10 }, "radius": 3 }),
    );
    assert_eq!(
        added(PLATE, &circle),
        vec!["        c1 = circle(center = [30 mm, 10 mm], radius = 3 mm)"]
    );
    let slot = applied(
        PLATE,
        "sketch.slot",
        json!({ "sketch": "s1", "type": "straight", "start": { "x": 10, "y": 30 }, "end": { "x": 30, "y": 30 }, "width": 6 }),
    );
    let new = added(PLATE, &slot);
    assert_eq!(
        new.iter().filter(|l| l.contains("= arc(")).count(),
        2,
        "{new:?}"
    );
    assert_eq!(
        new.iter()
            .filter(|l| l.contains("constrain tangent"))
            .count(),
        4,
        "{new:?}"
    );
    // What the sketch model lacks says so.
    let spline = run(PLATE, "sketch.spline", json!({ "sketch": "s1" }));
    assert_eq!(spline["ok"], false);
    assert!(
        spline["message"]
            .as_str()
            .unwrap()
            .contains("isn't in the sketch model"),
        "{spline}"
    );
}

#[test]
fn relations_and_dimensions_take_canvas_picks() {
    let par = applied(
        PLATE,
        "sketch.constraint",
        json!({ "sketch": "s1", "relation": "parallel", "entities": ["sketch.s1.l1", "sketch.s1.l3"] }),
    );
    assert_eq!(
        added(PLATE, &par),
        vec!["        constrain parallel(l1, l3)"]
    );
    let len = applied(
        PLATE,
        "sketch.dimension",
        json!({ "sketch": "s1", "entities": ["sketch.s1.l1"], "value": 40 }),
    );
    assert_eq!(
        added(PLATE, &len),
        vec!["        constrain length(l1) = 40 mm"]
    );
    let dist = applied(
        PLATE,
        "sketch.dimension",
        json!({ "sketch": "s1", "entities": ["sketch.s1.l1.start", "sketch.s1.l2.end"], "value": 44.72 }),
    );
    assert_eq!(
        added(PLATE, &dist),
        vec!["        constrain distance(l1.start, l2.end) = 44.72 mm"]
    );
    let bad = run(
        PLATE,
        "sketch.constraint",
        json!({ "sketch": "s1", "relation": "parallel", "entities": ["sketch.s1.l1"] }),
    );
    assert_eq!(bad["ok"], false);
}

/// A plate with a line across it (l5, x = 20) and a circle on the right edge (c1).
const CROSSED: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n        l5 = line(start = [20 mm, -10 mm], end = [20 mm, 30 mm])\n        c1 = circle(center = [40 mm, 10 mm], radius = 5 mm)\n    }\n}\n";

#[test]
fn trim_removes_the_piece_around_the_click() {
    // The end of l5 outside the plate, above l3, goes: l5 now ends on l3.
    let top = applied(CROSSED, "sketch.trim", json!({ "sketch": "s1", "entities": ["sketch.s1.l5"], "at": { "x": 20, "y": 26 } }));
    assert!(top.contains("l5 = line(start = [20 mm, -10 mm], end = [20 mm, 20 mm])"), "{top}");
    // l1's piece left of l5 goes: l1 now starts on l5.
    let left = applied(CROSSED, "sketch.trim", json!({ "sketch": "s1", "entities": ["sketch.s1.l1"], "at": { "x": 10, "y": 0 } }));
    assert!(left.contains("l1 = line(start = [20 mm, 0 mm], end = [40 mm, 0 mm])"), "{left}");
    // Nothing crosses l4 but its own corners: a click on it deletes it.
    let gone = applied(CROSSED, "sketch.trim", json!({ "sketch": "s1", "entities": ["sketch.s1.l4"], "at": { "x": 0, "y": 10 } }));
    assert!(!gone.contains("l4 = line"), "{gone}");
    // The middle of l2, inside the circle, goes: l2 splits at the circle.
    let split = applied(CROSSED, "sketch.trim", json!({ "sketch": "s1", "entities": ["sketch.s1.l2"], "at": { "x": 40, "y": 10 } }));
    assert!(split.contains("l2 = line(start = [40 mm, 0 mm], end = [40 mm, 5 mm])"), "{split}");
    assert!(split.contains("line(start = [40 mm, 15 mm], end = [40 mm, 20 mm])"), "{split}");
    // The circle's half inside the plate goes: an arc remains, outside.
    let arc = applied(CROSSED, "sketch.trim", json!({ "sketch": "s1", "entities": ["sketch.s1.c1"], "at": { "x": 35, "y": 10 } }));
    assert!(arc.contains("c1 = arc(center = [40 mm, 10 mm], start = [40 mm, 5 mm], end = [40 mm, 15 mm])"), "{arc}");
}

#[test]
fn extend_runs_a_line_end_to_the_next_curve() {
    let text = CROSSED.replace("l5 = line(start = [20 mm, -10 mm], end = [20 mm, 30 mm])", "l5 = line(start = [20 mm, 5 mm], end = [20 mm, 12 mm])");
    let up = applied(&text, "sketch.extend", json!({ "sketch": "s1", "entities": ["sketch.s1.l5"], "at": { "x": 20, "y": 11 } }));
    assert!(up.contains("l5 = line(start = [20 mm, 5 mm], end = [20 mm, 20 mm])"), "{up}");
    let down = applied(&text, "sketch.extend", json!({ "sketch": "s1", "entities": ["sketch.s1.l5"], "at": { "x": 20, "y": 6 } }));
    assert!(down.contains("l5 = line(start = [20 mm, 0 mm], end = [20 mm, 12 mm])"), "{down}");
    let none = run(&text.replace("l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n", ""), "sketch.extend", json!({ "sketch": "s1", "entities": ["sketch.s1.l5"], "at": { "x": 20, "y": 11 } }));
    assert_eq!(none["ok"], false, "{none}");
}
