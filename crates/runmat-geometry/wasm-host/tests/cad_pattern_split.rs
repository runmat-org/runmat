//! Face patterns, fillet patterns, splits by a curved face or a body, and body copies, turns
//! and indents on OCCT through `runmat_cad_call`. Volumes and body counts are checked against
//! closed forms.

use serde_json::{json, Value};
use std::f64::consts::PI;

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// A 40 x 20 x 10 mm plate with a 10 x 10 x 5 mm boss, a 6 x 6 x 3 mm pocket, a 3 mm pin
/// (radius) 4 mm tall, and a separate 5 mm cube that overlaps the plate's corner by 2 mm.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    sketch s2 on feature.body.face.end {\n        m1 = line(start = [2 mm, 2 mm], end = [12 mm, 2 mm])\n        m2 = line(start = [12 mm, 2 mm], end = [12 mm, 12 mm])\n        m3 = line(start = [12 mm, 12 mm], end = [2 mm, 12 mm])\n        m4 = line(start = [2 mm, 12 mm], end = [2 mm, 2 mm])\n    }\n    feature boss = extrude(profile = sketch.s2, distance = 5 mm, mode = add)\n    sketch s3 on feature.body.face.end {\n        p1 = line(start = [16 mm, 4 mm], end = [22 mm, 4 mm])\n        p2 = line(start = [22 mm, 4 mm], end = [22 mm, 10 mm])\n        p3 = line(start = [22 mm, 10 mm], end = [16 mm, 10 mm])\n        p4 = line(start = [16 mm, 10 mm], end = [16 mm, 4 mm])\n    }\n    feature pocket = extrude(profile = sketch.s3, distance = 3 mm, reverse, mode = cut)\n    sketch s4 on feature.body.face.end {\n        c1 = circle(center = [32 mm, 10 mm], radius = 3 mm)\n    }\n    feature pin = extrude(profile = sketch.s4, distance = 4 mm, mode = add)\n    sketch s5 on origin.xy {\n        q1 = line(start = [38 mm, -3 mm], end = [43 mm, -3 mm])\n        q2 = line(start = [43 mm, -3 mm], end = [43 mm, 2 mm])\n        q3 = line(start = [43 mm, 2 mm], end = [38 mm, 2 mm])\n        q4 = line(start = [38 mm, 2 mm], end = [38 mm, -3 mm])\n    }\n    feature cube = extrude(profile = sketch.s5, distance = 5 mm, mode = new)\n}\n";

const PLATE_V: f64 = 8000.0 + 500.0 - 108.0 + 9.0 * PI * 4.0;
const CUBE_V: f64 = 125.0;

/// The part after the command: its text, its body count and its total volume (mm³).
fn after(command: &str, values: Value) -> (String, usize, f64) {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/plate.part", PLATE, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    let compiled = call("compile", &["/p/plate.part", &text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{command}: {errors:?}\n{text}");
    let bodies = compiled["display"]["data"]["bodies"]
        .as_array()
        .map_or(0, Vec::len);
    let mass = call(
        "analyze",
        &[
            "/p/plate.part",
            &text,
            "{}",
            &json!({ "command": "evaluate.mass-properties" }).to_string(),
        ],
    );
    assert_eq!(mass["ok"], true, "{mass}");
    (
        text,
        bodies,
        mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9,
    )
}

fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 0.05, "{got} mm³, want {want}");
}

fn boss_faces() -> Value {
    json!([
        "feature.boss.face.end",
        "feature.boss.face.side(s2.m1)",
        "feature.boss.face.side(s2.m2)",
        "feature.boss.face.side(s2.m3)",
        "feature.boss.face.side(s2.m4)"
    ])
}

#[test]
fn the_unchanged_plate_has_its_closed_form_volume() {
    let (_, bodies, v) = after(
        "part.body-move",
        json!({ "bodies": ["feature.cube.body"], "dx": 0, "dy": 0, "dz": 0 }),
    );
    assert_eq!(bodies, 2);
    close(v, PLATE_V + CUBE_V);
}

#[test]
fn a_boss_patterned_by_its_faces_repeats_its_material() {
    // One copy 6 mm along y overlaps the seed by 10 x 4 mm, so it adds 10 x 6 x 5 mm.
    let (text, bodies, v) = after(
        "part.pattern",
        json!({ "layout": "table", "seeds": boss_faces(), "table": "b: 0, 6, 0" }),
    );
    assert!(text.contains("faces = ["), "{text}");
    assert_eq!(bodies, 2, "{text}");
    close(v, PLATE_V + CUBE_V + 300.0);
}

#[test]
fn a_pocket_patterned_by_its_faces_cuts_again() {
    let faces = json!([
        "feature.pocket.face.end",
        "feature.pocket.face.side(s3.p1)",
        "feature.pocket.face.side(s3.p2)",
        "feature.pocket.face.side(s3.p3)",
        "feature.pocket.face.side(s3.p4)"
    ]);
    let (_, _, v) = after(
        "part.pattern",
        json!({ "layout": "linear", "seeds": faces, "direction": "origin.y", "spacing": 8, "instances": 2 }),
    );
    close(v, PLATE_V + CUBE_V - 108.0);
}

#[test]
fn a_mirror_of_faces_reflects_the_boss() {
    // About the yz plane the boss's image lies at x < 0, clear of the plate: 500 mm³ more.
    let (_, _, v) = after(
        "part.mirror",
        json!({ "plane": "origin.yz", "seeds": boss_faces() }),
    );
    close(v, PLATE_V + CUBE_V + 500.0);
}

#[test]
fn a_fillet_patterns_with_its_boss() {
    // Pattern the boss 26 mm along x, round one vertical boss edge, then pattern that round
    // the same way: the copy's matching edge (at x = 38 mm, y = 2 mm) is rounded too.
    let base = PLATE.replace(
        "    sketch s3 on",
        "    feature row = pattern(features = [boss], layout = linear(direction = origin.x, spacing = 26 mm, count = 2))\n    feature r1 = fillet(edges = [edges_between(feature.boss.face.side(s2.m1), feature.boss.face.side(s2.m2))], radius = 1 mm)\n    sketch s3 on",
    );
    let request = json!({ "command": "part.pattern", "values": { "layout": "linear", "seeds": ["feature:r1"], "direction": "origin.x", "spacing": 26, "instances": 2 } }).to_string();
    let out = call("edit", &["/p/plate.part", &base, "{}", &request]);
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    let compiled = call("compile", &["/p/plate.part", &text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    let mass = |t: &str| {
        call(
            "analyze",
            &[
                "/p/plate.part",
                t,
                "{}",
                &json!({ "command": "evaluate.mass-properties" }).to_string(),
            ],
        )["data"]["total"]["volume"]
            .as_f64()
            .unwrap()
            * 1e9
    };
    // Each 1 mm round on a 5 mm tall vertical edge removes (1 - pi/4) x 5 mm³.
    let corner = (1.0 - PI / 4.0) * 5.0;
    close(mass(&text), mass(&base) - corner);
}

#[test]
fn a_body_splits_by_a_cylindrical_face() {
    // The pin's cylinder, extended down through the plate, cuts a 3 mm column out of it.
    let (text, bodies, v) = after(
        "part.split",
        json!({ "body": "feature.body.body", "tool": "feature.pin.face.side(s4.c1)" }),
    );
    assert!(text.contains("split("), "{text}");
    assert_eq!(bodies, 3, "{text}");
    close(v, PLATE_V + CUBE_V);
}

#[test]
fn a_body_splits_by_another_body() {
    // The cube cuts the plate's 2 x 2 x 5 mm corner off as its own piece; the cube is kept.
    let (_, bodies, v) = after(
        "part.split",
        json!({ "body": "feature.body.body", "tool": "feature.cube.body" }),
    );
    assert_eq!(bodies, 3);
    close(v, PLATE_V + CUBE_V);
}

#[test]
fn bodies_copy_turn_and_indent() {
    // Two copies of the cube, 20 mm apart along x, then a quarter turn of the cube.
    let (_, bodies, v) = after(
        "part.body-move",
        json!({ "bodies": ["feature.cube.body"], "dx": 20, "dy": 0, "dz": 0, "copy": true, "copies": 2 }),
    );
    assert_eq!(bodies, 4);
    close(v, PLATE_V + 3.0 * CUBE_V);
    let (_, _, v) = after(
        "part.body-move",
        json!({ "bodies": ["feature.cube.body"], "axis": "origin.z", "angle": 90 }),
    );
    close(v, PLATE_V + CUBE_V);
    // The cube pressed into the plate with no clearance removes the 2 x 2 x 5 mm overlap; a
    // 0.5 mm clearance removes more.
    let (_, bodies, v) = after(
        "part.body-indent",
        json!({ "target": "feature.body.body", "tool": "feature.cube.body", "clearance": 0 }),
    );
    assert_eq!(bodies, 2);
    close(v, PLATE_V + CUBE_V - 20.0);
    let (_, _, v2) = after(
        "part.body-indent",
        json!({ "target": "feature.body.body", "tool": "feature.cube.body", "clearance": 0.5 }),
    );
    assert!(v2 < v - 1.0, "{v2} vs {v}");
}
