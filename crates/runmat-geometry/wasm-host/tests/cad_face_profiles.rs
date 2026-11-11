//! Profiles from faces on OCCT through `runmat_cad_call`: Extrude and Revolve take a planar face
//! as the profile and build a prism or revolution of its boundary, holes included, with the
//! usual end conditions. Volumes are checked against closed forms.

use serde_json::{json, Value};
use std::f64::consts::PI;

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// A 40 x 20 x 10 mm plate with a 4 mm hole through it, and a 10 x 10 x 5 mm boss joined on top.
/// No output is named, so the volume counts every body, new bodies included.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature h = hole(face = feature.body.face.end, at = { a = point(30 mm, 10 mm, 10 mm) }, diameter = 4 mm, through_all)\n    sketch s2 on feature.body.face.end {\n        m1 = line(start = [5 mm, 5 mm], end = [15 mm, 5 mm])\n        m2 = line(start = [15 mm, 5 mm], end = [15 mm, 15 mm])\n        m3 = line(start = [15 mm, 15 mm], end = [5 mm, 15 mm])\n        m4 = line(start = [5 mm, 15 mm], end = [5 mm, 5 mm])\n    }\n    feature boss = extrude(profile = sketch.s2, distance = 5 mm, mode = add)\n}\n";

/// The part after the command: its text and its total volume (mm³).
fn after(command: &str, values: Value) -> (String, f64) {
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
        mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9,
    )
}

const HOLE: f64 = PI * 4.0; // the 4 mm hole's section, mm²
const BASE: f64 = 40.0 * 20.0 * 10.0 - HOLE * 10.0 + 10.0 * 10.0 * 5.0;

fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 0.05, "{got} mm³, want {want}");
}

#[test]
fn a_face_extrudes_with_its_hole() {
    // The plate's top face (the hole cuts it; the boss covers a 10 x 10 square of it) up 5 mm.
    let (text, v) = after(
        "part.extrude",
        json!({ "profile": ["feature.body.face.end"], "op": "add", "distance": 5 }),
    );
    assert!(
        text.contains("extrude(profile = feature.body.face.end"),
        "{text}"
    );
    close(v, BASE + (800.0 - 100.0 - HOLE) * 5.0);
    // As a new body: the same prism, separate.
    let (_, v) = after(
        "part.extrude",
        json!({ "profile": ["feature.body.face.end"], "op": "new", "distance": 5 }),
    );
    close(v, BASE + (800.0 - 100.0 - HOLE) * 5.0);
}

#[test]
fn a_face_cuts_through_all_and_to_a_depth() {
    // The boss's top face removed through all: a 10 x 10 hole through boss and plate.
    let (_, v) = after(
        "part.extrude",
        json!({ "profile": ["feature.boss.face.end"], "op": "remove", "end": "through_all" }),
    );
    close(v, BASE - 100.0 * 15.0);
    let (_, v) = after(
        "part.extrude",
        json!({ "profile": ["feature.boss.face.end"], "op": "remove", "distance": 3 }),
    );
    close(v, BASE - 100.0 * 3.0);
    // A thin wall applies to a face too: a 1 mm wall along the 10 x 10 face's edges, 4 mm up,
    // on one side of them: 36 mm² inside the edges or 44 mm² outside.
    let (_, v) = after(
        "part.extrude",
        json!({ "profile": ["feature.boss.face.end"], "op": "new", "distance": 4, "thin": true, "thin_thickness": 1, "thin_side": "one" }),
    );
    assert!(
        (v - (BASE + 36.0 * 4.0)).abs() < 0.05 || (v - (BASE + 44.0 * 4.0)).abs() < 0.05,
        "{v}"
    );
}

#[test]
fn a_face_revolves_about_an_edge_in_its_plane() {
    // The boss's top face, 10 x 10 at z = 15, a quarter turn about its edge along x at y = 5:
    // Pappus, area 100 at 5 mm from the axis.
    let edge = "edges_between(feature.boss.face.side(s2.m1), feature.boss.face.end)";
    let (text, v) = after(
        "part.revolve",
        json!({ "profile": ["feature.boss.face.end"], "axis": [edge], "angle": 90, "op": "new" }),
    );
    assert!(
        text.contains("revolve1 = revolve(") && text.contains("profile = feature.boss.face.end"),
        "{text}"
    );
    close(v, BASE + 100.0 * 5.0 * PI / 2.0);
}
