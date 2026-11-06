//! Direct edits on OCCT through `runmat_cad_call`: each Direct command writes a `direct.*`
//! feature, and the part regenerates with the face moved or the hole healed over.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// A 40 x 20 x 10 mm plate with a 4 mm hole through it.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature h = hole(face = feature.body.face.end, at = { a = point(30 mm, 10 mm, 10 mm) }, diameter = 4 mm, through_all)\n    output plate = { body = feature.body.body, material = \"material.aluminum_6061\" }\n}\n";

/// The part after the command, compiled: its text and its mass report.
fn after(command: &str, values: Value) -> (String, Value) {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/plate.part", PLATE, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    let compiled = call("compile", &["/p/plate.part", &text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"].as_array().into_iter().flatten().filter(|d| d["severity"] == "error").collect();
    assert!(errors.is_empty(), "{command}: {errors:?}\n{text}");
    let mass = call("analyze", &["/p/plate.part", &text, "{}", &json!({ "command": "evaluate.mass-properties" }).to_string()]);
    assert_eq!(mass["ok"], true, "{mass}");
    (text, mass["data"]["total"].clone())
}

#[test]
fn offset_face_grows_the_plate_and_delete_face_heals_the_hole() {
    let volume = |total: &Value| total["volume"].as_f64().unwrap() * 1e9; // mm³
    let hole = std::f64::consts::PI * 2.0 * 2.0 * 10.0;
    // The top face 2 mm up: 40 x 20 x 12 mm less a 12 mm deep hole.
    let (text, total) = after("direct.offset-face", json!({ "faces": ["feature.body.face.end"], "distance": 2 }));
    assert!(text.contains("direct.move_faces("), "{text}");
    let grown = 40.0 * 20.0 * 12.0 - std::f64::consts::PI * 4.0 * 12.0;
    assert!((volume(&total) - grown).abs() < 0.5, "{} mm³, want {grown}", volume(&total));
    // The hole's wall deleted and healed: the solid plate.
    let (text, total) = after("direct.delete-face", json!({ "faces": ["feature.h.face.wall(a)"], "heal": "heal" }));
    assert!(text.contains("direct.delete_faces("), "{text}");
    assert!((volume(&total) - 8000.0).abs() < 0.5, "{} mm³, want 8000 (the hole was {hole:.1})", volume(&total));
}
