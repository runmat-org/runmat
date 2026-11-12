//! Direct edits on OCCT through `runmat_cad_call`: each Direct command writes a `direct.*`
//! feature, and the part regenerates with the face moved or the hole healed over.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// A 40 x 20 x 10 mm plate with a 4 mm hole through it.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature h = hole(face = feature.body.face.end, at = { a = point(30 mm, 10 mm, 10 mm) }, diameter = 4 mm, through_all)\n    output plate = { body = feature.body.body, material = \"material.aluminum_6061\" }\n}\n";

/// The same plate with its hole cut through a profile of two half circles, so the hole's wall is
/// two faces, as a hole read from a STEP file often is.
const SPLIT_HOLE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    sketch hs on origin.xy {\n        a1 = arc(center = [30 mm, 10 mm], start = [32 mm, 10 mm], end = [28 mm, 10 mm])\n        a2 = arc(center = [30 mm, 10 mm], start = [28 mm, 10 mm], end = [32 mm, 10 mm])\n    }\n    feature bore = extrude(profile = sketch.hs, through_all, mode = cut)\n    output plate = { body = feature.body.body, material = \"material.aluminum_6061\" }\n}\n";

/// The part after the command, compiled: its text and its mass report.
fn after(command: &str, values: Value) -> (String, Value) {
    after_on(PLATE, command, values)
}

fn after_on(part: &str, command: &str, values: Value) -> (String, Value) {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/plate.part", part, "{}", &request]);
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
    (text, mass["data"]["total"].clone())
}

#[test]
fn offset_face_grows_the_plate_and_delete_face_heals_the_hole() {
    let volume = |total: &Value| total["volume"].as_f64().unwrap() * 1e9; // mm³
    let hole = std::f64::consts::PI * 2.0 * 2.0 * 10.0;
    // The top face 2 mm up: 40 x 20 x 12 mm less a 12 mm deep hole.
    let (text, total) = after(
        "direct.offset-face",
        json!({ "faces": ["feature.body.face.end"], "distance": 2 }),
    );
    assert!(text.contains("direct.move_faces("), "{text}");
    let grown = 40.0 * 20.0 * 12.0 - std::f64::consts::PI * 4.0 * 12.0;
    assert!(
        (volume(&total) - grown).abs() < 0.5,
        "{} mm³, want {grown}",
        volume(&total)
    );
    // The hole's wall deleted and healed: the solid plate.
    let (text, total) = after(
        "direct.delete-face",
        json!({ "faces": ["feature.h.face.wall(a)"], "heal": "heal" }),
    );
    assert!(text.contains("direct.delete_faces("), "{text}");
    assert!(
        (volume(&total) - 8000.0).abs() < 0.5,
        "{} mm³, want 8000 (the hole was {hole:.1})",
        volume(&total)
    );
}

/// Delete face on one face of a hole whose wall is two faces removes the whole hole: the part
/// regenerates as the solid plate, through the shared dispatcher (`runmat_cad_call` edit).
#[test]
fn delete_face_on_half_of_a_split_hole_wall_removes_the_hole() {
    let volume = |total: &Value| total["volume"].as_f64().unwrap() * 1e9; // mm³
    let mass = call(
        "analyze",
        &[
            "/p/plate.part",
            SPLIT_HOLE,
            "{}",
            &json!({ "command": "evaluate.mass-properties" }).to_string(),
        ],
    );
    assert_eq!(mass["ok"], true, "{mass}");
    let before = mass["data"]["total"].clone();
    let hole = std::f64::consts::PI * 2.0 * 2.0 * 10.0;
    assert!(
        (volume(&before) - (8000.0 - hole)).abs() < 0.5,
        "{} mm³, want the plate with its hole",
        volume(&before)
    );
    let (text, total) = after_on(
        SPLIT_HOLE,
        "direct.delete-face",
        json!({ "faces": ["feature.bore.face.side(hs.a1)"], "heal": "heal" }),
    );
    assert!(text.contains("direct.delete_faces("), "{text}");
    assert!(
        (volume(&total) - 8000.0).abs() < 0.5,
        "{} mm³, want 8000: the hole is gone",
        volume(&total)
    );
}

/// Delete face on the plate's top face: the side walls stand parallel and the hole's wall runs
/// along them, so no neighbour can close over the gap. The edit is written, and the part says
/// what went wrong on that line rather than keeping the face and reporting success.
#[test]
fn delete_face_that_nothing_can_close_over_is_a_named_failure() {
    let request = json!({
        "command": "direct.delete-face",
        "values": { "faces": ["feature.body.face.end"], "heal": "heal" }
    })
    .to_string();
    let out = call("edit", &["/p/plate.part", PLATE, "{}", &request]);
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    let line = "feature delete_face1 = direct.delete_faces(faces = [feature.body.face.end])";
    assert!(text.contains(line), "{text}");
    let compiled = call("compile", &["/p/plate.part", &text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert_eq!(errors.len(), 1, "{errors:?}");
    let message = errors[0]["message"].as_str().unwrap();
    assert!(message.contains("pick every face of the hole, boss or pocket"), "{message}");
    // The diagnostic points at the delete feature's line.
    let start = errors[0]["start"].as_u64().unwrap() as usize;
    let end = errors[0]["end"].as_u64().unwrap() as usize;
    assert!(text[start..end].contains("direct.delete_faces("), "{}", &text[start..end]);
}
