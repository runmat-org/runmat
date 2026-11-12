//! Direct edits on imported STEP solids through `runmat_cad_call`: Offset face moves a face of
//! a solid read from a STEP file the way it moves a face of a native solid, and the part's volume
//! changes by the face's area times the distance.
//!
//! Fixture: `fixtures/step/nist_ctc_03_asme1_ap242-e2.stp`, NIST's CTC-03 test part (see
//! `fixtures/step/README.md` for its source and licence).

use serde_json::{json, Value};

const CTC_03: &str = include_str!("fixtures/step/nist_ctc_03_asme1_ap242-e2.stp");

const CTC_03_PART: &str = "runmat 0.7\nstd 1\n\npart ctc_03 {\n    feature step = import(source = \"ctc_03.stp\", digest = \"unrecorded\")\n}\n";

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn errors(compiled: &Value) -> Vec<Value> {
    compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .cloned()
        .collect()
}

/// The part's volume in mm³.
fn volume(path: &str, text: &str, files: &Value) -> f64 {
    let mass = call(
        "analyze",
        &[
            path,
            text,
            &files.to_string(),
            &json!({ "command": "evaluate.mass-properties" }).to_string(),
        ],
    );
    assert_eq!(mass["ok"], true, "{mass}");
    mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9
}

/// Runs a command on the part, then compiles the result: its text and volume in mm³.
fn edit(path: &str, text: &str, files: &Value, command: &str, values: Value) -> (String, f64) {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &[path, text, &files.to_string(), &request]);
    assert_eq!(out["ok"], true, "{command}: {out}");
    let edited = out["files"][0]["text"].as_str().unwrap().to_owned();
    let compiled = call("compile", &[path, &edited, &files.to_string(), "{}"]);
    let errs = errors(&compiled);
    assert!(errs.is_empty(), "{command}: {errs:?}\n{edited}");
    let failed: Vec<&Value> = compiled["features"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|f| f["status"] == "failed")
        .collect();
    assert!(failed.is_empty(), "{command}: failed features {failed:?}\n{edited}");
    let v = volume(path, &edited, files);
    (edited, v)
}

/// The faces of the compiled part's display.
fn faces(compiled: &Value) -> Vec<Value> {
    compiled["display"]["data"]["bodies"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|b| b["faces"].as_array().cloned().unwrap_or_default())
        .collect()
}

/// A display face's area in mm², from its triangles.
fn area(face: &Value) -> f64 {
    let p: Vec<f64> = face["positions"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() * 1000.0).collect();
    let t: Vec<usize> = face["triangles"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let at = |i: usize| [p[3 * i], p[3 * i + 1], p[3 * i + 2]];
    t.chunks(3)
        .map(|c| {
            let (a, b, d) = (at(c[0]), at(c[1]), at(c[2]));
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let x = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
        })
        .sum()
}

fn compile(path: &str, text: &str, files: &Value) -> Value {
    let compiled = call("compile", &[path, text, &files.to_string(), "{}"]);
    let errs = errors(&compiled);
    assert!(errs.is_empty(), "{errs:?}");
    compiled
}

/// A 40 x 20 x 10 mm plate.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n}\n";

/// The same plate with its two long top edges rounded at 3 mm: the top face's neighbours are tangent to it.
const ROUNDED: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end), edges_between(feature.body.face.side(s1.l3), feature.body.face.end)], radius = 3 mm)\n}\n";

#[test]
fn offset_face_on_a_native_plate_adds_area_times_distance() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files = json!({ "/p/plate.part": PLATE });
    let (text, v) = edit("/p/plate.part", PLATE, &files, "direct.offset-face", json!({ "faces": ["feature.body.face.end"], "distance": 2 }));
    assert!(text.contains("direct.move_faces("), "{text}");
    assert!((v - 40.0 * 20.0 * 12.0).abs() < 0.5, "{v} mm³");
}

#[test]
fn offset_face_with_rounded_edges_moves_the_rounds_with_the_face() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files = json!({ "/p/plate.part": ROUNDED });
    let before = volume("/p/plate.part", ROUNDED, &files);
    let (_, v) = edit("/p/plate.part", ROUNDED, &files, "direct.offset-face", json!({ "faces": ["feature.body.face.end"], "distance": 2 }));
    // The rounds move up with the face and the side walls grow 2 mm: the plate's full 40 x 20
    // section, 2 mm thick, is added.
    assert!((v - before - 40.0 * 20.0 * 2.0).abs() < 1.0, "{before} → {v} mm³");
}

#[test]
fn offset_face_on_a_plate_read_from_step() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files = json!({ "/p/plate.part": PLATE });
    let request = json!({ "command": "manage.export", "values": { "format": "step_ap242" } });
    let out = call("export", &["/p/plate.part", PLATE, &files.to_string(), &request.to_string()]);
    assert_eq!(out["ok"], true, "{out}");
    let step = out["text"].as_str().unwrap().to_owned();
    let housing = "runmat 0.7\nstd 1\n\npart housing {\n    feature step = import(source = \"plate.step\", digest = \"unrecorded\")\n}\n";
    let files = json!({ "/p/plate.step": step, "/p/housing.part": housing });
    let compiled = compile("/p/housing.part", housing, &files);
    let top = faces(&compiled)
        .into_iter()
        .find(|f| f["label"].as_str().is_some_and(|l| l.contains("face.end")))
        .expect("the plate's end face keeps its name");
    let reference = top["reference"].as_str().unwrap().to_owned();
    let (_, v) = edit("/p/housing.part", housing, &files, "direct.offset-face", json!({ "faces": [reference], "distance": 2 }));
    assert!((v - 40.0 * 20.0 * 12.0).abs() < 0.5, "{v} mm³");
}

/// NIST CTC-03's large top face, 2 mm up: its edges are rounds into the side walls and fillets
/// into the flanges, and they move up with it.
#[test]
fn offset_face_on_ctc_03_top_face_moves_it_2_mm() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files = json!({ "/p/ctc_03.stp": CTC_03, "/p/ctc_03.part": CTC_03_PART });
    let compiled = compile("/p/ctc_03.part", CTC_03_PART, &files);
    let top_name = "feature.step.face(\"Test Case 3 Baseline/face68\")";
    let top = faces(&compiled).into_iter().find(|f| f["reference"] == top_name).expect("face68");
    assert_eq!(top["surface"]["surface"], "plane");
    let top_area = area(&top);
    let z0 = top["positions"][2].as_f64().unwrap() * 1000.0;
    let before = volume("/p/ctc_03.part", CTC_03_PART, &files);
    let (text, v) = edit("/p/ctc_03.part", CTC_03_PART, &files, "direct.offset-face", json!({ "faces": [top_name], "distance": 2 }));
    let grown = v - before;
    // At least the face's own area times 2 mm; the rounds along its edges move up too, adding
    // their footprint (a few percent of the face).
    assert!(grown > 2.0 * top_area * 0.999 && grown < 2.0 * top_area * 1.15, "{before} → {v} mm³ ({grown}), top face {top_area} mm²\n{text}");
    // The moved top face is 2 mm higher, and keeps its name.
    let after = compile("/p/ctc_03.part", &text, &files);
    let moved = faces(&after).into_iter().find(|f| f["reference"] == top_name).expect("face68 after the move");
    let z: Vec<f64> = moved["positions"].as_array().unwrap().chunks(3).map(|p| p[2].as_f64().unwrap() * 1000.0).collect();
    assert!(z.iter().all(|z| (z - z0 - 2.0).abs() < 0.01), "top face z {:?}, was {z0}", &z[..4]);
}

/// When a face can't move, the part's error says why in plain words: a round's own face has
/// tangent flat neighbours on both sides, which can't follow it.
#[test]
fn offset_face_that_cannot_move_names_the_reason() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files = json!({ "/p/plate.part": ROUNDED });
    let compiled = compile("/p/plate.part", ROUNDED, &files);
    let round = faces(&compiled)
        .into_iter()
        .find(|f| f["surface"]["surface"] == "cylinder")
        .expect("a round");
    let request = json!({ "command": "direct.offset-face", "values": { "faces": [round["reference"]], "distance": 1 } }).to_string();
    let out = call("edit", &["/p/plate.part", ROUNDED, &files.to_string(), &request]);
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap();
    let compiled = call("compile", &["/p/plate.part", text, &files.to_string(), "{}"]);
    let messages: Vec<String> = errors(&compiled).iter().map(|d| d["message"].as_str().unwrap_or("").to_owned()).collect();
    assert!(
        messages.iter().any(|m| m.contains("the faces could not be offset: ") && m.contains("tangent")),
        "{messages:?}"
    );
}
