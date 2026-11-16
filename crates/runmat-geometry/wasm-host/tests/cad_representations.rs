//! A part's analysis representations on OCCT, through `runmat_cad_call` (the Desktop worker's
//! entry): `simulate.defeature` and `simulate.representation` write the statement, the compile
//! draws the representation the options ask for, and the design's own geometry stays as it is.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// 100 × 60 × 10 mm plate with a 6 mm hole, a 20 mm hole, a 1.5 mm fillet on one top edge and
/// a 0.5 mm chamfer on another.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 60 mm])\n        l3 = line(start = [100 mm, 60 mm], end = [0 mm, 60 mm])\n        l4 = line(start = [0 mm, 60 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature small = hole(face = feature.body.face.end, at = { a = point(20 mm, 30 mm, 10 mm) }, diameter = 6 mm, through_all)\n    feature large = hole(face = feature.body.face.end, at = { a = point(70 mm, 30 mm, 10 mm) }, diameter = 20 mm, through_all)\n    feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)], radius = 1.5 mm)\n    feature bevel = chamfer(edges = [edges_between(feature.body.face.side(s1.l3), feature.body.face.end)], distance = 0.5 mm)\n    region top : face = { select = [feature.body.face.start], expect = exactly_one }\n    output plate = { body = feature.body.body }\n}\n";

fn edit(text: &str, command: &str, values: Value) -> Value {
    let request = json!({ "command": command, "values": values }).to_string();
    call("edit", &["/p/plate.part", text, "{}", &request])
}

fn compile(text: &str, representation: &str) -> Value {
    let options = json!({ "representation": representation }).to_string();
    call("compile", &["/p/plate.part", text, "{}", &options])
}

fn faces(compiled: &Value) -> usize {
    compiled["display"]["data"]["bodies"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["faces"].as_array().unwrap().len())
        .sum()
}

#[test]
fn defeature_removes_small_holes_fillets_and_chamfers_and_lists_them() {
    let out = edit(
        PLATE,
        "simulate.defeature",
        json!({ "name": "simple", "max_fillet": 2.0, "max_hole": 10.0, "max_chamfer": 1.0, "keep_regions": true }),
    );
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(
        text.contains("representation simple : defeatured = { max_fillet = 2 mm, max_hole = 10 mm, max_chamfer = 1 mm }"),
        "{text}"
    );
    let summary = out["summary"].as_str().unwrap();
    assert!(summary.contains("fillet round, r 1.5 mm"), "{summary}");
    assert!(summary.contains("chamfer bevel, 0.5 mm"), "{summary}");
    assert!(summary.contains("hole small, ⌀ 6 mm"), "{summary}");
    assert!(!summary.contains("large"), "the 20 mm hole is above the threshold: {summary}");
    // The design is unchanged; the representation draws fewer faces.
    let full = compile(&text, "full");
    let simple = compile(&text, "simple");
    assert!(full.get("representation").is_none());
    assert_eq!(simple["representation"]["kind"], "defeatured", "{}", simple["representation"]);
    assert!(faces(&simple) < faces(&full), "{} < {}", faces(&simple), faces(&full));
    assert_eq!(full["projection"]["data"]["document"]["features"].as_array().unwrap().len(), 6);
}

#[test]
fn a_thin_plate_has_a_mid_surface_with_its_thickness() {
    let out = edit(PLATE, "simulate.representation", json!({ "kind": "midsurface", "name": "shell" }));
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(text.contains("representation shell : midsurface = {"), "{text}");
    let shell = compile(&text, "shell");
    let report = &shell["representation"];
    assert_eq!(report["kind"], "midsurface", "{report}");
    let t = report["thicknesses"][0].as_f64().unwrap();
    assert!((t - 0.010).abs() < 1e-6, "{t}");
    assert!(faces(&shell) >= 1);
}

#[test]
fn a_bar_is_a_beam_and_a_holed_plate_is_refused_by_name() {
    let bar = "runmat 0.7\nstd 1\n\npart bar {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 10 mm])\n        l3 = line(start = [20 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 300 mm)\n}\n";
    let request = json!({ "command": "simulate.representation", "values": { "kind": "beam", "name": "spar" } }).to_string();
    let out = call("edit", &["/p/bar.part", bar, "{}", &request]);
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    let options = json!({ "representation": "spar" }).to_string();
    let compiled = call("compile", &["/p/bar.part", &text, "{}", &options]);
    let beam = &compiled["representation"]["beam"];
    assert!((beam["length"].as_f64().unwrap() - 0.3).abs() < 1e-6, "{beam}");
    assert!((beam["area"].as_f64().unwrap() - 200e-6).abs() < 1e-9, "{beam}");
    // 20 × 10 mm: b h³ / 12 about each axis.
    let major = 10.0 * 20f64.powi(3) / 12.0 * 1e-12;
    let minor = 20.0 * 10f64.powi(3) / 12.0 * 1e-12;
    assert!((beam["i_major"].as_f64().unwrap() - major).abs() < 1e-3 * major, "{beam}");
    assert!((beam["i_minor"].as_f64().unwrap() - minor).abs() < 1e-3 * minor, "{beam}");
    // The plate's fillet and chamfer break the prism: refused, naming a face, nothing written.
    let refused = edit(PLATE, "simulate.representation", json!({ "kind": "beam", "name": "spar" }));
    assert_eq!(refused["ok"], false, "{refused}");
    assert!(refused["message"].as_str().unwrap().contains("isn't prismatic"), "{refused}");
    // Full writes nothing.
    let full = edit(PLATE, "simulate.representation", json!({ "kind": "full" }));
    assert_eq!(full["ok"], true, "{full}");
    assert_eq!(full["files"], json!([]));
}

#[test]
fn a_component_at_beam_draws_its_parts_beam() {
    let bar = "runmat 0.7\nstd 1\n\npart bar {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 10 mm])\n        l3 = line(start = [20 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 300 mm)\n    representation spar : beam = {}\n}\n";
    let frame = "runmat 0.7\nstd 1\n\nassembly frame {\n    use part bar as a\n}\n";
    let files = json!({ "/p/bar.part": bar }).to_string();
    let pick = |c: &Value| format!("occurrence:{}", c["projection"]["data"]["occurrences"].as_array().unwrap().iter().find(|o| o["path"] == "a").unwrap()["uuid"].as_str().unwrap());
    let compiled = call("compile", &["/p/frame.assembly", frame, &files, "{}"]);
    let request = json!({ "command": "assembly.representation", "values": { "components": [pick(&compiled)], "representation": "beam" } }).to_string();
    let out = call("edit", &["/p/frame.assembly", frame, &files, &request]);
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(text.contains("representation = beam"), "{text}");
    let compiled = call("compile", &["/p/frame.assembly", &text, &files, "{}"]);
    let prototypes = compiled["display"]["data"]["prototypes"].as_array().unwrap();
    let body = &prototypes[0]["part"]["bodies"][0];
    assert_eq!(body["reference"], "representation.spar", "{body}");
    assert_eq!(body["faces"].as_array().unwrap().len(), 2, "the section at both ends");
    assert_eq!(body["edges"].as_array().unwrap().len(), 1, "the centerline");
}

