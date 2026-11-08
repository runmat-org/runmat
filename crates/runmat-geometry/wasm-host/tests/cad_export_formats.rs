//! The CAD host's Export call (`export` through `runmat_cad_call`, the browser worker's entry) on
//! OCCT, one test per format, each file read back: STL and glTF by their binary layouts, 3MF by
//! the design tools' reader, IGES by OCCT's exact import (its volume), and DXF of a face by the design
//! tools' DXF reader. The plate is 80 x 50 x 10 mm with a 10 mm hole.

use base64::Engine;
use runmat_geometry_core::{ExactMassPropertiesEvaluator, GeometryEvaluationControl, GeometryEvaluationError};
use serde_json::{json, Value};

const PLATE: &str = r#"runmat 0.7
std 1

part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
        c = circle(center = [40 mm, 25 mm], radius = 5 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
}
"#;

fn call(f: &str, args: [&str; 4]) -> Value {
    let request = json!({ "fn": f, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn export(values: Value) -> Value {
    runmat_geometry_wasm_host::cad::install_occt();
    let request = json!({ "command": "manage.export", "values": values }).to_string();
    let out = call("export", ["/p/plate.part", PLATE, "{}", &request]);
    assert_eq!(out["ok"], true, "{out}");
    out
}

fn bytes(out: &Value) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(out["base64"].as_str().expect("binary")).expect("base64")
}

/// Extents (max - min) of points, per axis.
fn extents(points: impl Iterator<Item = [f64; 3]>) -> [f64; 3] {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
}

fn near(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() < tol)
}

#[test]
fn stl_is_binary_millimetres_of_the_whole_plate() {
    let out = export(json!({ "format": "stl", "resolution": "fine" }));
    assert_eq!(out["name"], "plate.stl");
    let b = bytes(&out);
    let n = u32::from_le_bytes(b[80..84].try_into().unwrap()) as usize;
    assert_eq!(b.len(), 84 + 50 * n, "binary STL length");
    assert!(n > 20, "{n} triangles (the hole needs more than a box's 12)");
    let f = |at: usize| f32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as f64;
    let pts = (0..n).flat_map(|t| (0..3).map(move |v| 84 + 50 * t + 12 + 12 * v)).map(|at| [f(at), f(at + 4), f(at + 8)]);
    let e = extents(pts);
    assert!(near(e, [80.0, 50.0, 10.0], 1e-3), "{e:?}");
}

#[test]
fn three_mf_reads_back_as_one_millimetre_object() {
    let out = export(json!({ "format": "3mf" }));
    assert_eq!(out["name"], "plate.3mf");
    let model = runmat_design_tools::formats::three_mf::read::read(&bytes(&out)).expect("3MF reads back");
    assert_eq!(model.unit, "millimeter");
    let meshes: Vec<_> = model
        .objects
        .iter()
        .filter_map(|o| match &o.content {
            runmat_design_tools::formats::three_mf::Content::Mesh { vertices, triangles } => Some((vertices, triangles)),
            _ => None,
        })
        .collect();
    assert_eq!(meshes.len(), 1);
    let e = extents(meshes[0].0.iter().copied());
    assert!(near(e, [80.0, 50.0, 10.0], 1e-6), "{e:?}");
}

#[test]
fn gltf_is_a_glb_in_metres_with_y_up() {
    let out = export(json!({ "format": "gltf" }));
    assert_eq!(out["name"], "plate.glb");
    let b = bytes(&out);
    assert_eq!(&b[0..4], b"glTF");
    let json_len = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
    let doc: Value = serde_json::from_slice(&b[20..20 + json_len]).expect("glTF JSON");
    let a = &doc["accessors"][0];
    let e: Vec<f64> = (0..3).map(|k| a["max"][k].as_f64().unwrap() - a["min"][k].as_f64().unwrap()).collect();
    // Model z (10 mm) is glTF y.
    assert!(near([e[0], e[1], e[2]], [0.08, 0.01, 0.05], 1e-6), "{e:?}");
}

#[test]
fn iges_reads_back_as_the_plates_solid() {
    let out = export(json!({ "format": "iges" }));
    assert_eq!(out["name"], "plate.igs");
    let text = out["text"].as_str().expect("IGES is text");
    let first = text.lines().next().unwrap_or("");
    assert!(first.len() >= 73 && &first[72..73] == "S", "IGES start section: {first:?}");
    // OCCT's exact import (the browser host reads only STEP from a request's files): one solid
    // whose volume is the plate's, 80 x 50 x 10 mm less the 10 mm hole.
    let imported = runmat_geometry_io::import_exact_cad(
        "plate.igs",
        text.as_bytes(),
        runmat_geometry_io::GeometryFormat::Iges,
        &runmat_geometry_io::ExactCadImportOptions::default(),
        &runmat_geometry_io::GeometryImportContext::new(),
    )
    .expect("IGES imports back");
    assert_eq!(imported.topology.bodies.len(), 1);
    let volume = runmat_geometry_io::OcctExactEvaluator::new(&imported)
        .unwrap()
        .mass_properties(&imported.topology.bodies[0].mass_properties_evaluator_id, &Unlimited)
        .unwrap()
        .volume_m3
        .canonical_value();
    let plate = (80.0 * 50.0 * 10.0 - std::f64::consts::PI * 25.0 * 10.0) * 1e-9;
    assert!((volume - plate).abs() < 1e-9 * 0.5, "{volume} m³, want {plate}");
}

#[test]
fn dxf_of_the_top_face_is_its_outline_and_hole() {
    let out = export(json!({ "format": "dxf_face", "face": ["feature.body.face.end"] }));
    assert_eq!(out["name"], "plate-face.dxf");
    let sheet = runmat_design_tools::exchange::dxf::read(out["text"].as_str().expect("DXF is text")).expect("DXF reads back");
    let loops: Vec<Vec<[f64; 2]>> = sheet
        .entities
        .iter()
        .filter_map(|e| match &e.shape {
            runmat_design_tools::formats::sheet::Shape::Polyline { points, closed: true } => Some(points.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(loops.len(), 2, "the outline and the hole");
    let ext = |l: &Vec<[f64; 2]>| extents(l.iter().map(|p| [p[0], p[1], 0.0]));
    let mut sizes: Vec<[f64; 3]> = loops.iter().map(ext).collect();
    sizes.sort_by(|a, b| b[0].total_cmp(&a[0]));
    let outline = [sizes[0][0].max(sizes[0][1]), sizes[0][0].min(sizes[0][1])];
    assert!((outline[0] - 80.0).abs() < 1e-3 && (outline[1] - 50.0).abs() < 1e-3, "{sizes:?}");
    assert!((sizes[1][0] - 10.0).abs() < 0.05 && (sizes[1][1] - 10.0).abs() < 0.05, "hole {sizes:?}");
}

#[test]
fn dxf_of_a_curved_face_is_refused() {
    runmat_geometry_wasm_host::cad::install_occt();
    // The hole's wall, by the reference the canvas picks it with.
    let v = call("compile", ["/p/plate.part", PLATE, "{}", "{}"]);
    let faces = v["display"]["data"]["bodies"][0]["faces"].as_array().cloned().unwrap_or_default();
    let wall = faces
        .iter()
        .filter_map(|f| f["reference"].as_str())
        .find(|r| r.contains("(c)") || r.ends_with(".c)"))
        .unwrap_or_else(|| panic!("no hole wall in {:?}", faces.iter().map(|f| f["reference"].clone()).collect::<Vec<_>>()))
        .to_owned();
    let request = json!({ "command": "manage.export", "values": { "format": "dxf_face", "face": [wall] } }).to_string();
    let out = call("export", ["/p/plate.part", PLATE, "{}", &request]);
    assert_eq!(out["ok"], false, "{out}");
    assert!(out["message"].as_str().unwrap().contains("planar"), "{out}");
}

struct Unlimited;

impl GeometryEvaluationControl for Unlimited {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

#[test]
fn mesh_resolution_sets_how_finely_the_hole_is_cut() {
    let triangles = |resolution: &str| {
        let b = bytes(&export(json!({ "format": "stl", "resolution": resolution })));
        u32::from_le_bytes(b[80..84].try_into().unwrap())
    };
    let (fine, standard, coarse) = (triangles("fine"), triangles("standard"), triangles("coarse"));
    assert!(fine > standard && standard > coarse, "fine {fine}, standard {standard}, coarse {coarse}");
    // A resolution the form doesn't offer is refused.
    runmat_geometry_wasm_host::cad::install_occt();
    let request = json!({ "command": "manage.export", "values": { "format": "stl", "resolution": "draft" } }).to_string();
    let out = call("export", ["/p/plate.part", PLATE, "{}", &request]);
    assert_eq!(out["ok"], false, "{out}");
}
