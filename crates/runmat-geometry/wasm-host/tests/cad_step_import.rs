//! STEP in and out on OCCT through `runmat_cad_call`, the browser worker's entry: a part
//! exported as exact AP242, then imported into another part from the request's files (the
//! browser host has no project folder, so the STEP text travels with the request).

use runmat_cad::regen::kernel::{ExchangeDocument, ExchangeFormat, ExchangePart};
use runmat_cad_wasm::runtime::regenerate_part;
use serde_json::Value;

const PLATE: &str = r#"runmat 0.7
std 1

part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
}
"#;

const IMPORTED: &str = r#"runmat 0.7
std 1

part housing {
    feature step = import(source = "plate.step", digest = "unrecorded")
}
"#;

fn call(path: &str, text: &str, files: &Value) -> Value {
    let request = serde_json::json!({ "fn": "compile", "args": [path, text, files.to_string(), "{}"] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

#[test]
fn a_part_exports_exact_ap242_and_imports_back_from_the_requests_files() {
    runmat_geometry_wasm_host::cad::install_occt();
    let part = regenerate_part("/p/plate.part", PLATE).expect("compiles");
    let doc = ExchangeDocument {
        name: "plate".into(),
        parts: vec![ExchangePart {
            name: "plate".into(),
            bodies: part.result.state.bodies.iter().map(|b| &b.handle).collect(),
            color: None,
            material: None,
            face_names: Vec::new(),
            face_colors: Vec::new(),
        }],
        occurrences: Vec::new(),
    };
    let step = part.kernel.export(&doc, ExchangeFormat::Ap242).expect("OCCT writes STEP");
    let text = String::from_utf8(step).expect("STEP is text");
    let header = &text[..text.find("ENDSEC").unwrap_or(text.len())];
    assert!(header.contains("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING"), "not AP242: {header}");
    assert!(text.contains("ADVANCED_FACE") && !text.contains("FACETED_BREP"), "not an exact B-rep");

    let files = serde_json::json!({ "/p/plate.step": text, "/p/housing.part": IMPORTED });
    let v = call("/p/housing.part", IMPORTED, &files);
    let errors: Vec<&Value> = v["diagnostics"].as_array().into_iter().flatten().filter(|d| d["severity"] == "error").collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(v["kernel"], "occt");
    let bodies = v["display"]["data"]["bodies"].as_array().cloned().unwrap_or_default();
    assert_eq!(bodies.len(), 1, "{}", v["display"]["data"]["problems"]);
    let b = &v["display"]["data"]["bounds"];
    let ext = |k: usize| b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap();
    assert!((ext(0) - 0.08).abs() < 1e-6 && (ext(1) - 0.05).abs() < 1e-6 && (ext(2) - 0.01).abs() < 1e-6, "bounds {b}");
}
