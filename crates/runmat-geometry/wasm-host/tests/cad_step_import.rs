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

const PAIR: &str = r#"runmat 0.7
std 1

assembly pair {
    use part plate from "plate.part" as lower
    use part plate from "plate.part" as upper {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 10 mm] } }
    }
}
"#;

const STACK: &str = r#"runmat 0.7
std 1

assembly stack {
    use assembly pair from "pair.assembly" as base
    use assembly pair from "pair.assembly" as top {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 20 mm] } }
    }
}
"#;

/// An assembly of a sub-assembly placed twice keeps its hierarchy in AP242 (the sub-assembly
/// is one product used twice, holding its two plates), names its faces, and reads back at the
/// size it was written.
#[test]
fn an_assembly_exports_its_hierarchy_and_face_names_as_ap242() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files: std::collections::BTreeMap<String, String> = [
        ("/p/plate.part", PLATE),
        ("/p/pair.assembly", PAIR),
        ("/p/stack.assembly", STACK),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let step = runmat_cad_wasm::exchange::assembly_step("/p/stack.assembly", STACK, &files)
        .expect("exports")
        .expect("OCCT writes STEP");
    let text = String::from_utf8(step).expect("STEP is text");
    assert!(text.contains("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING"), "not AP242");
    // stack → base, top (2 uses of pair); pair → lower, upper (2 uses of plate): 4 usages.
    let usages = text.matches("NEXT_ASSEMBLY_USAGE_OCCURRENCE").count();
    assert_eq!(usages, 4, "usages {usages}");
    for product in ["'stack'", "'pair'", "'plate'"] {
        assert!(text.contains(&format!("PRODUCT({product}")), "no product {product}");
    }
    assert!(text.contains("feature.body.face.end"), "face names are not written");

    // Read back: the stack is 80 x 50 x 40 mm (four 10 mm plates).
    let mut files = files;
    files.insert("/p/stack.step".into(), text);
    let housing = "runmat 0.7\nstd 1\n\npart housing {\n    feature step = import(source = \"stack.step\", digest = \"unrecorded\")\n}\n";
    files.insert("/p/housing.part".into(), housing.into());
    let v = call("/p/housing.part", housing, &serde_json::to_value(&files).unwrap());
    let b = &v["display"]["data"]["bounds"];
    let ext = |k: usize| b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap();
    assert!((ext(0) - 0.08).abs() < 1e-6 && (ext(1) - 0.05).abs() < 1e-6 && (ext(2) - 0.04).abs() < 1e-6, "bounds {b}");
}
