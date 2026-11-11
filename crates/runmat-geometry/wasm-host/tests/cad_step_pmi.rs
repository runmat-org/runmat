//! A part's semantic PMI through STEP AP242 on OCCT: exported with the exact B-rep and
//! linked to the named faces, conforming to the AP242 PMI entity definitions, read back
//! item for item, and imported as the PMI of a part built on the STEP file.

use runmat_cad::pmi::{Characteristic, MaterialCondition};
use runmat_cad::step::p21::parse;
use runmat_cad::step::pmi::{read, schema};
use serde_json::Value;

const BRACKET: &str = r#"runmat 0.7
std 1

part bracket {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    datum A = { on = feature.body.face.start }
    datum B = { on = feature.body.face.side(s.l1) }
    tolerance top_flat = { characteristic = flatness, on = [feature.body.face.end], tolerance = 0.02 mm }
    tolerance top_parallel = { characteristic = parallelism, on = [feature.body.face.end], tolerance = 0.05 mm, datums = [{ label = "A" }] }
    tolerance side_square = { characteristic = perpendicularity, on = [feature.body.face.side(s.l2)], tolerance = 0.1 mm, modifier = mmc, datums = [{ label = "A" }, { label = "B", modifier = mmc }] }
    size thickness = { on = [feature.body.face.start, feature.body.face.end], size = 10 mm ± [0.1 mm, 0.05 mm] }
    location width = { from = feature.body.face.side(s.l4), to = feature.body.face.side(s.l2), distance = 80 mm ± 0.2 mm }
}
"#;

fn call(path: &str, text: &str, files: &Value) -> Value {
    let request =
        serde_json::json!({ "fn": "compile", "args": [path, text, files.to_string(), "{}"] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

#[test]
fn part_pmi_exports_as_ap242_semantic_pmi_and_imports_back() {
    runmat_geometry_wasm_host::cad::install_occt();
    let step = runmat_cad_wasm::exchange::part_step("/p/bracket.part", BRACKET)
        .expect("exports")
        .expect("OCCT writes STEP");
    let text = String::from_utf8(step).unwrap();
    assert!(
        text.contains("ADVANCED_FACE('feature.body.face.end'"),
        "faces are named"
    );
    let data = parse(&text).expect("parses");
    assert_eq!(schema::check(&data), Vec::<String>::new());

    let pmi = read(&text).expect("reads");
    let labels: Vec<&str> = pmi.datums.iter().map(|d| d.label.as_str()).collect();
    assert_eq!(labels, ["A", "B"]);
    assert_eq!(pmi.datums[0].faces, ["feature.body.face.start"]);
    assert_eq!(pmi.tolerances.len(), 3, "{:?}", pmi.tolerances);
    let flat = &pmi.tolerances[0];
    assert_eq!(
        (flat.characteristic, flat.faces.as_slice()),
        (
            Characteristic::Flatness,
            &["feature.body.face.end".to_string()][..]
        )
    );
    assert!((flat.value - 0.02).abs() < 1e-9);
    let square = &pmi.tolerances[2];
    assert_eq!(square.modifier, MaterialCondition::Mmc);
    assert_eq!(square.datums.len(), 2);
    assert_eq!(square.datums[1].modifier, MaterialCondition::Mmc);
    let size = &pmi.sizes[0];
    assert_eq!(size.faces.len(), 2);
    assert!(
        (size.limits.nominal - 10.0).abs() < 1e-9
            && (size.limits.lower + 0.05).abs() < 1e-9
            && (size.limits.upper - 0.1).abs() < 1e-9,
        "{size:?}"
    );
    let width = &pmi.locations[0];
    assert!((width.limits.nominal - 80.0).abs() < 1e-9, "{width:?}");

    // Import: a part on the STEP file with the PMI on its faces, which resolves on OCCT.
    let statements = runmat_cad_wasm::exchange::pmi::part_statements(&pmi, "step");
    assert!(
        statements.contains("size = 10 mm ± [0.1 mm, 0.05 mm]")
            && statements.contains("distance = 80 mm ± 0.2 mm"),
        "{statements}"
    );
    let housing = format!(
        "runmat 0.7\nstd 1\n\npart housing {{\n    feature step = import(source = \"bracket.step\", digest = \"unrecorded\")\n{statements}}}\n"
    );
    let files = serde_json::json!({ "/p/bracket.step": text, "/p/housing.part": housing });
    let v = call("/p/housing.part", &housing, &files);
    let errors: Vec<&Value> = v["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{housing}");
    let doc = &v["projection"]["data"]["document"];
    assert_eq!(
        doc["tolerances"].as_array().map(Vec::len),
        Some(3),
        "{housing}"
    );
    assert_eq!(doc["datums"].as_array().map(Vec::len), Some(2));
    assert_eq!(doc["sizes"].as_array().map(Vec::len), Some(1));
    assert_eq!(doc["locations"].as_array().map(Vec::len), Some(1));
}

#[test]
fn a_size_that_does_not_evaluate_refuses_the_step_export() {
    runmat_geometry_wasm_host::cad::install_occt();
    // `+0.1` here is arithmetic on a length, not a tolerance: the size has no value.
    let bad = BRACKET.replace("10 mm ± [0.1 mm, 0.05 mm]", "10 mm + 0.1");
    let err = runmat_cad_wasm::exchange::part_step("/p/bracket.part", &bad).expect_err("refused");
    assert!(err.contains("thickness"), "{err}");
}
