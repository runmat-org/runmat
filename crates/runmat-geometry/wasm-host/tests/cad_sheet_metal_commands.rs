//! Sheet Metal commands on a plate from the canvas, on OCCT through `runmat_cad_call`: miter
//! flange, break corner, unfold and fold. Each command writes its `sheet_metal.*` feature from
//! the form's values and the part regenerates; volumes and extents check the geometry.

use serde_json::{json, Value};

/// An 80 x 50 mm rectangle on the XY plane; sketch `m` on the YZ plane is a miter flange's
/// profile: 15 mm up from the top of a 1.5 mm plate at the start of l1.
const SKETCHES: &str = "runmat 0.7\nstd 1\n\npart sheet {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    sketch m on origin.yz {\n        a = line(start = [0 mm, 1.5 mm], end = [0 mm, 16.5 mm])\n    }\n}\n";

fn call(function: &str, args: &[&str]) -> Value {
    runmat_geometry_wasm_host::cad::install_occt();
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn edit(text: &str, command: &str, values: Value) -> String {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/sheet.part", text, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}\n{text}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

/// The compiled part, which must have no errors.
fn compile(text: &str) -> Value {
    let compiled = call("compile", &["/p/sheet.part", text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"].as_array().into_iter().flatten().filter(|d| d["severity"] == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    compiled
}

fn volume(text: &str) -> f64 {
    compile(text);
    let mass = call("analyze", &["/p/sheet.part", text, "{}", &json!({ "command": "evaluate.mass-properties" }).to_string()]);
    assert_eq!(mass["ok"], true, "{mass}");
    mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9
}

/// The folded model's extents in mm (flat patterns left out).
fn extents(text: &str) -> [f64; 3] {
    let b = &compile(text)["display"]["data"]["bounds"];
    [0, 1, 2].map(|k| (b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap()) * 1e3)
}

/// Every edge reference of the compiled part's bodies.
fn edge_names(text: &str) -> Vec<String> {
    let compiled = compile(text);
    compiled["display"]["data"]["bodies"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|b| b["edges"].as_array().cloned().unwrap_or_default())
        .filter_map(|e| e["reference"].as_str().map(str::to_owned))
        .collect()
}

/// The edge whose reference names every one of `parts`.
fn edge(text: &str, parts: &[&str]) -> String {
    let names = edge_names(text);
    names
        .iter()
        .find(|n| parts.iter().all(|p| n.contains(p)))
        .unwrap_or_else(|| panic!("no edge naming {parts:?} in {names:#?}"))
        .clone()
}

fn plate() -> String {
    edit(SKETCHES, "sheet-metal.base-flange", json!({ "profile": ["sketch:s1"], "thickness": 1.5, "radius": 1, "bend_rule": "k_factor", "k": 0.44 }))
}

fn near(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: {got:.3}, want {want:.3}");
}

#[test]
fn a_miter_flange_runs_along_the_plates_top_edges() {
    let plate = plate();
    near(volume(&plate), 80.0 * 50.0 * 1.5, 1.0, "the plate");
    let top = |side: &str| edge(&plate, &["sheet(plate, ordinal = 1)", side]);
    let text = edit(&plate, "sheet-metal.miter-flange", json!({ "profile": ["sketch:m"], "edges": [top("sheet_edge(l1)"), top("sheet_edge(l2)")], "gap": 0 }));
    assert!(text.contains("sheet_metal.miter_flange("), "{text}");
    // The wall stands bend radius + thickness out from each side and 15 mm above the plate.
    let e = extents(&text);
    near(e[0], 80.0 + 2.5, 0.2, "x");
    near(e[1], 50.0 + 2.5, 0.2, "y");
    near(e[2], 16.5, 0.2, "z");
}

#[test]
fn a_broken_corner_takes_its_chamfer_off_the_plate() {
    let plate = plate();
    let corner = edge(&plate, &["sheet_edge(l1)", "sheet_edge(l2)"]);
    let text = edit(&plate, "sheet-metal.break-corner", json!({ "edges": [corner], "break": "chamfer", "distance": 4 }));
    assert!(text.contains("sheet_metal.break_corner("), "{text}");
    // A 4 x 4 mm triangle off one corner, through the 1.5 mm sheet.
    near(volume(&text), 6000.0 - 0.5 * 4.0 * 4.0 * 1.5, 0.5, "chamfered");
    let text = edit(&plate, "sheet-metal.break-corner", json!({ "edges": [corner], "break": "fillet", "distance": 4 }));
    near(volume(&text), 6000.0 - (16.0 - std::f64::consts::PI * 4.0) * 1.5, 0.5, "rounded");
}

/// Every face reference of the compiled part's bodies.
fn face_names(text: &str) -> Vec<String> {
    compile(text)["display"]["data"]["bodies"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|b| b["faces"].as_array().cloned().unwrap_or_default())
        .filter_map(|f| f["reference"].as_str().map(str::to_owned))
        .collect()
}

#[test]
fn unfold_flattens_a_bent_sheet_and_fold_bends_it_back() {
    // An L section, 30 mm up and 50 mm along, swept 40 mm: one bend.
    let chain = "runmat 0.7\nstd 1\n\npart sheet {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 30 mm], end = [0 mm, 0 mm])\n        l2 = line(start = [0 mm, 0 mm], end = [50 mm, 0 mm])\n    }\n}\n";
    let bent = edit(chain, "sheet-metal.base-flange", json!({ "profile": ["sketch:s1"], "thickness": 1.5, "radius": 1, "depth": 40, "bend_rule": "k_factor", "k": 0.44 }));
    let before = extents(&bent);
    let faces = face_names(&bent);
    let fixed = faces
        .iter()
        .find(|n| n.contains("sheet(l2"))
        .unwrap_or_else(|| panic!("no face of l2's flat in {faces:#?}"))
        .clone();
    let unfolded = edit(&bent, "sheet-metal.unfold", json!({ "fixed": [fixed] }));
    assert!(unfolded.contains("sheet_metal.unfold("), "{unfolded}");
    let flat = extents(&unfolded);
    // Flat: one extent is the thickness, and the developed length passes the 50 mm flat.
    let thin = flat.iter().cloned().fold(f64::INFINITY, f64::min);
    near(thin, 1.5, 0.01, "flat");
    assert!(flat.iter().any(|e| *e > 70.0), "the 30 mm flat lies out along the 50 mm one: {flat:?}");
    let folded = edit(&unfolded, "sheet-metal.fold", json!({}));
    assert!(folded.contains("sheet_metal.fold("), "{folded}");
    let again = extents(&folded);
    for k in 0..3 {
        near(again[k], before[k], 1e-3, "folded back");
    }
}
