//! A board as STEP AP242 on OCCT (`board.export`), through `runmat_cad_call`'s `export`, the
//! Desktop's path: the drone's flight controller exports one named product per solid (the board,
//! its solder mask, each fitted component, and the outer copper when asked), and the file reads
//! back through the STEP importer with that many solids, under those names, inside the box the
//! board outline and its tallest part make. A package whose STEP model is in the project places
//! the model in place of its body box.

use runmat_cad::regen::kernel::{ExchangeDocument, ExchangeFormat, ExchangePart};
use runmat_cad_wasm::runtime::regenerate_part;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const FC: &str = "/p/drone/electrical/flight_controller/flight_controller.circuit.layout";

fn project() -> BTreeMap<String, String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runmat-cad/wasm/tests/drone");
    let mut all = BTreeMap::new();
    read_tree(&dir, "/p/drone", &mut all);
    assert!(all.contains_key(FC), "the drone project is checked in at {dir:?}");
    all
}

fn read_tree(dir: &std::path::Path, at: &str, all: &mut BTreeMap<String, String>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if e.path().is_dir() {
            read_tree(&e.path(), &format!("{at}/{name}"), all);
        } else if let Ok(text) = std::fs::read_to_string(e.path()) {
            all.insert(format!("{at}/{name}"), text);
        }
    }
}

/// `board.export` on the layout, as the Desktop's Export STEP runs it.
fn export(files: &BTreeMap<String, String>, values: Value) -> Value {
    let request = json!({ "command": "board.export", "values": values });
    let call = json!({
        "fn": "export",
        "args": [FC, files[FC], serde_json::to_string(files).unwrap(), request.to_string()],
    });
    let out: Value =
        serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&call.to_string())).unwrap();
    assert_eq!(out["ok"], true, "{out}");
    out
}

/// The product names in a STEP file, in order.
fn products(step: &str) -> Vec<String> {
    step.match_indices("PRODUCT('")
        .filter(|(i, _)| *i > 0 && matches!(step.as_bytes()[i - 1], b'=' | b' '))
        .map(|(i, m)| {
            let rest = &step[i + m.len()..];
            rest[..rest.find('\'').unwrap()].to_owned()
        })
        .collect()
}

/// The STEP file read back through the importer: its solids' count and the box they fill (m).
fn read_back(step: &str) -> (usize, [f64; 3], [f64; 3]) {
    const IMPORTED: &str = "runmat 0.7\nstd 1\n\npart board_model {\n    feature step = import(source = \"board.step\", digest = \"unrecorded\")\n}\n";
    let files = json!({ "/q/board.step": step, "/q/board_model.part": IMPORTED });
    let call = json!({ "fn": "compile", "args": ["/q/board_model.part", IMPORTED, files.to_string(), "{}"] });
    let v: Value =
        serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&call.to_string())).unwrap();
    let errors: Vec<&Value> = v["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let bodies = v["display"]["data"]["bodies"].as_array().unwrap().len();
    let b = &v["display"]["data"]["bounds"];
    let at = |s: &str| [0, 1, 2].map(|k| b[s][k].as_f64().unwrap());
    (bodies, at("min"), at("max"))
}

/// What the 3D tab draws: the outline's x/y box, the lowest and highest component body (m), and
/// the designators of the components with a body.
struct Shown {
    min: [f64; 3],
    max: [f64; 3],
    /// The board's top face (m).
    top: f64,
    bodies: BTreeSet<String>,
}

fn shown(files: &BTreeMap<String, String>) -> Shown {
    let display = runmat_cad_wasm::board::board_display(FC, files).expect("the board compiles");
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut bodies = BTreeSet::new();
    let mut top = f64::NEG_INFINITY;
    for body in &display.bodies {
        let name = body.reference.as_deref().unwrap_or("");
        let part = !matches!(
            name,
            "Board edge" | "Solder mask" | "Copper" | "Pads" | "Drills" | "Silkscreen" | "Terminals" | "Leads" | "Pins"
        );
        for face in &body.faces {
            if part {
                if let Some(r) = face.reference.as_deref().and_then(|r| r.strip_prefix("component:")) {
                    bodies.insert(r.to_owned());
                }
            }
            if name == "Board edge" || part {
                for p in face.positions.chunks_exact(3) {
                    if name == "Board edge" {
                        top = top.max(p[2] as f64);
                    }
                    for k in 0..3 {
                        let v = p[k] as f64;
                        // The board edge sets x and y; component bodies set z.
                        if k < 2 && name != "Board edge" {
                            continue;
                        }
                        min[k] = min[k].min(v);
                        max[k] = max[k].max(v);
                    }
                }
            }
        }
    }
    Shown { min, max, top, bodies }
}

#[test]
fn the_flight_controller_exports_one_named_solid_per_part_and_reads_back_in_its_box() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files = project();
    let out = export(&files, json!({}));
    assert_eq!(out["name"], "flight_controller.step");
    let step = out["text"].as_str().unwrap();
    assert!(step.contains("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING"), "not AP242");
    assert!(step.contains("ADVANCED_FACE") && !step.contains("FACETED_BREP"), "not an exact B-rep");

    let names = products(step);
    let tab = shown(&files);
    assert!(tab.bodies.len() > 30, "{:?}", tab.bodies);
    let mut want: BTreeSet<String> = tab.bodies.clone();
    want.extend(["board", "solder mask top", "solder mask bottom"].map(String::from));
    let got: BTreeSet<String> = names.iter().cloned().collect();
    assert_eq!(got, want);
    assert_eq!(names.len(), want.len(), "each product once: {names:?}");
    // Copper is off by default.
    assert!(!names.iter().any(|n| n.starts_with("copper")), "{names:?}");

    let (solids, min, max) = read_back(step);
    assert_eq!(solids, want.len(), "one solid per product");
    for k in 0..2 {
        assert!((min[k] - tab.min[k]).abs() < 1e-5 && (max[k] - tab.max[k]).abs() < 1e-5, "x/y {min:?}..{max:?} vs the outline {:?}..{:?}", tab.min, tab.max);
    }
    // The tallest part on top sets the top; the bottom mask (or a bottom part) the bottom.
    assert!((max[2] - tab.max[2]).abs() < 1e-5, "top {} vs the tallest part {}", max[2], tab.max[2]);
    assert!((min[2] - tab.min[2].min(-0.02e-3)).abs() < 1e-5, "bottom {} vs {}", min[2], tab.min[2]);
}

#[test]
fn copper_is_written_per_outer_layer_when_asked_and_the_mask_can_be_left_out() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files = project();
    let out = export(&files, json!({ "copper": true, "mask": false }));
    let names = products(out["text"].as_str().unwrap());
    assert!(names.contains(&"copper F.Cu".to_owned()) && names.contains(&"copper B.Cu".to_owned()), "{names:?}");
    assert!(!names.iter().any(|n| n.starts_with("solder mask")), "{names:?}");
    assert!(out["notes"].as_array().unwrap().iter().all(|n| !n.as_str().unwrap().contains("copper items")), "{}", out["notes"]);
}

const CHIP: &str = r#"runmat 0.7
std 1

part chip {
    sketch s on origin.xy {
        l1 = line(start = [-0.5 mm, -0.25 mm], end = [0.5 mm, -0.25 mm])
        l2 = line(start = l1.end, end = [0.5 mm, 0.25 mm])
        l3 = line(start = l2.end, end = [-0.5 mm, 0.25 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 0.5 mm)
}
"#;

#[test]
fn a_package_model_in_the_project_replaces_its_body_box() {
    runmat_geometry_wasm_host::cad::install_occt();
    let chip = regenerate_part("/m/chip.part", CHIP).expect("compiles");
    let doc = ExchangeDocument {
        name: "chip".into(),
        parts: vec![ExchangePart {
            name: "chip".into(),
            bodies: chip.result.state.bodies.iter().map(|b| &b.handle).collect(),
            color: None,
            material: None,
            face_names: Vec::new(),
            face_colors: Vec::new(),
        }],
        occurrences: Vec::new(),
    };
    let model = String::from_utf8(chip.kernel.export(&doc, ExchangeFormat::Ap242).unwrap()).unwrap();
    let mut files = project();
    let before = shown(&files);
    // The 0402 capacitors take the model, lifted 5 mm so the box it reads back in shows it.
    let layout = files[FC].replace(
        r#"model = { path = "${RUNMAT_3DMODELS}/C_0402_1005Metric_IPCN.step" }"#,
        r#"model = { path = "models/c0402.step", offset = [0, 0, 5] }"#,
    );
    assert_ne!(layout, files[FC], "the 0402 package names a model");
    files.insert(FC.into(), layout);
    files.insert("/p/drone/electrical/flight_controller/models/c0402.step".into(), model);
    let deps = runmat_cad_wasm::runtime::dependencies(FC, &files[FC]);
    assert!(deps.iter().any(|d| d.ends_with("/models/c0402.step")), "the CAD host gathers the model: {deps:?}");

    let out = export(&files, json!({}));
    let notes = out["notes"].to_string();
    assert!(!notes.contains("c0402"), "{notes}");
    let (_, _, max) = read_back(out["text"].as_str().unwrap());
    // The model's 0.5 mm box, 5 mm up from the mask on the board's top face, is the tallest thing.
    let want = before.top + 0.02e-3 + 5.5e-3;
    assert!(want > before.max[2], "{want} vs {}", before.max[2]);
    assert!((max[2] - want).abs() < 1e-5, "the model's top {} vs {want}", max[2]);
}
