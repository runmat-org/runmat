//! The simulation-page hero parts (`runmat-cad/wasm/tests/sim/<example id>/`) on OCCT: each
//! part compiles with no error and every feature regenerates, and its STEP export carries the
//! exact product (body) and face names its study selects, one solid per body.
//!
//! - `RUNMAT_SIM_STEP_DIR=<folder>` writes each export to `<folder>/<id>.step`.
//! - `RUNMAT_SIM_EXAMPLES=<runmat-examples>/fea` also checks that each example folder's design
//!   source is the same text as the copy here.
//! - `RUNMAT_SIM_ONLY=<id>` runs one part; `RUNMAT_SIM_SRC=<folder>` reads the parts from another
//!   copy of `sim/` (a variant under study).

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Each hero: its example id, its design file, the STEP products (with how many products carry
/// each name), the solids in all, and the face names (with how many faces carry each at least).
struct Hero {
    id: &'static str,
    file: &'static str,
    products: &'static [(&'static str, usize)],
    solids: usize,
    faces: &'static [(&'static str, usize)],
}

const HEROES: &[Hero] = &[Hero {
    id: "hydraulic-manifold",
    file: "manifold.part",
    products: &[("Block", 1), ("Oil", 2)],
    solids: 3,
    faces: &[("Port P", 1), ("Port T", 1), ("Port A", 1), ("Port B", 1), ("Passage wall", 4)],
}, Hero {
    id: "battery-cold-plate",
    file: "m01_cold_plate.part",
    products: &[
        ("M01 cell 1", 1), ("M01 cell 2", 1), ("M01 cell 3", 1), ("M01 cell 4", 1),
        ("M01 cell 5", 1), ("M01 cell 6", 1), ("M01 cell 7", 1), ("M01 cell 8", 1),
        ("Gap filler", 8), ("Cold plate", 1), ("Coolant", 1),
    ],
    solids: 18,
    faces: &[
        ("Coolant inlet", 1), ("Coolant outlet", 1), ("Channel wall", 16),
        ("Plate channel wall", 16), ("M01 cell bottom", 8),
    ],
}, Hero {
    id: "paddle-flow-switch",
    file: "flow_switch.part",
    products: &[("Tee", 1), ("Oil", 1), ("Paddle", 1)],
    solids: 3,
    faces: &[
        ("Inlet", 1), ("Outlet", 1), ("Pipe wall", 3), ("Paddle wetted", 5),
        ("Paddle surface", 5), ("Paddle root", 1),
    ],
}, Hero {
    id: "exhaust-muffler",
    file: "muffler.part",
    products: &[("Shell", 1), ("Air", 1)],
    solids: 2,
    faces: &[("Inlet", 1), ("Outlet", 1), ("Walls", 9)],
}, Hero {
    id: "hall-current-sensor",
    file: "sensor.part",
    products: &[("Core", 1), ("Primary", 1), ("Air", 1)],
    solids: 3,
    faces: &[("Primary in", 1), ("Primary out", 1), ("Gap", 2), ("Outer boundary", 6)],
}, Hero {
    id: "six-bolt-busbar",
    file: "busbar.part",
    products: &[("Busbar", 1)],
    solids: 1,
    faces: &[
        ("Cell bolt 1", 1), ("Cell bolt 2", 1), ("Cell bolt 3", 1), ("Cell bolt 4", 1),
        ("Cell bolt 5", 1), ("Cell bolt 6", 1), ("Output bolt", 1), ("Exterior", 10),
    ],
}, Hero {
    id: "brake-disc",
    file: "disc.part",
    products: &[("Rotor", 1)],
    solids: 1,
    faces: &[
        ("Friction face inboard", 1), ("Friction face outboard", 1), ("Vanes", 144),
        ("Bolt holes", 5), ("Exterior", 8),
    ],
}];

fn root() -> PathBuf {
    std::env::var("RUNMAT_SIM_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runmat-cad/wasm/tests/sim"))
}

fn compile(path: &str, text: &str) -> Value {
    let mut files = BTreeMap::new();
    files.insert(path.to_string(), text.to_string());
    let files = serde_json::to_string(&files).unwrap();
    let request = serde_json::json!({ "fn": "compile", "args": [path, text, files, "{}"] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

/// `PRODUCT('<name>'` occurrences, the solids, and every `ADVANCED_FACE('<name>'`.
fn step_names(text: &str) -> (BTreeMap<String, usize>, usize, BTreeMap<String, usize>) {
    let count = |prefix: &str| {
        let mut out = BTreeMap::new();
        for piece in text.split(prefix).skip(1) {
            if let Some(end) = piece.find('\'') {
                *out.entry(piece[..end].to_string()).or_insert(0) += 1;
            }
        }
        out
    };
    // A solid with an inner cavity (air round a core it encloses) is a BREP_WITH_VOIDS.
    let solids = text.matches("MANIFOLD_SOLID_BREP(").count() + text.matches("BREP_WITH_VOIDS(").count();
    (count("PRODUCT('"), solids, count("ADVANCED_FACE('"))
}

#[test]
fn the_hero_parts_compile_and_export_their_study_names() {
    runmat_geometry_wasm_host::cad::install_occt();
    let only = std::env::var("RUNMAT_SIM_ONLY").ok();
    let mut failures = Vec::new();
    for hero in HEROES {
        if only.as_ref().is_some_and(|o| o != hero.id) {
            continue;
        }
        let source = root().join(hero.id).join(hero.file);
        let text = std::fs::read_to_string(&source).unwrap_or_else(|e| panic!("{}: {e}", source.display()));
        if let Ok(examples) = std::env::var("RUNMAT_SIM_EXAMPLES") {
            let copy = PathBuf::from(examples).join(hero.id).join(hero.file);
            match std::fs::read_to_string(&copy) {
                Ok(t) if t == text => {}
                Ok(_) => failures.push(format!("{}: differs from {}", hero.id, copy.display())),
                Err(e) => failures.push(format!("{}: {e}", copy.display())),
            }
        }
        let path = format!("/p/{}/{}", hero.id, hero.file);
        let started = std::time::Instant::now();
        let v = compile(&path, &text);
        let diagnostics = v["diagnostics"].as_array().cloned().unwrap_or_default();
        let errors: Vec<&str> =
            diagnostics.iter().filter(|d| d["severity"] == "error").filter_map(|d| d["message"].as_str()).collect();
        let warnings: Vec<&str> =
            diagnostics.iter().filter(|d| d["severity"] != "error").filter_map(|d| d["message"].as_str()).collect();
        eprintln!("{}: {:.2?}, {} errors, warnings {warnings:?}", hero.id, started.elapsed(), errors.len());
        if !errors.is_empty() {
            failures.push(format!("{}: {errors:?}", hero.id));
            continue;
        }
        let failed: Vec<String> = v["projection"]["data"]["index"]["features"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter(|f| f["status"].as_str().is_some_and(|s| s != "ok" && s != "regenerated"))
            .map(|f| format!("{} {}", f["id"], f["status"]))
            .collect();
        if std::env::var("RUNMAT_SIM_DEBUG").is_ok() {
            for b in v["display"]["data"]["bodies"].as_array().cloned().unwrap_or_default() {
                let faces: Vec<String> = b["faces"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .map(|f| format!("{}{:?}", f["reference"].as_str().unwrap_or("-"), f["regions"]))
                    .collect();
                eprintln!("  body {} {}: {faces:?}", b["index"], b["reference"]);
            }
        }
        let problems = v["display"]["data"]["problems"].as_array().map(|p| p.len()).unwrap_or(0);
        if !failed.is_empty() || problems > 0 {
            failures.push(format!("{}: features {failed:?}, problems {}", hero.id, v["display"]["data"]["problems"]));
        }
        let step = match runmat_cad_wasm::exchange::part_step(&path, &text) {
            Ok(Some(step)) => String::from_utf8(step).expect("STEP is text"),
            Ok(None) => panic!("OCCT writes STEP"),
            Err(e) => {
                failures.push(format!("{}: export: {e}", hero.id));
                continue;
            }
        };
        if let Ok(dir) = std::env::var("RUNMAT_SIM_STEP_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(format!("{dir}/{}.step", hero.id), &step).unwrap();
        }
        let (products, solids, faces) = step_names(&step);
        eprintln!("  products {products:?}, {solids} solids, faces {faces:?}");
        for (name, n) in hero.products {
            if products.get(*name) != Some(n) {
                failures.push(format!("{}: product {name} x{:?}, want {n}", hero.id, products.get(*name)));
            }
        }
        if solids != hero.solids {
            failures.push(format!("{}: {solids} solids, want {}", hero.id, hero.solids));
        }
        for (name, n) in hero.faces {
            if faces.get(*name).copied().unwrap_or(0) < *n {
                failures.push(format!("{}: face {name} x{:?}, want at least {n}", hero.id, faces.get(*name)));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
