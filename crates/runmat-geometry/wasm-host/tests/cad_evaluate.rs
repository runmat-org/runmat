//! Evaluate commands on OCCT through `runmat_cad_call` (the Desktop worker's entry): mass
//! properties of a part and of the camera drone from its materials, and the configuration a
//! part compiles in.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    input length : length = 100 mm\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.length, 0 mm])\n        l2 = line(start = [input.length, 0 mm], end = [input.length, 20 mm])\n        l3 = line(start = [input.length, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = { body = feature.body.body, part_number = \"BK-100\", material = \"material.aluminum_6061\" }\n    configuration long = { length = 200 mm, part_number = \"BK-200\" }\n}\n";

#[test]
fn a_block_weighs_its_volume_at_its_materials_density_in_each_configuration() {
    let mass = |configuration: &str| {
        let request = json!({ "command": "evaluate.mass-properties", "values": { "configuration": configuration } }).to_string();
        let out = call("analyze", &["/p/block.part", BLOCK, "{}", &request]);
        assert_eq!(out["ok"], true, "{out}");
        out["data"]["total"]["mass"].as_f64().unwrap()
    };
    // 100 × 20 × 10 mm of 6061 at 2700 kg/m³ is 54 g; the long configuration twice that.
    assert!(
        (mass("default") - 0.054).abs() < 1e-6,
        "{}",
        mass("default")
    );
    assert!((mass("long") - 0.108).abs() < 1e-6, "{}", mass("long"));
    // The compile takes the active configuration: its part number and geometry.
    let compiled = call(
        "compile",
        &["/p/block.part", BLOCK, "{}", "{\"configuration\":\"long\"}"],
    );
    let data = &compiled["projection"]["data"];
    assert_eq!(data["index"]["configuration"], "long", "{compiled}");
    assert_eq!(data["document"]["parts"][0]["part_number"], "BK-200");
}

fn drone() -> BTreeMap<String, String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runmat-cad/wasm/tests/drone");
    let mut all = BTreeMap::new();
    read_tree(&dir, "/p/drone", &mut all);
    assert!(all.contains_key("/p/drone/drone.assembly"), "the drone project");
    all
}

/// Every file under `dir`, keyed `<at>/<relative path>` (the drone's folders are its modules).
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

#[test]
fn the_drone_has_an_assembly_mass_and_centre_of_gravity() {
    let all = drone();
    let files = serde_json::to_string(&all).unwrap();
    let path = "/p/drone/drone.assembly";
    let request = json!({ "command": "evaluate.mass-properties" }).to_string();
    let out = call("analyze", &[path, &all[path], &files, &request]);
    assert_eq!(out["ok"], true, "{out}");
    let total = &out["data"]["total"];
    let mass = total["mass"].as_f64().unwrap();
    let parts = out["data"]["parts"].as_array().unwrap();
    eprintln!(
        "drone: {} · {} parts · problems {:?}",
        out["summary"],
        parts.len(),
        out["problems"]
    );
    assert!(parts.len() > 20, "{} part occurrences", parts.len());
    // A sub-kilogram camera drone: its structure and the parts with materials weigh something.
    assert!(mass > 0.05 && mass < 3.0, "{mass} kg");
    // The inertia tensor about the centre of gravity is symmetric with positive moments.
    let i = |r: usize, c: usize| total["inertia"][r][c].as_f64().unwrap();
    for k in 0..3 {
        assert!(i(k, k) > 0.0);
    }
    assert!((i(0, 1) - i(1, 0)).abs() <= 1e-9 * i(0, 0).max(1e-12));
}

#[test]
fn interference_and_clearance_between_placed_blocks() {
    let assembly = "runmat 0.7\nstd 1\n\nassembly row {\n    use part block as a\n    use part block as near {\n        placement = { placement = offset, offset = { translation = [0 mm, 20.5 mm, 0 mm] } }\n    }\n    use part block as into {\n        placement = { placement = offset, offset = { translation = [0 mm, -10 mm, 0 mm] } }\n    }\n}\n";
    let files = json!({ "/p/block.part": BLOCK }).to_string();
    let run = |command: &str| {
        let request = json!({ "command": command, "values": { "minimum": 1.0 } }).to_string();
        let out = call("analyze", &["/p/row.assembly", assembly, &files, &request]);
        assert_eq!(out["ok"], true, "{out}");
        out
    };
    let interference = run("evaluate.interference");
    let overlaps = interference["data"]["overlaps"].as_array().unwrap();
    assert_eq!(overlaps.len(), 1, "{interference}");
    // a and into share a 100 × 10 × 10 mm slab.
    assert!(
        (overlaps[0]["volume"].as_f64().unwrap() - 1e-5).abs() < 1e-8,
        "{interference}"
    );
    let clearance = run("evaluate.clearance");
    let near = clearance["data"]["near"].as_array().unwrap();
    assert_eq!(near.len(), 1, "{clearance}");
    assert!(
        (near[0]["distance"].as_f64().unwrap() - 0.0005).abs() < 1e-7,
        "{clearance}"
    );
    // A row selects its occurrences by the ids the Outline and canvas use.
    for item in clearance["rows"][0]["items"].as_array().expect("row items") {
        let id = item.as_str().unwrap().strip_prefix("occurrence:").expect("an occurrence");
        assert!(id.len() == 32 && id.chars().all(|c| c.is_ascii_hexdigit()), "{clearance}");
    }
}

/// A board is an exact solid in interference and clearance checks: the daq board (90 x 60 mm,
/// 1.6 mm thick, components on top) placed through a block shares the block's 100 x 20 mm
/// footprint over its thickness and the bodies inside it; placed 0.5 mm off the block's side,
/// it comes within the asked clearance without sharing material.
#[test]
fn a_board_interferes_and_clears_as_its_outline_and_component_bodies() {
    let daq = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runmat-circuit/tests/projects/daq");
    let mut all = BTreeMap::new();
    all.insert("/p/block.part".to_owned(), BLOCK.to_owned());
    for name in ["daq.circuit", "daq.circuit.layout", "daq.circuit.lock"] {
        all.insert(format!("/p/{name}"), std::fs::read_to_string(daq.join(name)).unwrap());
    }
    let files = serde_json::to_string(&all).unwrap();
    let run = |offset: &str, command: &str| {
        let assembly = format!("runmat 0.7\nstd 1\n\nassembly enclosure {{\n    use part block as wall\n    use circuit daq as board {{\n        placement = {{ placement = offset, offset = {{ translation = {offset} }} }}\n    }}\n}}\n");
        let request = json!({ "command": command, "values": { "minimum": 1.0 } }).to_string();
        let out = call("analyze", &["/p/enclosure.assembly", &assembly, &files, &request]);
        assert_eq!(out["ok"], true, "{out}");
        assert!(out["data"]["errors"].as_array().is_none_or(|e| e.is_empty()), "{out}");
        assert_eq!(out["data"]["parts"], 2, "{out}");
        out
    };
    // Through the block's mid-height: the laminate alone shares 90 x 20 x 1.6 mm (2.88 cm^3).
    let through = run("[0 mm, 0 mm, 4 mm]", "evaluate.interference");
    let overlaps = through["data"]["overlaps"].as_array().unwrap();
    assert_eq!(overlaps.len(), 1, "{through}");
    let volume = overlaps[0]["volume"].as_f64().unwrap();
    assert!(volume > 2.8e-6 && volume < 1.2e-5, "{volume} m^3");
    // Beside it: the board's edge 0.5 mm from the block's side.
    let beside = run("[0 mm, 20.5 mm, 0 mm]", "evaluate.clearance");
    assert!(beside["data"]["overlaps"].as_array().unwrap().is_empty(), "{beside}");
    let near = beside["data"]["near"].as_array().unwrap();
    assert_eq!(near.len(), 1, "{beside}");
    let gap = near[0]["distance"].as_f64().unwrap();
    assert!(gap > 0.0 && gap <= 0.0005 + 1e-7, "{gap} m");
}

#[test]
fn study_checks_stop_on_overlaps_and_missing_densities_and_warn_on_open_contacts() {
    // `near` stands 0.05 mm off `a`, inside the 0.1 mm contact gap; `into` overlaps `a`.
    let assembly = |into: bool| {
        let third = if into { "    use part block as into {\n        placement = { placement = offset, offset = { translation = [0 mm, -10 mm, 0 mm] } }\n    }\n" } else { "" };
        format!("runmat 0.7\nstd 1\n\nassembly row {{\n    use part block as a\n    use part block as near {{\n        placement = {{ placement = offset, offset = {{ translation = [0 mm, 20.05 mm, 0 mm] }} }}\n    }}\n{third}}}\n")
    };
    let run = |text: &str, block: &str| {
        let files = json!({ "/p/block.part": block }).to_string();
        let request = json!({ "command": "simulate.check" }).to_string();
        let out = call("analyze", &["/p/row.assembly", text, &files, &request]);
        assert_eq!(out["ok"], true, "{out}");
        out
    };
    let checks = |out: &Value| out["data"]["checks"].as_array().unwrap().iter().map(|c| (c["check"].as_str().unwrap().to_owned(), c["severity"].as_str().unwrap().to_owned())).collect::<Vec<_>>();
    let open = run(&assembly(false), BLOCK);
    assert_eq!(checks(&open), [("fit".to_owned(), "warning".to_owned())], "{open}");
    assert!(open["summary"].as_str().unwrap().contains("ready to mesh"), "{open}");
    let overlapping = run(&assembly(true), BLOCK);
    assert!(checks(&overlapping).contains(&("fit".to_owned(), "error".to_owned())), "{overlapping}");
    assert!(overlapping["summary"].as_str().unwrap().contains("to fix before the study meshes"), "{overlapping}");
    let bare = run(&assembly(false), &BLOCK.replace(", material = \"material.aluminum_6061\"", ""));
    assert_eq!(checks(&bare).iter().filter(|c| c.0 == "mass").count(), 2, "{bare}");
    // Blocks that touch (a 0 mm gap) are a contact, for information; no warning.
    let touching = run(&assembly(false).replace("20.05 mm", "20 mm"), BLOCK);
    assert_eq!(checks(&touching), [("fit".to_owned(), "info".to_owned())], "{touching}");
    assert!(touching["summary"].as_str().unwrap().contains("ready to mesh · 0 warnings"), "{touching}");
}
#[test]
fn a_shelled_drafted_cover_names_every_face_in_the_draft_analysis() {
    let cover = "runmat 0.7\nstd 1\n\npart cover {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 25 mm)\n    feature dr = draft(\n        faces = [feature.body.face.side(s1.l1), feature.body.face.side(s1.l3)],\n        neutral = origin.xy,\n        angle = 3 deg\n    )\n    feature hollow = shell(body = feature.body.body, open_faces = [feature.body.face.start], thickness = 2 mm)\n    output cover = { body = feature.body.body, material = \"material.aluminum\" }\n}\n";
    let request = json!({ "command": "evaluate.draft", "values": { "direction": ["origin:top"], "angle": 1 } }).to_string();
    let out = call("analyze", &["/p/cover.part", cover, "{}", &request]);
    assert_eq!(out["ok"], true, "{out}");
    let faces = out["data"]["faces"].as_array().unwrap();
    // 5 outer faces, 5 inner walls and the rim the open bottom leaves.
    assert_eq!(faces.len(), 11, "{out}");
    let unnamed: Vec<&str> = faces.iter().filter_map(|f| f["face"].as_str()).filter(|n| n.starts_with("face ")).collect();
    assert!(unnamed.is_empty(), "unnamed faces {unnamed:?} in {out}");
    assert!(faces.iter().any(|f| f["face"].as_str().unwrap().contains("hollow")), "{out}");
}
