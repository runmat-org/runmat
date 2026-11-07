//! A 40-copper-layer board end to end (claims row PCB-25). The stack-up sets the copper layer
//! count: two headers on a 40-layer board compile, the autorouter joins them on two inner
//! layers, DRC passes, Gerber, ODB++ and IPC-2581 write all 40 copper layers, the 3D model is
//! the stack's thickness, and the KiCad export refuses the board with KiCad's 32-layer limit
//! instead of dropping layers.

use runmat_circuit::fab::set::FabRequest;
use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files, KicadExportError};
use runmat_circuit::syntax::board::parse_board;
use serde_json::{json, Value};

const CIRCUIT: &str = "/p/deep.circuit";
const LAYOUT: &str = "/p/deep.circuit.layout";
const COPPER: u32 = 40;

const CIRCUIT_TEXT: &str = r#"runmat 0.7
std 1

// Two headers joined through the inner layers of a 40-layer board.
circuit deep {
    root = main

    sheet main {
        component J1 = { part = "HDR-2x05" }
        component J2 = { part = "HDR-2x05" }
        connect = [net.A, "J1.1", "J2.1"]
        connect = [net.B, "J1.2", "J2.2"]
        no_erc = ["J1.3", "J1.4", "J1.5", "J1.6", "J1.7", "J1.8", "J1.9", "J1.10", "J2.3", "J2.4", "J2.5", "J2.6", "J2.7", "J2.8", "J2.9", "J2.10"]
    }
}
"#;

const LAYOUT_TEXT: &str = r#"runmat 0.7
std 1

use circuit deep

layout deep {
    outline = [[0, 0], [40, 0], [40, 20], [0, 20]]
    limits = { clearance = 0.15, track = 0.15, via_diameter = 0.45, drill = 0.2, annular = 0.1, hole_to_hole = 0.25, copper_to_edge = 0.3, silk_to_pad = 0.1, microvia_drill = 0.1 }
}
"#;

fn request(command: &str, values: Value, selection: &[&str]) -> EditRequest {
    EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: selection.iter().map(|s| s.to_string()).collect(),
        options: CompileOptions::default(),
    }
}

/// Runs `command` on `path` and takes every file it changes.
fn run(files: &mut Files, path: &str, command: &str, values: Value, selection: &[&str]) -> String {
    let out = runtime::edit(path, &files[path], files, &request(command, values, selection))
        .unwrap_or_else(|e| panic!("{command}: {} {:?}", e.message, e.diagnostics));
    for f in out.files {
        files.insert(f.path.clone(), f.text.clone());
    }
    out.summary
}

fn board(files: &Files) -> runmat_circuit::layout::Board {
    parse_board(LAYOUT, &files[LAYOUT]).board.expect("the layout reads")
}

fn copper_names() -> Vec<String> {
    let mut names = vec!["F.Cu".to_string()];
    names.extend((1..COPPER - 1).map(|i| format!("In{i}.Cu")));
    names.push("B.Cu".into());
    names
}

/// The project: the circuit, its board synced, stacked to 40 layers, the headers placed and
/// nets A and B autorouted on In19.Cu and In20.Cu.
fn build() -> Files {
    let mut files = Files::new();
    files.insert(CIRCUIT.into(), CIRCUIT_TEXT.into());
    files.insert(LAYOUT.into(), LAYOUT_TEXT.into());
    run(&mut files, LAYOUT, "board.layer-stack", json!({ "layers": COPPER, "thickness": 4.0 }), &[]);
    let index = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    let changes: Vec<String> = index.index.as_ref().expect("compiles").eco.groups.iter().flat_map(|g| g.rows.iter()).filter(|r| r.status != "error").map(|r| format!("eco:{}", r.id)).collect();
    run(&mut files, LAYOUT, "schematic.update-board", json!({ "changes": changes }), &[]);
    for (reference, x, y) in [("J1", 8.0, 10.0), ("J2", 32.0, 10.0)] {
        let b = board(&files);
        let f = b.footprint_by_ref(reference).unwrap_or_else(|| panic!("no {reference}"));
        let (fx, fy) = (runmat_circuit::geom::to_mm(f.at.x), runmat_circuit::geom::to_mm(f.at.y));
        let round = |v: f64| (v * 1e6).round() / 1e6;
        let turn = (-f.rotation).rem_euclid(360.0);
        run(&mut files, LAYOUT, "board.move", json!({ "dx": round(x - fx), "dy": round(y - fy), "rotation": turn }), &[&format!("component:{reference}")]);
    }
    run(&mut files, LAYOUT, "board.autoroute", json!({ "nets": ["net:A", "net:B"], "layers": "In19.Cu, In20.Cu" }), &[]);
    files
}

#[test]
fn a_forty_layer_board_compiles_routes_on_inner_layers_and_passes_drc() {
    let files = build();
    assert!(files[LAYOUT].contains("layer \"In38.Cu\""), "the stack-up prints every inner layer");
    let out = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let index = out.index.expect("the board compiles");
    let b = index.board.as_ref().expect("a board");
    let copper: Vec<String> = b.layers.iter().filter(|l| l.kind == "copper").map(|l| l.name.clone()).collect();
    assert_eq!(copper, copper_names());
    for net in ["A", "B"] {
        let layers: Vec<&str> = b.tracks.iter().filter(|t| t.net.as_deref() == Some(net)).map(|t| t.layer.as_str()).collect();
        assert!(!layers.is_empty(), "{net} is routed");
        assert!(layers.iter().all(|l| ["In19.Cu", "In20.Cu"].contains(l)), "{net} runs on the inner layers it was given: {layers:?}");
    }
    assert!(b.unrouted.is_empty(), "every connection is routed: {:?}", b.unrouted);
    let drc: Vec<String> = b.drc.iter().map(|r| format!("{} {}", r.code, r.message)).collect();
    assert!(drc.is_empty(), "DRC passes: {drc:?}");
    // The 3D model is the 40-layer stack's thickness.
    let board = board(&files);
    let solid = runmat_circuit::board3d::board_solid(&board);
    assert!((solid.thickness - runmat_circuit::geom::to_mm(board.stackup.thickness())).abs() < 1e-9);
    assert!((solid.thickness - 4.0).abs() < 0.05, "{} mm", solid.thickness);
    // The 3D tab's model: the substrate spans the whole stack, the headers' pins out both faces.
    let model = runmat_circuit::board3d::board_model(&board);
    let substrate = model.bodies.iter().find(|b| b.reference == "Board edge").expect("the substrate");
    let zs: Vec<f64> = substrate.faces.iter().flat_map(|f| f.positions.chunks(3).map(|p| p[2])).collect();
    let span = zs.iter().cloned().fold(f64::MIN, f64::max) - zs.iter().cloned().fold(f64::MAX, f64::min);
    assert!((span - solid.thickness).abs() < 1e-6, "substrate {span} mm");
    assert!(model.bounds.max[2] - model.bounds.min[2] > solid.thickness, "the model reaches past both faces");
}

#[test]
fn a_forty_layer_board_writes_every_copper_layer_to_gerber_odb_and_ipc2581() {
    let files = build();
    let set = runtime::export_fabrication(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default(), &FabRequest::default(), None)
        .unwrap_or_else(|d| panic!("{d:?}"));
    let names = copper_names();
    // Gerber: one file per copper layer with its X2 position.
    for (i, name) in names.iter().enumerate() {
        let path = format!("gerber/deep-{}.gbr", name.replace('.', "_"));
        let text = set.files.get(&path).unwrap_or_else(|| panic!("{path} is written"));
        let pos = if i == 0 { "Top" } else if i + 1 == names.len() { "Bot" } else { "Inr" };
        assert!(text.contains(&format!("%TF.FileFunction,Copper,L{},{pos}*%", i + 1)), "{path}");
    }
    let gerber_copper = set.files.keys().filter(|k| k.starts_with("gerber/") && k.ends_with("_Cu.gbr")).count();
    assert_eq!(gerber_copper, COPPER as usize);
    // The routed inner layers carry copper; an unrouted inner layer has only the header pads.
    let size = |name: &str| set.files[&format!("gerber/deep-{name}.gbr")].len();
    assert!(size("In19_Cu") > size("In5_Cu") || size("In20_Cu") > size("In5_Cu"));
    // ODB++: 40 signal rows in the matrix and a feature file per copper layer.
    let matrix = set.files.iter().find(|(k, _)| k.ends_with("/matrix/matrix")).map(|(_, v)| v).expect("ODB++ matrix");
    assert_eq!(matrix.matches("TYPE=SIGNAL").count(), COPPER as usize);
    for layer in ["top", "inner1", "inner19", "inner38", "bottom"] {
        assert!(set.files.keys().any(|k| k.ends_with(&format!("/steps/pcb/layers/{layer}/features"))), "ODB++ {layer}");
    }
    // IPC-2581: a signal layer per copper layer, by name.
    let ipc = &set.files["ipc2581/deep.xml"];
    assert_eq!(ipc.matches("layerFunction=\"SIGNAL\"").count(), COPPER as usize);
    for name in &names {
        assert!(ipc.contains(&format!("<Layer name=\"{name}\" layerFunction=\"SIGNAL\"")), "IPC-2581 {name}");
    }
}

#[test]
fn kicad_export_refuses_a_board_deeper_than_kicad_holds() {
    let files = build();
    let err = runtime::export_kicad(CIRCUIT, &files[CIRCUIT], &files, &CompileOptions::default(), "deep").expect_err("40 layers is past KiCad's limit");
    assert_eq!(
        err,
        KicadExportError::Unsupported(
            "KiCad supports at most 32 copper layers; this board has 40. Export to ODB++, IPC-2581 or Gerber instead.".into()
        )
    );
    // The board writer refuses on its own too, and a 32-layer board still writes.
    let mut board = board(&files);
    assert!(runmat_circuit::kicad::pcb::write_board(&board).is_err());
    board.stackup = runmat_circuit::layout::Stackup::standard(32, runmat_circuit::geom::mm(3.2), 1.0, 0.5);
    board.tracks.clear();
    board.vias.clear();
    let text = runmat_circuit::kicad::pcb::write_board(&board).expect("32 layers fit KiCad");
    assert!(text.contains("\"In30.Cu\"") && text.contains("(31 \"B.Cu\" signal)"), "KiCad's full 32-layer table");
}
