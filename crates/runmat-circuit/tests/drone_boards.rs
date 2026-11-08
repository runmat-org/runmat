//! The quadcopter showcase's boards (`runmat-cad/wasm/tests/drone/electrical`): the flight controller and
//! the 4-in-1 ESC, built with the product's board commands. `RUNMAT_DRONE_BUILD=1` rebuilds both
//! layouts from their start files with the scripts in `drone/` (run in release); the checks run
//! on the checked-in files.

mod drone {
    pub mod build;
    pub mod esc;
    pub mod fc;
}

use std::path::PathBuf;

/// A board's folder in the drone project (`electrical/flight_controller`, `electrical/esc`), or
/// under `RUNMAT_DRONE_BOARDS_SRC` to rebuild from a working copy.
fn here(name: &str) -> PathBuf {
    std::env::var_os("RUNMAT_DRONE_BOARDS_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runmat-cad/wasm/tests/drone/electrical"))
        .join(name)
}

/// Where a rebuild writes: `RUNMAT_DRONE_OUT`, else the board's checked-in folder.
fn out_dir(name: &str) -> PathBuf {
    std::env::var_os("RUNMAT_DRONE_OUT").map(PathBuf::from).unwrap_or_else(|| here(name))
}

#[test]
fn rebuild_when_asked() {
    let Ok(which) = std::env::var("RUNMAT_DRONE_BUILD") else { return };
    if which.contains("fc") || which == "1" {
        let b = drone::fc::build(&here("flight_controller"));
        // `fc-courtyards` also prints each footprint's courtyard, for checking a placement.
        if which.contains("courtyards") {
            b.courtyards();
        }
        b.report();
        b.write(&out_dir("flight_controller"), "flight_controller");
    }
    if which == "esc-report" {
        // The checked-in (or RUNMAT_DRONE_BOARDS_SRC) layout's findings, without rebuilding it.
        let dir = here("esc");
        let mut b = drone::build::Build::new(&dir, "esc", &std::fs::read_to_string(dir.join("esc.circuit.layout")).unwrap());
        if let Ok(lock) = std::fs::read_to_string(dir.join("esc.circuit.lock")) {
            b.files.insert("/p/esc.circuit.lock".into(), lock);
        }
        b.report();
        return;
    }
    if which.contains("esc") || which == "1" {
        // `esc-place` stops after placement and prints the courtyards; `esc-channel` after
        // routing channel 0.
        use drone::esc::Stage;
        let stage = if which.contains("place") { Stage::Place } else if which.contains("channel") { Stage::Channel } else { Stage::Board };
        let b = drone::esc::build(&here("esc"), stage);
        if stage != Stage::Board {
            b.courtyards();
        }
        b.report();
        b.write(&out_dir("esc"), "esc");
    }
}

fn check(name: &str) {
    use runmat_circuit::runtime::{compile, CompileOptions};
    let dir = here(name);
    let files: runmat_circuit::runtime::Files = [".circuit", ".circuit.layout", ".circuit.lock"]
        .iter()
        .filter_map(|x| Some((format!("/p/{name}{x}"), std::fs::read_to_string(dir.join(format!("{name}{x}"))).ok()?)))
        .collect();
    let layout = format!("/p/{name}.circuit.layout");
    let Some(text) = files.get(&layout) else { panic!("{name}: no layout checked in") };
    let out = compile(&layout, text, &files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let index = out.index.expect("compiles");
    let board = index.board.as_ref().expect("a board");
    let placed: std::collections::BTreeSet<&str> = board.footprints.iter().map(|f| f.reference.as_str()).collect();
    let missing: Vec<&str> = index.components.iter().map(|c| c.designator.as_str()).filter(|d| !placed.contains(d)).collect();
    assert!(missing.is_empty(), "{name}: not on the board: {missing:?}");
    let drc: Vec<String> = board.drc.iter().map(|r| format!("{} {}", r.code, r.message)).collect();
    assert!(drc.is_empty(), "{name}: DRC {drc:?}");
    assert!(board.unrouted.is_empty(), "{name}: {} unrouted", board.unrouted.len());
    assert!(index.erc.is_empty(), "{name}: ERC {:?}", index.erc.iter().map(|r| &r.message).collect::<Vec<_>>());
    assert_eq!(index.eco.total, 0, "{name}: board out of sync with the schematic");
}

#[test]
fn the_flight_controller_is_placed_routed_and_clean() {
    check("flight_controller");
}

#[test]
fn the_esc_is_placed_routed_and_clean() {
    check("esc");
    // Six copper layers in 1.6 mm, 2 oz outside for the phase current.
    let dir = here("esc");
    let text = std::fs::read_to_string(dir.join("esc.circuit.layout")).unwrap();
    let board = runmat_circuit::syntax::board::parse_board("/p/esc.circuit.layout", &text).board.expect("the layout reads");
    assert_eq!(board.copper_count(), 6);
    let weights: Vec<f64> = board
        .stackup
        .layers
        .iter()
        .filter_map(|l| match l.kind {
            runmat_circuit::layout::StackKind::Copper { weight_oz } => Some(weight_oz),
            _ => None,
        })
        .collect();
    assert_eq!(weights, [2.0, 1.0, 1.0, 1.0, 1.0, 2.0]);
    assert!((runmat_circuit::geom::to_mm(board.stackup.thickness()) - 1.6).abs() < 0.01);
}
