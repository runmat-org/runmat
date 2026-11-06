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

/// A board's folder in the drone project (`electrical/flight_controller`, `electrical/esc`).
fn here(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runmat-cad/wasm/tests/drone/electrical").join(name)
}

#[test]
fn rebuild_when_asked() {
    let Ok(which) = std::env::var("RUNMAT_DRONE_BUILD") else { return };
    if which.contains("fc") || which == "1" {
        let b = drone::fc::build(&here("flight_controller"));
        b.report();
        b.write(&here("flight_controller"), "flight_controller");
    }
    if which.contains("esc") || which == "1" {
        let b = drone::esc::build(&here("esc"));
        b.report();
        b.write(&here("esc"), "esc");
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

