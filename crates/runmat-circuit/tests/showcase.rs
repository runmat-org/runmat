//! The showcase board, `tests/projects/daq`: a three-channel, 16-bit data-acquisition board
//! (op-amp driver, SAR ADC and reference per channel, simultaneous-sampling SPI into a BGA-256
//! FPGA, USB-C power with two buck regulators) on six layers, placed, fanned out, routed and
//! poured with the product's own board commands.
//!
//! The project is checked in at `tests/projects/daq/` and in the Desktop at
//! `desktop/src/app/components/circuit/fixtures/daq/`. Set `RUNMAT_SHOWCASE_BUILD=1` to rebuild
//! it from `tests/showcase/start.circuit.layout` with the command script in
//! `showcase/build.rs` and rewrite both copies.

mod showcase {
    pub mod build;
}

use std::path::PathBuf;

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects/daq")
}

#[test]
fn rebuild_when_asked() {
    if std::env::var("RUNMAT_SHOWCASE_BUILD").is_err() {
        return;
    }
    showcase::build::build(&here());
}

fn files() -> runmat_circuit::runtime::Files {
    ["daq.circuit", "daq.circuit.layout", "daq.circuit.lock"]
        .iter()
        .map(|name| {
            (
                format!("/p/{name}"),
                std::fs::read_to_string(here().join(name)).expect("the showcase is checked in"),
            )
        })
        .collect()
}

/// Every component of the schematic is on the board, the board passes DRC with no finding,
/// nothing is left unrouted, ERC is clean and the board is in sync with the schematic.
#[test]
fn the_showcase_board_is_placed_routed_and_clean() {
    use runmat_circuit::runtime::{compile, CompileOptions};
    let files = files();
    let layout = "/p/daq.circuit.layout";
    let out = compile(layout, &files[layout], &files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let index = out.index.expect("compiles");
    let board = index.board.as_ref().expect("a board");
    let placed: std::collections::BTreeSet<&str> =
        board.footprints.iter().map(|f| f.reference.as_str()).collect();
    let missing: Vec<&str> = index
        .components
        .iter()
        .map(|c| c.designator.as_str())
        .filter(|d| !placed.contains(d))
        .collect();
    assert!(missing.is_empty(), "not on the board: {missing:?}");
    assert!(index.components.len() >= 70, "{} components", index.components.len());
    let outline = runmat_circuit::syntax::board::parse_board(layout, &files[layout])
        .board
        .expect("reads")
        .outline
        .bbox();
    for f in &board.footprints {
        let at = runmat_circuit::geom::Point::mm(f.x_mm, f.y_mm);
        assert!(outline.contains(at), "{} at {at} is off the board", f.reference);
    }
    let drc: Vec<String> = board
        .drc
        .iter()
        .map(|r| format!("{} {} {:?}", r.code, r.message, r.items))
        .collect();
    assert!(drc.is_empty(), "DRC: {drc:#?}");
    assert!(board.unrouted.is_empty(), "unrouted: {:?}", board.unrouted.len());
    let erc: Vec<String> = index
        .erc
        .iter()
        .map(|r| format!("{} {} {:?}", r.code, r.message, r.items))
        .collect();
    assert!(erc.is_empty(), "ERC: {erc:#?}");
    assert_eq!(index.eco.total, 0, "the board is in sync with the schematic");
    // Six copper layers, 1.6 mm, routed on both outer layers, BGA fanned out, poured planes.
    assert_eq!(board.layers.iter().filter(|l| l.kind == "copper").count(), 6);
    assert!(board.tracks.len() > 150 && board.vias.len() > 100);
    assert!(board.zones.len() >= 7);
}
