//! EAGLE XML import on a hand-written fixture (tests/import/eagle: a header, a resistor and
//! an LED on a two-layer board, the LED mirrored to the bottom).

use runmat_circuit::import::eagle;
use runmat_circuit::runtime::{compile, CompileOptions, Files};
use std::collections::BTreeMap;

fn fixture(names: &[&str]) -> BTreeMap<String, String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/import/eagle");
    names.iter().map(|n| (n.to_string(), std::fs::read_to_string(format!("{dir}/{n}")).unwrap())).collect()
}

#[test]
fn a_schematic_and_board_import_to_a_circuit_that_compiles() {
    let out = eagle::import(&fixture(&["blinky.sch", "blinky.brd"]), "boards/blinky").unwrap();
    let names: Vec<&str> = out.files.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "boards/blinky-footprints/connectors_HDR2.kicad_mod",
            "boards/blinky-footprints/leds_LED0603.kicad_mod",
            "boards/blinky-footprints/passives_0603.kicad_mod",
            "boards/blinky.circuit",
            "boards/blinky.circuit.layout",
        ]
    );
    let circuit = &out.files["boards/blinky.circuit"];
    // Each part becomes a declaration: from its MPN attribute, else its device and value.
    assert!(circuit.contains(r#"part "LED-RED-0603""#), "{circuit}");
    assert!(circuit.contains(r#"part "R0603-1k""#), "{circuit}");
    assert!(circuit.contains("blinky-footprints/passives_0603.kicad_mod"), "{circuit}");
    assert!(circuit.contains("net_class power"), "{circuit}");
    assert!(circuit.contains("variant lite"), "{circuit}");
    let r = &out.report;
    assert_eq!(r.carried["components"], 3, "{:?}", r.lines());
    assert_eq!(r.carried["nets"], 3, "{:?}", r.lines());
    assert_eq!(r.carried["tracks"], 9, "{:?}", r.lines());
    assert_eq!(r.carried["vias"], 1);
    assert_eq!(r.carried["zones"], 1);
    assert!(r.approximated.keys().any(|k| k.contains("2 sheets")), "{:?}", r.lines());

    let files: Files = out.files.iter().map(|(k, v)| (format!("/p/{k}"), v.clone())).collect();
    let layout = "/p/boards/blinky.circuit.layout";
    let compiled = compile(layout, &files[layout], &files, &CompileOptions::default());
    let errors: Vec<_> = compiled.diagnostics.iter().filter(|d| d.severity == "error").map(|d| &d.message).collect();
    assert!(errors.is_empty(), "{errors:#?}");
    let index = compiled.index.expect("the imported board compiles");
    let board = index.board.expect("the layout has a board");
    // The wires reach every pad: nothing is left unrouted, the mirrored LED's pads included.
    assert!(board.unrouted.is_empty(), "{:?}", board.unrouted);
    let drc: Vec<_> = board.drc.iter().filter(|r| r.severity == "error").map(|r| (&r.code, &r.message)).collect();
    assert!(drc.is_empty(), "{drc:#?}");
    // Every pad carries its net, and the LED sits on the bottom.
    let parsed = runmat_circuit::syntax::board::parse_board(layout, &files[layout]).board.unwrap();
    let pads: usize = parsed.footprints.values().map(|f| f.pad_nets.len()).sum();
    assert_eq!(pads, 6);
    let led = parsed.footprints.values().find(|f| f.reference == "D1").unwrap();
    assert_eq!(led.side, runmat_circuit::layout::Side::Bottom);
    assert_eq!(parsed.outline.bbox().width(), 20_000_000);
}

#[test]
fn a_file_that_isnt_eagle_xml_is_refused_with_the_reason() {
    let mut files = BTreeMap::new();
    files.insert("old.brd".to_string(), "\u{10}\u{0}binary".to_string());
    let err = eagle::import(&files, "old").unwrap_err();
    assert!(err.contains("before version 6"), "{err}");
}
