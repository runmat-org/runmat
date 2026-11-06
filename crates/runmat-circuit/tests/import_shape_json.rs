//! Shape-string JSON import on hand-written fixtures (tests/import/shape_json): a header, a
//! resistor and an LED, the LED rotated and on the bottom; a via, a copper area and a board text.

use runmat_circuit::import::shape_json;
use runmat_circuit::runtime::{compile, CompileOptions, Files};
use std::collections::BTreeMap;

fn fixture(names: &[&str]) -> BTreeMap<String, String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/import/shape_json");
    names.iter().map(|n| (n.to_string(), std::fs::read_to_string(format!("{dir}/{n}")).unwrap())).collect()
}

#[test]
fn a_schematic_and_board_import_to_a_circuit_that_compiles() {
    let out = shape_json::import(&fixture(&["blinky_sch.json", "blinky_pcb.json"]), "boards/blinky").unwrap();
    let names: Vec<&str> = out.files.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "boards/blinky-footprints/0603.kicad_mod",
            "boards/blinky-footprints/HDR-1x2.kicad_mod",
            "boards/blinky-footprints/LED0603.kicad_mod",
            "boards/blinky.circuit",
            "boards/blinky.circuit.layout",
        ]
    );
    let circuit = &out.files["boards/blinky.circuit"];
    assert!(circuit.contains(r#"part "R0603-1K00""#), "{circuit}");
    assert!(circuit.contains(r#"part "LED0603-red""#), "{circuit}");
    assert!(circuit.contains("blinky-footprints/LED0603.kicad_mod"), "{circuit}");
    let r = &out.report;
    assert_eq!(r.format, "shape-string JSON");
    assert_eq!(r.carried["components"], 3, "{:?}", r.lines());
    // VCC and LED_A by their labels, GND by its flag on the middle of a wire.
    assert_eq!(r.carried["nets"], 3, "{:?}", r.lines());
    assert_eq!(r.carried["footprints placed"], 3);
    assert_eq!(r.carried["tracks"], 8, "{:?}", r.lines());
    assert_eq!(r.carried["vias"], 1);
    assert_eq!(r.carried["zones"], 1);
    assert_eq!(r.carried["board texts"], 1);
    assert_eq!(r.not_carried["solid regions"], 1, "{:?}", r.lines());

    let files: Files = out.files.iter().map(|(k, v)| (format!("/p/{k}"), v.clone())).collect();
    let layout = "/p/boards/blinky.circuit.layout";
    let compiled = compile(layout, &files[layout], &files, &CompileOptions::default());
    let errors: Vec<_> = compiled.diagnostics.iter().filter(|d| d.severity == "error").map(|d| &d.message).collect();
    assert!(errors.is_empty(), "{errors:#?}");
    let board = compiled.index.expect("the imported board compiles").board.expect("the layout has a board");
    // Every track ends on its pad, the bottom LED's included: nothing is unrouted.
    assert!(board.unrouted.is_empty(), "{:?}", board.unrouted);
    let drc: Vec<_> = board.drc.iter().filter(|r| r.severity == "error").map(|r| (&r.code, &r.message)).collect();
    assert!(drc.is_empty(), "{drc:#?}");
    let parsed = runmat_circuit::syntax::board::parse_board(layout, &files[layout]).board.unwrap();
    let led = parsed.footprints.values().find(|f| f.reference == "D1").unwrap();
    assert_eq!(led.side, runmat_circuit::layout::Side::Bottom);
    assert_eq!(led.rotation, 90.0);
    // 80 × 40 canvas units of 10 mil.
    assert_eq!(parsed.outline.bbox().width(), 20_320_000);
    assert_eq!(parsed.outline.bbox().min.x, 0);
}

#[test]
fn a_document_that_isnt_a_schematic_or_board_is_refused_with_the_reason() {
    let mut files = BTreeMap::new();
    files.insert("lib.json".to_string(), r#"{"head":{"docType":"4"},"shape":[]}"#.to_string());
    let err = shape_json::import(&files, "x").unwrap_err();
    assert!(err.contains("type 4"), "{err}");
    files.insert("lib.json".to_string(), "<eagle/>".to_string());
    assert!(shape_json::import(&files, "x").unwrap_err().contains("isn't JSON"));
}
