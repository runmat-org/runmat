//! Pipe-record import on hand-written fixtures (tests/import/pipe_record): a header, a
//! resistor and an LED, the LED rotated and on the bottom; a via, a polygon pour, a board
//! text, a net class with its width rule. The binary test packs the same records into a
//! compound file, the way the binary files hold them.

use runmat_circuit::import::pipe_record::{self, binary};
use runmat_circuit::import::Imported;
use runmat_circuit::runtime::{compile, CompileOptions, Files};
use std::collections::BTreeMap;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/import/pipe_record/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// The imported files compile with nothing unrouted and no DRC error.
fn compiles(out: &Imported) -> runmat_circuit::layout::Board {
    let files: Files = out.files.iter().map(|(k, v)| (format!("/p/{k}"), v.clone())).collect();
    let layout = "/p/boards/blinky.circuit.layout";
    let compiled = compile(layout, &files[layout], &files, &CompileOptions::default());
    let errors: Vec<_> = compiled.diagnostics.iter().filter(|d| d.severity == "error").map(|d| &d.message).collect();
    assert!(errors.is_empty(), "{errors:#?}\n{}", files["/p/boards/blinky.circuit"]);
    let board = compiled.index.expect("the imported board compiles").board.expect("the layout has a board");
    assert!(board.unrouted.is_empty(), "{:?}", board.unrouted);
    let drc: Vec<_> = board.drc.iter().filter(|r| r.severity == "error").map(|r| (&r.code, &r.message)).collect();
    assert!(drc.is_empty(), "{drc:#?}");
    runmat_circuit::syntax::board::parse_board(layout, &files[layout]).board.unwrap()
}

#[test]
fn an_ascii_schematic_and_board_import_to_a_circuit_that_compiles() {
    let files: BTreeMap<String, String> = [("blinky.SchDoc", fixture("blinky.SchDoc")), ("blinky.PcbDoc", fixture("blinky.PcbDoc"))].into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
    let out = pipe_record::import(&files, "boards/blinky").unwrap();
    let names: Vec<&str> = out.files.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "boards/blinky-footprints/HDR1X2.kicad_mod",
            "boards/blinky-footprints/LED0603.kicad_mod",
            "boards/blinky-footprints/R0603.kicad_mod",
            "boards/blinky.circuit",
            "boards/blinky.circuit.layout",
        ]
    );
    let circuit = &out.files["boards/blinky.circuit"];
    // Parts by part number, else library reference and value; the LED's current footprint.
    assert!(circuit.contains(r#"part "LED-0603-RED""#), "{circuit}");
    assert!(circuit.contains(r#"part "Res-1k""#), "{circuit}");
    assert!(circuit.contains("blinky-footprints/LED0603.kicad_mod"), "{circuit}");
    let r = &out.report;
    assert_eq!(r.format, "pipe-record ASCII");
    assert_eq!(r.carried["components"], 3, "{:?}", r.lines());
    // VCC and LED_A by their labels, GND by its power port on the middle of a wire.
    assert_eq!(r.carried["nets"], 3, "{:?}", r.lines());
    assert_eq!(r.carried["footprints placed"], 3);
    assert_eq!(r.carried["tracks"], 7, "{:?}", r.lines());
    assert_eq!(r.carried["vias"], 1);
    assert_eq!(r.carried["zones"], 1);
    assert_eq!(r.carried["board texts"], 1);
    assert_eq!(r.carried["net classes"], 1);
    assert_eq!(r.not_carried["solid regions"], 1, "{:?}", r.lines());
    assert_eq!(r.not_carried["dimensions"], 1);
    assert_eq!(r.not_carried["ShortCircuit rules"], 1);
    assert_eq!(r.not_carried["no-ERC markers"], 1);

    let board = compiles(&out);
    let led = board.footprints.values().find(|f| f.reference == "D1").unwrap();
    assert_eq!(led.side, runmat_circuit::layout::Side::Bottom);
    assert_eq!(led.rotation, 90.0);
    // 1000 × 800 mil, from the outline's corner.
    assert_eq!(board.outline.bbox().width(), 25_400_000);
    assert_eq!(board.outline.bbox().min.x, 0);
    assert_eq!(board.limits.min_clearance, runmat_circuit::geom::mm(0.2032));
    assert_eq!(board.nets.iter().find(|n| n.name == "GND").and_then(|n| n.class.as_deref()), Some("Power"));
}

#[test]
fn the_binary_files_import_their_text_records_and_report_the_binary_ones() {
    // The schematic's records in a FileHeader stream; the board's by kind, in their storages.
    let sch_text = fixture("blinky.SchDoc");
    let sch_lines: Vec<&str> = sch_text.lines().collect();
    let sch = binary::write_compound(&[("FileHeader", &binary::write_records(&sch_lines))]);
    let pcb_text = fixture("blinky.PcbDoc");
    let of = |kind: &str| -> Vec<u8> {
        let lines: Vec<String> = pcb_text.lines().filter(|l| l.starts_with(&format!("|RECORD={kind}|"))).map(|l| l.replacen(&format!("|RECORD={kind}"), "", 1)).collect();
        binary::write_records(&lines.iter().map(String::as_str).collect::<Vec<_>>())
    };
    let (board, nets, classes, rules, comps) = (of("Board"), of("Net"), of("Class"), of("Rule"), of("Component"));
    let pads = vec![1u8; 700];
    let pcb = binary::write_compound(&[("Board6/Data", &board), ("Nets6/Data", &nets), ("Classes6/Data", &classes), ("Rules6/Data", &rules), ("Components6/Data", &comps), ("Pads6/Data", &pads)]);
    let files: BTreeMap<String, Vec<u8>> = [("blinky.SchDoc".to_owned(), sch), ("blinky.PcbDoc".to_owned(), pcb)].into_iter().collect();
    let out = binary::import(&files, "boards/blinky").unwrap();
    let r = &out.report;
    assert_eq!(r.format, "pipe-record binary");
    assert_eq!(r.carried["components"], 3, "{:?}", r.lines());
    assert_eq!(r.carried["nets"], 3);
    assert_eq!(r.carried["footprints placed"], 3);
    assert_eq!(r.not_carried["pads (binary structures)"], 1, "{:?}", r.lines());
    assert!(out.files["boards/blinky.circuit.layout"].contains("D1"));

    let err = binary::import(&[("x.PcbDoc".to_owned(), b"|RECORD=Board|".to_vec())].into_iter().collect(), "x").unwrap_err();
    assert!(err.contains("not a compound file"), "{err}");
}

#[test]
fn a_document_without_records_is_refused_by_the_import() {
    let files: BTreeMap<String, String> = [("notes.txt".to_owned(), "hello".to_owned())].into_iter().collect();
    let err = pipe_record::import(&files, "x").unwrap_err();
    assert!(err.contains("isn't a pipe-record document"), "{err}");
}
