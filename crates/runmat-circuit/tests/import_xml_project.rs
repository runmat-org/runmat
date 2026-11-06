//! XML project import on a hand-written fixture (tests/import/xml_project: a header, a resistor and
//! an LED on a two-layer board, the LED mirrored to the bottom and turned 90°, a fiducial, and a copper polygon extending the header's pad 2).

use runmat_circuit::import::xml_project;
use runmat_circuit::runtime::{compile, CompileOptions, Files};
use std::collections::BTreeMap;

mod import_check;

fn fixture(names: &[&str]) -> BTreeMap<String, String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/import/xml_project");
    names.iter().map(|n| (n.to_string(), std::fs::read_to_string(format!("{dir}/{n}")).unwrap())).collect()
}

#[test]
fn a_schematic_and_board_import_to_a_circuit_that_compiles() {
    let out = xml_project::import(&fixture(&["blinky.sch", "blinky.brd"]), "boards/blinky").unwrap();
    let names: Vec<&str> = out.files.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "boards/blinky-footprints/connectors_HDR2.kicad_mod",
            "boards/blinky-footprints/leds_LED0603.kicad_mod",
            "boards/blinky-footprints/marks_FID1.kicad_mod",
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
    assert_eq!(r.carried["tracks"], 10, "{:?}", r.lines());
    assert_eq!(r.carried["vias"], 1);
    assert_eq!(r.carried["zones"], 1);
    assert!(r.approximated.keys().any(|k| k.contains("2 sheets")), "{:?}", r.lines());
    // A fiducial (a package without pins) is a footprint on the board, not a component.
    assert!(!circuit.contains("FD1"), "{circuit}");
    assert!(r.approximated.keys().any(|k| k.contains("without pins")), "{:?}", r.lines());

    let files: Files = out.files.iter().map(|(k, v)| (format!("/p/{k}"), v.clone())).collect();
    let layout = "/p/boards/blinky.circuit.layout";
    let compiled = compile(layout, &files[layout], &files, &CompileOptions::default());
    let errors: Vec<_> = compiled.diagnostics.iter().filter(|d| d.severity == "error").map(|d| &d.message).collect();
    assert!(errors.is_empty(), "{errors:#?}");
    let errors = import_check::design_errors(&files, "/p/boards/blinky.circuit", Some(layout));
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
    // J1's bottom copper polygon is more copper of pad 2 (the GND wire ends on it).
    let j1 = parsed.footprints.values().find(|f| f.reference == "J1").unwrap();
    assert_eq!(j1.package.pads.iter().filter(|p| p.name == "2").count(), 2);
    let led = parsed.footprints.values().find(|f| f.reference == "D1").unwrap();
    assert_eq!(led.side, runmat_circuit::layout::Side::Bottom);
    // MR90 in the file: rotate 90°, then mirror. Its anode lands at (10, 4.2), where the wire ends.
    let anode = led.pads(2).into_iter().find(|p| p.name == "A").unwrap();
    assert_eq!(anode.center, runmat_circuit::geom::Point::new(10_000_000, 4_200_000));
    assert_eq!(parsed.outline.bbox().width(), 20_000_000);
}

#[test]
fn a_file_that_isnt_an_xml_project_is_refused_with_the_reason() {
    let mut files = BTreeMap::new();
    files.insert("old.brd".to_string(), "\u{10}\u{0}binary".to_string());
    let err = xml_project::import(&files, "old").unwrap_err();
    assert!(err.contains("before version 6"), "{err}");
}

/// `RUNMAT_XML_PROJECT_CORPUS=<dir>`: every schematic and board pair under it (same stem, same
/// folder) imports, passes the design check without errors and leaves nothing unrouted. The corpus stays
/// outside the repository; without the variable the test checks nothing.
#[test]
fn a_local_corpus_imports_and_compiles_when_named() {
    let Ok(root) = std::env::var("RUNMAT_XML_PROJECT_CORPUS") else { return };
    let mut pairs = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(root)];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "brd") && p.with_extension("sch").exists() {
                pairs.push(p);
            }
        }
    }
    pairs.sort();
    let mut failures = Vec::new();
    for brd in &pairs {
        let sch = brd.with_extension("sch");
        let stem: String = brd.file_stem().unwrap().to_string_lossy().chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
        let mut files = BTreeMap::new();
        for p in [&sch, brd] {
            files.insert(p.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(p).unwrap());
        }
        let out = match xml_project::import(&files, &stem) {
            Ok(o) => o,
            Err(e) => { failures.push(format!("{}: {e}", brd.display())); continue; }
        };
        let files: Files = out.files.iter().map(|(k, v)| (format!("/p/{k}"), v.clone())).collect();
        let layout = format!("/p/{stem}.circuit.layout");
        let compiled = compile(&layout, &files[&layout], &files, &CompileOptions::default());
        let mut errors: Vec<_> = compiled.diagnostics.iter().filter(|d| d.severity == "error").map(|d| d.message.clone()).collect();
        errors.extend(import_check::design_errors(&files, &format!("/p/{stem}.circuit"), Some(&layout)));
        let board = compiled.index.and_then(|i| i.board);
        let unrouted: Vec<String> = board.iter().flat_map(|b| b.unrouted.iter().map(|u| format!("{u:?}"))).collect();
        let drc = board.iter().flat_map(|b| b.drc.iter()).filter(|r| r.severity == "error").count();
        let codes: BTreeMap<String, usize> = board.iter().flat_map(|b| b.drc.iter()).filter(|r| r.severity == "error").fold(BTreeMap::new(), |mut m, r| {
            *m.entry(format!("{}: {}", r.code, r.message.chars().take(90).collect::<String>())).or_default() += 1;
            m
        });
        eprintln!("{}: {} errors, {} unrouted, {drc} DRC errors {codes:#?}; {:?}", brd.display(), errors.len(), unrouted.len(), out.report.lines());
        if !errors.is_empty() || !unrouted.is_empty() {
            failures.push(format!("{}: {errors:#?} unrouted {unrouted:#?}", brd.display()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
