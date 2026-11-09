//! Pipe-record import on hand-written fixtures (tests/import/pipe_record): a header, a
//! resistor and an LED, the LED rotated and on the bottom; a via, a polygon pour, a solid
//! copper region, a board text, a net class with its width rule. The binary test packs the
//! same records into a compound file the way the binary files hold them (text records in
//! their storages, pads, tracks, vias, texts and regions as binary structures) and checks
//! that it prints the same layout as the ASCII files.

use runmat_circuit::import::pipe_record::{self, binary, primitives, Record};
use runmat_circuit::import::Imported;
use runmat_circuit::runtime::{compile, CompileOptions, Files};
use std::collections::BTreeMap;

mod import_check;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/import/pipe_record/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// The imported files compile with nothing unrouted and no DRC error.
fn compiles(out: &Imported) -> runmat_circuit::layout::Board {
    let files: Files = out
        .files
        .iter()
        .map(|(k, v)| (format!("/p/{k}"), v.clone()))
        .collect();
    let layout = "/p/boards/blinky.circuit.layout";
    let compiled = compile(layout, &files[layout], &files, &CompileOptions::default());
    let errors: Vec<_> = compiled
        .diagnostics
        .iter()
        .filter(|d| d.severity == "error")
        .map(|d| &d.message)
        .collect();
    assert!(
        errors.is_empty(),
        "{errors:#?}\n{}",
        files["/p/boards/blinky.circuit"]
    );
    let errors = import_check::design_errors(&files, "/p/boards/blinky.circuit", Some(layout));
    assert!(
        errors.is_empty(),
        "{errors:#?}\n{}",
        files["/p/boards/blinky.circuit"]
    );
    let board = compiled
        .index
        .expect("the imported board compiles")
        .board
        .expect("the layout has a board");
    assert!(board.unrouted.is_empty(), "{:?}", board.unrouted);
    let drc: Vec<_> = board
        .drc
        .iter()
        .filter(|r| r.severity == "error")
        .map(|r| (&r.code, &r.message))
        .collect();
    assert!(drc.is_empty(), "{drc:#?}");
    runmat_circuit::syntax::board::parse_board(layout, &files[layout])
        .board
        .unwrap()
}

#[test]
fn an_ascii_schematic_and_board_import_to_a_circuit_that_compiles() {
    let files: BTreeMap<String, String> = [
        ("blinky.SchDoc", fixture("blinky.SchDoc")),
        ("blinky.PcbDoc", fixture("blinky.PcbDoc")),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
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
    assert!(
        circuit.contains("blinky-footprints/LED0603.kicad_mod"),
        "{circuit}"
    );
    let r = &out.report;
    assert_eq!(r.format, ".SchDoc / .PcbDoc (ASCII)");
    assert_eq!(r.carried["components"], 3, "{:?}", r.lines());
    // VCC and LED_A by their labels, GND by its power port on the middle of a wire.
    assert_eq!(r.carried["nets"], 3, "{:?}", r.lines());
    assert_eq!(r.carried["footprints placed"], 3);
    assert_eq!(r.carried["tracks"], 7, "{:?}", r.lines());
    assert_eq!(r.carried["vias"], 1);
    // The polygon pour, and the solid copper region over J1's GND pad.
    assert_eq!(r.carried["zones"], 2);
    assert_eq!(r.carried["board texts"], 1);
    assert_eq!(r.carried["net classes"], 1);
    assert_eq!(
        r.approximated["solid copper regions are carried as pours of their net"],
        1,
        "{:?}",
        r.lines()
    );
    assert_eq!(r.not_carried["dimensions"], 1);
    assert_eq!(r.not_carried["ShortCircuit rules"], 1);
    assert_eq!(r.not_carried["no-ERC markers"], 1);

    let board = compiles(&out);
    let led = board
        .footprints
        .values()
        .find(|f| f.reference == "D1")
        .unwrap();
    assert_eq!(led.side, runmat_circuit::layout::Side::Bottom);
    assert_eq!(led.rotation, 90.0);
    // 1000 × 800 mil, from the outline's corner.
    assert_eq!(board.outline.bbox().width(), 25_400_000);
    assert_eq!(board.outline.bbox().min.x, 0);
    assert_eq!(board.limits.min_clearance, runmat_circuit::geom::mm(0.2032));
    // The pour cutout inside the bottom GND pour: a keepout for pours on the bottom layer.
    assert_eq!(r.carried["pour cutouts"], 1, "{:?}", r.lines());
    let cut = board
        .keepouts
        .values()
        .next()
        .expect("the pour cutout is a keepout");
    assert_eq!(cut.layers, [1]);
    assert!(
        cut.rules.pours && !cut.rules.tracks && !cut.rules.vias && !cut.rules.pads,
        "{:?}",
        cut.rules
    );
    assert_eq!(cut.outline.bbox().width(), runmat_circuit::geom::mm(1.27));
    // Pour connect rules: GND joins solid; through-hole and SMD pads elsewhere by thermal
    // relief with their spoke width and gap; vias solid. The rule on a footprint is reported.
    use runmat_circuit::board_rules::{props, RuleValue};
    let rule = |id: &str| {
        board.rules.iter().find(|r| r.id == id).unwrap_or_else(|| {
            panic!(
                "{id}: {:?}",
                board.rules.iter().map(|r| &r.id).collect::<Vec<_>>()
            )
        })
    };
    let gnd = rule("PolygonConnect_1");
    assert_eq!(gnd.selector, "*:where(*:net(\"GND\"))");
    assert!(
        gnd.decls.contains(&(
            props::ZONE_CONNECT.to_string(),
            RuleValue::Text("solid".into())
        )),
        "{gnd:?}"
    );
    let tht = rule("PolygonConnect thpad");
    assert_eq!(tht.selector, "*:where(pad.tht)");
    assert!(
        tht.decls.contains(&(
            props::SPOKE_WIDTH.to_string(),
            RuleValue::Len(runmat_circuit::geom::mm(0.3048))
        )),
        "{tht:?}"
    );
    assert!(
        tht.decls.contains(&(
            props::THERMAL_GAP.to_string(),
            RuleValue::Len(runmat_circuit::geom::mm(0.254))
        )),
        "{tht:?}"
    );
    assert!(gnd.priority > tht.priority);
    assert_eq!(rule("PolygonConnect via").selector, "*:where(via)");
    assert_eq!(
        out.report.carried["pour connect rules"],
        4,
        "{:?}",
        out.report.lines()
    );
    assert_eq!(
        out.report.not_carried["pour connect rules scoped to HasFootprint('J1') and All"],
        1,
        "{:?}",
        out.report.lines()
    );
    assert_eq!(
        board
            .nets
            .iter()
            .find(|n| n.name == "GND")
            .and_then(|n| n.class.as_deref()),
        Some("Power")
    );
}

/// A length field of `r` in 1/10000 mil, as the binary records hold it.
fn mil(r: &Record, key: &str) -> [u8; 4] {
    primitives::le_mil(r.len(key) as f64 / 25_400.0)
}

fn layer_of(r: &Record, key: &str) -> u8 {
    primitives::layer_number(r.str(key)).unwrap_or_else(|| panic!("layer {}", r.str(key)))
}

/// One ASCII board record in its binary form: kind byte and blocks.
fn encode(r: &Record) -> (u8, Vec<Vec<u8>>) {
    // A via without a layer spans them all.
    let layer = if r.has("LAYER") {
        layer_of(r, "LAYER")
    } else {
        74
    };
    let head = |rest: &[u8]| {
        primitives::head(
            layer,
            r.index("NET").map(|n| n as u16),
            r.index("COMPONENT").map(|n| n as u16),
            rest,
        )
    };
    let cat = |parts: &[&[u8]]| parts.concat();
    let pad_to = |mut b: Vec<u8>, n: usize| {
        b.resize(n, 0);
        b
    };
    match r.kind() {
        "Pad" => {
            let shape = if r.str("SHAPE") == "ROUND" { 1 } else { 2 };
            let size = cat(&[&mil(r, "XSIZE"), &mil(r, "YSIZE")]);
            let rest = cat(&[
                &mil(r, "X"),
                &mil(r, "Y"),
                &size,
                &size,
                &size,
                &mil(r, "HOLESIZE"),
                &[shape; 3],
                &r.num("ROTATION").to_le_bytes(),
                &[r.flag("PLATED") as u8],
            ]);
            let name = cat(&[&[r.str("NAME").len() as u8], r.str("NAME").as_bytes()]);
            (
                2,
                vec![
                    name,
                    vec![0],
                    b"\x04|&|0".to_vec(),
                    vec![0],
                    pad_to(head(&rest), 185),
                    vec![],
                ],
            )
        }
        "Track" => (
            4,
            vec![pad_to(
                head(&cat(&[
                    &mil(r, "X1"),
                    &mil(r, "Y1"),
                    &mil(r, "X2"),
                    &mil(r, "Y2"),
                    &mil(r, "WIDTH"),
                ])),
                49,
            )],
        ),
        "Via" => (
            3,
            vec![pad_to(
                head(&cat(&[
                    &mil(r, "X"),
                    &mil(r, "Y"),
                    &mil(r, "DIAMETER"),
                    &mil(r, "HOLESIZE"),
                    &[layer_of(r, "STARTLAYER"), layer_of(r, "ENDLAYER")],
                ])),
                321,
            )],
        ),
        "Text" => {
            let rest = cat(&[
                &mil(r, "X"),
                &mil(r, "Y"),
                &mil(r, "HEIGHT"),
                &[0, 0],
                &r.num("ROTATION").to_le_bytes(),
                &[r.flag("MIRROR") as u8],
                &mil(r, "WIDTH"),
            ]);
            (
                5,
                vec![
                    pad_to(head(&rest), 252),
                    cat(&[&[r.str("TEXT").len() as u8], r.str("TEXT").as_bytes()]),
                ],
            )
        }
        "Region" => {
            let params = format!("|KIND={}|\0", r.str("KIND"));
            let mut ring: Vec<u8> = Vec::new();
            let n = (0..).take_while(|i| r.has(&format!("VX{i}"))).count();
            ring.extend((n as u32).to_le_bytes());
            for i in 0..n {
                for k in ["VX", "VY"] {
                    ring.extend(
                        (r.len(&format!("{k}{i}")) as f64 / 25_400.0 * 10_000.0).to_le_bytes(),
                    );
                }
            }
            (
                11,
                vec![cat(&[
                    &head(&[0; 5]),
                    &(params.len() as u32).to_le_bytes(),
                    params.as_bytes(),
                    &ring,
                ])],
            )
        }
        other => panic!("no binary form for {other}"),
    }
}

#[test]
fn the_binary_files_import_the_same_board_as_the_ascii_ones() {
    // The schematic's records in a FileHeader stream; the board's text records by kind, in
    // their storages, and its primitives as binary structures.
    let sch_text = fixture("blinky.SchDoc");
    let sch_lines: Vec<&str> = sch_text.lines().collect();
    let sch = binary::write_compound(&[("FileHeader", &binary::write_records(&sch_lines))]);
    let pcb_text = fixture("blinky.PcbDoc");
    let of = |kind: &str| -> Vec<u8> {
        let lines: Vec<String> = pcb_text
            .lines()
            .filter(|l| l.starts_with(&format!("|RECORD={kind}|")))
            .map(|l| l.replacen(&format!("|RECORD={kind}"), "", 1))
            .collect();
        binary::write_records(&lines.iter().map(String::as_str).collect::<Vec<_>>())
    };
    let prims = |kind: &str| -> Vec<u8> {
        let recs: Vec<_> = pipe_record::records(&pcb_text)
            .into_iter()
            .filter(|r| r.kind() == kind)
            .map(|r| encode(&r))
            .collect();
        primitives::write(&recs)
    };
    let (board, nets, classes, rules, comps, polygons) = (
        of("Board"),
        of("Net"),
        of("Class"),
        of("Rule"),
        of("Component"),
        of("Polygon"),
    );
    let (pads, tracks, vias, texts, regions) = (
        prims("Pad"),
        prims("Track"),
        prims("Via"),
        prims("Text"),
        prims("Region"),
    );
    // A board keeps its version text in a FileHeader stream too.
    let header = binary::write_records(&["|KIND=PCB board|"]);
    let pcb = binary::write_compound(&[
        ("FileHeader", &header),
        ("Board6/Data", &board),
        ("Nets6/Data", &nets),
        ("Classes6/Data", &classes),
        ("Rules6/Data", &rules),
        ("Components6/Data", &comps),
        ("Polygons6/Data", &polygons),
        ("Pads6/Data", &pads),
        ("Tracks6/Data", &tracks),
        ("Vias6/Data", &vias),
        ("Texts6/Data", &texts),
        ("Regions6/Data", &regions),
        ("ComponentBodies6/Data", &[1, 2, 3]),
    ]);
    let files: BTreeMap<String, Vec<u8>> = [
        ("blinky.SchDoc".to_owned(), sch),
        ("blinky.PcbDoc".to_owned(), pcb),
    ]
    .into_iter()
    .collect();
    let out = binary::import(&files, "boards/blinky").unwrap();
    let r = &out.report;
    assert_eq!(r.format, ".SchDoc / .PcbDoc (binary)");
    for (kind, n) in [
        ("components", 3),
        ("nets", 3),
        ("footprints placed", 3),
        ("tracks", 7),
        ("vias", 1),
        ("zones", 2),
        ("pour cutouts", 1),
        ("board texts", 1),
        ("net classes", 1),
    ] {
        assert_eq!(r.carried.get(kind), Some(&n), "{kind}: {:?}", r.lines());
    }
    assert_eq!(
        r.not_carried["3D bodies of components (binary structures)"],
        1,
        "{:?}",
        r.lines()
    );
    let board = compiles(&out);
    let led = board
        .footprints
        .values()
        .find(|f| f.reference == "D1")
        .unwrap();
    assert_eq!(
        (led.side, led.rotation),
        (runmat_circuit::layout::Side::Bottom, 90.0)
    );
    let j1 = board
        .footprints
        .values()
        .find(|f| f.reference == "J1")
        .unwrap();
    assert_eq!(j1.package.pads.len(), 2);
    assert_eq!(
        j1.package.pads[0].drill.as_ref().map(|d| d.diameter),
        Some(runmat_circuit::geom::mm(1.016))
    );
    // The ASCII board and the binary one print the same layout.
    let ascii: BTreeMap<String, String> = [
        ("blinky.SchDoc", sch_text.clone()),
        ("blinky.PcbDoc", pcb_text.clone()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    let from_ascii = pipe_record::import(&ascii, "boards/blinky").unwrap();
    assert_eq!(
        out.files["boards/blinky.circuit.layout"],
        from_ascii.files["boards/blinky.circuit.layout"]
    );

    let err = binary::import(
        &[("x.PcbDoc".to_owned(), b"|RECORD=Board|".to_vec())]
            .into_iter()
            .collect(),
        "x",
    )
    .unwrap_err();
    assert!(err.contains("not a compound file"), "{err}");
}

#[test]
fn a_document_without_records_is_refused_by_the_import() {
    let files: BTreeMap<String, String> = [("notes.txt".to_owned(), "hello".to_owned())]
        .into_iter()
        .collect();
    let err = pipe_record::import(&files, "x").unwrap_err();
    assert!(
        err.contains("isn't an ASCII .SchDoc or .PcbDoc document"),
        "{err}"
    );
}

/// A board alone imports with an empty circuit stub, so its layout passes the design check;
/// the missing schematic is reported.
#[test]
fn a_board_alone_imports_to_a_layout_that_compiles() {
    let files: BTreeMap<String, String> = [("blinky.PcbDoc".to_owned(), fixture("blinky.PcbDoc"))]
        .into_iter()
        .collect();
    let out = pipe_record::import(&files, "boards/blinky").unwrap();
    assert!(
        out.report
            .not_carried
            .keys()
            .any(|k| k.starts_with("the schematic")),
        "{:?}",
        out.report.lines()
    );
    compiles(&out);
}
