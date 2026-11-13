//! Lengths through series parts on a real board: a memory byte lane (`tests/projects/ddr`),
//! an MCU driving eight data lines through 10 Ω series resistors to a 2x5 header. Each
//! line's two nets (MCU to resistor, resistor to connector) are one signal in a delay group,
//! and Tune group brings every line to the group's delay.
//! `RUNMAT_DDR_BUILD=1` rebuilds the board from its start file with the board commands (layer
//! stack, update from the schematic, place, pour, fan out, autoroute) and writes the group's
//! target 10 ps beyond its slowest line.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::board::parse_board;
use serde_json::{json, Value};
use std::path::PathBuf;

const NAME: &str = "ddr_lane";
const LAYOUT: &str = "/p/ddr_lane.circuit.layout";
const CIRCUIT: &str = "/p/ddr_lane.circuit";
const RULES_AFTER: &str = "    net_class supply = { track_width = 0.25 mm, clearance = 0.1 mm }\n";

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects/ddr")
}

fn read(file: &str) -> Option<String> {
    std::fs::read_to_string(dir().join(file)).ok()
}

fn request(command: &str, values: Value, selection: &[&str]) -> EditRequest {
    EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: selection.iter().map(|s| s.to_string()).collect(),
        options: CompileOptions::default(),
    }
}

fn run(files: &mut Files, path: &str, command: &str, values: Value, selection: &[&str]) -> String {
    let out = runtime::edit(
        path,
        &files[path],
        files,
        &request(command, values, selection),
    )
    .unwrap_or_else(|e| panic!("{command}: {} {:?}", e.message, e.diagnostics));
    for f in out.files {
        files.insert(f.path.clone(), f.text.clone());
    }
    out.summary
}

fn board(files: &Files) -> runmat_circuit::layout::Board {
    parse_board(LAYOUT, &files[LAYOUT])
        .board
        .expect("the layout reads")
}

fn index(files: &Files) -> runmat_circuit::index::BoardIndex {
    let out = runtime::compile(LAYOUT, &files[LAYOUT], files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    out.index.expect("compiles").board.expect("a board")
}

/// (designator, x, y, rotation), board millimetres from the board's center.
fn placement() -> Vec<(String, f64, f64, f64)> {
    let mut v = vec![
        ("U1".to_owned(), -11.0, 0.0, 0.0),
        ("C1".to_owned(), -11.0, -7.5, 0.0),
        ("J1".to_owned(), 15.0, 0.0, 0.0),
    ];
    for k in 0..8 {
        v.push((format!("R{}", k + 1), 0.0, 8.75 - 2.5 * k as f64, 0.0));
    }
    v
}

fn build() -> Files {
    let mut files = Files::new();
    files.insert(CIRCUIT.into(), read("ddr_lane.circuit").unwrap());
    files.insert(
        LAYOUT.into(),
        read("ddr_lane.start.circuit.layout").unwrap(),
    );
    run(&mut files, CIRCUIT, "schematic.annotate", json!({}), &[]);
    run(
        &mut files,
        LAYOUT,
        "board.layer-stack",
        json!({ "layers": 4, "thickness": 1.6 }),
        &[],
    );
    let out = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    let changes: Vec<String> = out
        .index
        .as_ref()
        .expect("compiles")
        .eco
        .groups
        .iter()
        .flat_map(|g| g.rows.iter())
        .filter(|r| r.status != "error")
        .map(|r| format!("eco:{}", r.id))
        .collect();
    run(
        &mut files,
        LAYOUT,
        "schematic.update-board",
        json!({ "changes": changes }),
        &[],
    );
    for (reference, x, y, r) in placement() {
        let b = board(&files);
        let f = b
            .footprint_by_ref(&reference)
            .unwrap_or_else(|| panic!("no {reference}"));
        let (fx, fy) = (
            runmat_circuit::geom::to_mm(f.at.x),
            runmat_circuit::geom::to_mm(f.at.y),
        );
        let turn = (r - f.rotation).rem_euclid(360.0);
        let round = |v: f64| (v * 1e6).round() / 1e6;
        run(
            &mut files,
            LAYOUT,
            "board.move",
            json!({ "dx": round(x - fx), "dy": round(y - fy), "rotation": turn }),
            &[&format!("component:{reference}")],
        );
    }
    let all = [[-19.6, -11.6], [19.6, -11.6], [19.6, 11.6], [-19.6, 11.6]];
    run(
        &mut files,
        LAYOUT,
        "board.pour",
        json!({ "net": "net:GND", "layer": "In1.Cu", "points": all, "connect": "relief" }),
        &[],
    );
    run(
        &mut files,
        LAYOUT,
        "board.pour",
        json!({ "net": "net:+3V3", "layer": "In2.Cu", "points": all, "connect": "relief" }),
        &[],
    );
    run(
        &mut files,
        LAYOUT,
        "board.fanout",
        json!({ "component": ["component:U1", "component:C1"] }),
        &[],
    );
    let lines: Vec<String> = (0..8)
        .flat_map(|k| [format!("net:MDQ{k}"), format!("net:DQ{k}")])
        .collect();
    run(
        &mut files,
        LAYOUT,
        "board.autoroute",
        json!({ "nets": lines }),
        &[],
    );
    // The byte lane: every DQ line through its resistor, in ps, 10 ps beyond the slowest now.
    let with_rule = |files: &Files, target: f64| {
        let rules = format!(
            "    rule byte0 = {{ select = \":net(DQ*)\", match_group = byte0, match_target = {target} ps, match_tolerance = 2 ps, through_parts = true }}\n    rule series = {{ select = \"footprint#R1, footprint#R2, footprint#R3, footprint#R4, footprint#R5, footprint#R6, footprint#R7, footprint#R8\", through_delay = 5 ps }}\n"
        );
        files[LAYOUT].replacen(RULES_AFTER, &format!("{RULES_AFTER}{rules}"), 1)
    };
    let mut probe = files.clone();
    probe.insert(LAYOUT.into(), with_rule(&files, 1000.0));
    let slowest = index(&probe).length_groups[0]
        .members
        .iter()
        .map(|m| m.delay_ps)
        .fold(0.0, f64::max);
    let text = with_rule(&files, (slowest + 10.0).round());
    files.insert(LAYOUT.into(), text);
    files
}

fn checked_in() -> Files {
    let mut files = Files::new();
    for x in [".circuit", ".circuit.layout", ".circuit.lock"] {
        if let Some(text) = read(&format!("{NAME}{x}")) {
            files.insert(format!("/p/{NAME}{x}"), text);
        }
    }
    files
}

#[test]
fn rebuild_when_asked() {
    if std::env::var_os("RUNMAT_DDR_BUILD").is_none() {
        return;
    }
    let files = build();
    let b = index(&files);
    eprintln!(
        "drc {:#?}",
        b.drc
            .iter()
            .map(|r| format!("{} {}", r.code, r.message))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "unrouted {:?}",
        b.unrouted.iter().map(|a| a.net.clone()).collect::<Vec<_>>()
    );
    for (path, file) in [
        (LAYOUT.to_owned(), format!("{NAME}.circuit.layout")),
        (
            format!("/p/{NAME}.circuit.lock"),
            format!("{NAME}.circuit.lock"),
        ),
        (CIRCUIT.to_owned(), format!("{NAME}.circuit")),
    ] {
        std::fs::write(dir().join(file), &files[&path]).unwrap();
    }
}

#[test]
fn tune_group_brings_each_line_through_its_resistor_to_the_lanes_delay() {
    let mut files = checked_in();
    let b = index(&files);
    assert!(
        b.unrouted.is_empty(),
        "{:?}",
        b.unrouted.iter().map(|a| &a.net).collect::<Vec<_>>()
    );
    let g = b
        .length_groups
        .iter()
        .find(|g| g.name == "byte0")
        .expect("the byte0 group");
    assert_eq!(
        (g.unit.as_str(), g.written, g.members.len()),
        ("ps", true, 8),
        "{g:?}"
    );
    // Each member is one signal: its connector net, its resistor and the MCU side.
    for (k, m) in g.members.iter().enumerate() {
        assert_eq!(m.net, format!("DQ{k}"));
        assert_eq!(m.through, vec![format!("R{}", k + 1)], "{m:?}");
        assert_eq!(m.status, "short", "{m:?}");
        // Copper on both sides of the resistor, plus its 5 ps.
        let b2 = board(&files);
        let own = runmat_circuit::pair::net_track_length(&b2, b2.net(&m.net).unwrap());
        let mcu = runmat_circuit::pair::net_track_length(&b2, b2.net(&format!("MDQ{k}")).unwrap());
        assert!(
            (m.length_mm - (own + mcu)).abs() < 0.05,
            "{m:?} {own} {mcu}"
        );
    }
    assert_eq!(
        b.drc.iter().filter(|r| r.code == "length-match").count(),
        8,
        "{:#?}",
        b.drc
    );
    let summary = run(
        &mut files,
        LAYOUT,
        "board.tune-group",
        json!({ "group": "byte0", "pattern": "trombone" }),
        &[],
    );
    assert!(
        summary.starts_with("Tuned byte0: ") && !summary.contains("not tuned"),
        "{summary}"
    );
    let b = index(&files);
    let g = b.length_groups.iter().find(|g| g.name == "byte0").unwrap();
    assert!(g.members.iter().all(|m| m.status == "matched"), "{g:#?}");
    assert!(
        b.drc.is_empty(),
        "{:#?}",
        b.drc
            .iter()
            .map(|r| format!("{} {}", r.code, r.message))
            .collect::<Vec<_>>()
    );
    if let Some(out) = std::env::var_os("RUNMAT_DDR_OUT") {
        std::fs::write(
            PathBuf::from(out).join(format!("{NAME}.tuned.circuit.layout")),
            &files[LAYOUT],
        )
        .unwrap();
    }
}
