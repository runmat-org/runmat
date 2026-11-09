//! A differential pair on a real project, through the Board tab's commands: the USB demo board
//! (`tests/projects/usb`) has every net routed except its declared D+/D- pair. Diff pair routes
//! the pair at its gap and Tune length meanders one net, each leaving the board DRC clean.
//! `RUNMAT_PAIR_BUILD=1` rebuilds the board from its start file with the same commands the
//! drone boards use (place, pour, fan out, autoroute every other net).

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::board::parse_board;
use serde_json::{json, Value};
use std::path::PathBuf;

const NAME: &str = "usb_bridge";
const LAYOUT: &str = "/p/usb_bridge.circuit.layout";
const CIRCUIT: &str = "/p/usb_bridge.circuit";

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects/usb")
}

fn read(file: &str) -> Option<String> {
    std::fs::read_to_string(dir().join(file)).ok()
}

/// (designator, x, y, rotation), board millimetres from the board's center.
const PLACE: &[(&str, f64, f64, f64)] = &[
    ("U1", -5.0, 0.0, 0.0),
    ("C3", -5.0, -6.0, 0.0),
    ("C4", -5.0, 6.0, 0.0),
    ("C5", -1.6, -6.4, 0.0),
    ("R1", -11.2, 1.0, 90.0),
    ("R2", -11.2, -2.5, 90.0),
    ("J1", 9.9, 3.0, 90.0),
    ("U2", 7.0, -6.0, 0.0),
    ("C1", 10.6, -6.0, 90.0),
    ("C2", 3.4, -6.0, 90.0),
];

fn request(command: &str, values: Value, selection: &[&str]) -> EditRequest {
    EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: selection.iter().map(|s| s.to_string()).collect(),
        options: CompileOptions::default(),
    }
}

/// Runs `command` on `path` and takes every file it changes; the summary comes back.
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

/// The board's DRC codes and the nets left unrouted.
fn findings(files: &Files) -> (Vec<String>, Vec<String>) {
    let out = runtime::compile(LAYOUT, &files[LAYOUT], files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let index = out.index.expect("the board compiles");
    let b = index.board.as_ref().expect("a board");
    (
        b.drc
            .iter()
            .map(|r| format!("{} {}", r.code, r.message))
            .collect(),
        b.unrouted.iter().map(|a| a.net.clone()).collect(),
    )
}

fn build() -> Files {
    let mut files = Files::new();
    files.insert(CIRCUIT.into(), read("usb_bridge.circuit").unwrap());
    files.insert(
        LAYOUT.into(),
        read("usb_bridge.start.circuit.layout").unwrap(),
    );
    run(&mut files, CIRCUIT, "schematic.annotate", json!({}), &[]);
    run(
        &mut files,
        LAYOUT,
        "board.layer-stack",
        json!({ "layers": 2, "thickness": 1.6 }),
        &[],
    );
    let index = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    let changes: Vec<String> = index
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
    for (reference, x, y, r) in PLACE {
        let b = board(&files);
        let f = b
            .footprint_by_ref(reference)
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
    let all = [[-14.6, -9.6], [14.6, -9.6], [14.6, 9.6], [-14.6, 9.6]];
    run(
        &mut files,
        LAYOUT,
        "board.pour",
        json!({ "net": "net:GND", "layer": "B.Cu", "points": all, "connect": "relief" }),
        &[],
    );
    // The header's GND pad drops to the bottom plane above the header, so no GND track crosses
    // in front of the pair's pads.
    let j1 = board(&files)
        .footprint_by_ref("J1")
        .unwrap()
        .pads(2)
        .into_iter()
        .find(|p| p.name == "4")
        .unwrap()
        .center;
    let (x, y) = (
        runmat_circuit::geom::to_mm(j1.x),
        runmat_circuit::geom::to_mm(j1.y),
    );
    run(
        &mut files,
        LAYOUT,
        "board.route",
        json!({ "net": "net:GND", "points": [[x, y], [x, y + 1.2]], "width": 0.25, "layer": "Top", "mode": "ignore", "corner": "any" }),
        &[],
    );
    run(
        &mut files,
        LAYOUT,
        "board.place-via",
        json!({ "points": [[x, y + 1.2]], "net": "net:GND", "size": "0.2/0.45" }),
        &[],
    );
    let smd: Vec<String> = ["U1", "U2", "C1", "C2", "C3", "C4", "C5", "R1", "R2"]
        .iter()
        .map(|r| format!("component:{r}"))
        .collect();
    run(
        &mut files,
        LAYOUT,
        "board.fanout",
        json!({ "component": smd }),
        &[],
    );
    let others: Vec<String> = board(&files)
        .nets
        .iter()
        .map(|n| n.name.clone())
        .filter(|n| !n.starts_with("USB_"))
        .map(|n| format!("net:{n}"))
        .collect();
    run(
        &mut files,
        LAYOUT,
        "board.autoroute",
        json!({ "nets": others }),
        &[],
    );
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
    if std::env::var_os("RUNMAT_PAIR_BUILD").is_none() {
        return;
    }
    let files = build();
    let b = board(&files);
    for f in b.footprints.values() {
        for pad in f.pads(b.copper_count()) {
            let r = pad.prim.bbox();
            let net = pad
                .net
                .map(|n| b.net_name(n).to_owned())
                .unwrap_or_default();
            if !net.is_empty() && (f.reference != "U1" || pad.name.starts_with("3")) {
                eprintln!(
                    "pad {}.{} {:.3} {:.3} {net}",
                    f.reference,
                    pad.name,
                    runmat_circuit::geom::to_mm((r.min.x + r.max.x) / 2),
                    runmat_circuit::geom::to_mm((r.min.y + r.max.y) / 2)
                );
            }
        }
    }
    eprintln!("{:?}", findings(&files));
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
fn diff_pair_routes_the_usb_pair_at_its_gap_and_tune_length_meanders_it() {
    let mut files = checked_in();
    // The board as checked in: everything routed but the pair.
    let (drc, open) = findings(&files);
    assert!(drc.iter().all(|f| f.starts_with("unrouted")), "{drc:?}");
    assert!(
        !open.is_empty() && open.iter().all(|n| n == "USB_DP" || n == "USB_DN"),
        "{open:?}"
    );
    let summary = run(
        &mut files,
        LAYOUT,
        "board.diff-pair",
        json!({ "pair": "net:USB_DP", "width": 0.12, "gap": 0.15 }),
        &[],
    );
    assert!(
        summary.starts_with("Routed USB on F.Cu at 0.120 mm width, 0.150 mm gap: USB_DP"),
        "{summary}"
    );
    let (drc, open) = findings(&files);
    assert!(drc.is_empty(), "{drc:?}");
    assert!(open.is_empty(), "{open:?}");
    let b = board(&files);
    let length = |net: &str| runmat_circuit::pair::net_track_length(&b, b.net(net).unwrap());
    let (dp, dn) = (length("USB_DP"), length("USB_DN"));
    assert!((dp - dn).abs() < 1.0, "skew {dp} {dn}");
    // The board projection gives each pair net its partner, its length and the skew, for
    // Properties.
    let out = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    let index = out.index.unwrap();
    let rows = &index.board.as_ref().unwrap().nets;
    let row = rows.iter().find(|r| r.name == "USB_DP").unwrap();
    assert_eq!(row.pair.as_deref(), Some("USB_DN"));
    assert!(
        (row.length_mm - dp).abs() < 1e-3 && (row.skew_mm.unwrap() - (dp - dn).abs()).abs() < 1e-3,
        "{row:?} {dp} {dn}"
    );
    let gnd = rows.iter().find(|r| r.name == "GND").unwrap();
    assert!(gnd.pair.is_none() && gnd.skew_mm.is_none(), "{gnd:?}");
    assert_eq!(
        rows.iter().find(|r| r.name == "VBUS").unwrap().width_mm,
        Some(0.25)
    );
    // Tune both nets to 16 mm (a length budget the pair must meet): each gets meanders on its
    // outer side, and the board stays clean, skew included.
    for net in ["USB_DN", "USB_DP"] {
        let summary = run(
            &mut files,
            LAYOUT,
            "board.tune-length",
            json!({ "net": format!("net:{net}"), "target": 16.0, "tolerance": 0.05 }),
            &[],
        );
        eprintln!("{summary}");
        assert!(summary.starts_with(&format!("Tuned {net}")), "{summary}");
        let b = board(&files);
        let after = runmat_circuit::pair::net_track_length(&b, b.net(net).unwrap());
        assert!((after - 16.0).abs() < 0.05, "{net}: {after}");
    }
    let (drc, open) = findings(&files);
    assert!(drc.is_empty(), "{drc:?}");
    assert!(open.is_empty(), "{open:?}");
    // Meanders only add length.
    let shorter = runtime::edit(
        LAYOUT,
        &files[LAYOUT],
        &files,
        &request(
            "board.tune-length",
            json!({ "net": "net:USB_DN", "target": 10.0 }),
            &[],
        ),
    );
    assert!(shorter.is_err_and(|e| e.message.contains("longer than")));
    if let Some(out) = std::env::var_os("RUNMAT_PAIR_OUT") {
        std::fs::write(
            PathBuf::from(out).join("usb_bridge-tuned.circuit.layout"),
            &files[LAYOUT],
        )
        .unwrap();
    }
}

/// The Desktop runs board commands through a session, which keeps the board between edits; the
/// pair and the board tools need the circuit too, so the session hands them to the project.
#[test]
fn a_session_runs_diff_pair_and_tune_length() {
    let files = checked_in();
    let mut session = runtime::Session::new();
    let out = session
        .edit(
            LAYOUT,
            &files[LAYOUT],
            &files,
            &request("board.diff-pair", json!({ "pair": "net:USB_DN" }), &[]),
        )
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert!(
        out.summary.starts_with("Routed USB on F.Cu"),
        "{}",
        out.summary
    );
    let mut files = files;
    for f in out.files {
        files.insert(f.path.clone(), f.text.clone());
    }
    let out = session
        .edit(
            LAYOUT,
            &files[LAYOUT],
            &files,
            &request(
                "board.tune-length",
                json!({ "net": "net:USB_DN", "target": 16.0, "tolerance": 0.05 }),
                &[],
            ),
        )
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert!(out.summary.starts_with("Tuned USB_DN"), "{}", out.summary);
    for command in ["board.stitch", "board.teardrops"] {
        // Each reaches its own runner: whatever it says, it is not the session's refusal.
        if let Err(e) = session.edit(
            LAYOUT,
            &files[LAYOUT],
            &files,
            &request(command, json!({}), &[]),
        ) {
            assert!(
                !e.message.contains("isn't a model edit"),
                "{command}: {}",
                e.message
            );
        }
    }
}
