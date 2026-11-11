//! The autorouter on a wide power-class net that has no room to join its islands: the finished
//! drone ESC (runmat-cad/wasm/tests/drone/electrical/esc) with the centre column's GND vias taken
//! out, so the bulk capacitors' and the regulator's ground pads (C1.2, C4.2, U1.2) are islands.
//! GND's class asks for 1 mm tracks and 0.2 mm clearance. Each autoroute pass must leave DRC
//! as clean as it found it, count as routed only connections that joined islands, and add no
//! copper when it joins nothing, so repeated passes give the same board. The same steps on
//! VBAT route clean.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::board::parse_board;
use serde_json::json;
use std::path::PathBuf;

const LAYOUT: &str = "/p/esc.circuit.layout";

/// The ESC's files with `net`'s vias within 1.5 mm of the board's centre line (x) removed.
fn esc_without_centre_vias(net: &str) -> (Files, usize) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runmat-cad/wasm/tests/drone/electrical/esc");
    let mut files = Files::new();
    for name in ["esc.circuit", "esc.circuit.layout", "esc.circuit.lock"] {
        files.insert(format!("/p/{name}"), std::fs::read_to_string(dir.join(name)).unwrap());
    }
    let prefix = format!("via {net} = {{ at = [");
    let text = files[LAYOUT].clone();
    let kept: Vec<&str> = text
        .lines()
        .filter(|l| {
            let Some(rest) = l.trim_start().strip_prefix(&prefix) else { return true };
            let x: f64 = rest.split(',').next().unwrap().trim().parse().unwrap();
            x.abs() > 1.5
        })
        .collect();
    let removed = text.lines().count() - kept.len();
    files.insert(LAYOUT.to_owned(), kept.join("\n") + "\n");
    (files, removed)
}

struct State {
    /// DRC findings other than the unrouted ones.
    drc: Vec<String>,
    unrouted: usize,
    tracks: usize,
    vias: usize,
}

fn state(files: &Files) -> State {
    let out = runtime::compile(LAYOUT, &files[LAYOUT], files, &CompileOptions::default());
    let index = out.index.expect("compiles");
    let board = index.board.as_ref().expect("a board");
    let parsed = parse_board(LAYOUT, &files[LAYOUT]).board.unwrap();
    State {
        drc: board.drc.iter().filter(|r| r.code != "unrouted").map(|r| format!("{} {}", r.code, r.message)).collect(),
        unrouted: board.unrouted.len(),
        tracks: parsed.tracks.len(),
        vias: parsed.vias.len(),
    }
}

/// "Routed N of M ..." → N.
fn routed(summary: &str) -> usize {
    summary.strip_prefix("Routed ").and_then(|s| s.split(' ').next()).and_then(|n| n.parse().ok()).expect(summary)
}

fn autoroute(files: &mut Files, net: &str) -> String {
    let request = EditRequest {
        command: "board.autoroute".into(),
        values: json!({ "nets": [format!("net:{net}")] }).as_object().cloned().unwrap(),
        selection: vec![],
        options: CompileOptions::default(),
    };
    let out = runtime::edit(LAYOUT, &files[LAYOUT], files, &request).unwrap_or_else(|e| panic!("{}", e.message));
    for f in out.files {
        files.insert(f.path, f.text);
    }
    out.summary
}

/// Three passes on `net`: none adds a DRC finding, "routed" never exceeds the connections that
/// closed, and once a pass joins nothing the board stops changing.
fn passes(net: &str) -> State {
    let (mut files, removed) = esc_without_centre_vias(net);
    assert!(removed > 0, "{net}: no centre vias to remove");
    let before = state(&files);
    assert!(before.drc.is_empty(), "{net}: DRC before routing {:?}", before.drc);
    assert!(before.unrouted > 0, "{net}: nothing to route");
    let mut last = before;
    let mut settled: Option<String> = None;
    for pass in 0..3 {
        let summary = autoroute(&mut files, net);
        let now = state(&files);
        eprintln!("{net} pass {pass}: {summary} | tracks {} vias {} unrouted {} drc {}", now.tracks, now.vias, now.unrouted, now.drc.len());
        assert!(now.drc.is_empty(), "{net} pass {pass}: {summary}: DRC {:?}", now.drc);
        let joined = last.unrouted.saturating_sub(now.unrouted);
        assert!(routed(&summary) <= joined, "{net} pass {pass}: \"{summary}\" but {joined} connection(s) closed");
        if joined == 0 {
            assert_eq!((now.tracks, now.vias), (last.tracks, last.vias), "{net} pass {pass}: {summary}: copper added that joins nothing");
        }
        if let Some(text) = &settled {
            assert_eq!(&files[LAYOUT], text, "{net} pass {pass}: a repeated pass changed the board");
        }
        if now.unrouted == 0 || joined == 0 {
            settled = Some(files[LAYOUT].clone());
        }
        last = now;
    }
    last
}

#[test]
fn a_blocked_power_net_adds_no_copper_and_keeps_clearance() {
    passes("GND");
}

#[test]
fn a_power_net_with_room_routes_clean() {
    let end = passes("VBAT");
    assert_eq!(end.unrouted, 0, "VBAT left unrouted");
}

/// With no via asked for, the router's vias meet the board's limits: on the ESC with the limits
/// raised to a 0.3 mm drill, 0.6 mm vias and a 0.15 mm ring, every via the VBAT pass adds is at
/// least that size.
#[test]
fn new_vias_meet_the_board_limits() {
    let (mut files, _) = esc_without_centre_vias("VBAT");
    let raised = files[LAYOUT].replace("via_diameter = 0.45, drill = 0.2, annular = 0.1", "via_diameter = 0.6, drill = 0.3, annular = 0.15");
    assert_ne!(raised, files[LAYOUT], "the ESC's limits line changed");
    files.insert(LAYOUT.to_owned(), raised);
    let before: Vec<_> = parse_board(LAYOUT, &files[LAYOUT]).board.unwrap().vias.values().map(|v| v.at).collect();
    autoroute(&mut files, "VBAT");
    let board = parse_board(LAYOUT, &files[LAYOUT]).board.unwrap();
    let added: Vec<_> = board.vias.values().filter(|v| !before.contains(&v.at)).collect();
    assert!(!added.is_empty(), "the VBAT pass added no vias");
    for v in added {
        assert!(v.drill >= runmat_circuit::geom::mm(0.3) && v.diameter >= runmat_circuit::geom::mm(0.6), "via {:?} is {} over {}", v.at, v.diameter, v.drill);
    }
}
