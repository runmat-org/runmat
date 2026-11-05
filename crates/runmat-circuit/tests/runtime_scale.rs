//! The document runtime at scale: a board of 50k+ primitives opened, edited and recompiled
//! through a [`Session`], as the Desktop's worker does. A small edit (move one footprint,
//! delete one track) and its recompile stay within the budget, and the session's projection
//! equals a fresh compile of the edited files.
//!
//! Run with `--release --nocapture` for the timing lines.

use runmat_circuit::fixture::{self, LargeParams};
use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files, Session};
use runmat_circuit::syntax::board::print_board;
use runmat_circuit::syntax::print_circuit;
use runmat_design_syntax::document::UseItem;
use std::time::{Duration, Instant};

const CIRCUIT: &str = "/p/big.circuit";
const LAYOUT: &str = "/p/big.circuit.layout";

/// The edit budget: edit and recompile under 0.5 s in release. Debug builds run the same
/// code unoptimised, about ten times slower.
fn budget() -> Duration {
    if cfg!(debug_assertions) {
        Duration::from_secs(5)
    } else {
        Duration::from_millis(500)
    }
}

fn project() -> (Files, usize) {
    let params = LargeParams {
        channels: 700,
        copper_layers: 8,
        route_every: 1,
        pours: true,
    };
    let (_, mut design, _, mut board) = fixture::large(&params);
    design.name = "big".into();
    board.name = "big".into();
    let primitives = board.footprints.len()
        + board.tracks.len()
        + board.vias.len()
        + board.zones.len()
        + board.all_pads().len();
    let uses = vec![UseItem {
        symbol: Some(("circuit".into(), "big".into())),
        from: "./big.circuit".into(),
        alias: None,
        revision: None,
        span: Default::default(),
    }];
    let mut files = Files::new();
    files.insert(CIRCUIT.into(), print_circuit(&design).expect("prints"));
    files.insert(LAYOUT.into(), print_board(&board, &uses));
    (files, primitives)
}

fn request(command: &str, values: serde_json::Value, selection: &[String]) -> EditRequest {
    EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: selection.to_vec(),
        options: CompileOptions::default(),
    }
}

/// Runs `request` on the session and recompiles the edited layout: the time both take.
fn edit_and_compile(
    session: &mut Session,
    files: &mut Files,
    request: &EditRequest,
) -> (Duration, Duration) {
    let t = Instant::now();
    let out = session
        .edit(LAYOUT, &files[LAYOUT], files, request)
        .expect("the edit runs");
    let edit = t.elapsed();
    let change = out
        .files
        .iter()
        .find(|f| f.path == LAYOUT)
        .expect("the layout changes");
    assert!(!change.reprinted, "a small edit changes a few lines");
    assert!(change.edits.len() <= 2, "{} edits", change.edits.len());
    files.insert(LAYOUT.into(), change.text.clone());
    let t = Instant::now();
    let compiled = session.compile(LAYOUT, &files[LAYOUT], files, &CompileOptions::default());
    let compile = t.elapsed();
    assert!(
        compiled.diagnostics.is_empty(),
        "{:?}",
        compiled.diagnostics
    );
    assert!(!session.timings.whole, "the recompile patched the board");
    eprintln!("  {:?}", session.timings);
    (edit, compile)
}

#[test]
fn small_edits_on_a_large_board_stay_within_budget() {
    let t = Instant::now();
    let (mut files, primitives) = project();
    eprintln!(
        "fixture: {primitives} primitives, layout {} bytes, in {:?}",
        files[LAYOUT].len(),
        t.elapsed()
    );
    assert!(primitives >= 50_000, "{primitives} primitives");
    let mut session = Session::new();
    let t = Instant::now();
    let first = session.compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    eprintln!("first compile {:?}: {:?}", t.elapsed(), session.timings);
    assert!(first.diagnostics.is_empty(), "{:?}", first.diagnostics);
    let index = first.index.expect("compiles");
    let board = index.board.as_ref().expect("has a board");

    // Move one footprint by half a millimetre.
    let fp = board.footprints[board.footprints.len() / 2]
        .reference
        .clone();
    let mv = request(
        "board.move",
        serde_json::json!({ "dx": 0.5, "dy": 0.0 }),
        &[format!("component:{fp}")],
    );
    let (edit, compile) = edit_and_compile(&mut session, &mut files, &mv);
    eprintln!("move {fp}: edit {edit:?}, recompile {compile:?}");
    assert!(edit + compile < budget(), "move took {:?}", edit + compile);
    // The patched projection is the one a fresh compile gives.
    let fresh = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    assert!(
        session.index() == fresh.index.as_ref(),
        "the session's projection after a move differs from a fresh compile"
    );

    // Delete one track.
    let track = session.index().unwrap().board.as_ref().unwrap().tracks[100].id;
    let del = request(
        "board.delete",
        serde_json::json!({}),
        &[format!("track:{track}")],
    );
    let (edit, compile) = edit_and_compile(&mut session, &mut files, &del);
    eprintln!("delete track {track}: edit {edit:?}, recompile {compile:?}");
    assert!(
        edit + compile < budget(),
        "delete took {:?}",
        edit + compile
    );
    let fresh = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    let (a, b) = (
        session.index().unwrap().board.as_ref().unwrap(),
        fresh.index.as_ref().unwrap().board.as_ref().unwrap(),
    );
    // A fresh read numbers items in source order; the session keeps the ids it had.
    assert_eq!(a.tracks.len(), b.tracks.len());
    assert_eq!(a.drc.len(), b.drc.len());
    assert_eq!(a.zones.len(), b.zones.len());
    for (x, y) in a.zones.iter().zip(&b.zones) {
        assert_eq!(x.polygons, y.polygons, "fills after a delete");
    }
}
