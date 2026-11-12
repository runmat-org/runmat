//! The document runtime at scale: a board of 50k+ primitives opened, edited and recompiled
//! through a [`Session`], as the Desktop's worker does. A small edit (move one footprint,
//! delete one track) and its recompile stay within the budget, and the session's projection
//! equals a fresh compile of the edited files.
//!
//! Run with `--release --nocapture` for the timing lines.

use runmat_circuit::fixture;
use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files, Session};
use std::time::{Duration, Instant};

const LAYOUT: &str = fixture::LARGE_LAYOUT;

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
    fixture::large_files(&fixture::scale_params())
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

/// The session's delta from projection `since` (which was `before`) is the one comparing the
/// whole projections gives, and applying it to `before` gives the current projection.
fn check_delta(session: &Session, before: &runmat_circuit::index::CircuitIndex, since: u64) {
    let runtime::Since::Delta(delta) = session.since(since) else {
        panic!(
            "a patched recompile has its delta: {:?}",
            session.since(since)
        );
    };
    let after = session.index().expect("compiled");
    assert_eq!(
        *delta,
        runtime::delta::index_delta(before, after),
        "the patch's delta is the one a whole compare gives"
    );
    assert!(
        runtime::delta::apply_index_delta(before, delta).unwrap() == *after,
        "the delta applied to the old projection differs from the new one"
    );
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
    let first_fill = session.timings.fill;
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
    let before = session.index().cloned().expect("compiled");
    let since = session.generation();
    let (edit, compile) = edit_and_compile(&mut session, &mut files, &mv);
    eprintln!("move {fp}: edit {edit:?}, recompile {compile:?}");
    // The worker sends the change, not the projection: the patch's own delta, a few rows, and
    // applying them gives the new projection exactly.
    let after = session.index().expect("compiled");
    let t = Instant::now();
    let runtime::Since::Delta(delta) = session.since(since) else {
        panic!("a patched move has its delta: {:?}", session.since(since));
    };
    let delta_text = delta.to_string();
    let delta_time = t.elapsed();
    let whole_text = serde_json::to_string(after).unwrap();
    eprintln!(
        "move delta: {} bytes, written in {delta_time:?}; whole projection {} bytes",
        delta_text.len(),
        whole_text.len()
    );
    let t = Instant::now();
    let compared = runtime::delta::index_delta(&before, after);
    eprintln!("  (comparing whole projections instead: {:?})", t.elapsed());
    assert_eq!(
        *delta, compared,
        "the patch's delta is the one a whole compare gives"
    );
    for part in ["fields", "board"] {
        for (k, v) in delta[part].as_object().into_iter().flatten() {
            for (k2, v2) in v.as_object().into_iter().flatten() {
                eprintln!("  {part}.{k}.{k2}: {} bytes", v2.to_string().len());
            }
            if !v.is_object() {
                eprintln!("  {part}.{k}: {} bytes", v.to_string().len());
            }
        }
    }
    assert!(
        delta_text.len() < 64 * 1024,
        "delta {} bytes, projection {}",
        delta_text.len(),
        whole_text.len()
    );
    assert!(
        runtime::delta::apply_index_delta(&before, delta).unwrap() == *after,
        "the delta applied to the old projection differs from the new one"
    );
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
    let (before, since) = (session.index().cloned().unwrap(), session.generation());
    let (edit, compile) = edit_and_compile(&mut session, &mut files, &del);
    check_delta(&session, &before, since);
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

    // A package statement changes (as an ECO's footprint swap does): only the footprints
    // that use it re-read, and rules, fills, DRC and the index patch from them.
    let parsed = runmat_circuit::syntax::board::parse_board(LAYOUT, &files[LAYOUT]);
    let b = parsed.board.as_ref().expect("reads");
    let mut users: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for f in b.footprints.values() {
        *users.entry(f.package.name.as_str()).or_default() += 1;
    }
    let (name, n) = users
        .iter()
        .min_by_key(|(_, n)| **n)
        .map(|(k, n)| (k.to_string(), *n))
        .unwrap();
    let text = &files[LAYOUT];
    // The printer quotes a name that isn't a plain word.
    let at = [
        format!("package {name} = {{"),
        format!("package \"{name}\" = {{"),
    ]
    .iter()
    .find_map(|head| text.find(head.as_str()).map(|at| at + head.len()))
    .expect("the package block");
    let edited = format!(
        "{}\n        description = \"edited\",{}",
        &text[..at],
        &text[at..]
    );
    let edited = if runmat_circuit::syntax::board::parse_board(LAYOUT, &edited)
        .diagnostics
        .is_empty()
    {
        edited
    } else {
        // The block already has a description: change its text instead.
        let d = at
            + text[at..].find("description = \"").expect("a description")
            + "description = \"".len();
        format!("{}edited {}", &text[..d], &text[d..])
    };
    files.insert(LAYOUT.into(), edited);
    let (before, since) = (session.index().cloned().unwrap(), session.generation());
    let t = Instant::now();
    let compiled = session.compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    let recompile = t.elapsed();
    assert!(
        compiled.diagnostics.is_empty(),
        "{:?}",
        compiled.diagnostics
    );
    assert!(!session.timings.whole, "a package change patches the board");
    eprintln!(
        "package {name} ({n} footprints) edited: recompile {recompile:?}: {:?}",
        session.timings
    );
    check_delta(&session, &before, since);
    assert!(recompile < budget(), "package edit took {recompile:?}");
    let fresh = runtime::compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    let (a, b) = (
        session.index().unwrap().board.as_ref().unwrap(),
        fresh.index.as_ref().unwrap().board.as_ref().unwrap(),
    );
    assert_eq!(a.footprints.len(), b.footprints.len());
    assert_eq!(a.drc.len(), b.drc.len());
    for (x, y) in a.zones.iter().zip(&b.zones) {
        assert_eq!(x.polygons, y.polygons, "fills after a package edit");
    }
    // The fills reuse their tiles: well under a fresh fill.
    assert!(
        session.timings.fill * 4 < first_fill,
        "fill {:?} after a package edit, {first_fill:?} fresh",
        session.timings.fill
    );
}
