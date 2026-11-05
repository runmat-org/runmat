//! A session's recompile tells a client holding the last projection what changed
//! ([`Session::since`]): nothing when the files are the same, a patch's own delta after an item
//! change, and the delta from the projection it replaced after a whole recompile. Each delta is
//! the one comparing the whole projections gives, and applying it gives the new projection.

use runmat_circuit::index::CircuitIndex;
use runmat_circuit::runtime::{self, CompileOptions, Files, Session, Since};

const CIRCUIT: &str = "/p/io.circuit";
const LAYOUT: &str = "/p/io.circuit.layout";

fn files() -> Files {
    ["io.circuit", "io.circuit.layout", "io.circuit.lock"]
        .iter()
        .map(|name| {
            let text = std::fs::read_to_string(format!("{}/tests/projects/io/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            (format!("/p/{name}"), text)
        })
        .collect()
}

/// Recompiles `files` and checks the delta from the projection before.
fn recompile(session: &mut Session, files: &Files, expect_whole: bool) {
    let before: CircuitIndex = session.index().cloned().expect("compiled");
    let since = session.generation();
    let out = session.compile(LAYOUT, &files[LAYOUT], files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(session.timings.whole, expect_whole, "{:?}", session.timings);
    let Since::Delta(delta) = session.since(since) else {
        panic!("a recompile after a change has a delta: {:?}", session.since(since));
    };
    let after = session.index().unwrap();
    assert_eq!(*delta, runtime::delta::index_delta(&before, after));
    assert!(runtime::delta::apply_index_delta(&before, delta).unwrap() == *after);
    assert_eq!(session.since(since + 1), Since::Same);
}

/// `text` with the first line starting with `word` changed by `f`.
fn change_line(text: &str, word: &str, f: impl Fn(&str) -> String) -> String {
    let mut done = false;
    text.split_inclusive('\n')
        .map(|l| {
            if !done && l.trim_start().starts_with(word) {
                done = true;
                f(l)
            } else {
                l.to_owned()
            }
        })
        .collect()
}

#[test]
fn recompiles_send_what_changed_since_the_projection_a_client_holds() {
    let mut files = files();
    let mut session = Session::new();
    let first = session.compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    assert!(first.diagnostics.is_empty(), "{:?}", first.diagnostics);
    let g = session.generation();
    assert_eq!(session.since(g - 1), Since::Whole, "the first projection has no delta");
    // The same files: the projection is the one the client holds.
    session.compile(LAYOUT, &files[LAYOUT], &files, &CompileOptions::default());
    assert_eq!(session.generation(), g);
    assert_eq!(session.since(g), Since::Same);

    // An item change patches the board, and the patch keeps its delta.
    let moved = change_line(&files[LAYOUT], "footprint", |l| l.replacen("at = [", "at = [1", 1));
    files.insert(LAYOUT.into(), moved);
    recompile(&mut session, &files, false);
    let wider = change_line(&files[LAYOUT], "track", |l| {
        let w = l.find("width = ").unwrap() + "width = ".len();
        let end = w + l[w..].find([',', ' ', '}']).unwrap();
        format!("{}0.333{}", &l[..w], &l[end..])
    });
    files.insert(LAYOUT.into(), wider);
    recompile(&mut session, &files, false);
    let without_via = change_line(&files[LAYOUT], "via", |_| String::new());
    files.insert(LAYOUT.into(), without_via);
    recompile(&mut session, &files, false);

    // A schematic change recompiles whole, with the delta from the projection it replaced.
    let circuit = format!("{}\n// a note\n", files[CIRCUIT]);
    files.insert(CIRCUIT.into(), circuit);
    recompile(&mut session, &files, true);

    // A client two projections behind gets the projection whole.
    assert_eq!(session.since(session.generation() - 2), Since::Whole);
    // An error is a new projection with none.
    let broken = format!("{}\n}}}}\n", files[LAYOUT]);
    let out = session.compile(LAYOUT, &broken, &files, &CompileOptions::default());
    assert!(!out.diagnostics.is_empty());
    assert!(session.index().is_none() || session.since(session.generation() - 1) == Since::Whole);
}
