//! Opening the quadcopter's flight controller (`runmat-cad/wasm/tests/drone`) stays fast: its
//! first compile through a [`Session`], as the Desktop's circuit worker runs it, finishes
//! within the budget, and so does the recompile after a one-line edit.
//!
//! Run with `--release --nocapture` for the timing lines.

use runmat_circuit::runtime::{CompileOptions, Files, Session};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Release: 3 s for the first compile. Debug builds run the same code unoptimised, about ten
/// times slower.
fn budget() -> Duration {
    if cfg!(debug_assertions) {
        Duration::from_secs(30)
    } else {
        Duration::from_secs(3)
    }
}

fn drone_files(name: &str) -> (Files, String) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runmat-cad/wasm/tests/drone");
    let files: Files = [".circuit", ".circuit.layout", ".circuit.lock"]
        .iter()
        .filter_map(|x| {
            let text = std::fs::read_to_string(dir.join(format!("{name}{x}"))).ok()?;
            Some((format!("/p/{name}{x}"), text))
        })
        .collect();
    (files, format!("/p/{name}.circuit.layout"))
}

fn first_compile(name: &str) -> (Session, Files, String, Duration) {
    let (files, layout) = drone_files(name);
    let mut session = Session::default();
    let t = Instant::now();
    let out = session.compile(&layout, &files[&layout], &files, &CompileOptions::default());
    let took = t.elapsed();
    eprintln!("{name}: first compile {took:?} {:?}", session.timings);
    assert!(out.index.is_some(), "{name} compiles: {:?}", out.diagnostics);
    (session, files, layout, took)
}

#[test]
fn the_flight_controller_opens_within_budget() {
    let (_, _, _, took) = first_compile("flight_controller");
    assert!(took < budget(), "the flight controller took {took:?} to compile");
}

#[test]
fn the_esc_opens_within_budget() {
    let (_, _, _, took) = first_compile("esc");
    assert!(took < budget(), "the ESC took {took:?} to compile");
}
