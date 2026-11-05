//! The airliner suite (`aircraft/suite.rs`) on this crate's own `compile` entry: the grid test
//! double. The same suite runs on OCCT through the geometry host's `runmat_cad_call`
//! (`runmat-geometry-wasm-host` tests/cad_aircraft.rs).

fn entry(path: &str, text: &str, files: &str) -> String {
    runmat_cad_wasm::compile(path, text, files, "{}")
}

mod aircraft_text;
#[path = "aircraft/suite.rs"]
mod suite;
