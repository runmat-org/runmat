//! The display-mesh suite (`display_mesh/suite.rs`) on this crate's own `compile` entry, with
//! no kernel installed: the grid test double. The same suite runs on OCCT through the geometry
//! host's `runmat_cad_call` (`runmat-geometry-wasm-host` tests/cad_display_mesh.rs).

fn entry(path: &str, text: &str) -> String {
    runmat_cad_wasm::compile(path, text, "{}", "{}")
}

#[path = "display_mesh/suite.rs"]
mod suite;
