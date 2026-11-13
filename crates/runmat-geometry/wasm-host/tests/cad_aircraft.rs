//! The airliner suite on OCCT, through `runmat_cad_call` (the Desktop and browser worker's
//! entry). `RUNMAT_AIRCRAFT_DIR=<folder>` writes the files there.

fn entry(path: &str, text: &str, files: &str) -> String {
    let request = serde_json::json!({ "fn": "compile", "args": [path, text, files, "{}"] });
    runmat_geometry_wasm_host::cad::call(&request.to_string())
}

#[path = "../../../runmat-cad/wasm/tests/aircraft_text/mod.rs"]
mod aircraft_text;
#[path = "../../../runmat-cad/wasm/tests/aircraft/suite.rs"]
mod suite;
