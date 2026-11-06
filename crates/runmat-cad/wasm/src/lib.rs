//! WebAssembly entry points of the CAD design layer: `.part` and `.assembly` files compile to
//! the projection the Desktop's Outline and Properties read and the display its model canvas
//! draws, from their own text.
//!
//! Every function takes and returns JSON text. `files` is a JSON object of workspace path →
//! file text: the files [`dependencies`] listed, as the caller read them.
//!
//! - `dependencies(path, text)` → `["/p/library/clip.part", …]`
//! - `compile(path, text, files, options)` (`options`: `{ configuration? }`, a part's active
//!   configuration) → `{ projection?, display?, diagnostics }`, where
//!   `projection` is `{ kind: "part", data: { document, index } }` or
//!   `{ kind: "assembly", data: <assembly index> }` and `display` is `{ kind, data }` with the
//!   part display (`part_display` with `sketch_display`) or the assembly display.
//!   A `.drawing` compiles to `{ kind: "drawing", data: <drawing index> }` with `models`: the
//!   display of each part or assembly its views show, by the file name the view wrote
//!   ([`drawing`]).
//! - `analyze(path, text, files, request)` → an Evaluate command's results table
//!   ([`analyze`]).
//! - `edit(path, text, files, request)` → `{ ok: true, files, summary }` or
//!   `{ ok: false, message, diagnostics }`: a catalog command run from the canvas, written as
//!   the smallest text change ([`edit`]).
//!
//! Geometry comes from the kernel installed with [`kernel::install`]: OCCT in the product (the
//! browser and Desktop worker run this runtime inside the OCCT geometry host,
//! `runmat-geometry-wasm-host`). Without one, the grid reference kernel stands in as the test
//! double. Every compile result names its kernel (`kernel`). Malformed input is reported as a
//! diagnostic, never a panic.

pub mod analyze;
pub mod board;
mod configured;
pub mod drawing;
pub mod edit;
pub mod evaluate;
pub mod exchange;
pub mod features;
pub mod hlr;
pub mod kernel;
pub mod mass;
pub mod picks;
pub mod runtime;

#[cfg(feature = "bindings")]
use wasm_bindgen::prelude::*;

/// The files `path` needs beyond itself.
#[cfg(feature = "bindings")]
#[wasm_bindgen]
pub fn dependencies(path: &str, text: &str) -> String {
    serde_json::to_string(&runtime::dependencies(path, text)).unwrap_or_else(|_| "[]".into())
}

/// The projection and display of the part or assembly `path`.
#[cfg(feature = "bindings")]
#[wasm_bindgen]
pub fn compile(path: &str, text: &str, files: &str, options: &str) -> String {
    let files = match runtime::parse_files(files) {
        Ok(f) => f,
        Err(message) => return runtime::error_json(path, &message),
    };
    let configuration = runtime::compile_options(options);
    runtime::compile_json_in(path, text, &files, configuration.as_deref())
}

/// Runs an Evaluate command (mass properties, interference) on the part or assembly `path`.
#[cfg(feature = "bindings")]
#[wasm_bindgen]
pub fn analyze(path: &str, text: &str, files: &str, request: &str) -> String {
    match runtime::parse_files(files) {
        Ok(files) => analyze::analyze_json(path, text, &files, request),
        Err(message) => serde_json::json!({ "ok": false, "message": message }).to_string(),
    }
}

/// Runs a catalog command on the part or assembly `path`.
#[cfg(feature = "bindings")]
#[wasm_bindgen]
pub fn edit(path: &str, text: &str, files: &str, request: &str) -> String {
    match runtime::parse_files(files) {
        Ok(files) => edit::edit_json(path, text, &files, request),
        Err(message) => {
            serde_json::json!({ "ok": false, "message": message, "diagnostics": [] }).to_string()
        }
    }
}
