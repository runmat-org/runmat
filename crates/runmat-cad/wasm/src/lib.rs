//! WebAssembly entry points of the CAD design layer: `.part` and `.assembly` files compile to
//! the projection the Desktop's Outline and Properties read and the display its model canvas
//! draws, from their own text.
//!
//! Every function takes and returns JSON text. `files` is a JSON object of workspace path →
//! file text: the files [`dependencies`] listed, as the caller read them.
//!
//! - `dependencies(path, text)` → `["/p/library/clip.part", …]`
//! - `compile(path, text, files, options)` → `{ projection?, display?, diagnostics }`, where
//!   `projection` is `{ kind: "part", data: { document, index } }` or
//!   `{ kind: "assembly", data: <assembly index> }` and `display` is `{ kind, data }` with the
//!   part display (`part_display` with `sketch_display`) or the assembly display.
//!
//! Geometry comes from the reference kernel ([`runmat_cad::regen::fake::FakeKernel`]): exact
//! for prismatic parts, sampled for curved tools. The OCCT kernel slots in behind the same
//! `GeometryKernel` and `Tessellator` traits when it is built for the browser or reached
//! through the Tauri host; the JSON shapes stay the same. Malformed input is reported as a
//! diagnostic, never a panic.

pub mod runtime;

use wasm_bindgen::prelude::*;

/// The files `path` needs beyond itself.
#[wasm_bindgen]
pub fn dependencies(path: &str, text: &str) -> String {
    serde_json::to_string(&runtime::dependencies(path, text)).unwrap_or_else(|_| "[]".into())
}

/// The projection and display of the part or assembly `path`.
#[wasm_bindgen]
pub fn compile(path: &str, text: &str, files: &str, options: &str) -> String {
    let files = match runtime::parse_files(files) {
        Ok(f) => f,
        Err(message) => return runtime::error_json(path, &message),
    };
    let _ = options;
    runtime::compile_json(path, text, &files)
}
