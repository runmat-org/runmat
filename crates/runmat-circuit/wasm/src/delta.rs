//! Recompiles that send what changed: `compile_delta(path, text, files, options, base)`.
//!
//! The module's session numbers each projection. A caller that holds the one before gets
//! `{ seq, delta, diagnostics }` ([`runmat_circuit::runtime::delta`]): the rows and fields that
//! changed since. Any other caller (the first call, or one whose `base` is older) gets
//! `{ seq, index, diagnostics }`, the whole projection. After a small edit on a
//! 50k-primitive board the delta is kilobytes where the projection is megabytes.

use crate::{error_json, files, parse, SESSION};
use runmat_circuit::runtime::CompileOptions;
use wasm_bindgen::prelude::*;

/// The projection of the circuit `path` belongs to, as a delta from projection `base` (a `seq`
/// an earlier call returned) when the module can give it, else whole
/// ([`runmat_circuit::runtime::Session::compile_since`]).
#[wasm_bindgen]
pub fn compile_delta(path: &str, text: &str, files_json: &str, options: &str, base: &str) -> String {
    let (files, options) = match (files(files_json), parse::<CompileOptions>(options, "options")) {
        (Ok(f), Ok(o)) => (f, o),
        (Err(m), _) | (_, Err(m)) => return error_json(&m),
    };
    let base: Option<u64> = base.trim().parse().ok();
    SESSION.with(|s| s.borrow_mut().compile_since(path, text, &files, &options, base))
}

/// The 50k-primitive scale fixture's source files (`runmat_circuit::fixture::large_files`), as a
/// JSON object of path → text, for the Desktop's end-to-end edit timing.
#[wasm_bindgen]
pub fn large_project() -> String {
    let (files, _) = runmat_circuit::fixture::large_files(&runmat_circuit::fixture::scale_params());
    serde_json::to_string(&files).unwrap_or_default()
}
