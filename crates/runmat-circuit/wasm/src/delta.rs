//! Recompiles that send what changed: `compile_delta(path, text, files, options, base)`.
//!
//! The module keeps the last projection it sent, numbered. A caller that holds projection
//! `base` gets `{ seq, delta, diagnostics }` ([`runmat_circuit::runtime::delta`]): the rows and
//! fields that changed since. Any other caller (the first call, or one whose `base` isn't the
//! last sent) gets `{ seq, index, diagnostics }`, the whole projection. After a small edit on a
//! 50k-primitive board the delta is kilobytes where the projection is megabytes.

use crate::{error_json, parse, SESSION};
use runmat_circuit::index::CircuitIndex;
use runmat_circuit::runtime::{delta, CompileOptions, Files};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

thread_local! {
    /// The last projection sent, and its number.
    static SENT: RefCell<(u64, Option<CircuitIndex>)> = const { RefCell::new((0, None)) };
}

/// The projection of the circuit `path` belongs to, as a delta from projection `base` (a `seq`
/// an earlier call returned) when the module still holds it, else whole.
#[wasm_bindgen]
pub fn compile_delta(path: &str, text: &str, files: &str, options: &str, base: &str) -> String {
    let (files, options) = match (parse::<Files>(files, "files"), parse::<CompileOptions>(options, "options")) {
        (Ok(f), Ok(o)) => (f, o),
        (Err(m), _) | (_, Err(m)) => return error_json(&m),
    };
    let out = SESSION.with(|s| s.borrow_mut().compile(path, text, &files, &options));
    let base: Option<u64> = base.trim().parse().ok();
    SENT.with(|sent| {
        let mut sent = sent.borrow_mut();
        let seq = sent.0 + 1;
        let diagnostics = serde_json::to_string(&out.diagnostics).unwrap_or_else(|_| "[]".into());
        // Written straight to text: the whole projection is the large case.
        let body = match (&out.index, &sent.1) {
            (Some(new), Some(old)) if base == Some(sent.0) => format!(",\"delta\":{}", delta::index_delta(old, new)),
            (Some(new), _) => format!(",\"index\":{}", serde_json::to_string(new).unwrap_or_else(|_| "null".into())),
            (None, _) => String::new(),
        };
        *sent = (seq, out.index);
        format!("{{\"seq\":{seq},\"diagnostics\":{diagnostics}{body}}}")
    })
}

/// The 50k-primitive scale fixture's source files (`runmat_circuit::fixture::large_files`), as a
/// JSON object of path → text, for the Desktop's end-to-end edit timing.
#[wasm_bindgen]
pub fn large_project() -> String {
    let (files, _) = runmat_circuit::fixture::large_files(&runmat_circuit::fixture::scale_params());
    serde_json::to_string(&files).unwrap_or_default()
}
