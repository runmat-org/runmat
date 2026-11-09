//! WebAssembly entry points of the PCB document runtime ([`runmat_circuit::runtime`]).
//!
//! Every function takes and returns JSON text, so the module needs no glue beyond
//! `wasm-bindgen`'s strings. `files` is a JSON object of workspace path → file text: the
//! files [`dependencies`] listed, as the caller read them.
//!
//! - `dependencies(path, text)` → `["/p/io.circuit", …]`
//! - `compile(path, text, files, options)` → `{ index?, diagnostics }`; the module keeps the
//!   last circuit's state, so recompiling after a small edit patches it
//! - `compile_delta(path, text, files, options, base)` → `{ seq, index? | delta?, diagnostics }`:
//!   what changed since projection `base` ([`delta`])
//! - `review(path, text, files, options)` → `{ eco?, back_annotation?, diagnostics }`
//! - `edit(path, text, files, request)` → `{ ok: true, files, summary }` or
//!   `{ ok: false, message, diagnostics }`; the open file's change has `text: null` when its
//!   edits give the new text (applied from the last to the first to `text`)
//! - `board_model()` → the last compiled board in 3D as a part display, or `null`
//!
//! A file whose text is `null` in `files` is the text the module last received for that path:
//! the caller sends a large file's text once, not on every call ([`files`]).
//!
//! Malformed JSON input is reported the same way as a failed edit, never as a panic.

mod delta;
mod simulate;
pub use delta::{compile_delta, large_project};
pub use simulate::{simulate_begin, simulate_end, simulate_step, simulation_setup};

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files, Session};
use serde::Serialize;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

thread_local! {
    /// The module's runtime session: the last circuit's compiled state, so a small edit
    /// recompiles only what it changed (`runmat_circuit::runtime::Session`).
    pub(crate) static SESSION: RefCell<Session> = RefCell::new(Session::new());
}

fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| error_json(&format!("the result can't be written: {e}")))
}

thread_local! {
    /// The text of every file the module has received, by path.
    static HELD: RefCell<Files> = RefCell::new(Files::new());
}

/// The files of a call: each path's text as sent, or the text last sent for it when `null`.
pub(crate) fn files(text: &str) -> Result<Files, String> {
    let sent: std::collections::BTreeMap<String, Option<String>> = parse(text, "files")?;
    HELD.with(|held| {
        let mut held = held.borrow_mut();
        sent.into_iter()
            .map(|(path, text)| match text {
                Some(text) => {
                    held.insert(path.clone(), text.clone());
                    Ok((path, text))
                }
                None => match held.get(&path) {
                    Some(text) => Ok((path, text.clone())),
                    None => Err(format!("the files aren't valid: {path} was sent as unchanged, but the module never received it")),
                },
            })
            .collect()
    })
}

pub(crate) fn error_json(message: &str) -> String {
    serde_json::json!({ "ok": false, "message": message, "diagnostics": [] }).to_string()
}

pub(crate) fn parse<T: serde::de::DeserializeOwned + Default>(
    text: &str,
    what: &str,
) -> Result<T, String> {
    if text.trim().is_empty() {
        return Ok(T::default());
    }
    serde_json::from_str(text).map_err(|e| format!("the {what} aren't valid: {e}"))
}

/// The files `path` needs beyond itself.
#[wasm_bindgen]
pub fn dependencies(path: &str, text: &str) -> String {
    json(&runtime::dependencies(path, text))
}

/// The projection of the circuit `path` belongs to.
#[wasm_bindgen]
pub fn compile(path: &str, text: &str, files: &str, options: &str) -> String {
    let (files, options) = match (
        crate::files(files),
        parse::<CompileOptions>(options, "options"),
    ) {
        (Ok(f), Ok(o)) => (f, o),
        (Err(m), _) | (_, Err(m)) => return error_json(&m),
    };
    SESSION.with(|s| s.borrow_mut().compile_json(path, text, &files, &options))
}

/// Both ECO directions under a run's comparator settings.
#[wasm_bindgen]
pub fn review(path: &str, text: &str, files: &str, options: &str) -> String {
    let (files, options) = match (
        crate::files(files),
        parse::<CompileOptions>(options, "options"),
    ) {
        (Ok(f), Ok(o)) => (f, o),
        (Err(m), _) | (_, Err(m)) => return error_json(&m),
    };
    json(&runtime::review(path, text, &files, &options))
}

/// Runs a catalog command: each changed file's text and its minimal edits.
#[wasm_bindgen]
pub fn edit(path: &str, text: &str, files: &str, request: &str) -> String {
    let (files, request) = match (
        crate::files(files),
        parse::<EditRequest>(request, "request"),
    ) {
        (Ok(f), Ok(r)) => (f, r),
        (Err(m), _) | (_, Err(m)) => return error_json(&m),
    };
    match SESSION.with(|s| s.borrow_mut().edit(path, text, &files, &request)) {
        Ok(out) => {
            let mut v = serde_json::to_value(&out).unwrap_or_default();
            // A large board's text goes back once, as the caller's own edits.
            for f in v["files"].as_array_mut().into_iter().flatten() {
                if f["path"] == path && f["reprinted"] == false {
                    f["text"] = serde_json::Value::Null;
                }
            }
            v["ok"] = serde_json::Value::Bool(true);
            v.to_string()
        }
        Err(e) => {
            let mut v = serde_json::to_value(&e).unwrap_or_default();
            v["ok"] = serde_json::Value::Bool(false);
            v.to_string()
        }
    }
}

/// A catalog part's previews for the library panel: its schematic symbol, footprint and body
/// box (`runmat_circuit::library::preview`), from a part search hit's `part` record. `files`
/// holds a footprint-file package's text by the path the record writes.
#[wasm_bindgen]
pub fn library_preview(part: &str, files: &str) -> String {
    let part: runmat_circuit::library::catalog::CatalogPart = match serde_json::from_str(part) {
        Ok(p) => p,
        Err(e) => return error_json(&format!("the part record isn't valid: {e}")),
    };
    let files: std::collections::BTreeMap<String, String> = match parse(files, "files") {
        Ok(f) => f,
        Err(m) => return error_json(&m),
    };
    match runmat_circuit::library::preview::preview_with(&part, &|p| files.get(p).cloned()) {
        Ok(p) => json(&serde_json::json!({ "ok": true, "preview": p })),
        Err(m) => error_json(&m),
    }
}

/// A KiCad symbol library opened in place (`runmat_circuit::library::kicad_lib`): its parts,
/// each on the footprint file its symbol names, rebased from the library's folder `lib_dir` to
/// the circuit's folder `base`, and the symbols left out (power symbols, symbols without a
/// footprint).
#[wasm_bindgen]
pub fn kicad_library(text: &str, lib_dir: &str, base: &str) -> String {
    use runmat_circuit::library::kicad_lib;
    match kicad_lib::read(text) {
        Ok((parts, skipped)) => {
            let parts: Vec<_> = parts
                .iter()
                .map(|p| kicad_lib::rebase(p, lib_dir, base))
                .collect();
            json(&serde_json::json!({ "ok": true, "parts": parts, "skipped": skipped }))
        }
        Err(m) => error_json(&m),
    }
}

/// The last compiled board in 3D, as a part display (`runmat_circuit::board3d`), or `null`
/// when the last compile had no board.
#[wasm_bindgen]
pub fn board_model() -> String {
    SESSION.with(|s| match s.borrow().board_model() {
        Some(m) => json(&m),
        None => "null".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_json_is_an_error_result() {
        let out: serde_json::Value =
            serde_json::from_str(&edit("a.circuit.layout", "", "{", "")).unwrap();
        assert_eq!(out["ok"], false);
        assert!(out["message"].as_str().unwrap().contains("files"));
    }

    #[test]
    fn a_layout_without_its_circuit_is_a_diagnostic() {
        let text = "runmat 0.7\nstd 1\n\nuse circuit io\n\nlayout io {\n}\n";
        // The circuit is named by symbol, so the host lists the layout's folder.
        let deps: Vec<String> =
            serde_json::from_str(&dependencies("/p/io.circuit.layout", text)).unwrap();
        assert_eq!(deps, vec!["/p/".to_owned()]);
        let out: serde_json::Value =
            serde_json::from_str(&compile("/p/io.circuit.layout", text, "{}", "")).unwrap();
        assert!(out.get("index").is_none());
        assert!(out["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains("no `circuit io` in this folder"));
    }
}

/// The large fixture design (`runmat_circuit::fixture::large`) with `channels` channels on a
/// routed, poured 8-layer board, as `{ kind: "circuit", data }`: the Desktop's board scale
/// scene draws it to measure frame times.
#[wasm_bindgen]
pub fn large_board(channels: u32) -> String {
    let params = runmat_circuit::fixture::LargeParams {
        channels,
        ..Default::default()
    };
    let index = runmat_circuit::fixture::large_index(&params);
    format!(
        "{{\"kind\":\"circuit\",\"data\":{}}}",
        runmat_circuit::index::to_json(&index)
    )
}
