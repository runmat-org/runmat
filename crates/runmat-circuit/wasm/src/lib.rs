//! WebAssembly entry points of the PCB document runtime ([`runmat_circuit::runtime`]).
//!
//! Every function takes and returns JSON text, so the module needs no glue beyond
//! `wasm-bindgen`'s strings. `files` is a JSON object of workspace path → file text: the
//! files [`dependencies`] listed, as the caller read them.
//!
//! - `dependencies(path, text)` → `["/p/io.circuit", …]`
//! - `compile(path, text, files, options)` → `{ index?, diagnostics }`
//! - `review(path, text, files, options)` → `{ eco?, back_annotation?, diagnostics }`
//! - `edit(path, text, files, request)` → `{ ok: true, files, summary }` or
//!   `{ ok: false, message, diagnostics }`
//!
//! Malformed JSON input is reported the same way as a failed edit, never as a panic.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| error_json(&format!("the result can't be written: {e}")))
}

fn error_json(message: &str) -> String {
    serde_json::json!({ "ok": false, "message": message, "diagnostics": [] }).to_string()
}

fn parse<T: serde::de::DeserializeOwned + Default>(text: &str, what: &str) -> Result<T, String> {
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
        parse::<Files>(files, "files"),
        parse::<CompileOptions>(options, "options"),
    ) {
        (Ok(f), Ok(o)) => (f, o),
        (Err(m), _) | (_, Err(m)) => return error_json(&m),
    };
    json(&runtime::compile(path, text, &files, &options))
}

/// Both ECO directions under a run's comparator settings.
#[wasm_bindgen]
pub fn review(path: &str, text: &str, files: &str, options: &str) -> String {
    let (files, options) = match (
        parse::<Files>(files, "files"),
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
        parse::<Files>(files, "files"),
        parse::<EditRequest>(request, "request"),
    ) {
        (Ok(f), Ok(r)) => (f, r),
        (Err(m), _) | (_, Err(m)) => return error_json(&m),
    };
    match runtime::edit(path, text, &files, &request) {
        Ok(out) => {
            let mut v = serde_json::to_value(&out).unwrap_or_default();
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
        let text = "runmat 0.7\nstd 1\n\nuse circuit io from \"./io.circuit\"\n\nlayout io {\n}\n";
        let deps: Vec<String> =
            serde_json::from_str(&dependencies("/p/io.circuit.layout", text)).unwrap();
        assert_eq!(
            deps,
            vec!["/p/io.circuit".to_owned(), "/p/io.circuit.lock".to_owned()]
        );
        let out: serde_json::Value =
            serde_json::from_str(&compile("/p/io.circuit.layout", text, "{}", "")).unwrap();
        assert!(out.get("index").is_none());
        assert!(out["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains("/p/io.circuit"));
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
