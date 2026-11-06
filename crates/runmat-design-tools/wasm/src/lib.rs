//! WebAssembly entry point of the design tool layer ([`runmat_design_tools::call`]), for the
//! browser when no server project is open (the sandbox): the agent's file overview and its
//! design tools then run here, on the workspace files the page reads.
//!
//! - `design_tool(name, input, files)` → `{ data }` or `{ error: { code, message } }`, the shape
//!   the server's `POST /v1/projects/{id}/design-tools/{tool}` route answers.
//! - `dependencies(path, text)` → `["/p/daq.circuit.lock", …]`: the files `path` needs, which the
//!   page reads (and what they need) into `files`.
//! - `related(path, project)` → the project files an overview of `path` reads to find what uses
//!   it, from the project's paths (a JSON array).
//!
//! `input` is the tool's input as JSON text (`{"path":"/p/daq.circuit"}`); `files` is a JSON
//! object of workspace path → text holding the file, what it needs and, for `design.overview`,
//! the project's other design files ([`runmat_design_tools::overview::related_candidates`]).
//!
//! The module answers for ECAD files only (`.circuit`, `.circuit.layout`). Parts, assemblies and
//! drawings regenerate on the OCCT kernel, which the tool layer reaches natively on the server;
//! here they answer `kernel_unavailable`, so no answer comes from the test kernel.

use runmat_design_tools::{call, DesignKind, Files};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

/// The tools this host runs on ECAD files: the read-only ones, and `design.export`, which
/// answers the exported files for the page to write (the Desktop's Fabrication outputs).
/// `design.render` draws through the server, and `design.propose_edit` and import stay with the
/// project.
const TOOLS: &[&str] = &["design.overview", "design.check", "design.inspect", "design.erc", "design.drc", "design.simulate", "design.export"];

#[wasm_bindgen]
pub fn design_tool(name: &str, input: &str, files: &str) -> String {
    answer(name, input, files).to_string()
}

#[wasm_bindgen]
pub fn dependencies(path: &str, text: &str) -> String {
    Value::from(runmat_design_tools::dependencies(path, text)).to_string()
}

#[wasm_bindgen]
pub fn related(path: &str, project: &str) -> String {
    let project: Vec<String> = serde_json::from_str(project).unwrap_or_default();
    Value::from(runmat_design_tools::overview::related_candidates(path, &project)).to_string()
}

/// The tool's answer as JSON: `{ data }` or `{ error }`.
pub fn answer(name: &str, input: &str, files: &str) -> Value {
    match run(name, input, files) {
        Ok(data) => json!({ "data": data }),
        Err((code, message)) => json!({ "error": { "code": code, "message": message } }),
    }
}

fn run(name: &str, input: &str, files: &str) -> Result<Value, (String, String)> {
    if !TOOLS.contains(&name) {
        return Err(("unsupported".into(), format!("{name} runs with a server project")));
    }
    let input: Value = serde_json::from_str(input).map_err(|e| ("invalid_arguments".to_owned(), format!("the input isn't valid JSON: {e}")))?;
    let files: Files = serde_json::from_str(files).map_err(|e| ("invalid_arguments".to_owned(), format!("the files aren't a JSON object of path to text: {e}")))?;
    let path = input.get("path").and_then(Value::as_str).unwrap_or_default();
    match DesignKind::of(path) {
        Some(DesignKind::Circuit | DesignKind::Layout) => {}
        Some(_) => {
            return Err((
                "kernel_unavailable".into(),
                "Parts, assemblies and drawings regenerate on OCCT, which the design tools run with a server project".into(),
            ))
        }
        None => return Err(("invalid_arguments".into(), format!("{path} isn't a circuit or a board"))),
    }
    call(name, input, &files).map_err(|e| (e.code, e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAQ: &str = "../../runmat-circuit/tests/projects/daq";

    fn daq_files() -> Value {
        let mut files = serde_json::Map::new();
        for name in ["daq.circuit", "daq.circuit.lock", "daq.circuit.layout"] {
            let text = std::fs::read_to_string(format!("{}/{DAQ}/{name}", env!("CARGO_MANIFEST_DIR"))).expect("daq fixture");
            files.insert(format!("/p/{name}"), Value::String(text));
        }
        Value::Object(files)
    }

    #[test]
    fn the_overview_of_a_circuit_answers_from_its_files() {
        let out = answer("design.overview", r#"{"path":"/p/daq.circuit"}"#, &daq_files().to_string());
        let data = out.get("data").unwrap_or_else(|| panic!("an overview, not {out}"));
        assert_eq!(data["path"], "/p/daq.circuit");
        // The board uses the schematic, so it links back to it.
        assert!(data.to_string().contains("/p/daq.circuit.layout"), "{data}");
    }

    #[test]
    fn it_lists_what_a_circuit_needs_and_what_may_use_it() {
        let needs: Vec<String> = serde_json::from_str(&dependencies("/p/daq.circuit", "circuit daq {}")).unwrap();
        assert!(needs.iter().all(|p| p.starts_with("/p/")), "{needs:?}");
        let related: Vec<String> = serde_json::from_str(&related("/p/daq.circuit", r#"["/p/daq.circuit","/p/daq.circuit.layout","/p/notes.md","/p/arm.fea"]"#)).unwrap();
        assert_eq!(related, ["/p/arm.fea", "/p/daq.circuit.layout"]);
    }

    #[test]
    fn mcad_files_and_other_tools_refuse_with_their_reason() {
        let part = answer("design.overview", r#"{"path":"/p/a.part"}"#, r#"{"/p/a.part":""}"#);
        assert_eq!(part["error"]["code"], "kernel_unavailable");
        let render = answer("design.render", r#"{"path":"/p/daq.circuit"}"#, "{}");
        assert_eq!(render["error"]["code"], "unsupported");
        let bad = answer("design.check", "{", "{}");
        assert_eq!(bad["error"]["code"], "invalid_arguments");
    }

    #[test]
    fn a_board_exports_its_fabrication_set_for_the_page_to_write() {
        let input = r#"{"path":"/p/daq.circuit.layout","format":"fabrication","fabrication":{"gerber":false,"drill":false,"drillMap":false,"drawing":false,"pickPlace":false,"bom":false,"ipc2581":false,"odb":false,"ipc356":false}}"#;
        let out = answer("design.export", input, &daq_files().to_string());
        let summary = &out["data"]["fabrication"];
        assert!(summary["layers"].as_array().map(|l| !l.is_empty()).unwrap_or(false), "{out}");
        assert_eq!(summary["variants"][0], Value::Null);
    }
}
