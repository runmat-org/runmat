//! WebAssembly entry point of the design tool layer ([`runmat_design_tools::call`]), for the
//! browser when no server project is open (the sandbox): the agent's file overview and its
//! design tools then run here, on the workspace files the page reads.
//!
//! - `design_tool(name, input, files)` → `{ data }` or `{ error: { code, message } }`, the shape
//!   the server's `POST /v1/projects/{id}/design-tools/{tool}` route answers.
//! - `design_tool_progress(name, input, files, onProgress)`: the same, calling `onProgress(step)`
//!   as each stage of a fabrication export starts.
//! - `dependencies(path, text)` → `["/p/daq.circuit.lock", …]`: the files `path` needs, which the
//!   page reads (and what they need) into `files`.
//! - `related(path, project)` → the project files an overview of `path` reads to find what uses
//!   it, from the project's paths (a JSON array).
//!
//! `input` is the tool's input as JSON text (`{"path":"/p/daq.circuit"}`); `files` is a JSON
//! object of workspace path → text holding the file, what it needs and, for `design.overview`,
//! the project's other design files ([`runmat_design_tools::overview::related_candidates`]).
//!
//! The module answers for ECAD files (`.circuit`, `.circuit.layout`). Parts, assemblies and
//! drawings regenerate on OCCT, so the page runs their tools in the CAD host the editor compiles
//! them in (`runmat-geometry-wasm-host`, its `designTool` call: the same
//! [`runmat_design_tools::call_json`] on the kernel that host installs). Asked for one here, the
//! module answers `geometry_host`, so no answer comes from the grid test kernel.

use runmat_design_tools::{call, DesignKind, Files};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

/// The answer for a part, assembly or drawing, which the page sends to the CAD host instead.
const MCAD_HERE: &str = "Parts, assemblies and drawings are checked in the editor's geometry host, not in this module.";
/// The same for a STEP import.
const STEP_HERE: &str = "STEP files are imported in the editor's geometry host, not in this module.";

/// The tools this host runs on ECAD files: the read-only ones, and `design.export`, which
/// answers the exported files for the page to write (the Desktop's Fabrication outputs).
/// `design.import` reads KiCad, XML project, shape-string JSON and pipe-record files into staged circuit sources here (the Desktop's
/// Import dialog); STEP import stays with the project. `design.propose_edit` stages a circuit or
/// board change here, so the sandbox's agent changes a schematic or board the way it does in a
/// project (the page keeps or undoes it). `design.render` draws through the server.
const TOOLS: &[&str] = &["design.overview", "design.check", "design.inspect", "design.erc", "design.drc", "design.simulate", "design.export", "design.import", "design.propose_edit"];

#[wasm_bindgen]
pub fn design_tool(name: &str, input: &str, files: &str) -> String {
    answer(name, input, files).to_string()
}

#[wasm_bindgen]
extern "C" {
    /// A JavaScript function the page passes for progress.
    #[derive(Clone)]
    #[wasm_bindgen(js_name = Function)]
    pub type ProgressCallback;
    #[wasm_bindgen(method, js_name = call)]
    fn call1(this: &ProgressCallback, context: &JsValue, step: &str);
}

/// [`design_tool`], calling `on_progress(step)` as each stage of a fabrication export starts
/// (`board`, each output's id, `preflight`; see `runmat_circuit::fab::progress`).
#[wasm_bindgen]
pub fn design_tool_progress(name: &str, input: &str, files: &str, on_progress: &ProgressCallback) -> String {
    let callback = on_progress.clone();
    runmat_design_tools::progress::reporting(move |step| callback.call1(&JsValue::NULL, step), || answer(name, input, files)).to_string()
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
    // No OCCT here: an overview lists the parts, assemblies and drawings it links unchecked.
    runmat_design_tools::overview::check_linked_models(false);
    if !TOOLS.contains(&name) {
        return Err(("unsupported".into(), format!("{name} runs with a server project")));
    }
    let input: Value = serde_json::from_str(input).map_err(|e| ("invalid_arguments".to_owned(), format!("the input isn't valid JSON: {e}")))?;
    let files: Files = serde_json::from_str(files).map_err(|e| ("invalid_arguments".to_owned(), format!("the files aren't a JSON object of path to text: {e}")))?;
    if name == "design.import" {
        return match input.get("format").and_then(Value::as_str) {
            Some("step") => Err(("geometry_host".into(), STEP_HERE.into())),
            _ => call(name, input, &files).map_err(|e| (e.code, e.message)),
        };
    }
    let path = input.get("path").and_then(Value::as_str).unwrap_or_default();
    match DesignKind::of(path) {
        Some(DesignKind::Circuit | DesignKind::Layout) => {}
        Some(_) => {
            return Err(("geometry_host".into(), MCAD_HERE.into()))
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
    fn an_xml_project_imports_here_with_its_report_and_step_waits_for_a_project() {
        let dir = format!("{}/../../runmat-circuit/tests/import/xml_project", env!("CARGO_MANIFEST_DIR"));
        let file = |n: &str| json!({ "name": n, "content": std::fs::read_to_string(format!("{dir}/{n}")).unwrap() });
        let input = json!({ "format": "xml_project", "target": "/p/blinky", "files": [file("blinky.sch"), file("blinky.brd")] });
        let out = answer("design.import", &input.to_string(), "{}");
        let data = out.get("data").unwrap_or_else(|| panic!("a staged change, not {out}"));
        assert!(data["files"].as_array().unwrap().iter().any(|f| f["path"] == "/p/blinky.circuit.layout"));
        assert_eq!(data["report"]["format"], ".sch / .brd");
        assert_eq!(data["report"]["carried"]["components"], 3);
        let step = answer("design.import", r#"{"format":"step","target":"/p/a.part","files":[]}"#, "{}");
        assert_eq!(step["error"]["code"], "geometry_host");
    }

    #[test]
    fn mcad_files_go_to_the_geometry_host_and_other_tools_refuse_with_their_reason() {
        let part = answer("design.overview", r#"{"path":"/p/a.part"}"#, r#"{"/p/a.part":""}"#);
        assert_eq!(part["error"]["code"], "geometry_host");
        // The message names a capability, never an internal (kernel, wasm, server project).
        let said = part["error"]["message"].as_str().unwrap().to_lowercase();
        for internal in ["occt", "kernel", "wasm", "server project"] {
            assert!(!said.contains(internal), "{said}");
        }
        let render = answer("design.render", r#"{"path":"/p/daq.circuit"}"#, "{}");
        assert_eq!(render["error"]["code"], "unsupported");
        let bad = answer("design.check", "{", "{}");
        assert_eq!(bad["error"]["code"], "invalid_arguments");
    }

    #[test]
    fn a_circuit_change_stages_here_and_a_part_change_goes_to_the_geometry_host() {
        let files = daq_files();
        let text = files["/p/daq.circuit"].as_str().unwrap();
        // A second copy of the gain resistor, wired to nothing new: the change stages with the file's new text.
        let anchor = "        component RG = { part = \"R0402-1K\" }\n";
        assert!(text.contains(anchor), "the daq fixture names RG");
        let new = text.replacen(anchor, &format!("{anchor}        component R900 = {{ part = \"R0402-1K\" }}\n"), 1);
        let input = json!({ "path": "/p/daq.circuit", "text": new }).to_string();
        let out = answer("design.propose_edit", &input, &files.to_string());
        let data = out.get("data").unwrap_or_else(|| panic!("a staged change, not {out}"));
        assert_eq!(data["files"][0]["path"], "/p/daq.circuit", "{data}");
        assert!(data["files"][0]["text"].as_str().unwrap().contains("R900"), "{data}");
        // A part's change regenerates the model: the page sends it to the geometry host.
        let part = answer("design.propose_edit", r#"{"path":"/p/a.part","text":"part a {}"}"#, r#"{"/p/a.part":""}"#);
        assert_eq!(part["error"]["code"], "geometry_host");
    }

    #[test]
    fn an_assembly_that_uses_the_board_is_listed_without_compiling_it_here() {
        let mut files = daq_files();
        files.as_object_mut().unwrap().insert(
            "/box.assembly".into(),
            Value::String("runmat 0.7\nstd 1\n\nassembly box {\n    use circuit daq from \"./p\" as daq {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 0 mm] } }\n    }\n}\n".into()),
        );
        let out = answer("design.overview", r#"{"path":"/p/daq.circuit"}"#, &files.to_string());
        let links = out["data"]["links"].as_array().unwrap_or_else(|| panic!("links, not {out}"));
        let assembly = links.iter().find(|l| l["path"] == "/box.assembly").unwrap_or_else(|| panic!("the assembly's link, not {out}"));
        assert_eq!(assembly["level"], "info", "{assembly}");
        assert_eq!(assembly["status"], runmat_design_tools::overview::NOT_CHECKED_HERE, "{assembly}");
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
