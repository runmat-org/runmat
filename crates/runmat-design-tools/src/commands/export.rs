//! Export commands as files: `manage.export`, `part.export-body` and `assembly.export` (STEP
//! AP242, IGES, STL, 3MF, glTF, a face's DXF: [`crate::exchange::host`]), `drawing.export`
//! (PDF, DXF, DWG or SVG of the current sheet or of every sheet), `schematic.export-pdf` and
//! `board.fab-outputs` ([`crate::exchange::export`]); `assembly.pack-and-share` copies the
//! assembly and every file it references into a folder ([`pack`]).
//!
//! The answer is `{ ok: true, name, mime, text | base64, files }`: `files` lists every file
//! written (`{ name, mime, text | base64 }`), and the first one's fields stand at the top for
//! callers that write one file.

use super::{failed, mark_unhandled, RunRequest};
use crate::exchange::{ExportFormat, ExportInput, ExportedFile};
use crate::Files;
use runmat_design_commands::EditorKind;
use serde_json::{json, Value};

fn file_json(f: &ExportedFile) -> Value {
    if f.encoding == "base64" {
        json!({ "name": f.name, "mime": f.mime_type, "base64": f.content })
    } else {
        json!({ "name": f.name, "mime": f.mime_type, "text": f.content })
    }
}

fn answer(files: Vec<Value>) -> Value {
    let Some(first) = files.first().cloned() else {
        return failed("the export wrote no file");
    };
    let mut out = first;
    out["ok"] = true.into();
    out["files"] = Value::Array(files);
    out
}

fn string<'a>(request: &'a RunRequest, id: &str) -> Option<&'a str> {
    request.values.get(id).and_then(Value::as_str).filter(|s| !s.is_empty())
}

fn configuration(request: &RunRequest) -> Option<String> {
    string(request, "configuration").filter(|c| *c != "default").map(str::to_owned)
}

fn exported(input: &ExportInput, files: &Files) -> Value {
    match crate::exchange::export(input, files) {
        Ok(out) => answer(out.files.iter().map(file_json).collect()),
        Err(e) => failed(e.message),
    }
}

fn input(path: &str, format: ExportFormat) -> ExportInput {
    serde_json::from_value(json!({ "path": path, "format": format })).expect("an export input")
}

/// Runs the export `request.command` on `path`.
pub fn run(path: &str, text: &str, files: &Files, request: &RunRequest, editor: EditorKind) -> Value {
    let mut files = files.clone();
    files.insert(path.to_owned(), text.to_owned());
    match (editor, request.command.as_str()) {
        (EditorKind::Drawing, "drawing.export") => drawing(path, text, &files, request),
        (EditorKind::Assembly, "assembly.pack-and-share") => pack(path, text, &files, request),
        (EditorKind::Part | EditorKind::Assembly, _) => {
            let body = json!({ "command": request.command, "values": request.values });
            let out: Value = serde_json::from_str(&crate::exchange::host::export_json(path, text, &files, &body.to_string()))
                .unwrap_or_else(|e| failed(format!("the export answer isn't JSON: {e}")));
            if out["ok"] != true {
                return out;
            }
            let mut file = out.clone();
            if let Some(o) = file.as_object_mut() {
                o.remove("ok");
            }
            answer(vec![file])
        }
        (EditorKind::Circuit | EditorKind::CircuitLayout, "schematic.export-pdf") => exported(&input(path, ExportFormat::Pdf), &files),
        (EditorKind::Circuit | EditorKind::CircuitLayout, "board.fab-outputs") => exported(&input(path, ExportFormat::Fabrication), &files),
        (_, other) => {
            mark_unhandled();
            failed(format!("`{other}` has no handler for this file"))
        }
    }
}

/// Pack and share: the assembly and every file it references, transitively, copied into the
/// folder `values.folder` beside it, each at its path relative to the assembly's folder (a file
/// from outside that folder goes in under its own name), so the folder opens on its own.
fn pack(path: &str, text: &str, files: &Files, request: &RunRequest) -> Value {
    let Some(folder) = string(request, "folder").map(|f| f.trim_matches('/').to_owned()).filter(|f| !f.is_empty() && !f.contains("..")) else {
        return failed("Pack and share needs a folder name");
    };
    let base = &path[..path.rfind('/').map(|i| i + 1).unwrap_or(0)];
    let mut todo = vec![(path.to_owned(), text.to_owned())];
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    let mut missing = Vec::new();
    while let Some((at, body)) = todo.pop() {
        if !seen.insert(at.clone()) {
            continue;
        }
        for dep in runmat_cad_wasm::runtime::dependency_files(&at, &body, files) {
            match files.get(&dep) {
                Some(t) => todo.push((dep, t.clone())),
                None if !seen.contains(&dep) => missing.push(dep),
                None => {}
            }
        }
        let relative = at.strip_prefix(base).map(str::to_owned).unwrap_or_else(|| at.rsplit('/').next().unwrap_or(&at).to_owned());
        out.push(json!({ "name": format!("{folder}/{relative}"), "mime": "text/plain", "text": body }));
    }
    if !missing.is_empty() {
        return failed(format!("Pack and share needs every referenced file; these weren't found: {}", missing.join(", ")));
    }
    answer(out)
}

/// The sheet names of a drawing, in order.
fn sheet_names(path: &str, text: &str, files: &Files) -> Vec<String> {
    let scope = runmat_cad_wasm::runtime::module_scope(path, text, files);
    runmat_cad::syntax::parse_drawing_in(path, text, &scope)
        .document
        .map(|d| d.sheets.iter().map(|s| s.name.clone()).collect())
        .unwrap_or_default()
}

fn drawing(path: &str, text: &str, files: &Files, request: &RunRequest) -> Value {
    let format = match string(request, "format").unwrap_or("pdf") {
        "pdf" => ExportFormat::Pdf,
        "dxf" => ExportFormat::Dxf,
        "dwg" => ExportFormat::Dwg,
        "svg" => ExportFormat::Svg,
        other => {
            // A format the catalog offers with no writer behind it.
            mark_unhandled();
            return failed(format!("a drawing exports as PDF, DXF, DWG or SVG, not `{other}`"));
        }
    };
    let all = string(request, "sheets").unwrap_or("all") == "all";
    let sheets: Vec<String> = if all { sheet_names(path, text, files) } else { string(request, "sheet").map(|s| vec![s.to_owned()]).unwrap_or_default() };
    let configuration = configuration(request);
    let one = |sheet: Option<&str>| -> Result<ExportedFile, String> {
        let mut input = input(path, format);
        input.sheet = sheet.map(str::to_owned);
        input.configuration = configuration.clone();
        let out = crate::exchange::export(&input, files).map_err(|e| e.message)?;
        out.files.into_iter().next().ok_or_else(|| "the export wrote no file".to_owned())
    };
    if sheets.len() <= 1 {
        return match one(sheets.first().map(String::as_str)) {
            Ok(f) => answer(vec![file_json(&f)]),
            Err(message) => failed(message),
        };
    }
    if format == ExportFormat::Pdf {
        // Every sheet as one page of one PDF.
        return match crate::exchange::drawing_sheets_pdf(path, text, files, &sheets, configuration.as_deref()) {
            Ok(f) => answer(vec![file_json(&f)]),
            Err(e) => failed(e.message),
        };
    }
    let mut out = Vec::new();
    for sheet in &sheets {
        match one(Some(sheet)) {
            Ok(f) => out.push(file_json(&f)),
            Err(message) => return failed(message),
        }
    }
    answer(out)
}
