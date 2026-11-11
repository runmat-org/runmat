//! The command registry: for every catalog command, the surfaces that run it and the handler
//! that runs it there. The Desktop (through the CAD host's `run`, `edit`, `analyze` and `export`
//! calls), the agent's design tools (`design.propose_edit` with a `command`) and the CLI
//! (`runmat-design run`) all dispatch through [`run`], so a command behaves the same everywhere.
//!
//! - [`route`]`(id, editor)` → the command's [`Route`]: its surfaces and its [`Handler`].
//! - [`run`]`(path, text, files, request)` → one answer, tagged by `kind`:
//!   `{ ok: true, kind: "edit", files, summary }` (a text change),
//!   `{ ok: true, kind: "table", title, summary, columns, rows, problems, data, … }` (a results
//!   table), `{ ok: true, kind: "file", name, mime, text | base64, files }` (an export), or
//!   `{ ok: false, message }`.
//!
//! A handler that has no code for a command marks the call ([`take_unhandled`]). The registry's
//! build check (`tests/command_registry.rs`) runs every catalog command on every surface its
//! route claims and fails when one is marked, so a toolbar never offers a command that only
//! answers "not here".

mod export;
mod report;

use crate::Files;
use runmat_design_commands::{Catalog, CommandDescriptor, EditOp, EditorKind};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::cell::Cell;
use std::sync::OnceLock;

/// Where a command can be run from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// The Desktop and browser editors' toolbars, menus and command search.
    Desktop,
    /// The agent's design tools (in-app and MCP).
    Agent,
    /// `runmat-design run`.
    Cli,
}

/// What runs a command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Handler {
    /// A model edit written as the smallest text change (the CAD and circuit edit dispatchers).
    Edit,
    /// An Evaluate command's results table (`runmat_cad_wasm::analyze`).
    Analyze,
    /// An assembly report: the bill of materials and Where-used ([`report`]).
    Report,
    /// A file made from the design ([`export`]).
    Export,
    /// A view, mode or agent command the Desktop shell runs on its canvas and panels; there is
    /// no file change or answer for another surface to show.
    Shell,
}

/// One command in one editor: who runs it and where.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Route {
    pub id: String,
    pub editor: EditorKind,
    pub surfaces: Vec<Surface>,
    pub handler: Handler,
}

/// A command to run: the catalog id, the values the form holds and the selection (Outline ids).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct RunRequest {
    pub command: String,
    #[serde(default)]
    pub values: Map<String, Value>,
    #[serde(default)]
    pub selection: Vec<String>,
}

thread_local! {
    static UNHANDLED: Cell<bool> = const { Cell::new(false) };
}

/// Marks the current call as one no handler took.
pub(crate) fn mark_unhandled() {
    UNHANDLED.with(|u| u.set(true));
}

/// Whether the last calls reached a dispatcher with no code for their command, clearing the
/// mark (the CAD and circuit dispatchers' marks included).
pub fn take_unhandled() -> bool {
    let own = UNHANDLED.with(|u| u.replace(false));
    let cad = runmat_cad_wasm::edit::take_unhandled();
    let circuit = runmat_circuit::runtime::take_unhandled();
    own | cad | circuit
}

/// The merged CAD and circuit catalog, the one the Desktop loads.
pub fn catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut c = runmat_cad::commands::catalog();
        c.merge(runmat_circuit::commands::catalog());
        c
    })
}

/// The editor a file opens in, by its extension.
pub fn editor_of(path: &str) -> Option<EditorKind> {
    let lower = path.to_ascii_lowercase();
    Some(if lower.ends_with(".circuit.layout") {
        EditorKind::CircuitLayout
    } else if lower.ends_with(".circuit") {
        EditorKind::Circuit
    } else if lower.ends_with(".part") {
        EditorKind::Part
    } else if lower.ends_with(".assembly") {
        EditorKind::Assembly
    } else if lower.ends_with(".drawing") {
        EditorKind::Drawing
    } else if lower.ends_with(".fea") {
        EditorKind::Fea
    } else {
        return None;
    })
}

fn is_circuit(editor: EditorKind) -> bool {
    matches!(editor, EditorKind::Circuit | EditorKind::CircuitLayout)
}

/// The handler of `command` in `editor`.
fn handler_of(command: &CommandDescriptor, editor: EditorKind) -> Handler {
    let id = command.id.as_str();
    match command.edit.op {
        _ if editor == EditorKind::Fea => Handler::Shell,
        EditOp::View
        | EditOp::EnterMode
        | EditOp::ExitMode
        | EditOp::DiscardMode
        | EditOp::Agent => Handler::Shell,
        // Rebuild recompiles the open file; the circuit checks and simulation show in the
        // circuit editor's panes (the agent has `design.drc`, `design.erc` and `design.simulate`).
        EditOp::Check if id == "design.rebuild" || is_circuit(editor) => Handler::Shell,
        EditOp::Analyze if report::handles(id) => Handler::Report,
        // Measure on a drawing or a board is the canvas's own measure tool.
        EditOp::Analyze
            if id == "view.measure"
                && !matches!(editor, EditorKind::Part | EditorKind::Assembly) =>
        {
            Handler::Shell
        }
        EditOp::Analyze if !is_circuit(editor) => Handler::Analyze,
        EditOp::Export => Handler::Export,
        _ => Handler::Edit,
    }
}

fn route_of(command: &CommandDescriptor, editor: EditorKind) -> Route {
    let handler = handler_of(command, editor);
    let surfaces = match handler {
        Handler::Shell => vec![Surface::Desktop],
        _ => vec![Surface::Desktop, Surface::Agent, Surface::Cli],
    };
    Route {
        id: command.id.clone(),
        editor,
        surfaces,
        handler,
    }
}

/// The route of catalog command `id` in `editor`; `None` when the catalog doesn't offer it there.
pub fn route(id: &str, editor: EditorKind) -> Option<Route> {
    let command = catalog().command(id)?;
    command
        .editors
        .contains(&editor)
        .then(|| route_of(command, editor))
}

/// Every catalog command in every editor that offers it.
pub fn routes() -> Vec<Route> {
    catalog()
        .commands
        .iter()
        .flat_map(|c| c.editors.iter().map(move |e| route_of(c, *e)))
        .collect()
}

fn failed(message: impl Into<String>) -> Value {
    json!({ "ok": false, "message": message.into() })
}

fn tagged(mut answer: Value, kind: &str) -> Value {
    if answer["ok"] == true {
        if let Some(o) = answer.as_object_mut() {
            o.insert("kind".into(), kind.into());
        }
    }
    answer
}

/// Runs `request` on the file `path` (text `text`, with the files it needs in `files`).
pub fn run(path: &str, text: &str, files: &Files, request: &RunRequest) -> Value {
    run_or(path, text, files, request, Handler::Edit)
}

/// [`run`], with `fallback` running a command the catalog doesn't list for this file (the
/// Properties panel's `part.set-property`, a repair, a study's `simulate.check`): the host's
/// `edit`, `analyze` and `export` calls each pass their own.
pub fn run_or(
    path: &str,
    text: &str,
    files: &Files,
    request: &RunRequest,
    fallback: Handler,
) -> Value {
    let Some(editor) = editor_of(path) else {
        return failed(format!("{path} isn't a design file"));
    };
    let handler = route(&request.command, editor)
        .map(|r| r.handler)
        .unwrap_or(fallback);
    match handler {
        Handler::Edit => tagged(edit(path, text, files, request, editor), "edit"),
        Handler::Analyze => tagged(analyze(path, text, files, request), "table"),
        Handler::Report => tagged(report::run(path, text, files, request), "table"),
        Handler::Export => tagged(export::run(path, text, files, request, editor), "file"),
        Handler::Shell => {
            let label = catalog()
                .command(&request.command)
                .map(|c| c.label.as_str())
                .unwrap_or(request.command.as_str());
            failed(format!(
                "{label} runs in the editor's canvas: open {path} to use it"
            ))
        }
    }
}

/// [`run`] on JSON text: `files` is path → text and `request` a [`RunRequest`].
pub fn run_json(path: &str, text: &str, files: &Files, request: &str) -> String {
    run_json_or(path, text, files, request, Handler::Edit)
}

/// [`run_or`] on JSON text.
pub fn run_json_or(
    path: &str,
    text: &str,
    files: &Files,
    request: &str,
    fallback: Handler,
) -> String {
    match serde_json::from_str::<RunRequest>(request) {
        Ok(request) => run_or(path, text, files, &request, fallback).to_string(),
        Err(e) => json!({ "ok": false, "message": format!("the request isn't valid: {e}"), "diagnostics": [] }).to_string(),
    }
}

fn edit(path: &str, text: &str, files: &Files, request: &RunRequest, editor: EditorKind) -> Value {
    if is_circuit(editor) {
        let circuit = runmat_circuit::runtime::EditRequest {
            command: request.command.clone(),
            values: request.values.clone(),
            selection: request.selection.clone(),
            options: Default::default(),
        };
        return match runmat_circuit::runtime::edit(path, text, files, &circuit) {
            Ok(out) => json!({ "ok": true, "files": out.files, "summary": out.summary }),
            Err(e) => json!({ "ok": false, "message": e.message, "diagnostics": e.diagnostics }),
        };
    }
    let body = json!({ "command": request.command, "values": request.values, "selection": request.selection });
    serde_json::from_str(&runmat_cad_wasm::edit::edit_json(
        path,
        text,
        files,
        &body.to_string(),
    ))
    .unwrap_or_else(|e| failed(format!("the edit answer isn't JSON: {e}")))
}

/// Catalog ids that run another Evaluate command: Mold's Draft analysis is Evaluate's Draft.
const SAME_ANALYSIS: &[(&str, &str)] = &[("mold.draft-analysis", "evaluate.draft")];

fn analyze(path: &str, text: &str, files: &Files, request: &RunRequest) -> Value {
    let command = SAME_ANALYSIS
        .iter()
        .find(|(id, _)| *id == request.command)
        .map(|(_, to)| *to)
        .unwrap_or(&request.command);
    let body = json!({ "command": command, "values": request.values });
    let out: Value = serde_json::from_str(&runmat_cad_wasm::analyze::analyze_json(
        path,
        text,
        files,
        &body.to_string(),
    ))
    .unwrap_or_else(|e| failed(format!("the analysis answer isn't JSON: {e}")));
    // The analyze dispatcher's own fall-through.
    if out["message"]
        .as_str()
        .is_some_and(|m| m.contains("isn't an Evaluate command"))
    {
        mark_unhandled();
    }
    out
}
