//! The design tool layer: one set of deterministic tools over `.part`, `.assembly`, `.drawing`,
//! `.circuit` and `.circuit.layout` files, run by name through [`call`]. The `runmat` CLI, the CAD
//! host (`runmat-geometry-wasm-host`, its `designTool` call) and the servers call it.
//!
//! | Tool | What it returns |
//! |---|---|
//! | `design.check` | The regenerated file's diagnostics (root causes, the features that failed because of each, source ranges) and the resolved inputs. |
//! | `design.render` | A PNG of a view (named view or camera, display mode) through the Desktop's renderer. |
//! | `design.inspect` | Named faces, regions, published values, inputs, features, mates, the BOM, mass properties, components, nets and rooms, filtered by name or path. |
//! | `design.drc`, `design.erc` | Structured violations: rule, objects, location and source range. |
//! | `design.import` | KiCad projects and STEP files in, as staged RunMat sources. |
//! | `design.export` | STEP, STL, 3MF, glTF, DXF and DWG (drawing sheets and flat patterns), PDF (drawing sheets), KiCad out, a board's fabrication set, and KiCad, SPICE, EDIF and IPC-D-356A netlists. |
//! | `design.simulate` | A circuit's simulation results. |
//! | `design.overview` | What a file is, its state from the checks above, the files linked to it with their status, and the checks' counts. |
//! | `design.propose_edit` | Source text or a catalog command with its values, as a staged change with its diff and the diagnostics after it. |
//! | `design.seat_part` | A component seated against another (a face, a ray's surface or a connector) with a mate, as a staged change with the solved pose, its interference and its clearance. |
//!
//! Every tool is a pure function of its input and the workspace [`Files`] (path → text) the
//! caller read. [`dependencies`] lists what a file needs; the caller reads those files, and
//! theirs, until nothing is new. Nothing here touches a file system, a network or a clock, so
//! the server, the Desktop and tests get the same answer from the same files.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub mod check;
pub mod ci;
pub mod circuit_values;
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
pub mod commands;
#[cfg(not(target_arch = "wasm32"))]
pub mod workspace;
pub mod edit;
pub mod exchange;
pub mod formats;
pub mod inspect;
pub mod kernel;
mod locate;
pub mod overview;
#[cfg(feature = "render")]
pub mod render;
pub mod rules;
pub mod scene;
pub mod seat;
pub mod simulate;

pub use locate::{Locator, Range};
/// Progress while a board's fabrication set is made ([`runmat_circuit::fab::progress`]).
pub use runmat_circuit::fab::progress;

/// Workspace path → file text.
pub type Files = BTreeMap<String, String>;

/// The design file kinds the tools work on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DesignKind {
    Part,
    Assembly,
    Drawing,
    Circuit,
    Layout,
}

impl DesignKind {
    /// The kind of `path` by its extension.
    pub fn of(path: &str) -> Option<Self> {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".circuit.layout") {
            Some(Self::Layout)
        } else if lower.ends_with(".circuit") {
            Some(Self::Circuit)
        } else if lower.ends_with(".part") {
            Some(Self::Part)
        } else if lower.ends_with(".assembly") {
            Some(Self::Assembly)
        } else if lower.ends_with(".drawing") {
            Some(Self::Drawing)
        } else {
            None
        }
    }

    pub fn is_circuit(self) -> bool {
        matches!(self, Self::Circuit | Self::Layout)
    }

    pub fn is_model(self) -> bool {
        matches!(self, Self::Part | Self::Assembly)
    }
}

/// Whether `path` is a design file these tools work on.
pub fn is_design_path(path: &str) -> bool {
    DesignKind::of(path).is_some()
}

/// The files `path` needs beyond itself, resolved to workspace paths. Ask again for each file
/// read until nothing is new ([`dependency_closure`] does this over a reader).
pub fn dependencies(path: &str, text: &str) -> Vec<String> {
    match DesignKind::of(path) {
        // A part needs the STEP files its imports read and the folders of its `.m` values.
        Some(DesignKind::Part | DesignKind::Assembly) => {
            runmat_cad_wasm::runtime::dependencies(path, text)
        }
        Some(DesignKind::Drawing) => runmat_cad_wasm::drawing::dependencies(path, text),
        Some(DesignKind::Circuit | DesignKind::Layout) => {
            runmat_circuit::runtime::dependencies(path, text)
        }
        None => Vec::new(),
    }
}

/// The files `path` needs, resolved among `files` (which hold the folders [`dependencies`]
/// lists): a design named by symbol is the file that declares it, so an assembly's parts, a
/// drawing's models and a layout's circuit come back as file paths, and folder listings drop
/// out. The overview's links and every other consumer that compares paths read this.
pub fn dependency_files(path: &str, text: &str, files: &Files) -> Vec<String> {
    match DesignKind::of(path) {
        Some(DesignKind::Part) | None => Vec::new(),
        Some(DesignKind::Assembly) => runmat_cad_wasm::runtime::dependency_files(path, text, files),
        Some(DesignKind::Drawing) => runmat_cad_wasm::drawing::dependency_files(path, text, files),
        Some(DesignKind::Circuit | DesignKind::Layout) => {
            runmat_circuit::runtime::dependency_files(path, text, files)
        }
    }
}

/// Reads `path` and everything it needs through `read` (`None` for a file that doesn't exist),
/// breadth-first, each file once. A folder [`dependencies`] lists (`/p/esc/`) is each file of
/// `project` directly in it, as a host lists one. A file that doesn't exist is left out; the
/// tools report it where it matters (an assembly draws a missing part as its box and says so).
pub fn dependency_closure(
    path: &str,
    text: &str,
    project: &[String],
    mut read: impl FnMut(&str) -> Option<String>,
    max_files: usize,
) -> Files {
    let mut files = Files::new();
    files.insert(path.to_owned(), text.to_owned());
    let parent = |p: &str| match p.rsplit_once('/') {
        Some(("", _)) => "/".to_owned(),
        Some((d, _)) => d.to_owned(),
        None => String::new(),
    };
    let in_folder = |listing: &str| -> Vec<String> {
        let folder = match listing.trim_end_matches('/') {
            "" => "/",
            f => f,
        };
        project
            .iter()
            .filter(|p| parent(p) == folder)
            .cloned()
            .collect()
    };
    let mut queue: Vec<String> = dependencies(path, text);
    while let Some(next) = queue.pop() {
        if next.ends_with('/') {
            queue.extend(
                in_folder(&next)
                    .into_iter()
                    .filter(|p| !files.contains_key(p)),
            );
            continue;
        }
        if files.contains_key(&next) || files.len() >= max_files {
            continue;
        }
        if let Some(t) = read(&next) {
            queue.extend(dependencies(&next, &t));
            files.insert(next, t);
        }
    }
    files
}

/// A tool call that couldn't run: the input is malformed or names something that isn't there.
/// Design problems (a feature that fails, a DRC error) are results, not errors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolError {
    /// `invalid_arguments`, `unknown_tool`, `unsupported`, `file_missing`, `not_compiled`,
    /// `render_unavailable`.
    pub code: String,
    pub message: String,
}

impl ToolError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        ToolError {
            code: code.to_owned(),
            message: message.into(),
        }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new("invalid_arguments", message)
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for ToolError {}

pub type ToolResult = Result<Value, ToolError>;

/// Every tool [`call`] runs, by name.
pub const TOOL_NAMES: [&str; 11] = [
    "design.check",
    "design.render",
    "design.inspect",
    "design.drc",
    "design.erc",
    "design.export",
    "design.import",
    "design.simulate",
    "design.overview",
    "design.propose_edit",
    "design.seat_part",
];

/// Runs the tool `name` with `input` over `files` (which holds `input.path` and what it needs).
pub fn call(name: &str, input: Value, files: &Files) -> ToolResult {
    fn parse<T: serde::de::DeserializeOwned>(v: Value) -> Result<T, ToolError> {
        serde_json::from_value(v).map_err(|e| ToolError::invalid(e.to_string()))
    }
    fn out<T: Serialize>(v: T) -> ToolResult {
        serde_json::to_value(v).map_err(|e| ToolError::new("internal", e.to_string()))
    }
    kernel::install();
    match name {
        "design.check" => out(check::check(&parse(input)?, files)?),
        "design.inspect" => out(inspect::inspect(&parse(input)?, files)?),
        "design.drc" => out(rules::drc(&parse(input)?, files)?),
        "design.erc" => out(rules::erc(&parse(input)?, files)?),
        "design.export" => out(exchange::export(&parse(input)?, files)?),
        "design.import" => out(exchange::import(&parse(input)?, files)?),
        "design.propose_edit" => out(edit::propose(&parse(input)?, files)?),
        "design.simulate" => out(simulate::simulate(&parse(input)?, files)?),
        "design.overview" => out(overview::overview(&parse(input)?, files)?),
        "design.seat_part" => out(seat::seat(&parse(input)?, files)?),
        "design.render" => render_call(input, files),
        other => Err(ToolError::new(
            "unknown_tool",
            format!("{other} isn't a design tool"),
        )),
    }
}

/// [`call`] with JSON text in and out, the shape every host entry answers (the server's
/// design-tools route, the agent's design-tools wasm and the CAD host's `designTool`):
/// `{ "data": … }` or `{ "error": { "code", "message" } }`. `input` is the tool's input and
/// `files` a JSON object of workspace path → text. Parts, assemblies and drawings regenerate on
/// the kernel the host installed (`runmat_cad_wasm::kernel`): OCCT in the CAD host and on the
/// server, the grid test double in this crate's tests.
pub fn call_json(name: &str, input: &str, files: &str) -> Value {
    let parsed = serde_json::from_str::<Value>(input)
        .map_err(|e| ToolError::invalid(format!("the input isn't valid JSON: {e}")))
        .and_then(|input| {
            let files: Files = serde_json::from_str(files).map_err(|e| {
                ToolError::invalid(format!(
                    "the files aren't a JSON object of path to text: {e}"
                ))
            })?;
            call(name, input, &files)
        });
    match parsed {
        Ok(data) => serde_json::json!({ "data": data }),
        Err(e) => serde_json::json!({ "error": { "code": e.code, "message": e.message } }),
    }
}

#[cfg(feature = "render")]
fn render_call(input: Value, files: &Files) -> ToolResult {
    let input: scene::RenderInput =
        serde_json::from_value(input).map_err(|e| ToolError::invalid(e.to_string()))?;
    serde_json::to_value(render::render(&input, files)?)
        .map_err(|e| ToolError::new("internal", e.to_string()))
}

#[cfg(not(feature = "render"))]
fn render_call(_input: Value, _files: &Files) -> ToolResult {
    Err(ToolError::new(
        "render_unavailable",
        "this host renders through the server; call design.render there",
    ))
}

/// The text of `path` in `files`, or the `file_missing` error.
pub(crate) fn text_of<'a>(files: &'a Files, path: &str) -> Result<&'a str, ToolError> {
    files
        .get(path)
        .map(String::as_str)
        .ok_or_else(|| ToolError::new("file_missing", format!("{path} isn't in the workspace")))
}

/// The design kind of `path`, or `invalid_arguments`.
pub(crate) fn kind_of(path: &str) -> Result<DesignKind, ToolError> {
    DesignKind::of(path).ok_or_else(|| {
        ToolError::invalid(format!(
            "{path} isn't a design file (.part, .assembly, .drawing, .circuit or .circuit.layout)"
        ))
    })
}

/// Whether `name` matches `filter`: `*` matches any run of characters; without a `*` the filter
/// matches anywhere in the name. Case-insensitive.
pub fn name_matches(filter: Option<&str>, name: &str) -> bool {
    let Some(filter) = filter.filter(|f| !f.is_empty()) else {
        return true;
    };
    let f = filter.to_ascii_lowercase();
    let n = name.to_ascii_lowercase();
    if !f.contains('*') {
        return n.contains(&f);
    }
    let parts: Vec<&str> = f.split('*').collect();
    let mut at = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !n.starts_with(part) {
                return false;
            }
            at = part.len();
            continue;
        }
        match n[at..].find(part) {
            Some(k) => at += k + part.len(),
            None => return false,
        }
    }
    parts
        .last()
        .is_none_or(|last| last.is_empty() || n.ends_with(last))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_match_by_substring_or_glob() {
        assert!(name_matches(None, "anything"));
        assert!(name_matches(Some("mount"), "feature.base.face.mount_top"));
        assert!(name_matches(
            Some("feature.*.face.end"),
            "feature.boss.face.end"
        ));
        assert!(!name_matches(
            Some("feature.*.face.end"),
            "feature.boss.face.side"
        ));
        assert!(name_matches(Some("U*"), "U3"));
        assert!(!name_matches(Some("U*"), "R3"));
    }
}
