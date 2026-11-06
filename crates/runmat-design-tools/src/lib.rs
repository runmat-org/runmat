//! The design tool layer: one set of tools over `.part`, `.assembly`, `.drawing`, `.circuit`
//! and `.circuit.layout` files, declared here once and called by every client. The server's
//! MCP registry lists [`catalog`] next to the generic project tools; the in-app agent calls the
//! same [`call`].
//!
//! | Tool | What it does |
//! |---|---|
//! | `design.check` | Regenerates a file: the editor's diagnostics (root causes, the features that failed because of each, source ranges) and the resolved inputs. |
//! | `design.render` | A PNG of a view (named view or camera, display mode) through the Desktop's renderer. |
//! | `design.inspect` | Named faces, regions, published values, inputs, features, mates, the BOM, mass properties, components, nets and rooms, filtered by name or path. |
//! | `design.drc`, `design.erc` | Structured violations: rule, objects, location and source range. |
//! | `design.import` | KiCad projects and STEP files in, as staged RunMat sources. |
//! | `design.export` | STEP, STL, 3MF, glTF, DXF and DWG (drawing sheets and flat patterns), PDF (drawing sheets), KiCad out, a board's fabrication set, and KiCad, SPICE, EDIF and IPC-D-356A netlists. |
//! | `design.overview` | What a file is, its state from the checks above, the files linked to it and their status, the decisions its guides still need, and two or three next steps. |
//! | `design.propose_edit` | Source text or a catalog command with its values, as a staged change with its diff and the diagnostics after it. |
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
pub mod workspace;
pub mod edit;
pub mod exchange;
pub mod formats;
pub mod guides;
pub mod inspect;
pub mod kernel;
mod locate;
pub mod overview;
#[cfg(feature = "render")]
pub mod render;
pub mod rules;
pub mod scene;
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
        Some(DesignKind::Part) => Vec::new(),
        Some(DesignKind::Assembly) => runmat_cad_wasm::runtime::dependencies(path, text),
        Some(DesignKind::Drawing) => runmat_cad_wasm::drawing::dependencies(path, text),
        Some(DesignKind::Circuit | DesignKind::Layout) => {
            runmat_circuit::runtime::dependencies(path, text)
        }
        None => Vec::new(),
    }
}

/// Reads `path` and everything it needs through `read` (`None` for a file that doesn't exist),
/// breadth-first, each file once. A file that doesn't exist is left out; the tools report it
/// where it matters (an assembly draws a missing part as its box and says so).
pub fn dependency_closure(
    path: &str,
    text: &str,
    mut read: impl FnMut(&str) -> Option<String>,
    max_files: usize,
) -> Files {
    let mut files = Files::new();
    files.insert(path.to_owned(), text.to_owned());
    let mut queue: Vec<String> = dependencies(path, text);
    while let Some(next) = queue.pop() {
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

/// One tool's declaration: what MCP `tools/list` and the in-app agent's tool list show.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    /// `design.check`, `design.render`, …
    pub name: String,
    pub title: String,
    /// When a client should call it, in one or two sentences.
    pub description: String,
    /// JSON Schema of the input. Every input has `path`, the workspace path of the design file.
    pub input_schema: Value,
    /// JSON Schema of the `data` a successful call returns.
    pub output_schema: Value,
    /// The tool reads only: it changes no file (edits come back staged).
    pub read_only: bool,
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

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).expect("tool schema serializes")
}

fn definition<I: JsonSchema, O: JsonSchema>(
    name: &str,
    title: &str,
    description: &str,
) -> ToolDefinition {
    ToolDefinition {
        name: name.to_owned(),
        title: title.to_owned(),
        description: description.to_owned(),
        input_schema: schema::<I>(),
        output_schema: schema::<O>(),
        read_only: true,
    }
}

/// Every design tool, in the order clients list them.
pub fn catalog() -> Vec<ToolDefinition> {
    let mut tools = vec![
        definition::<check::CheckInput, check::CheckOutput>(
            "design.check",
            "Check design file",
            "Use this after changing a .part, .assembly, .drawing, .circuit or .circuit.layout file, or before relying on it: regenerates the file and returns the editor's diagnostics (each root cause with the features that failed because of it and the source range) and the resolved inputs.",
        ),
        definition::<inspect::InspectInput, inspect::InspectOutput>(
            "design.inspect",
            "Inspect design",
            "Use this to read a design's structure by name: named faces and regions, published values, inputs, features, occurrences, mates, the BOM and mass properties of parts and assemblies; components, nets, net classes, rooms and the BOM of circuits; sheets and views of drawings. Filter by name or path (`*` matches any run).",
        ),
        definition::<rules::RulesInput, rules::RulesOutput>(
            "design.drc",
            "Run design-rule check",
            "Use this to check a board (.circuit.layout, or a .circuit with a board): fills pours and returns each DRC violation with its rule, the objects involved, the copper layer, the board location in millimetres, the measured and required values and the source range of the rule.",
        ),
        definition::<rules::RulesInput, rules::RulesOutput>(
            "design.erc",
            "Run electrical-rule check",
            "Use this to check a schematic (.circuit, or the .circuit a layout uses): returns each ERC finding with its code, the pins, components and nets involved and the source range of the declaration.",
        ),
        definition::<exchange::ExportInput, exchange::ExportOutput>(
            "design.export",
            "Export design",
            "Use this to hand a design to another tool: parts and assemblies as STEP, STL, 3MF or glTF; drawing sheets and sheet-metal flat patterns as DXF or DWG, drawing sheets as PDF; circuits and boards as a KiCad project; a board's fabrication set (format `fabrication`: Gerber X2, Excellon drill and drill map, fabrication drawing, pick-and-place and BOM per variant, IPC-2581, ODB++, IPC-D-356A); a circuit's netlist as `kicad_netlist`, `spice`, `edif` or `ipc356`. Returns the files' bytes (base64 for binary formats).",
        ),
        definition::<exchange::ImportInput, exchange::ImportChange>(
            "design.import",
            "Import into design",
            "Use this to bring a KiCad project (.kicad_sch, .kicad_pcb), an EAGLE XML schematic or board (.sch, .brd; format `eagle`), a shape-string JSON schematic or board (standard edition; format `shape_json`) or a STEP file into the project as RunMat sources. Returns a staged change: the new files, their diagnostics, the project calls that keep it and, for EAGLE and shape-string JSON, a report of what was carried over, approximated or left out.",
        ),
        definition::<simulate::SimulateInput, simulate::SimulateOutput>(
            "design.simulate",
            "Simulate circuit",
            "Use this to simulate a .circuit: runs its `simulation` section (operating point, DC sweep, AC, transient and Monte Carlo over component tolerances) on RunMat's circuit engine and returns the probed nets' waveforms (transient and DC in volts, AC as dB and degrees), each measurement (Monte Carlo ones with mean, standard deviation and a histogram), the parts left out for want of a model, and the measured operations.",
        ),
        definition::<overview::OverviewInput, overview::OverviewOutput>(
            "design.overview",
            "Overview design file",
            "Use this first when a design file is opened or before planning work on it: says what the file is, its state from the real checks (regeneration, ERC and DRC counts, whether the board has the schematic's changes, unplaced parts, simulation sections), the project files linked to it (a part's studies and drawings, a schematic's board and the board's studies and enclosure) with their status, the decisions its guides still need, and two or three next steps, each a guide or a catalog command. Every fact names the check it came from.",
        ),
        definition::<edit::ProposeEditInput, edit::StagedChange>(
            "design.propose_edit",
            "Propose design edit",
            "Use this to change a design: give the new source text, or a catalog command (for example part.extrude, part.fillet, board.route) with its values and selection. Returns a staged change: the feature and dimension diff or the schematic-to-board change rows, the diagnostics after it, and the patch that keeps it and the patch that undoes it.",
        ),
    ];
    let render = definition::<scene::RenderInput, scene::RenderOutput>(
        "design.render",
        "Render design view",
        "Use this to see a part or assembly: renders a named view (isometric, front, back, left, right, top, bottom) or a camera, in a display mode (shaded, shaded_edges, wireframe, hidden_lines), at a size and theme, through the Desktop's renderer, to a PNG.",
    );
    tools.insert(1, render);
    tools
}

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
        "design.render" => render_call(input, files),
        other => Err(ToolError::new(
            "unknown_tool",
            format!("{other} isn't a design tool"),
        )),
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

    #[test]
    fn the_catalog_is_unique_and_described() {
        let tools = catalog();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), tools.len());
        assert_eq!(names.len(), 10);
        for t in &tools {
            assert!(t.name.starts_with("design."));
            assert!(t.description.starts_with("Use this"));
            assert_eq!(t.input_schema["type"], "object", "{}", t.name);
            assert!(
                t.read_only,
                "{} changes nothing; edits come back staged",
                t.name
            );
        }
    }
}
