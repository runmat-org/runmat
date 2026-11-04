//! The document runtime: what the Desktop and the browser call to open, check and edit a
//! circuit's files.
//!
//! A circuit is two source files, the schematic (`io.circuit`) and its board
//! (`io.circuit.layout`), plus the lock that pins auto-annotated designators
//! (`io.circuit.lock`, [`LockSection`] text). Either source opens the same design:
//! - a layout names its schematic with `use circuit <name> from "<file>"`;
//! - a schematic's board is the sibling `<stem>.circuit.layout`.
//!
//! The caller reads files; the runtime never touches a file system. [`dependencies`] lists
//! what a source needs, the caller reads those files (and theirs) into a [`Files`] map, and
//! [`compile`], [`review`] and [`edit`] take the map. An edit returns each changed file's
//! new text with the minimal text edits that make it (byte offsets converted to UTF-16 code
//! units, as editors count them).
//!
//! Errors are [`Diagnostic`]s with the file they belong to. Nothing falls back silently: a
//! file that can't be read or doesn't compile is an error the caller shows.

use crate::board_rules::{BoardGraph, BoardRules};
use crate::drc::Drc;
use crate::eco::{self, ApplyOptions, ComparatorOptions, Difference, EcoPlan, Schematic, Status};
use crate::elab::{elaborate, ElabOptions, Netlist};
use crate::erc::{self, ErcOptions};
use crate::ids::{fnv128, DerivedUuids, LockSection};
use crate::index::{add_board, add_eco, build_index, eco_index, CircuitIndex, EcoIndex};
use crate::ir::Design;
use crate::layout::{Board, ItemId, Side};
use crate::library::Pool;
use crate::review::{self, ReviewOptions};
use crate::room_formats::RoomFormatOptions;
use crate::syntax::board::{parse_board, ParsedBoard};
use crate::syntax::board_edit::{self, BoardEdit, EditFailure, SourceChange};
use crate::syntax::circuit_edit::diff_circuit;
use crate::syntax::{parse_circuit, resolve_uses};
use crate::zone::fill_zones;
use runmat_design_syntax::edit::TextEdit;
use runmat_design_syntax::{parse_source, Dialect, FeaSyntaxDiagnostic};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const CIRCUIT_SUFFIX: &str = ".circuit";
pub const LAYOUT_SUFFIX: &str = ".circuit.layout";
pub const LOCK_SUFFIX: &str = ".circuit.lock";

/// Workspace path → file text: the files a source needs, as the caller read them.
pub type Files = BTreeMap<String, String>;

/// A problem in one file, at a UTF-16 range.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub path: String,
    pub start: usize,
    pub end: usize,
    /// `error`, `warning` or `info`.
    pub severity: String,
    pub message: String,
}

impl Diagnostic {
    fn error(path: &str, text: Option<&str>, start: usize, end: usize, message: String) -> Self {
        let (start, end) = match text {
            Some(t) => (utf16_offset(t, start), utf16_offset(t, end)),
            None => (0, 0),
        };
        Diagnostic {
            path: path.to_owned(),
            start,
            end,
            severity: "error".into(),
            message,
        }
    }

    fn syntax(path: &str, text: &str, d: &FeaSyntaxDiagnostic) -> Self {
        Self::error(
            path,
            Some(text),
            d.span.start,
            d.span.end,
            d.message.clone(),
        )
    }
}

/// The byte offset `at` of `text` in UTF-16 code units.
pub fn utf16_offset(text: &str, at: usize) -> usize {
    let at = at.min(text.len());
    let head = &text[..floor_char(text, at)];
    if head.is_ascii() {
        head.len()
    } else {
        head.encode_utf16().count()
    }
}

fn floor_char(text: &str, mut at: usize) -> usize {
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// One text edit in UTF-16 code units.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Utf16Edit {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// A changed file: its new text and the edits from the old text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChange {
    pub path: String,
    pub text: String,
    pub edits: Vec<Utf16Edit>,
    /// The minimal edits did not reproduce the model, so the file was reprinted whole.
    pub reprinted: bool,
}

fn file_change(path: &str, old: &str, change: SourceChange) -> FileChange {
    // Offsets are the old text's, in source order; an insertion comes before a removal that
    // starts at the same place, so applying them from the last to the first gives the new text.
    let mut edits: Vec<Utf16Edit> = change
        .edits
        .iter()
        .map(|e: &TextEdit| Utf16Edit {
            start: utf16_offset(old, e.span.start),
            end: utf16_offset(old, e.span.end),
            text: e.text.clone(),
        })
        .collect();
    edits.sort_by_key(|e| (e.start, e.end));
    FileChange {
        path: path.to_owned(),
        text: change.source,
        edits,
        reprinted: change.reprinted,
    }
}

/* ------------------------------------------------------------------------------------------------
 * Paths
 * --------------------------------------------------------------------------------------------- */

/// `from` resolved against the folder of `base` (a file path), with `.` and `..` removed.
pub fn resolve_path(base: &str, from: &str) -> String {
    let joined = if from.starts_with('/') {
        from.to_owned()
    } else {
        match base.rsplit_once('/') {
            Some((dir, _)) => format!("{dir}/{from}"),
            None => from.to_owned(),
        }
    };
    let absolute = joined.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for part in joined.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|p| *p != "..") {
                    parts.pop();
                } else if !absolute {
                    parts.push("..");
                }
            }
            p => parts.push(p),
        }
    }
    let body = parts.join("/");
    if absolute {
        format!("/{body}")
    } else {
        body
    }
}

pub fn is_layout(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(LAYOUT_SUFFIX)
}

/// `io.circuit` → `io.circuit.layout`.
pub fn layout_path_for(circuit: &str) -> String {
    format!("{}{LAYOUT_SUFFIX}", stem(circuit))
}

/// `io.circuit` → `io.circuit.lock`.
pub fn lock_path_for(circuit: &str) -> String {
    format!("{}{LOCK_SUFFIX}", stem(circuit))
}

fn stem(circuit: &str) -> &str {
    let lower = circuit.to_ascii_lowercase();
    if lower.ends_with(CIRCUIT_SUFFIX) {
        &circuit[..circuit.len() - CIRCUIT_SUFFIX.len()]
    } else {
        circuit
    }
}

/// The `use` lines at the head of a source, read without parsing the rest (a 10 MB layout's
/// head is a few lines).
fn head_uses(dialect: Dialect, text: &str) -> Vec<runmat_design_syntax::document::UseItem> {
    let mut end = 0;
    for line in text.split_inclusive('\n') {
        let t = line.trim_start();
        let first = t.split_whitespace().next().unwrap_or("");
        if !(t.is_empty()
            || t.starts_with("//")
            || matches!(first, "use" | "runmat" | "std" | "import"))
        {
            break;
        }
        end += line.len();
    }
    let syntax = parse_source(dialect, &text[..end]);
    syntax
        .file()
        .items()
        .into_iter()
        .filter_map(|item| match item {
            runmat_design_syntax::ast::Item::Use(u) => crate::syntax::board::use_item(&u).ok(),
            _ => None,
        })
        .collect()
}

/// The circuit file a layout lays out (its `use circuit … from` line), resolved.
fn layout_circuit(path: &str, uses: &[runmat_design_syntax::document::UseItem]) -> Option<String> {
    uses.iter()
        .find(|u| u.symbol.as_ref().is_some_and(|(kind, _)| kind == "circuit"))
        .map(|u| resolve_path(path, &u.from))
}

/// The files `path` needs, beyond itself: for a layout, its circuit and that circuit's lock;
/// for a circuit, the circuits its `use` lines name, its lock and its sibling layout. The
/// caller reads each that exists and asks again for each file it read, until nothing is new.
pub fn dependencies(path: &str, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    if is_layout(path) {
        if let Some(circuit) = layout_circuit(path, &head_uses(Dialect::CircuitLayout, text)) {
            out.push(lock_path_for(&circuit));
            out.push(circuit);
        }
    } else if path.to_ascii_lowercase().ends_with(CIRCUIT_SUFFIX) {
        for u in head_uses(Dialect::Circuit, text) {
            if u.symbol.as_ref().is_some_and(|(kind, _)| kind != "library") {
                out.push(resolve_path(path, &u.from));
            }
        }
        out.push(lock_path_for(path));
        out.push(layout_path_for(path));
    }
    out.sort();
    out.dedup();
    out.retain(|p| p != path);
    out
}

/* ------------------------------------------------------------------------------------------------
 * Loading a circuit and its board
 * --------------------------------------------------------------------------------------------- */

/// The part pool the runtime elaborates against: the built-in library. Pools from
/// `use library` lines are resolved by the package system, which does not exist yet.
fn pool() -> Pool {
    crate::fixture::standard_pool()
}

struct Layout {
    path: String,
    text: String,
    parsed: ParsedBoard,
}

struct Project {
    circuit_path: String,
    circuit_text: String,
    /// The file's own design, before `use` lines bring sheets in.
    own: Design,
    design: Design,
    pool: Pool,
    lock: LockSection,
    lock_text: Option<String>,
    netlist: Netlist,
    layout: Option<Layout>,
}

impl Project {
    fn schematic(&self) -> Schematic<'_> {
        Schematic {
            design: &self.design,
            netlist: &self.netlist,
            pool: &self.pool,
            lock: &self.lock,
        }
    }

    fn board(&self) -> Option<&Board> {
        self.layout.as_ref().and_then(|l| l.parsed.board.as_ref())
    }

    fn layout_or_err(&self) -> Result<&Layout, String> {
        self.layout.as_ref().ok_or_else(|| {
            format!(
                "{} has no board yet (no {} next to it)",
                self.circuit_path,
                layout_path_for(&self.circuit_path)
            )
        })
    }
}

/// Options for one compile.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompileOptions {
    /// The variant to elaborate (the design's default when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    /// The comparator settings of a running Update board, over the board's saved ones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eco: Option<EcoSettings>,
}

/// A run's comparator settings: the difference types it ignores and the parameters it never
/// compares. Absent fields keep the board's saved settings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EcoSettings {
    /// Difference type id (`rooms`) → ignored.
    #[serde(default)]
    pub ignore: BTreeMap<String, bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignored_parameters: Option<Vec<String>>,
}

impl EcoSettings {
    /// The run settings a command's values carry (`ignore_rooms`, `ignored_parameters`).
    pub fn from_values(values: &serde_json::Map<String, Value>) -> Self {
        let mut s = EcoSettings::default();
        for d in Difference::ALL {
            if let Some(Value::Bool(on)) = values.get(&d.input_id()) {
                s.ignore.insert(difference_id(d), *on);
            }
        }
        if let Some(Value::Array(items)) = values.get("ignored_parameters") {
            s.ignored_parameters = Some(
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(str::to_owned)
                    .collect(),
            );
        }
        s
    }

    /// The saved settings with this run's over them.
    pub fn over(&self, saved: &ComparatorOptions) -> ComparatorOptions {
        let mut o = saved.clone();
        for d in Difference::ALL {
            if let Some(on) = self.ignore.get(&difference_id(d)) {
                o.set_ignore(d, *on);
            }
        }
        if let Some(names) = &self.ignored_parameters {
            o.ignored_parameters = names.iter().cloned().collect();
        }
        o
    }
}

/// `ignore_rooms` → `rooms`.
fn difference_id(d: Difference) -> String {
    d.as_str().to_owned()
}

fn load(
    path: &str,
    text: &str,
    files: &Files,
    options: &CompileOptions,
) -> Result<Project, Vec<Diagnostic>> {
    let (circuit_path, layout) = if is_layout(path) {
        let parsed = parse_board(path, text);
        if !parsed.diagnostics.is_empty() {
            return Err(parsed
                .diagnostics
                .iter()
                .map(|d| Diagnostic::syntax(path, text, d))
                .collect());
        }
        let Some(circuit) = layout_circuit(path, &parsed.uses) else {
            return Err(vec![Diagnostic::error(
                path,
                Some(text),
                0,
                0,
                "a layout names its schematic with `use circuit <name> from \"<file>.circuit\"`"
                    .into(),
            )]);
        };
        (
            circuit,
            Some(Layout {
                path: path.to_owned(),
                text: text.to_owned(),
                parsed,
            }),
        )
    } else {
        let layout_path = layout_path_for(path);
        let layout = match files.get(&layout_path) {
            Some(t) => {
                let parsed = parse_board(&layout_path, t);
                if !parsed.diagnostics.is_empty() {
                    return Err(parsed
                        .diagnostics
                        .iter()
                        .map(|d| Diagnostic::syntax(&layout_path, t, d))
                        .collect());
                }
                Some(Layout {
                    path: layout_path,
                    text: t.clone(),
                    parsed,
                })
            }
            None => None,
        };
        (path.to_owned(), layout)
    };
    let circuit_text = if circuit_path == path {
        text.to_owned()
    } else {
        files.get(&circuit_path).cloned().ok_or_else(|| {
            vec![Diagnostic::error(
                path,
                Some(text),
                0,
                0,
                format!("{circuit_path} can't be read; the layout's `use circuit` line names it"),
            )]
        })?
    };
    let parsed = parse_circuit(&circuit_path, &circuit_text);
    let Some(own) = parsed.design else {
        return Err(parsed
            .diagnostics
            .iter()
            .map(|d| Diagnostic::syntax(&circuit_path, &circuit_text, d))
            .collect());
    };
    let mut design = own.clone();
    let mut used_errors: Vec<Diagnostic> = Vec::new();
    let loader = |from: &str| -> Option<Design> {
        let p = resolve_path(&circuit_path, from);
        let t = files.get(&p)?;
        parse_circuit(&p, t).design
    };
    if let Err(message) = resolve_uses(&mut design, &loader) {
        used_errors.push(Diagnostic::error(
            &circuit_path,
            Some(&circuit_text),
            0,
            0,
            message,
        ));
    }
    // A used file with errors is reported in that file.
    for u in &own.uses {
        let p = resolve_path(&circuit_path, &u.from);
        if let Some(t) = files.get(&p) {
            used_errors.extend(
                parse_circuit(&p, t)
                    .diagnostics
                    .iter()
                    .map(|d| Diagnostic::syntax(&p, t, d)),
            );
        }
    }
    if !used_errors.is_empty() {
        return Err(used_errors);
    }
    let lock_path = lock_path_for(&circuit_path);
    let lock_text = files.get(&lock_path).cloned();
    let mut lock = match &lock_text {
        Some(t) => LockSection::from_text(t)
            .map_err(|m| vec![Diagnostic::error(&lock_path, Some(t), 0, 0, m)])?,
        None => LockSection::default(),
    };
    let pool = pool();
    let netlist = elaborate(
        &design,
        &pool,
        &ElabOptions {
            variant: options.variant.clone(),
        },
        Some(&lock),
    );
    lock.reconcile(
        &netlist.lock_items(),
        &mut DerivedUuids {
            namespace: fnv128(design.name.as_bytes()),
        },
    );
    Ok(Project {
        circuit_path,
        circuit_text,
        own,
        design,
        pool,
        lock,
        lock_text,
        netlist,
        layout,
    })
}

/* ------------------------------------------------------------------------------------------------
 * Compile
 * --------------------------------------------------------------------------------------------- */

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompileOutput {
    /// The projection, when the files compile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<CircuitIndex>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Compiles the circuit `path` belongs to: the schematic's netlist, ERC and review, and the
/// board's DRC and both ECO directions when it has a board.
pub fn compile(path: &str, text: &str, files: &Files, options: &CompileOptions) -> CompileOutput {
    let project = match load(path, text, files, options) {
        Ok(p) => p,
        Err(diagnostics) => {
            return CompileOutput {
                index: None,
                diagnostics,
            }
        }
    };
    let mut erc = project.netlist.diagnostics.clone();
    erc.extend(erc::run(&project.netlist, &ErcOptions::default()));
    let findings = review::run_all(&project.netlist, &project.pool, &ReviewOptions::default());
    let mut index = build_index(
        &project.design,
        &project.netlist,
        &project.pool,
        &erc,
        &findings,
    );
    let mut diagnostics = Vec::new();
    if let Some(layout) = &project.layout {
        let mut board = layout
            .parsed
            .board
            .clone()
            .expect("a layout without errors has a board");
        let (mut rules, errors) = BoardRules::new(&board.limits, &board.rules);
        for e in errors {
            diagnostics.push(Diagnostic::error(
                &layout.path,
                Some(&layout.text),
                0,
                0,
                format!("{e:?}"),
            ));
        }
        rules.rebuild(&BoardGraph::new(&board, &[]));
        let fills = fill_zones(&mut board, &rules, &[]);
        let drc = Drc::run(&board, &rules, &[], &fills);
        add_board(&mut index, &board, &rules, &drc);
        let (forward, back) = plans(&project, &board, options.eco.as_ref());
        add_eco(&mut index, &forward, &back, &board);
    }
    CompileOutput {
        index: Some(index),
        diagnostics,
    }
}

fn plans(project: &Project, board: &Board, run: Option<&EcoSettings>) -> (EcoPlan, EcoPlan) {
    let s = project.schematic();
    let forward_options = run.map_or_else(
        || board.sync.options.clone(),
        |r| r.over(&board.sync.options),
    );
    (
        eco::compare(&s, board, &forward_options),
        eco::back_annotate(&s, board, &board.sync.options),
    )
}

/// Both ECO directions with a run's comparator settings, compared afresh (no DRC): what the
/// review shows while a run's ignore settings change.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReviewOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eco: Option<EcoIndex>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub back_annotation: Option<EcoIndex>,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn review(path: &str, text: &str, files: &Files, options: &CompileOptions) -> ReviewOutput {
    let project = match load(path, text, files, options) {
        Ok(p) => p,
        Err(diagnostics) => {
            return ReviewOutput {
                eco: None,
                back_annotation: None,
                diagnostics,
            }
        }
    };
    let Some(board) = project.board() else {
        return ReviewOutput {
            eco: None,
            back_annotation: None,
            diagnostics: Vec::new(),
        };
    };
    let (forward, back) = plans(&project, board, options.eco.as_ref());
    ReviewOutput {
        eco: Some(eco_index(&forward, board)),
        back_annotation: Some(eco_index(&back, board)),
        diagnostics: Vec::new(),
    }
}

/* ------------------------------------------------------------------------------------------------
 * Edits
 * --------------------------------------------------------------------------------------------- */

/// A catalog command's run, as the Desktop sends it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EditRequest {
    pub command: String,
    #[serde(default)]
    pub values: serde_json::Map<String, Value>,
    /// Outline node ids (`component:R1`, `track:17`, `room:ch[0]`).
    #[serde(default)]
    pub selection: Vec<String>,
    #[serde(default)]
    pub options: CompileOptions,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EditOutput {
    pub files: Vec<FileChange>,
    /// One line on what the edit did.
    pub summary: String,
}

/// Why an edit did nothing: the files don't compile, or the edit can't be made.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EditError {
    pub message: String,
    pub diagnostics: Vec<Diagnostic>,
}

impl EditError {
    fn new(message: impl Into<String>) -> Self {
        EditError {
            message: message.into(),
            diagnostics: Vec::new(),
        }
    }
}

impl From<String> for EditError {
    fn from(m: String) -> Self {
        EditError::new(m)
    }
}

fn failure(f: EditFailure) -> EditError {
    EditError::new(f.to_string())
}

/// Runs a catalog command on the circuit `path` belongs to, as one change per file.
pub fn edit(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Result<EditOutput, EditError> {
    let project = load(path, text, files, &request.options).map_err(|diagnostics| EditError {
        message: format!(
            "the circuit has {} error(s), so the edit wasn't made",
            diagnostics.len()
        ),
        diagnostics,
    })?;
    match request.command.as_str() {
        "schematic.update-board" => update_board(&project, request),
        "board.update-schematic" => update_schematic(&project, request),
        "board.copy-room-formats" => room_formats(&project, request),
        command => {
            let layout = project.layout_or_err()?;
            let board = layout.parsed.board.as_ref().expect("read");
            let edit = board_edit(board, &layout.parsed.sites, command, request)?;
            let change =
                board_edit::board_command(&layout.path, &layout.text, &edit).map_err(failure)?;
            Ok(EditOutput {
                summary: format!("{} on {}", edit.command(), layout.path),
                files: vec![file_change(&layout.path, &layout.text, change)],
            })
        }
    }
}

/// The ECO lines a run selected: its `changes` value holds Outline node ids
/// (`eco:<line>`, `eco-back:<line>`).
fn selected_lines(values: &serde_json::Map<String, Value>, prefix: &str) -> BTreeSet<String> {
    strings(values.get("changes"))
        .into_iter()
        .filter_map(|id| id.strip_prefix(prefix).map(str::to_owned))
        .collect()
}

fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn flag(values: &serde_json::Map<String, Value>, id: &str, default: bool) -> bool {
    values.get(id).and_then(Value::as_bool).unwrap_or(default)
}

fn number(values: &serde_json::Map<String, Value>, id: &str) -> Option<f64> {
    values.get(id).and_then(Value::as_f64)
}

/// All or nothing: the selected lines that fail validation, named.
fn refuse(plan: &EcoPlan, board: &Board, selected: &BTreeSet<String>) -> Result<(), EditError> {
    if selected.is_empty() {
        return Err(EditError::new("select at least one change to apply"));
    }
    let unknown: Vec<&String> = selected
        .iter()
        .filter(|id| !plan.changes.iter().any(|c| &c.id == *id))
        .collect();
    if !unknown.is_empty() {
        return Err(EditError::new(format!(
            "{} selected change(s) are no longer pending: {}",
            unknown.len(),
            unknown
                .iter()
                .take(3)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let checks = eco::validate(plan, board, selected);
    let failures: Vec<String> = plan
        .changes
        .iter()
        .filter(|c| checks.get(&c.id).is_some_and(|k| k.status == Status::Error))
        .map(|c| {
            format!(
                "{} ({})",
                c.item,
                checks[&c.id].reason().unwrap_or_default()
            )
        })
        .collect();
    if failures.is_empty() {
        return Ok(());
    }
    Err(EditError::new(format!(
        "{} {} validation, so nothing was applied: {}{}",
        failures.len(),
        if failures.len() == 1 {
            "change fails"
        } else {
            "changes fail"
        },
        failures
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("; "),
        if failures.len() > 3 { "…" } else { "" }
    )))
}

fn update_board(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let board = project.board().expect("read");
    let run = EcoSettings::from_values(&request.values);
    let (plan, _) = plans(project, board, Some(&run));
    let selected = selected_lines(&request.values, "eco:");
    refuse(&plan, board, &selected)?;
    let options = ApplyOptions {
        remove_orphaned_copper: flag(&request.values, "remove_orphaned_copper", false),
    };
    let (change, report) = board_edit::apply_eco(
        &layout.path,
        &layout.text,
        &plan,
        &selected,
        &project.pool,
        &options,
    )
    .map_err(failure)?;
    Ok(EditOutput {
        summary: format!(
            "applied {} change(s); removed {} track(s) and {} via(s); {} left without a net",
            report.applied.len(),
            report.removed_tracks.len(),
            report.removed_vias.len(),
            report.unnetted.len()
        ),
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

fn update_schematic(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let board = project.board().expect("read");
    let plan = eco::back_annotate(&project.schematic(), board, &board.sync.options);
    let selected = selected_lines(&request.values, "eco-back:");
    refuse(&plan, board, &selected)?;
    let mut design = project.design.clone();
    let mut lock = project.lock.clone();
    let (board_change, applied) = board_edit::edit_board(&layout.path, &layout.text, |b| {
        eco::apply_back(&mut design, &mut lock, b, &plan, &selected).map_err(|e| format!("{e:?}"))
    })
    .map_err(failure)?;
    // The file's own sheets take the change; a change in a used file is not written here.
    let mut own_new = project.own.clone();
    for (name, sheet) in own_new.sheets.iter_mut() {
        if let Some(changed) = design.sheets.get(name) {
            *sheet = changed.clone();
        }
    }
    for (name, sheet) in &design.sheets {
        if !project.own.sheets.contains_key(name) && project.design.sheets.get(name) != Some(sheet)
        {
            return Err(EditError::new(format!(
                "sheet `{name}` comes from a used file; back-annotate there"
            )));
        }
    }
    let mut files = vec![file_change(&layout.path, &layout.text, board_change)];
    if own_new != project.own {
        let change = diff_circuit(
            &project.circuit_path,
            &project.circuit_text,
            &project.own,
            &own_new,
        )?;
        files.push(file_change(
            &project.circuit_path,
            &project.circuit_text,
            change,
        ));
    }
    let lock_text = lock.to_text();
    if project.lock_text.as_deref() != Some(lock_text.as_str()) && lock != project.lock {
        let path = lock_path_for(&project.circuit_path);
        let old = project.lock_text.clone().unwrap_or_default();
        files.push(FileChange {
            edits: vec![Utf16Edit {
                start: 0,
                end: utf16_offset(&old, old.len()),
                text: lock_text.clone(),
            }],
            path,
            text: lock_text,
            reprinted: true,
        });
    }
    Ok(EditOutput {
        summary: format!("back-annotated {} change(s)", applied.len()),
        files,
    })
}

/// `room:ch[0]` → `ch[0]`; a bare room name stays.
fn room_name(id: &str) -> String {
    id.strip_prefix("room:").unwrap_or(id).to_owned()
}

fn room_formats(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let layout = project.layout_or_err()?;
    let source = strings(request.values.get("source"))
        .into_iter()
        .chain(
            request
                .selection
                .iter()
                .filter(|s| s.starts_with("room:"))
                .cloned(),
        )
        .next()
        .map(|s| room_name(&s))
        .ok_or_else(|| EditError::new("pick the room to copy from"))?;
    let targets: Vec<String> = strings(request.values.get("targets"))
        .iter()
        .map(|s| room_name(s))
        .filter(|t| t != &source)
        .collect();
    let options = RoomFormatOptions {
        placement: flag(&request.values, "placement", true),
        routing: flag(&request.values, "routing", true),
        room_shape: flag(&request.values, "room_shape", true),
    };
    let (change, report) =
        board_edit::copy_room_formats(&layout.path, &layout.text, &source, &targets, &options)
            .map_err(failure)?;
    let placed: usize = report.copies.iter().map(|c| c.placed.len()).sum();
    let tracks: usize = report.copies.iter().map(|c| c.tracks_added.len()).sum();
    Ok(EditOutput {
        summary: format!(
            "copied {source} into {} room(s): {placed} part(s) placed, {tracks} track(s) copied",
            report.copies.len()
        ),
        files: vec![file_change(&layout.path, &layout.text, change)],
    })
}

/// An Outline node id or designator → the site or designator `apply_board_edit` resolves.
fn site_of(board: &Board, sites: &BTreeMap<ItemId, String>, id: &str) -> Result<String, EditError> {
    let (kind, rest) = id.split_once(':').unwrap_or(("", id));
    match kind {
        "component" | "footprint" | "" => {
            if board.footprint_by_ref(rest).is_some() {
                Ok(rest.to_owned())
            } else {
                Err(EditError::new(format!("{rest} is not on the board")))
            }
        }
        "track" | "via" | "pour" | "zone" | "fiducial" | "text" | "keepout" | "dimension" => {
            let n: u64 = rest
                .parse()
                .map_err(|_| EditError::new(format!("`{id}` is not a board item")))?;
            sites
                .get(&ItemId(n))
                .cloned()
                .ok_or_else(|| EditError::new(format!("`{id}` is not on the board")))
        }
        _ => Err(EditError::new(format!("`{id}` is not a board item"))),
    }
}

const MM: f64 = 1_000_000.0;

fn mm_value(values: &serde_json::Map<String, Value>, id: &str) -> Option<crate::geom::Coord> {
    number(values, id).map(|v| (v * MM).round() as crate::geom::Coord)
}

/// The model edit a board catalog command makes, from its values and the selection.
fn board_edit(
    board: &Board,
    sites: &BTreeMap<ItemId, String>,
    command: &str,
    request: &EditRequest,
) -> Result<BoardEdit, EditError> {
    let values = &request.values;
    let items = |field: &str| -> Result<Vec<String>, EditError> {
        let picked = strings(values.get(field));
        let ids = if picked.is_empty() {
            request.selection.clone()
        } else {
            picked
        };
        if ids.is_empty() {
            return Err(EditError::new("select the items first"));
        }
        ids.iter().map(|id| site_of(board, sites, id)).collect()
    };
    match command {
        "board.move" => Ok(BoardEdit::Move {
            items: items("items")?,
            dx: mm_value(values, "dx").unwrap_or(0),
            dy: mm_value(values, "dy").unwrap_or(0),
            rotation: number(values, "rotation").unwrap_or(0.0),
        }),
        "board.lock" => {
            let items = items("items")?;
            // Lock toggles: all locked unlocks, otherwise everything locks.
            let all_locked = items.iter().all(|s| {
                board.footprint_by_ref(s).is_some_and(|f| f.locked)
                    || sites.iter().find(|(_, x)| *x == s).is_some_and(|(id, _)| {
                        board.tracks.get(id).is_some_and(|t| t.locked)
                            || board.vias.get(id).is_some_and(|v| v.locked)
                            || board.zones.get(id).is_some_and(|z| z.locked)
                    })
            });
            Ok(BoardEdit::Lock {
                items,
                locked: !all_locked,
            })
        }
        "board.delete" | "edit.delete" => Ok(BoardEdit::Delete {
            items: items("items")?,
        }),
        "board.layer-stack" => Ok(BoardEdit::LayerStack {
            layers: number(values, "layers").map_or(board.copper_count(), |n| n as u8),
            thickness: mm_value(values, "thickness").unwrap_or_else(|| board.stackup.thickness()),
        }),
        "board.test-points" => {
            let nets = strings(values.get("nets"));
            let net = nets
                .first()
                .map(|n| n.strip_prefix("net:").unwrap_or(n).to_owned())
                .ok_or_else(|| EditError::new("pick the net for the test point"))?;
            if nets.len() > 1 {
                return Err(EditError::new("test points are placed one net at a time"));
            }
            let at = board
                .outline
                .outer
                .iter()
                .fold(None::<crate::geom::Point>, |acc, p| {
                    Some(acc.map_or(*p, |a| crate::geom::Point::new(a.x.min(p.x), a.y.min(p.y))))
                })
                .ok_or_else(|| {
                    EditError::new("the board has no outline to place a test point in")
                })?;
            Ok(BoardEdit::TestPoint {
                net,
                at: crate::geom::Point::new(
                    at.x + crate::geom::mm(2.0),
                    at.y + crate::geom::mm(2.0),
                ),
                side: if values.get("side").and_then(Value::as_str) == Some("top") {
                    Side::Top
                } else {
                    Side::Bottom
                },
                diameter: mm_value(values, "diameter").unwrap_or(crate::geom::mm(1.0)),
            })
        }
        "board.route" | "board.place-via" | "board.pour" | "board.keepout" | "board.room"
        | "board.dimension" | "board.outline" | "board.fill" => Err(EditError::new(format!(
            "{command} takes its points from the board view; draw it there"
        ))),
        other => Err(EditError::new(format!(
            "{other} isn't a model edit the circuit runtime runs"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_resolve_against_the_file_folder() {
        assert_eq!(
            resolve_path("/p/boards/io.circuit.layout", "./io.circuit"),
            "/p/boards/io.circuit"
        );
        assert_eq!(
            resolve_path("/p/boards/io.circuit.layout", "../sch/io.circuit"),
            "/p/sch/io.circuit"
        );
        assert_eq!(
            resolve_path("io.circuit.layout", "io.circuit"),
            "io.circuit"
        );
        assert_eq!(layout_path_for("/p/io.circuit"), "/p/io.circuit.layout");
        assert_eq!(lock_path_for("/p/io.circuit"), "/p/io.circuit.lock");
    }

    #[test]
    fn utf16_offsets_count_code_units() {
        let t = "a±b";
        assert_eq!(utf16_offset(t, 0), 0);
        assert_eq!(utf16_offset(t, 3), 2);
        assert_eq!(utf16_offset(t, t.len()), 3);
    }
}
