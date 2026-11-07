//! `design.overview`: what a design file is, where it stands, what it links to, what is still
//! open on it and what to do next, each fact from a check (doc 18 §9).
//!
//! The overview generalizes FEA guided setup's Setup overview to every design file. It runs the
//! checks a person would run and reports only what they return:
//! - [`check`](crate::check) regenerates the file (features, inputs, occurrences, views);
//! - [`erc`](crate::rules::erc) and [`drc`](crate::rules::drc) give a circuit's and a board's
//!   rule findings;
//! - [`inspect`](crate::inspect) gives board sync (changes the board hasn't taken, unplaced
//!   parts);
//! - the circuit compiler gives the design's simulation sections;
//! - [`guides::status`](crate::guides::status) gives the decisions each fitting guide still
//!   needs.
//!
//! Links go both ways. A file links to what it uses ([`dependencies`](crate::dependencies)) and
//! to the project files that use it: a board uses its schematic, a drawing and an assembly use
//! their parts, and a study (`.fea`) uses the file its `geometry { file = … }` names. The caller
//! reads those files first; [`related_candidates`] says which.
//!
//! Every [`Fact`] names the check it came from (`source`), so a client, a test or a person can
//! trace each line on the overview to the tool call that produced it.

mod facts;
mod links;
mod steps;

use crate::guides::{self, GuideCatalog};
use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use links::{related_candidates, study_geometry_files, MAX_RELATED, NOT_CHECKED_HERE, check_linked_models};

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OverviewInput {
    /// Workspace path of the design file.
    pub path: String,
}

/// How a fact or a link stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    /// The check passed.
    Ok,
    /// Something needs attention but nothing fails.
    Warning,
    /// The check failed.
    Error,
    /// A count or a description; nothing passes or fails.
    Info,
}

/// One line of the file's state, from one check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Fact {
    /// Stable id: `regeneration`, `erc`, `drc`, `board_sync`, `placement`, `simulation`, `views`.
    pub id: String,
    /// What it is about, as the overview labels it (`Electrical rules`).
    pub label: String,
    /// What the check found, in one line (`No errors, 4 warnings`).
    pub value: String,
    pub level: Level,
    /// The tool that produced it: `design.check`, `design.erc`, `design.drc`, `design.inspect`,
    /// `circuit.simulations`.
    pub source: String,
}

/// How a linked file relates to the overview's file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    /// The board laid out from this schematic.
    Board,
    /// The schematic this board is laid out from.
    Schematic,
    /// A `.fea` study whose geometry is this file (or its board).
    Study,
    /// A drawing whose views show this file.
    Drawing,
    /// An assembly that places this file.
    Assembly,
    /// A part or assembly this file uses (an assembly's parts, a drawing's model).
    Uses,
}

/// A related file and how it stands.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Link {
    pub path: String,
    pub relation: Relation,
    /// The link it was reached through (a schematic's thermal study is linked through its board).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub via: Option<String>,
    /// How the linked file stands, in one line (`No errors; 1 study: bracket_static`).
    pub status: String,
    pub level: Level,
    /// The check that produced `status`: `design.check` for design files, `fea.syntax` for a
    /// study (its syntax; the study's own check is `runmat check`).
    pub source: String,
}

/// A decision a fitting guide still needs, from its completion check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OpenDecision {
    pub guide: String,
    pub guide_title: String,
    pub decision: String,
    /// The question as the guide's card asks it.
    pub question: String,
}

/// What a next step does when chosen.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StepAction {
    /// Start a guide (an id in the guide catalog).
    Guide { id: String },
    /// Run a catalog command (an id in `design-commands.json`).
    Command { id: String },
}

/// A next step that fits the file's state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NextStep {
    /// What it does, as a button reads (`Update the board from the schematic`).
    pub label: String,
    /// The fact or link that makes it fit (`board_sync`), so the step can be traced to a check.
    pub because: String,
    pub action: StepAction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OverviewOutput {
    pub path: String,
    pub kind: DesignKind,
    /// What the file is, in one line, from the check's counts (`Schematic: 63 components, 48
    /// nets on 6 sheets`).
    pub about: String,
    pub state: Vec<Fact>,
    pub links: Vec<Link>,
    pub open_decisions: Vec<OpenDecision>,
    /// Two or three steps, most pressing first.
    pub next_steps: Vec<NextStep>,
    /// The checks that ran, in order.
    pub checks: Vec<String>,
}

pub fn overview(input: &OverviewInput, files: &Files) -> Result<OverviewOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    text_of(files, &input.path)?;
    let path = input.path.as_str();
    let mut ran = Ran::default();
    let (about, state) = facts::state(kind, path, files, &mut ran)?;
    let links = links::links(kind, path, files, &mut ran);
    let catalog = guides::catalog();
    let open_decisions = open_decisions(kind, path, files, &catalog, &state, &mut ran);
    let next_steps = steps::next_steps(kind, &state, &links, &open_decisions, &catalog);
    Ok(OverviewOutput {
        path: path.to_owned(),
        kind,
        about,
        state,
        links,
        open_decisions,
        next_steps,
        checks: ran.0,
    })
}

/// The checks that ran, each once, in order.
#[derive(Default)]
pub(crate) struct Ran(Vec<String>);

impl Ran {
    pub(crate) fn note(&mut self, tool: &str) {
        if !self.0.iter().any(|t| t == tool) {
            self.0.push(tool.to_owned());
        }
    }
}

/// The guides whose completion check reads this kind of file: those that write it, and those
/// that open from it. A circuit with a board also gets the board's guides, since the board's
/// state is part of the circuit's overview.
fn fitting_guides<'a>(
    kind: DesignKind,
    has_board: bool,
    catalog: &'a GuideCatalog,
) -> Vec<&'a guides::Guide> {
    let ext = |k: DesignKind| match k {
        DesignKind::Part => ".part",
        DesignKind::Assembly => ".assembly",
        DesignKind::Drawing => ".drawing",
        DesignKind::Circuit => ".circuit",
        DesignKind::Layout => ".circuit.layout",
    };
    catalog
        .guides
        .iter()
        .filter(|g| g.completion.tool == "design.check")
        .filter(|g| {
            g.extension == ext(kind)
                || (kind == DesignKind::Circuit && has_board && g.extension == ".circuit.layout")
                || g.opens_from.iter().any(|e| matches!(e, guides::EntryPoint::OpenFile { extensions } if extensions.iter().any(|x| x == ext(kind))))
        })
        .collect()
}

fn open_decisions(
    kind: DesignKind,
    path: &str,
    files: &Files,
    catalog: &GuideCatalog,
    state: &[Fact],
    ran: &mut Ran,
) -> Vec<OpenDecision> {
    let has_board = state.iter().any(|f| f.id == "board_sync");
    let mut out = Vec::new();
    for guide in fitting_guides(kind, has_board, catalog) {
        let Ok(status) = guides::status(guide, path, files) else {
            continue;
        };
        ran.note("guides.status");
        for id in &status.open {
            let Some(d) = guide.decisions.iter().find(|d| &d.id == id) else {
                continue;
            };
            if out.iter().any(|o: &OpenDecision| o.question == d.question) {
                continue;
            }
            out.push(OpenDecision {
                guide: guide.id.clone(),
                guide_title: guide.title.clone(),
                decision: d.id.clone(),
                question: d.question.clone(),
            });
        }
    }
    out
}

/// `n thing` or `n things`.
pub(crate) fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The file name of a workspace path.
pub(crate) fn name_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}
