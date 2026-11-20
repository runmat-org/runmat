//! `design.overview`: what a design file is, where it stands and what it links to, each fact
//! from a check (doc 18 §9).
//!
//! The overview generalizes FEA guided setup's Setup overview to every design file. It runs the
//! checks a person would run and reports only what they return:
//! - [`check`](crate::check) regenerates the file (features, inputs, occurrences, views);
//! - [`erc`](crate::rules::erc) and [`drc`](crate::rules::drc) give a circuit's and a board's
//!   rule findings;
//! - [`inspect`](crate::inspect) gives board sync (changes the board hasn't taken, unplaced
//!   parts);
//! - the circuit compiler gives the design's simulation sections.
//!
//! [`OverviewOutput::counts`] carries the check's counts (and, for a circuit or a board, the
//! inspection's totals), so a caller can compare them with its own requirements without
//! regenerating the file again.
//!
//! Links go both ways. A file links to what it uses ([`dependencies`](crate::dependencies)) and
//! to the project files that use it: a board uses its schematic, a drawing and an assembly use
//! their parts, and a study (`.fea`) uses the file its `geometry { file = … }` names. The caller
//! reads those files first; [`related_candidates`] says which.
//!
//! Every [`Fact`] names the check it came from (`source`), so a client, a test or a person can
//! trace each line on the overview to the tool call that produced it.

mod facts;
pub mod layout;
mod links;

use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use runmat_cad::index::FileState;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub use links::{
    check_linked_models, related_candidates, study_geometry_files, MAX_RELATED, NOT_CHECKED_HERE,
};

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

/// The counts the file's checks returned, for callers that compare them with requirements.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CheckCounts {
    /// The file regenerated with no errors.
    pub ok: bool,
    /// The check's summary line.
    pub summary: String,
    /// `design.check`'s counts and, for a circuit or a board, `design.inspect`'s totals (keys as
    /// those tools name them).
    pub totals: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OverviewOutput {
    pub path: String,
    pub kind: DesignKind,
    /// What the file holds (`design.check`'s `fileState`): empty or written, and for a part its
    /// kind. Guides and next steps that don't fit it aren't offered.
    pub file_state: FileState,
    /// What the file is, in one line, from the check's counts (`Schematic: 63 components, 48
    /// nets on 6 sheets`).
    pub about: String,
    pub state: Vec<Fact>,
    pub links: Vec<Link>,
    pub counts: CheckCounts,
    /// The checks that ran, in order.
    pub checks: Vec<String>,
    /// The project's folders and libraries, and where new files go.
    pub project: layout::ProjectLayout,
}

pub fn overview(input: &OverviewInput, files: &Files) -> Result<OverviewOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    text_of(files, &input.path)?;
    let path = input.path.as_str();
    let mut ran = Ran::default();
    let checked = facts::state(kind, path, files, &mut ran)?;
    let links = links::links(kind, path, files, &mut ran);
    Ok(OverviewOutput {
        path: path.to_owned(),
        kind,
        file_state: checked.file_state,
        about: checked.about,
        state: checked.facts,
        links,
        counts: checked.counts,
        checks: ran.0,
        project: layout::layout(path, files),
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

/// `n thing` or `n things`.
pub fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The file name of a workspace path.
pub fn name_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}
