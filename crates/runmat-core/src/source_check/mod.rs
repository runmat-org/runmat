//! Ordinary script/function/project checks, using the same project frontend as
//! `runmat check`. No execution, JIT, collector or foreign runtime is started.
mod host;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use host::{check_source, check_source_with_runtime_settings};
use runmat_config::runtime::{ConfigurationPolicy, LanguageCompatMode};
use runmat_static_analysis::{
    frontend::{AnalysisDomains, ResolutionEvidence},
    project::check_report::CheckDiagnostic,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum CheckTarget {
    Script {
        path: PathBuf,
    },
    /// `path` names the actual source file; `function` is its semantic function
    /// name, not executable source or an expression to evaluate.
    Function {
        path: PathBuf,
        function: String,
    },
    /// A manifest/directory inside the selected project. Check every indexed
    /// root-package .m source, including functions not reached by a script.
    Project {
        path: PathBuf,
    },
}
impl CheckTarget {
    pub(super) fn path(&self) -> &std::path::Path {
        match self {
            Self::Script { path } | Self::Function { path, .. } | Self::Project { path } => path,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceCheckRequest {
    pub target: CheckTarget,
    /// Host-owned user layer only. Project settings are read from the selected
    /// project's real manifest through the same captured filesystem authority.
    #[serde(default)]
    pub user_settings_json: Option<String>,
    #[serde(default)]
    pub policy: ConfigurationPolicy,
    #[serde(default)]
    pub deny_warnings: bool,
    #[serde(default)]
    pub search_paths: Vec<PathBuf>,
    /// Existing immutable/versioned filesystem contract, not a second snapshot.
    #[serde(default)]
    pub input_manifest: Option<runmat_filesystem::pinned::InputManifest>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckOutcome {
    Clean,
    Warnings,
    Failed,
}

#[derive(Debug, Serialize)]
pub struct CheckSummary {
    pub errors: usize,
    pub warnings: usize,
    pub warnings_denied: bool,
}

#[derive(Debug, Serialize)]
pub struct CheckedSource {
    pub source_id: usize,
    pub path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct CheckedDocument {
    pub path: PathBuf,
    pub function: Option<String>,
    pub outcome: CheckOutcome,
    pub diagnostics: Vec<CheckDiagnostic>,
    /// Source IDs in diagnostics/resolution are local to this source table.
    pub sources: Vec<CheckedSource>,
    pub analysis: AnalysisDomains,
    pub resolution: Vec<ResolutionEvidence>,
    pub analysis_incomplete_sources: Vec<PathBuf>,
    pub call_context_budget_exhausted: bool,
    pub call_context_precision_withheld: bool,
    pub summary: CheckSummary,
}

#[derive(Debug, Serialize)]
pub struct SourceCheckReport {
    pub schema_version: u16,
    pub document_kind: &'static str,
    pub path: PathBuf,
    pub outcome: CheckOutcome,
    pub language_compatibility: LanguageCompatMode,
    pub project_revision: Option<runmat_package::ProjectRevision>,
    /// Whole-project checks contain one canonical reached-closure analysis per
    /// indexed root. A shared dependency diagnostic can occur in several roots;
    /// don't combine source ordinals across these independent tables.
    pub documents: Vec<CheckedDocument>,
    pub summary: CheckSummary,
}

fn outcome(errors: usize, warnings: usize, deny: bool) -> CheckOutcome {
    if errors > 0 || (deny && warnings > 0) {
        CheckOutcome::Failed
    } else if warnings > 0 {
        CheckOutcome::Warnings
    } else {
        CheckOutcome::Clean
    }
}
