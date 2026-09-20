use runmat_analysis_language::{
    FeaDiagnostic, FeaDiagnosticSeverity, LogicalSourcePath, SourceAvailability,
};
use serde::{Deserialize, Serialize};

use super::CheckedFeaSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaCheckStatus {
    Valid,
    Invalid,
    Incomplete,
}

impl FeaCheckStatus {
    pub const fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }
}

impl std::fmt::Display for FeaCheckStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Valid => "valid",
            Self::Invalid => "invalid",
            Self::Incomplete => "incomplete",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeaRootSelector {
    Study { id: String },
    Sweep { id: String },
}

impl FeaRootSelector {
    pub fn id(&self) -> &str {
        match self {
            Self::Study { id } | Self::Sweep { id } => id,
        }
    }

    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Study { .. } => "study",
            Self::Sweep { .. } => "sweep",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSourceCheckResult {
    pub status: FeaCheckStatus,
    pub root_module: LogicalSourcePath,
    pub roots: Vec<FeaRootSelector>,
    pub diagnostics: Vec<FeaDiagnostic>,
}

impl CheckedFeaSource {
    pub fn check_result(&self) -> FeaSourceCheckResult {
        let diagnostics = self.diagnostics().cloned().collect::<Vec<_>>();
        let status = check_status(&diagnostics);
        FeaSourceCheckResult {
            status,
            root_module: self.root_module.clone(),
            roots: self.root_selectors(),
            diagnostics,
        }
    }

    pub fn diagnostics(&self) -> impl Iterator<Item = &FeaDiagnostic> {
        self.program
            .bound
            .graph
            .diagnostics
            .iter()
            .chain(&self.program.bound.diagnostics)
            .chain(&self.program.diagnostics)
    }

    pub fn root_selectors(&self) -> Vec<FeaRootSelector> {
        let module = self.root_module.as_str();
        let studies = self
            .program
            .studies
            .values()
            .filter(|study| study.module_path == module)
            .map(|study| FeaRootSelector::Study {
                id: study.study_id.clone(),
            });
        let sweeps = self
            .program
            .sweeps
            .values()
            .filter(|sweep| sweep.module_path == module)
            .map(|sweep| FeaRootSelector::Sweep {
                id: sweep.sweep_id.clone(),
            });
        studies.chain(sweeps).collect()
    }
}

fn check_status(diagnostics: &[FeaDiagnostic]) -> FeaCheckStatus {
    let mut unavailable_error = false;
    for diagnostic in diagnostics {
        if diagnostic.severity != FeaDiagnosticSeverity::Error {
            continue;
        }
        match diagnostic.availability {
            SourceAvailability::Available => return FeaCheckStatus::Invalid,
            SourceAvailability::Unavailable => unavailable_error = true,
        }
    }
    if unavailable_error {
        FeaCheckStatus::Incomplete
    } else {
        FeaCheckStatus::Valid
    }
}
