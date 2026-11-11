use runmat_analysis_language::{FeaDiagnostic, FeaDiagnosticSeverity, SourceAvailability};
use serde::{Deserialize, Serialize};

use super::CheckedFeaSource;

pub use runmat_fea_source::{FeaCheckStatus, FeaSourceCheckResult};

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

impl CheckedFeaSource {
    pub fn check_result(&self) -> FeaSourceCheckResult {
        let diagnostics = self.diagnostics().cloned().collect::<Vec<_>>();
        let status = check_status(&diagnostics);
        let program = self.semantic_program();
        let functions = self.program.bound.functions.values().collect::<Vec<_>>();
        let geometry = self
            .program
            .geometry_inspections
            .values()
            .map(|inspection| {
                (
                    &inspection.module,
                    &inspection.geometry,
                    &inspection.source_path,
                    &inspection.facts,
                )
            })
            .collect::<Vec<_>>();
        let revision = super::source_digest(
            &serde_json::to_vec(&(
                "runmat-fea-checked-v1",
                &self.root_module,
                self.program.bound.graph.catalog_fingerprint,
                self.program.bound.graph.source_digests(),
                functions,
                geometry,
                &self.execution_profile_facts,
                &program,
                &diagnostics,
            ))
            .expect("checked source facts are serializable"),
        );
        FeaSourceCheckResult {
            contract: runmat_fea_source::FeaSourceCheckContract::AssessmentsV1,
            revision,
            status,
            root_module: self.root_module.clone(),
            program,
            source_index: self.source_index(),
            geometry_previews: self
                .program
                .geometry_inspections
                .values()
                .map(|inspection| runmat_fea_source::FeaGeometryPreview {
                    module_path: inspection.module.clone(),
                    geometry_id: inspection.geometry.clone(),
                    source_path: inspection.source_path.clone(),
                })
                .collect(),
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
        // A sweep owns execution of its referenced study. Keep that study
        // available to explicit selectors, but do not offer it as a second
        // independent default root beside the sweep.
        let swept_studies: std::collections::BTreeSet<_> = self
            .program
            .sweeps
            .values()
            .filter(|sweep| sweep.module_path == module)
            .map(|sweep| (&sweep.study.module_path, &sweep.study.study_id))
            .collect();
        let studies = self
            .program
            .studies
            .values()
            .filter(|study| study.module_path == module)
            .filter(|study| !swept_studies.contains(&(&study.module_path, &study.study_id)))
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
