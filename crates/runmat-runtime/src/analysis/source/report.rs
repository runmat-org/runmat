use runmat_analysis_core::{AnalysisProcedure, AuthoredModel, PhysicsFamily};
use runmat_analysis_language::{
    FeaDiagnostic, FeaDiagnosticSeverity, LogicalSourcePath, SourceAvailability,
};
use runmat_geometry_core::{GeometryDefinitionSource, UnitSystem};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
    pub studies: Vec<FeaStudyCheckSummary>,
    pub diagnostics: Vec<FeaDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaStudyProcedureSummary {
    pub step_id: String,
    pub physics_family: PhysicsFamily,
    pub analysis_procedure: AnalysisProcedure,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
/// Counts the authored objects available to a study without exposing the full lowered model.
/// Model-owned counts include transitive model includes; `materials` counts declarations in the
/// checked module graph, and `signals` includes both direct and parameterized signal declarations.
pub struct FeaStudyObjectCounts {
    pub materials: usize,
    pub material_assignments: usize,
    pub constraints: usize,
    pub boundaries: usize,
    pub connectors: usize,
    pub interfaces: usize,
    pub signals: usize,
    pub excitation_cases: usize,
    pub excitations: usize,
    pub initial_conditions: usize,
    pub response_spectra: usize,
    pub sequence_steps: usize,
    pub outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaStudyCheckSummary {
    pub id: String,
    pub geometry_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry_units: Option<UnitSystem>,
    pub mesh_id: String,
    pub model_id: String,
    pub procedures: Vec<FeaStudyProcedureSummary>,
    pub counts: FeaStudyObjectCounts,
}

impl CheckedFeaSource {
    pub fn check_result(&self) -> FeaSourceCheckResult {
        let diagnostics = self.diagnostics().cloned().collect::<Vec<_>>();
        let status = check_status(&diagnostics);
        FeaSourceCheckResult {
            status,
            root_module: self.root_module.clone(),
            roots: self.root_selectors(),
            studies: self.study_summaries(),
            diagnostics,
        }
    }

    fn study_summaries(&self) -> Vec<FeaStudyCheckSummary> {
        let root_module = self.root_module.as_str();
        self.program
            .studies
            .values()
            .filter(|study| study.module_path == root_module)
            .map(|study| {
                let geometry = self.program.geometries.values().find(|geometry| {
                    geometry.module_path == study.geometry.definition.module_path
                        && geometry.geometry_id == study.geometry.definition.geometry_id
                });
                let (geometry_path, geometry_units) = match geometry.map(|value| &value.source) {
                    Some(GeometryDefinitionSource::Imported(source)) => {
                        (Some(source.logical_path.clone()), Some(source.units))
                    }
                    _ => (None, None),
                };
                let models = study_models(study, self.program.models.values());
                FeaStudyCheckSummary {
                    id: study.study_id.clone(),
                    geometry_id: study.geometry.definition.geometry_id.clone(),
                    geometry_path,
                    geometry_units,
                    mesh_id: study.mesh.mesh_id.clone(),
                    model_id: study.model.model.model_id.clone(),
                    procedures: study
                        .sequence
                        .steps
                        .iter()
                        .map(|step| FeaStudyProcedureSummary {
                            step_id: step.step_id.0.clone(),
                            physics_family: step.family,
                            analysis_procedure: step.procedure,
                        })
                        .collect(),
                    counts: count_study_objects(study, &models, self.program.materials.len()),
                }
            })
            .collect()
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

fn study_models<'a>(
    study: &runmat_analysis_core::AuthoredStudy,
    models: impl Iterator<Item = &'a AuthoredModel>,
) -> Vec<&'a AuthoredModel> {
    let catalog = models.collect::<Vec<_>>();
    let mut pending = vec![study.model.model.clone()];
    let mut visited = BTreeSet::new();
    let mut selected = Vec::new();
    while let Some(reference) = pending.pop() {
        if !visited.insert(reference.clone()) {
            continue;
        }
        let Some(model) = catalog
            .iter()
            .copied()
            .find(|model| model.reference() == reference)
        else {
            continue;
        };
        pending.extend(model.includes.iter().map(|include| include.model.clone()));
        selected.push(model);
    }
    selected
}

fn count_study_objects(
    study: &runmat_analysis_core::AuthoredStudy,
    models: &[&AuthoredModel],
    material_declarations: usize,
) -> FeaStudyObjectCounts {
    FeaStudyObjectCounts {
        materials: material_declarations,
        material_assignments: models
            .iter()
            .map(|model| model.material_assignments.len())
            .sum(),
        constraints: models.iter().map(|model| model.constraints.len()).sum(),
        boundaries: models.iter().map(|model| model.boundaries.len()).sum(),
        connectors: models.iter().map(|model| model.connectors.len()).sum(),
        interfaces: models.iter().map(|model| model.interfaces.len()).sum(),
        signals: models
            .iter()
            .map(|model| model.signals.len() + model.parameterized_signals.len())
            .sum(),
        excitation_cases: models
            .iter()
            .map(|model| model.excitation_cases.len())
            .sum(),
        excitations: models.iter().map(|model| model.excitations.len()).sum(),
        initial_conditions: models
            .iter()
            .map(|model| model.initial_conditions.len())
            .sum(),
        response_spectra: models
            .iter()
            .map(|model| model.response_spectra.len())
            .sum(),
        sequence_steps: study.sequence.steps.len(),
        outputs: study.outputs.len(),
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
