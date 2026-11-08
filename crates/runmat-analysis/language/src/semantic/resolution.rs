use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AuthoredExecutionPolicy, AuthoredExecutionPolicyRef, AuthoredMaterialRef, AuthoredModel,
    AuthoredModelRef, AuthoredStudy, AuthoredSweep, MaterialModel,
};
use runmat_geometry_core::GeometryDefinition;
use runmat_meshing_core::MeshDefinition;

use crate::{
    CheckedFeaProgram, FeaDiagnosticSeverity, FeaSymbolKey, LogicalSourcePath, SourceAvailability,
};

/// A runnable study root detached from compiler-internal syntax and symbol tables.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAnalysisStudy {
    pub study: AuthoredStudy,
    pub geometry: GeometryDefinition,
    /// Transitive geometry sources in dependency-first order. The selected geometry is not
    /// repeated. Imported, constructed, and generated roots have an empty closure.
    pub geometry_dependencies: Vec<GeometryDefinition>,
    pub mesh: MeshDefinition,
    pub models: BTreeMap<AuthoredModelRef, AuthoredModel>,
    /// Candidate declarations available to material-valued study inputs and sweep cases.
    /// Realization retains only materials actually assigned by the bound model.
    pub material_catalog: BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub execution_policy: Option<AuthoredExecutionPolicy>,
}

/// A runnable sweep plus the exact study closure each expanded case realizes.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAnalysisSweep {
    pub sweep: AuthoredSweep,
    pub study: ResolvedAnalysisStudy,
}

impl CheckedFeaProgram {
    pub fn resolve_study(
        &self,
        module_path: &str,
        study_id: &str,
    ) -> Result<ResolvedAnalysisStudy, AnalysisRootResolutionError> {
        self.require_complete_diagnostics()?;
        let (key, study) = find_study(self, module_path, study_id)?;
        self.resolve_study_from(key, study, None)
    }

    pub fn resolve_sweep(
        &self,
        module_path: &str,
        sweep_id: &str,
    ) -> Result<ResolvedAnalysisSweep, AnalysisRootResolutionError> {
        self.require_complete_diagnostics()?;
        let (sweep_key, sweep) = self
            .sweeps
            .iter()
            .find(|(_, sweep)| sweep.module_path == module_path && sweep.sweep_id == sweep_id)
            .ok_or_else(|| AnalysisRootResolutionError::MissingSweep {
                module_path: module_path.to_string(),
                sweep_id: sweep_id.to_string(),
            })?;
        let (study_key, study) = find_study(self, &sweep.study.module_path, &sweep.study.study_id)?;
        let policy = self.policy_for(sweep_key)?;
        let study = self.resolve_study_from(study_key, study, policy)?;
        sweep
            .validate(&study.study)
            .map_err(|error| AnalysisRootResolutionError::InvalidSweep(error.to_string()))?;
        Ok(ResolvedAnalysisSweep {
            sweep: sweep.clone(),
            study,
        })
    }

    fn resolve_study_from(
        &self,
        key: &(LogicalSourcePath, FeaSymbolKey),
        study: &AuthoredStudy,
        policy_override: Option<AuthoredExecutionPolicy>,
    ) -> Result<ResolvedAnalysisStudy, AnalysisRootResolutionError> {
        let geometry = self
            .geometries
            .values()
            .find(|geometry| {
                geometry.module_path == study.geometry.definition.module_path
                    && geometry.geometry_id == study.geometry.definition.geometry_id
            })
            .ok_or_else(|| AnalysisRootResolutionError::MissingGeometry {
                module_path: study.geometry.definition.module_path.clone(),
                geometry_id: study.geometry.definition.geometry_id.clone(),
            })?;
        let mesh = self
            .meshes
            .values()
            .find(|mesh| {
                mesh.module_path == study.mesh.module_path && mesh.mesh_id == study.mesh.mesh_id
            })
            .ok_or_else(|| AnalysisRootResolutionError::MissingMesh {
                module_path: study.mesh.module_path.clone(),
                mesh_id: study.mesh.mesh_id.clone(),
            })?;
        if mesh.geometry != study.geometry.definition {
            return Err(AnalysisRootResolutionError::MeshGeometryMismatch {
                mesh_id: mesh.mesh_id.clone(),
            });
        }
        let mut geometry_dependencies = Vec::new();
        let mut geometry_visiting = BTreeSet::new();
        collect_geometry_dependencies(
            geometry,
            &self.geometries,
            &mut geometry_visiting,
            &mut geometry_dependencies,
        )?;

        let available_models = self
            .models
            .values()
            .map(|model| (model.reference(), model))
            .collect::<BTreeMap<_, _>>();
        let mut models = BTreeMap::new();
        match &study.model {
            Some(instance) => {
                let mut visiting = BTreeSet::new();
                collect_model_closure(
                    &instance.model,
                    &available_models,
                    &mut visiting,
                    &mut models,
                )?;
                let root = models.get(&instance.model).ok_or_else(|| {
                    AnalysisRootResolutionError::MissingModel {
                        module_path: instance.model.module_path.clone(),
                        model_id: instance.model.model_id.clone(),
                    }
                })?;
                study.validate(root, geometry)
            }
            None => study.validate(
                &runmat_analysis_core::AuthoredModel::empty(&study.module_path, &study.study_id),
                geometry,
            ),
        }
        .map_err(|error| AnalysisRootResolutionError::InvalidStudy(error.to_string()))?;

        let material_catalog = self
            .materials
            .iter()
            .map(|((path, symbol), material)| {
                (
                    AuthoredMaterialRef {
                        module_path: path.as_str().to_string(),
                        material_id: symbol.name.clone(),
                    },
                    material.clone(),
                )
            })
            .collect();
        let execution_policy = match policy_override {
            Some(policy) => Some(policy),
            None => self.policy_for(key)?,
        };
        Ok(ResolvedAnalysisStudy {
            study: study.clone(),
            geometry: geometry.clone(),
            geometry_dependencies,
            mesh: mesh.clone(),
            models,
            material_catalog,
            execution_policy,
        })
    }

    fn policy_for(
        &self,
        key: &(LogicalSourcePath, FeaSymbolKey),
    ) -> Result<Option<AuthoredExecutionPolicy>, AnalysisRootResolutionError> {
        self.execution_bindings
            .get(key)
            .map(|reference| find_policy(self, reference).cloned())
            .transpose()
    }

    fn require_complete_diagnostics(&self) -> Result<(), AnalysisRootResolutionError> {
        let unavailable = self.diagnostics.iter().any(|diagnostic| {
            diagnostic.severity == FeaDiagnosticSeverity::Error
                && diagnostic.availability == SourceAvailability::Unavailable
        });
        let invalid = self
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == FeaDiagnosticSeverity::Error);
        if invalid {
            Err(AnalysisRootResolutionError::IncompleteDiagnostics {
                unavailable_external_facts: unavailable,
            })
        } else {
            Ok(())
        }
    }
}

fn collect_geometry_dependencies(
    geometry: &GeometryDefinition,
    available: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
    visiting: &mut BTreeSet<runmat_geometry_core::GeometryDefinitionRef>,
    resolved: &mut Vec<GeometryDefinition>,
) -> Result<(), AnalysisRootResolutionError> {
    let runmat_geometry_core::GeometryDefinitionSource::Derived(derived) = &geometry.source else {
        return Ok(());
    };
    if !visiting.insert(derived.source.clone()) {
        return Err(AnalysisRootResolutionError::GeometryDependencyCycle {
            module_path: derived.source.module_path.clone(),
            geometry_id: derived.source.geometry_id.clone(),
        });
    }
    let source = available
        .values()
        .find(|candidate| {
            candidate.module_path == derived.source.module_path
                && candidate.geometry_id == derived.source.geometry_id
        })
        .ok_or_else(|| AnalysisRootResolutionError::MissingGeometry {
            module_path: derived.source.module_path.clone(),
            geometry_id: derived.source.geometry_id.clone(),
        })?;
    collect_geometry_dependencies(source, available, visiting, resolved)?;
    visiting.remove(&derived.source);
    let source_reference = runmat_geometry_core::GeometryDefinitionRef {
        module_path: source.module_path.clone(),
        geometry_id: source.geometry_id.clone(),
    };
    if !resolved.iter().any(|candidate| {
        candidate.module_path == source_reference.module_path
            && candidate.geometry_id == source_reference.geometry_id
    }) {
        resolved.push(source.clone());
    }
    Ok(())
}

fn find_study<'a>(
    program: &'a CheckedFeaProgram,
    module_path: &str,
    study_id: &str,
) -> Result<(&'a (LogicalSourcePath, FeaSymbolKey), &'a AuthoredStudy), AnalysisRootResolutionError>
{
    program
        .studies
        .iter()
        .find(|(_, study)| study.module_path == module_path && study.study_id == study_id)
        .ok_or_else(|| AnalysisRootResolutionError::MissingStudy {
            module_path: module_path.to_string(),
            study_id: study_id.to_string(),
        })
}

fn find_policy<'a>(
    program: &'a CheckedFeaProgram,
    reference: &AuthoredExecutionPolicyRef,
) -> Result<&'a AuthoredExecutionPolicy, AnalysisRootResolutionError> {
    program
        .execution_policies
        .values()
        .find(|policy| {
            policy.module_path == reference.module_path && policy.policy_id == reference.policy_id
        })
        .ok_or_else(|| AnalysisRootResolutionError::MissingExecutionPolicy {
            module_path: reference.module_path.clone(),
            policy_id: reference.policy_id.clone(),
        })
}

fn collect_model_closure(
    reference: &AuthoredModelRef,
    available: &BTreeMap<AuthoredModelRef, &AuthoredModel>,
    visiting: &mut BTreeSet<AuthoredModelRef>,
    resolved: &mut BTreeMap<AuthoredModelRef, AuthoredModel>,
) -> Result<(), AnalysisRootResolutionError> {
    if resolved.contains_key(reference) {
        return Ok(());
    }
    if !visiting.insert(reference.clone()) {
        return Err(AnalysisRootResolutionError::ModelIncludeCycle {
            module_path: reference.module_path.clone(),
            model_id: reference.model_id.clone(),
        });
    }
    let model =
        available
            .get(reference)
            .ok_or_else(|| AnalysisRootResolutionError::MissingModel {
                module_path: reference.module_path.clone(),
                model_id: reference.model_id.clone(),
            })?;
    for include in &model.includes {
        collect_model_closure(&include.model, available, visiting, resolved)?;
    }
    visiting.remove(reference);
    resolved.insert(reference.clone(), (*model).clone());
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnalysisRootResolutionError {
    #[error("checked FEA program has unresolved errors (unavailable external facts: {unavailable_external_facts})")]
    IncompleteDiagnostics { unavailable_external_facts: bool },
    #[error("study `{module_path}::{study_id}` is unavailable")]
    MissingStudy {
        module_path: String,
        study_id: String,
    },
    #[error("sweep `{module_path}::{sweep_id}` is unavailable")]
    MissingSweep {
        module_path: String,
        sweep_id: String,
    },
    #[error("geometry `{module_path}::{geometry_id}` is unavailable")]
    MissingGeometry {
        module_path: String,
        geometry_id: String,
    },
    #[error("geometry dependency cycle reaches `{module_path}::{geometry_id}`")]
    GeometryDependencyCycle {
        module_path: String,
        geometry_id: String,
    },
    #[error("mesh `{module_path}::{mesh_id}` is unavailable")]
    MissingMesh {
        module_path: String,
        mesh_id: String,
    },
    #[error("mesh `{mesh_id}` targets a different geometry from its study")]
    MeshGeometryMismatch { mesh_id: String },
    #[error("model `{module_path}::{model_id}` is unavailable")]
    MissingModel {
        module_path: String,
        model_id: String,
    },
    #[error("model include cycle reaches `{module_path}::{model_id}`")]
    ModelIncludeCycle {
        module_path: String,
        model_id: String,
    },
    #[error("execution policy `{module_path}::{policy_id}` is unavailable")]
    MissingExecutionPolicy {
        module_path: String,
        policy_id: String,
    },
    #[error("resolved study is invalid: {0}")]
    InvalidStudy(String),
    #[error("resolved sweep is invalid: {0}")]
    InvalidSweep(String),
}
