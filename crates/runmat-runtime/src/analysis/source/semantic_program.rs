use runmat_analysis_core::{
    AuthoredExecutionPolicy, AuthoredExecutionPolicyRef, AuthoredModel, AuthoredStudy,
    AuthoredSweep, MaterialModel,
};
use runmat_analysis_language::{FeaSymbolKind, LogicalSourcePath};
use runmat_geometry_core::GeometryDefinition;
use runmat_meshing_core::MeshDefinition;
use serde::{Deserialize, Serialize};

use super::CheckedFeaSource;

/// Serializable semantic product of checking a `.fea` module graph.
///
/// Every definition stored here is the canonical domain object produced by the checker. This
/// envelope only replaces map keys that cannot be represented as JSON object keys; it does not
/// translate domain objects into a display-specific schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSemanticProgram {
    pub studies: Vec<AuthoredStudy>,
    pub sweeps: Vec<AuthoredSweep>,
    pub execution_policies: Vec<AuthoredExecutionPolicy>,
    pub execution_bindings: Vec<FeaExecutionBinding>,
    pub materials: Vec<FeaDeclaredMaterial>,
    pub geometries: Vec<GeometryDefinition>,
    pub meshes: Vec<MeshDefinition>,
    pub models: Vec<AuthoredModel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaDeclaredMaterial {
    pub module_path: LogicalSourcePath,
    pub material: MaterialModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaExecutionBinding {
    pub target: FeaExecutionTarget,
    pub policy: AuthoredExecutionPolicyRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeaExecutionTarget {
    Study {
        module_path: LogicalSourcePath,
        study_id: String,
    },
    Sweep {
        module_path: LogicalSourcePath,
        sweep_id: String,
    },
}

impl CheckedFeaSource {
    pub fn semantic_program(&self) -> FeaSemanticProgram {
        FeaSemanticProgram {
            studies: self.program.studies.values().cloned().collect(),
            sweeps: self.program.sweeps.values().cloned().collect(),
            execution_policies: self.program.execution_policies.values().cloned().collect(),
            execution_bindings: self
                .program
                .execution_bindings
                .iter()
                .filter_map(|((module_path, symbol), policy)| {
                    let target = match symbol.kind {
                        FeaSymbolKind::Study => FeaExecutionTarget::Study {
                            module_path: module_path.clone(),
                            study_id: symbol.name.clone(),
                        },
                        FeaSymbolKind::Sweep => FeaExecutionTarget::Sweep {
                            module_path: module_path.clone(),
                            sweep_id: symbol.name.clone(),
                        },
                        _ => return None,
                    };
                    Some(FeaExecutionBinding {
                        target,
                        policy: policy.clone(),
                    })
                })
                .collect(),
            materials: self
                .program
                .materials
                .iter()
                .map(|((module_path, _), material)| FeaDeclaredMaterial {
                    module_path: module_path.clone(),
                    material: material.clone(),
                })
                .collect(),
            geometries: self.program.geometries.values().cloned().collect(),
            meshes: self.program.meshes.values().cloned().collect(),
            models: self.program.models.values().cloned().collect(),
        }
    }
}
