use super::CheckedFeaSource;
use runmat_analysis_language::FeaSymbolKind;
pub use runmat_fea_source::{
    FeaDeclaredMaterial, FeaExecutionBinding, FeaExecutionTarget, FeaSemanticProgram,
};

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
