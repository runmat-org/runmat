use super::CheckedFeaSource;
use runmat_analysis_language::FeaSymbolKind;
pub use runmat_fea_source::{
    FeaDeclaredMaterial, FeaExecutionBinding, FeaExecutionTarget, FeaSemanticProgram,
    FeaStudyEvaluatedSignals,
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
            evaluated_signals: self.evaluated_signals(),
        }
    }

    /// Each study's parameterized signals evaluated at the study's own input values, with the
    /// same evaluation realization uses. Studies whose model has none are left out.
    fn evaluated_signals(&self) -> Vec<FeaStudyEvaluatedSignals> {
        self.program
            .studies
            .iter()
            .filter_map(|((module_path, _), study)| {
                let instance = study.model.as_ref()?;
                let model = self
                    .program
                    .models
                    .values()
                    .find(|model| model.reference() == instance.model)?;
                if model.parameterized_signals.is_empty() {
                    return None;
                }
                Some(FeaStudyEvaluatedSignals {
                    module_path: module_path.clone(),
                    study_id: study.study_id.clone(),
                    signals: study.evaluate_parameterized_signals(model),
                })
            })
            .collect()
    }
}
