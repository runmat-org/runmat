//! Setup dependencies are derived from canonical definitions, never labels or JSON keys.
use crate::{FeaObjectDefinition, FeaObjectDetails, FeaSourceIdentity, LogicalSourcePath};
use runmat_analysis_core::{
    AuthoredInputValue, AuthoredMaterialSource, AuthoredModelArgumentValue,
};
use runmat_analysis_language::{FeaSymbolKey, FeaSymbolKind, SourceIdentityError};

impl FeaObjectDetails {
    /// Primary setup inspection follows a study's mesh and composed models and
    /// their material declarations. Model definitions include loads, supports,
    /// signals and assignments. Geometry realization is intentionally separate.
    pub fn setup_dependencies(&self) -> Result<Vec<FeaSourceIdentity>, SourceIdentityError> {
        let declaration = |path: &str, kind, name: &str| -> Result<_, SourceIdentityError> {
            Ok(FeaSourceIdentity::Declaration {
                module_path: LogicalSourcePath::new(path)?,
                symbol: FeaSymbolKey {
                    kind,
                    name: name.into(),
                },
            })
        };
        let mut dependencies = Vec::new();
        match &self.definition {
            FeaObjectDefinition::Assessment(assessment) => {
                if let runmat_analysis_core::AssessmentRule::Function { function, .. } =
                    &assessment.rule
                {
                    dependencies.push(FeaSourceIdentity::Function {
                        module_path: LogicalSourcePath::new(&function.module_path)?,
                        function_id: function.symbol.clone(),
                    });
                }
                if let FeaSourceIdentity::StudyMember {
                    module_path,
                    study_id,
                    ..
                } = &self.identity
                {
                    dependencies.push(FeaSourceIdentity::StudyMember {
                        module_path: module_path.clone(),
                        study_id: study_id.clone(),
                        member_kind: crate::FeaStudyMemberKind::Output,
                        member_id: assessment.output_id.clone(),
                    });
                }
            }
            FeaObjectDefinition::Study(study) => {
                dependencies.push(declaration(
                    &study.mesh.module_path,
                    FeaSymbolKind::Mesh,
                    &study.mesh.mesh_id,
                )?);
                if let Some(model) = &study.model {
                    dependencies.push(declaration(
                        &model.model.module_path,
                        FeaSymbolKind::Model,
                        &model.model.model_id,
                    )?);
                }
                for argument in study.model.iter().flat_map(|model| &model.arguments) {
                    if let AuthoredModelArgumentValue::Constant(AuthoredInputValue::Material(
                        material,
                    )) = &argument.value
                    {
                        dependencies.push(declaration(
                            &material.module_path,
                            FeaSymbolKind::Material,
                            &material.material_id,
                        )?);
                    }
                }
            }
            FeaObjectDefinition::Model(model) => {
                for include in &model.includes {
                    dependencies.push(declaration(
                        &include.model.module_path,
                        FeaSymbolKind::Model,
                        &include.model.model_id,
                    )?);
                }
                let materials = model
                    .material_assignments
                    .iter()
                    .map(|assignment| &assignment.material)
                    .chain(
                        model
                            .layups
                            .iter()
                            .flat_map(|layup| layup.plies.iter().map(|ply| &ply.material)),
                    );
                for source in materials {
                    if let AuthoredMaterialSource::Declaration(material) = source {
                        dependencies.push(declaration(
                            &material.module_path,
                            FeaSymbolKind::Material,
                            &material.material_id,
                        )?);
                    }
                }
            }
            _ => {}
        }
        Ok(dependencies)
    }
}
