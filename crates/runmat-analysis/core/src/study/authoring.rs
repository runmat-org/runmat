use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{GeometryDefinitionRef, GeometryRegionRef};
use serde::{Deserialize, Serialize};

use crate::{
    validate_analysis_sequence, validate_authored_outputs, AnalysisSequence,
    AuthoredInputDefinition, AuthoredInputType, AuthoredInputValue, AuthoredModel,
    AuthoredModelRef, AuthoredOutput, AuthoredOutputExpression, AuthoredRegionSource,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredMeshRef {
    pub module_path: String,
    pub mesh_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredStudyInputRef {
    pub input_id: String,
    pub value_type: AuthoredInputType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source", content = "value")]
pub enum AuthoredModelArgumentValue {
    Constant(AuthoredInputValue),
    StudyInput(AuthoredStudyInputRef),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredModelArgument {
    pub input_id: String,
    pub value: AuthoredModelArgumentValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredModelInstance {
    pub model: AuthoredModelRef,
    pub arguments: Vec<AuthoredModelArgument>,
}

/// Runnable authored semantics before geometry, mesh, and model realization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredStudy {
    pub module_path: String,
    pub study_id: String,
    pub geometry: GeometryDefinitionRef,
    pub mesh: AuthoredMeshRef,
    pub model: AuthoredModelInstance,
    pub inputs: Vec<AuthoredInputDefinition>,
    pub outputs: Vec<AuthoredOutput>,
    pub sequence: AnalysisSequence,
}

impl AuthoredStudy {
    pub fn validate(&self, model: &AuthoredModel) -> Result<(), AuthoredStudyError> {
        token("study module path", &self.module_path)?;
        token("study id", &self.study_id)?;
        token("geometry module path", &self.geometry.module_path)?;
        token("geometry id", &self.geometry.geometry_id)?;
        token("mesh module path", &self.mesh.module_path)?;
        token("mesh id", &self.mesh.mesh_id)?;
        token("model module path", &self.model.model.module_path)?;
        token("model id", &self.model.model.model_id)?;
        if self.model.model.module_path != model.module_path
            || self.model.model.model_id != model.model_id
        {
            return Err(AuthoredStudyError::WrongModel);
        }

        let mut study_inputs = BTreeMap::new();
        for input in &self.inputs {
            input
                .validate()
                .map_err(|error| AuthoredStudyError::InvalidInput(error.to_string()))?;
            if study_inputs
                .insert(input.input_id.as_str(), input)
                .is_some()
            {
                return Err(AuthoredStudyError::DuplicateInput(input.input_id.clone()));
            }
        }

        let model_inputs = model
            .inputs
            .iter()
            .map(|input| (input.input_id.as_str(), input))
            .collect::<BTreeMap<_, _>>();
        let mut arguments = BTreeSet::new();
        for argument in &self.model.arguments {
            token("model argument id", &argument.input_id)?;
            if !arguments.insert(argument.input_id.as_str()) {
                return Err(AuthoredStudyError::DuplicateModelArgument(
                    argument.input_id.clone(),
                ));
            }
            let definition = model_inputs
                .get(argument.input_id.as_str())
                .ok_or_else(|| AuthoredStudyError::UnknownModelInput(argument.input_id.clone()))?;
            match &argument.value {
                AuthoredModelArgumentValue::Constant(value) => {
                    definition.validate_value(value).map_err(|error| {
                        AuthoredStudyError::InvalidModelArgument {
                            input_id: argument.input_id.clone(),
                            detail: error.to_string(),
                        }
                    })?;
                    if let AuthoredInputValue::Region(region) = value {
                        validate_region_owner(region, &self.geometry)?;
                    }
                }
                AuthoredModelArgumentValue::StudyInput(reference) => {
                    token("study input reference", &reference.input_id)?;
                    let input = study_inputs
                        .get(reference.input_id.as_str())
                        .ok_or_else(|| {
                            AuthoredStudyError::UnknownStudyInput(reference.input_id.clone())
                        })?;
                    if reference.value_type != input.value_type
                        || !input_type_assignable(definition.value_type, input.value_type)
                    {
                        return Err(AuthoredStudyError::ModelArgumentTypeMismatch(
                            argument.input_id.clone(),
                        ));
                    }
                }
            }
        }
        for input in &model.inputs {
            if input.is_required() && !arguments.contains(input.input_id.as_str()) {
                return Err(AuthoredStudyError::MissingModelArgument(
                    input.input_id.clone(),
                ));
            }
        }

        validate_analysis_sequence(&self.sequence)
            .map_err(|error| AuthoredStudyError::InvalidSequence(error.to_string()))?;
        validate_authored_outputs(&self.outputs, &self.sequence.steps)
            .map_err(|error| AuthoredStudyError::InvalidOutput(error.to_string()))?;
        for output in &self.outputs {
            validate_output_regions(&output.expression, &self.geometry, model)?;
        }
        Ok(())
    }
}

fn validate_output_regions(
    expression: &AuthoredOutputExpression,
    geometry: &GeometryDefinitionRef,
    model: &AuthoredModel,
) -> Result<(), AuthoredStudyError> {
    match expression {
        AuthoredOutputExpression::Projection { source, .. } => {
            validate_output_regions(source, geometry, model)
        }
        AuthoredOutputExpression::SpatialReduction { source, target, .. } => {
            validate_output_regions(source, geometry, model)?;
            match target {
                AuthoredRegionSource::Geometry(region) => validate_region_owner(region, geometry),
                AuthoredRegionSource::ModelInput(reference)
                    if model.inputs.iter().any(|input| {
                        input.input_id == reference.input_id
                            && input.value_type == reference.value_type
                            && matches!(input.value_type, AuthoredInputType::Region(_))
                    }) =>
                {
                    Ok(())
                }
                AuthoredRegionSource::ModelInput(_) => Err(AuthoredStudyError::InvalidOutput(
                    "output region references an unavailable model region input".into(),
                )),
            }
        }
        AuthoredOutputExpression::AnalysisField { .. }
        | AuthoredOutputExpression::OutputReference { .. } => Ok(()),
    }
}

fn validate_region_owner(
    region: &GeometryRegionRef,
    geometry: &GeometryDefinitionRef,
) -> Result<(), AuthoredStudyError> {
    if region.module_path == geometry.module_path && region.geometry_id == geometry.geometry_id {
        Ok(())
    } else {
        Err(AuthoredStudyError::RegionFromWrongGeometry)
    }
}

fn input_type_assignable(expected: AuthoredInputType, actual: AuthoredInputType) -> bool {
    expected == actual
        || matches!(
            (expected, actual),
            (AuthoredInputType::Real, AuthoredInputType::Integer)
                | (
                    AuthoredInputType::Complex,
                    AuthoredInputType::Integer | AuthoredInputType::Real
                )
        )
}

fn token(field: &'static str, value: &str) -> Result<(), AuthoredStudyError> {
    if value.trim().is_empty() {
        Err(AuthoredStudyError::InvalidToken(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredStudyError {
    #[error("{0} must not be empty")]
    InvalidToken(&'static str),
    #[error("study input `{0}` is duplicated")]
    DuplicateInput(String),
    #[error("study input is invalid: {0}")]
    InvalidInput(String),
    #[error("study model identity does not match the validated model")]
    WrongModel,
    #[error("model argument `{0}` is duplicated")]
    DuplicateModelArgument(String),
    #[error("model argument references unknown model input `{0}`")]
    UnknownModelInput(String),
    #[error("required model input `{0}` is not bound")]
    MissingModelArgument(String),
    #[error("model argument `{input_id}` is invalid: {detail}")]
    InvalidModelArgument { input_id: String, detail: String },
    #[error("model argument `{0}` has an incompatible study-input type")]
    ModelArgumentTypeMismatch(String),
    #[error("model argument references unknown study input `{0}`")]
    UnknownStudyInput(String),
    #[error("model region arguments must belong to the study geometry")]
    RegionFromWrongGeometry,
    #[error("study procedure sequence is invalid: {0}")]
    InvalidSequence(String),
    #[error("study output is invalid: {0}")]
    InvalidOutput(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnalysisProcedure, PhysicsFamily, ProcedureStep, SolutionSpace};
    use runmat_types::QuantityDimension;

    fn model() -> AuthoredModel {
        AuthoredModel {
            module_path: "project/model.fea".into(),
            model_id: "bracket".into(),
            inputs: vec![AuthoredInputDefinition {
                input_id: "load".into(),
                value_type: AuthoredInputType::Quantity(QuantityDimension::Force),
                default: None,
                minimum: None,
                maximum: None,
            }],
            includes: Vec::new(),
            sections: Vec::new(),
            layups: Vec::new(),
            orientations: Vec::new(),
            material_assignments: Vec::new(),
            property_assignments: Vec::new(),
            constraints: Vec::new(),
            boundaries: Vec::new(),
            connectors: Vec::new(),
            interfaces: Vec::new(),
            signals: Vec::new(),
            parameterized_signals: Vec::new(),
            excitations: Vec::new(),
            excitation_cases: Vec::new(),
            initial_conditions: Vec::new(),
        }
    }

    #[test]
    fn required_model_input_can_bind_one_exact_study_input() {
        let mut step = ProcedureStep::structural_static("static");
        step.family = PhysicsFamily::Structural;
        step.procedure = AnalysisProcedure::Static;
        step.solution_space = SolutionSpace::Direct;
        let study = AuthoredStudy {
            module_path: "project/main.fea".into(),
            study_id: "response".into(),
            geometry: GeometryDefinitionRef {
                module_path: "project/main.fea".into(),
                geometry_id: "part".into(),
            },
            mesh: AuthoredMeshRef {
                module_path: "project/main.fea".into(),
                mesh_id: "part".into(),
            },
            model: AuthoredModelInstance {
                model: AuthoredModelRef {
                    module_path: "project/model.fea".into(),
                    model_id: "bracket".into(),
                },
                arguments: vec![AuthoredModelArgument {
                    input_id: "load".into(),
                    value: AuthoredModelArgumentValue::StudyInput(AuthoredStudyInputRef {
                        input_id: "service_load".into(),
                        value_type: AuthoredInputType::Quantity(QuantityDimension::Force),
                    }),
                }],
            },
            inputs: vec![AuthoredInputDefinition {
                input_id: "service_load".into(),
                value_type: AuthoredInputType::Quantity(QuantityDimension::Force),
                default: None,
                minimum: None,
                maximum: None,
            }],
            outputs: Vec::new(),
            sequence: AnalysisSequence {
                sequence_id: "response".into(),
                steps: vec![step],
                dependencies: Vec::new(),
                transitions: Vec::new(),
                branches: Vec::new(),
            },
        };
        study.validate(&model()).unwrap();
    }
}
