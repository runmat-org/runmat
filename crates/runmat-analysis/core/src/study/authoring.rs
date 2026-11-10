use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    GeometryDefinition, GeometryDefinitionRef, GeometryParameterDefinition, GeometryParameterType,
    GeometryRegionRef,
};
use serde::{Deserialize, Serialize};

use crate::{
    lower_authored_observations, validate_analysis_sequence, validate_authored_outputs,
    AnalysisSequence, AuthoredInputDefinition, AuthoredInputType, AuthoredInputValue,
    AuthoredModel, AuthoredModelRef, AuthoredOutput, AuthoredOutputExpression,
    AuthoredRegionSource,
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
pub enum AuthoredGeometryArgumentValue {
    Constant(AuthoredInputValue),
    StudyInput(AuthoredStudyInputRef),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredGeometryArgument {
    pub parameter_id: String,
    pub value: AuthoredGeometryArgumentValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredGeometryInstance {
    pub definition: GeometryDefinitionRef,
    pub arguments: Vec<AuthoredGeometryArgument>,
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
    pub geometry: AuthoredGeometryInstance,
    pub mesh: AuthoredMeshRef,
    /// `None` exactly for a study whose every step is `line_parameters`: its cross-section
    /// geometry carries the dielectrics, and no model applies materials, loads or constraints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<AuthoredModelInstance>,
    pub inputs: Vec<AuthoredInputDefinition>,
    pub outputs: Vec<AuthoredOutput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assessments: Vec<crate::AuthoredAssessment>,
    pub sequence: AnalysisSequence,
}

impl AuthoredStudy {
    pub fn validate(
        &self,
        model: &AuthoredModel,
        geometry: &GeometryDefinition,
    ) -> Result<(), AuthoredStudyError> {
        token("study module path", &self.module_path)?;
        token("study id", &self.study_id)?;
        token(
            "geometry module path",
            &self.geometry.definition.module_path,
        )?;
        token("geometry id", &self.geometry.definition.geometry_id)?;
        token("mesh module path", &self.mesh.module_path)?;
        token("mesh id", &self.mesh.mesh_id)?;
        let line_parameters = !self.sequence.steps.is_empty()
            && self
                .sequence
                .steps
                .iter()
                .all(|step| step.procedure == crate::AnalysisProcedure::LineParameters);
        let section = matches!(
            geometry.source,
            runmat_geometry_core::GeometryDefinitionSource::CrossSection(_)
        );
        if line_parameters != section {
            return Err(AuthoredStudyError::CrossSectionProcedure);
        }
        let no_arguments = Vec::new();
        let model_arguments = match &self.model {
            Some(instance) => {
                if line_parameters {
                    return Err(AuthoredStudyError::UnexpectedModel);
                }
                token("model module path", &instance.model.module_path)?;
                token("model id", &instance.model.model_id)?;
                if instance.model.module_path != model.module_path
                    || instance.model.model_id != model.model_id
                {
                    return Err(AuthoredStudyError::WrongModel);
                }
                &instance.arguments
            }
            None if line_parameters => &no_arguments,
            None => return Err(AuthoredStudyError::MissingModel),
        };
        if self.geometry.definition.module_path != geometry.module_path
            || self.geometry.definition.geometry_id != geometry.geometry_id
        {
            return Err(AuthoredStudyError::WrongGeometry);
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

        validate_geometry_arguments(&self.geometry.arguments, geometry, &study_inputs)?;

        let model_inputs = model
            .inputs
            .iter()
            .map(|input| (input.input_id.as_str(), input))
            .collect::<BTreeMap<_, _>>();
        let mut arguments = BTreeSet::new();
        for argument in model_arguments {
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
                        validate_region_owner(region, &self.geometry.definition)?;
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
        crate::validate_assessments(
            &self.assessments,
            &crate::authored_output_types(&self.outputs, &self.sequence.steps)
                .map_err(|e| AuthoredStudyError::InvalidOutput(e.to_string()))?,
        )
        .map_err(AuthoredStudyError::InvalidOutput)?;
        lower_authored_observations(&self.outputs, &self.sequence.steps)
            .map_err(|error| AuthoredStudyError::InvalidOutput(error.to_string()))?;
        for output in &self.outputs {
            validate_output_regions(&output.expression, &self.geometry.definition, model)?;
        }
        Ok(())
    }
}

fn validate_geometry_arguments(
    arguments: &[AuthoredGeometryArgument],
    geometry: &GeometryDefinition,
    study_inputs: &BTreeMap<&str, &AuthoredInputDefinition>,
) -> Result<(), AuthoredStudyError> {
    let parameters = geometry
        .parameters
        .iter()
        .map(|parameter| (parameter.parameter_id.as_str(), parameter))
        .collect::<BTreeMap<_, _>>();
    let mut bound = BTreeSet::new();
    for argument in arguments {
        token("geometry argument id", &argument.parameter_id)?;
        if !bound.insert(argument.parameter_id.as_str()) {
            return Err(AuthoredStudyError::DuplicateGeometryArgument(
                argument.parameter_id.clone(),
            ));
        }
        let parameter = parameters
            .get(argument.parameter_id.as_str())
            .ok_or_else(|| {
                AuthoredStudyError::UnknownGeometryParameter(argument.parameter_id.clone())
            })?;
        let expected = geometry_parameter_input_type(parameter).ok_or_else(|| {
            AuthoredStudyError::UnsupportedGeometryParameter(argument.parameter_id.clone())
        })?;
        match &argument.value {
            AuthoredGeometryArgumentValue::Constant(value) if value.matches(expected) => {
                value
                    .validate()
                    .map_err(|error| AuthoredStudyError::InvalidGeometryArgument {
                        parameter_id: argument.parameter_id.clone(),
                        detail: error.to_string(),
                    })?;
            }
            AuthoredGeometryArgumentValue::Constant(_) => {
                return Err(AuthoredStudyError::GeometryArgumentTypeMismatch(
                    argument.parameter_id.clone(),
                ));
            }
            AuthoredGeometryArgumentValue::StudyInput(reference) => {
                let input = study_inputs
                    .get(reference.input_id.as_str())
                    .ok_or_else(|| {
                        AuthoredStudyError::UnknownStudyInput(reference.input_id.clone())
                    })?;
                if input.value_type != reference.value_type
                    || !input_type_assignable(expected, input.value_type)
                {
                    return Err(AuthoredStudyError::GeometryArgumentTypeMismatch(
                        argument.parameter_id.clone(),
                    ));
                }
            }
        }
    }
    if let Some(parameter) = geometry.parameters.iter().find(|parameter| {
        parameter.required
            && parameter.default.is_none()
            && !bound.contains(parameter.parameter_id.as_str())
    }) {
        return Err(AuthoredStudyError::MissingGeometryArgument(
            parameter.parameter_id.clone(),
        ));
    }
    Ok(())
}

pub fn geometry_parameter_input_type(
    parameter: &GeometryParameterDefinition,
) -> Option<AuthoredInputType> {
    Some(match parameter.parameter_type {
        GeometryParameterType::Boolean => AuthoredInputType::Boolean,
        GeometryParameterType::Scalar => AuthoredInputType::Real,
        GeometryParameterType::Length => {
            AuthoredInputType::Quantity((runmat_types::PhysicalDimension::LENGTH).into())
        }
        GeometryParameterType::Angle => {
            AuthoredInputType::Quantity(runmat_types::physical::DynamicDimension::quantity_type(
                runmat_types::physical::QuantityKind::Angle,
            ))
        }
        GeometryParameterType::Text => AuthoredInputType::String,
        GeometryParameterType::Vector3 | GeometryParameterType::QuantityVector3(_) => return None,
    })
}

fn validate_output_regions(
    expression: &AuthoredOutputExpression,
    geometry: &GeometryDefinitionRef,
    model: &AuthoredModel,
) -> Result<(), AuthoredStudyError> {
    match expression {
        AuthoredOutputExpression::Projection { source, .. }
        | AuthoredOutputExpression::Recovery { source, .. }
        | AuthoredOutputExpression::Interpolation { source, .. }
        | AuthoredOutputExpression::Sampling { source, .. }
        | AuthoredOutputExpression::SeriesReduction { source, .. } => {
            validate_output_regions(source, geometry, model)
        }
        AuthoredOutputExpression::PortReduction {
            source,
            first,
            second,
            ..
        } => {
            validate_output_regions(source, geometry, model)?;
            validate_output_region(first, geometry, model)?;
            validate_output_region(second, geometry, model)
        }
        AuthoredOutputExpression::SpatialReduction { source, target, .. }
        | AuthoredOutputExpression::SpatialSelection { source, target } => {
            validate_output_regions(source, geometry, model)?;
            validate_output_region(target, geometry, model)
        }
        AuthoredOutputExpression::AnalysisField { .. }
        | AuthoredOutputExpression::LineParameter { .. }
        | AuthoredOutputExpression::OutputReference { .. } => Ok(()),
    }
}

fn validate_output_region(
    target: &AuthoredRegionSource,
    geometry: &GeometryDefinitionRef,
    model: &AuthoredModel,
) -> Result<(), AuthoredStudyError> {
    {
        {
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
    #[error("study geometry identity does not match the validated geometry")]
    WrongGeometry,
    #[error("study requires a `model`")]
    MissingModel,
    #[error("a `line_parameters` study takes no `model`: its cross-section carries the dielectrics")]
    UnexpectedModel,
    #[error("a `cross_section` geometry is solved only by `electromagnetic line_parameters`, which needs a `cross_section` geometry")]
    CrossSectionProcedure,
    #[error("geometry argument `{0}` is duplicated")]
    DuplicateGeometryArgument(String),
    #[error("geometry argument references unknown parameter `{0}`")]
    UnknownGeometryParameter(String),
    #[error("required geometry parameter `{0}` is not bound")]
    MissingGeometryArgument(String),
    #[error("geometry parameter `{0}` cannot be bound from a study input")]
    UnsupportedGeometryParameter(String),
    #[error("geometry argument `{0}` has an incompatible study-input type")]
    GeometryArgumentTypeMismatch(String),
    #[error("geometry argument `{parameter_id}` is invalid: {detail}")]
    InvalidGeometryArgument {
        parameter_id: String,
        detail: String,
    },
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
mod tests;
