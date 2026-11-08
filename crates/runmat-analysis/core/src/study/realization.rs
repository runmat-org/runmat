use std::collections::BTreeMap;

use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionRef, GeometryDefinitionSource,
    GeometryScalarExpression,
};
use serde::{Deserialize, Serialize};

use crate::{
    AnalysisSequence, AuthoredInitialConditionTarget, AuthoredInputValue, AuthoredMaterialSource,
    AuthoredModel, AuthoredModelArgumentValue, AuthoredModelError, AuthoredModelRef,
    AuthoredPropertyAssignment, AuthoredPropertyTarget, AuthoredRegionSource,
};

use super::{AuthoredGeometryArgumentValue, AuthoredMeshRef, AuthoredStudy, AuthoredStudyError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredStudyInputBinding {
    pub input_id: String,
    pub value: AuthoredInputValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealizedGeometryInstance {
    pub definition: GeometryDefinitionRef,
    /// Captured source definition, independent of a later ambient geometry lookup.
    pub source: GeometryDefinitionSource,
    pub bindings: BTreeMap<String, GeometryAuthoringValue>,
}

/// A self-contained study after all declaration inputs have concrete values and every included
/// model has been captured by exact source identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealizedStudy {
    pub module_path: String,
    pub study_id: String,
    pub geometry: RealizedGeometryInstance,
    pub mesh: AuthoredMeshRef,
    /// The realized root model selected by the study. A study without a model (a
    /// `line_parameters` study) realizes an empty model named after the study.
    pub model: AuthoredModel,
    /// Transitive included models in dependency-first order. The root model is not repeated here.
    pub included_models: Vec<AuthoredModel>,
    pub study_inputs: Vec<AuthoredStudyInputBinding>,
    pub outputs: Vec<crate::AuthoredOutput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assessments: Vec<crate::AuthoredAssessment>,
    pub sequence: AnalysisSequence,
}

mod initial_fields;

impl AuthoredStudy {
    pub fn realize(
        &self,
        model: &AuthoredModel,
        geometry: &GeometryDefinition,
        provided_inputs: &[AuthoredStudyInputBinding],
    ) -> Result<RealizedStudy, AuthoredStudyRealizationError> {
        let catalog = BTreeMap::from([(model.reference(), model.clone())]);
        self.realize_with_models(&catalog, geometry, provided_inputs)
    }

    /// Realize the study against an exact catalog of authored models.
    ///
    /// All transitive includes must be present. The returned composition is dependency ordered and
    /// detached from the catalog, so execution never depends on a later ambient name lookup.
    pub fn realize_with_models(
        &self,
        models: &BTreeMap<AuthoredModelRef, AuthoredModel>,
        geometry: &GeometryDefinition,
        provided_inputs: &[AuthoredStudyInputBinding],
    ) -> Result<RealizedStudy, AuthoredStudyRealizationError> {
        let empty;
        let model = match &self.model {
            Some(instance) => models.get(&instance.model).ok_or_else(|| {
                AuthoredStudyRealizationError::MissingModel(instance.model.clone())
            })?,
            None => {
                empty = AuthoredModel::empty(&self.module_path, &self.study_id);
                &empty
            }
        };
        self.validate(model, geometry)
            .map_err(AuthoredStudyRealizationError::InvalidStudy)?;
        let included_models = resolve_included_models(model, models)?;
        let study_inputs = bind_study_inputs(self, provided_inputs)?;
        let geometry = realize_geometry(&self.geometry, geometry, &study_inputs)?;
        let model_inputs = bind_model_inputs(self, model, &study_inputs)?;
        let realized_model = realize_model(model.clone(), &model_inputs)?;
        let mut outputs = self.outputs.clone();
        for output in &mut outputs {
            substitute_output_regions(&mut output.expression, &model_inputs)?;
        }
        Ok(RealizedStudy {
            module_path: self.module_path.clone(),
            study_id: self.study_id.clone(),
            geometry,
            mesh: self.mesh.clone(),
            model: realized_model,
            included_models,
            study_inputs: self
                .inputs
                .iter()
                .map(|definition| AuthoredStudyInputBinding {
                    input_id: definition.input_id.clone(),
                    value: study_inputs[&definition.input_id].clone(),
                })
                .collect(),
            outputs,
            assessments: self.assessments.clone(),
            sequence: self.sequence.clone(),
        })
    }
}

fn realize_geometry(
    instance: &super::AuthoredGeometryInstance,
    definition: &GeometryDefinition,
    study_inputs: &BTreeMap<String, AuthoredInputValue>,
) -> Result<RealizedGeometryInstance, AuthoredStudyRealizationError> {
    let arguments = instance
        .arguments
        .iter()
        .map(|argument| (argument.parameter_id.as_str(), &argument.value))
        .collect::<BTreeMap<_, _>>();
    let mut bindings = BTreeMap::new();
    for parameter in &definition.parameters {
        let value = match arguments.get(parameter.parameter_id.as_str()) {
            Some(AuthoredGeometryArgumentValue::Constant(value)) => geometry_authoring_value(value),
            Some(AuthoredGeometryArgumentValue::StudyInput(reference)) => study_inputs
                .get(&reference.input_id)
                .and_then(geometry_authoring_value),
            None => parameter.default.clone(),
        }
        .ok_or_else(|| {
            AuthoredStudyRealizationError::InvalidGeometryArgument(parameter.parameter_id.clone())
        })?;
        bindings.insert(parameter.parameter_id.clone(), value);
    }
    Ok(RealizedGeometryInstance {
        definition: instance.definition.clone(),
        source: definition.source.clone(),
        bindings,
    })
}

fn geometry_authoring_value(value: &AuthoredInputValue) -> Option<GeometryAuthoringValue> {
    Some(match value {
        AuthoredInputValue::Boolean(value) => GeometryAuthoringValue::Boolean(*value),
        AuthoredInputValue::Integer(value) => {
            GeometryAuthoringValue::Scalar(GeometryScalarExpression::Literal {
                value: *value as f64,
            })
        }
        AuthoredInputValue::Real(value) => {
            GeometryAuthoringValue::Scalar(GeometryScalarExpression::Literal { value: *value })
        }
        AuthoredInputValue::Quantity(value) => {
            GeometryAuthoringValue::Scalar(GeometryScalarExpression::Canonical { value: *value })
        }
        AuthoredInputValue::String(value) => GeometryAuthoringValue::Text(value.clone()),
        AuthoredInputValue::Complex { .. }
        | AuthoredInputValue::Material(_)
        | AuthoredInputValue::Region(_) => return None,
    })
}

fn resolve_included_models(
    root: &AuthoredModel,
    models: &BTreeMap<AuthoredModelRef, AuthoredModel>,
) -> Result<Vec<AuthoredModel>, AuthoredStudyRealizationError> {
    let mut visiting = BTreeMap::new();
    let mut ordered = Vec::new();
    for include in &root.includes {
        visit_included_model(&include.model, models, &mut visiting, &mut ordered)?;
    }
    Ok(ordered)
}

fn visit_included_model(
    reference: &AuthoredModelRef,
    models: &BTreeMap<AuthoredModelRef, AuthoredModel>,
    states: &mut BTreeMap<AuthoredModelRef, VisitState>,
    ordered: &mut Vec<AuthoredModel>,
) -> Result<(), AuthoredStudyRealizationError> {
    match states.get(reference) {
        Some(VisitState::Complete) => return Ok(()),
        Some(VisitState::Visiting) => {
            return Err(AuthoredStudyRealizationError::ModelIncludeCycle(
                reference.clone(),
            ));
        }
        None => {}
    }
    let model = models
        .get(reference)
        .ok_or_else(|| AuthoredStudyRealizationError::MissingModel(reference.clone()))?;
    if model.reference() != *reference {
        return Err(AuthoredStudyRealizationError::WrongCatalogIdentity(
            reference.clone(),
        ));
    }
    model
        .validate()
        .map_err(AuthoredStudyRealizationError::InvalidIncludedModel)?;
    if !model.inputs.is_empty() {
        return Err(AuthoredStudyRealizationError::ParameterizedIncludedModel(
            reference.clone(),
        ));
    }
    states.insert(reference.clone(), VisitState::Visiting);
    for include in &model.includes {
        visit_included_model(&include.model, models, states, ordered)?;
    }
    states.insert(reference.clone(), VisitState::Complete);
    ordered.push(realize_model(model.clone(), &BTreeMap::new())?);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Complete,
}

fn realize_model(
    mut model: AuthoredModel,
    inputs: &BTreeMap<String, AuthoredInputValue>,
) -> Result<AuthoredModel, AuthoredStudyRealizationError> {
    substitute_model_inputs(&mut model, inputs)?;
    let parameterized = std::mem::take(&mut model.parameterized_signals);
    for signal in parameterized {
        model.signals.push(signal.realize(inputs).map_err(|error| {
            AuthoredStudyRealizationError::ParameterizedSignal {
                signal_id: signal.signal_id.0,
                detail: error.to_string(),
            }
        })?);
    }
    for signal in &mut model.signals {
        signal
            .bind_parameter_axis(&model.inputs)
            .map_err(
                |detail| AuthoredStudyRealizationError::ParameterizedSignal {
                    signal_id: signal.signal_id().0.clone(),
                    detail,
                },
            )?;
    }
    model.inputs.clear();
    model
        .validate()
        .map_err(AuthoredStudyRealizationError::InvalidRealizedModel)?;
    Ok(model)
}

fn bind_study_inputs(
    study: &AuthoredStudy,
    provided: &[AuthoredStudyInputBinding],
) -> Result<BTreeMap<String, AuthoredInputValue>, AuthoredStudyRealizationError> {
    let definitions = study
        .inputs
        .iter()
        .map(|definition| (definition.input_id.as_str(), definition))
        .collect::<BTreeMap<_, _>>();
    let mut values = BTreeMap::new();
    for binding in provided {
        if binding.input_id.trim().is_empty() || values.contains_key(&binding.input_id) {
            return Err(AuthoredStudyRealizationError::DuplicateStudyInput(
                binding.input_id.clone(),
            ));
        }
        let definition = definitions.get(binding.input_id.as_str()).ok_or_else(|| {
            AuthoredStudyRealizationError::UnknownStudyInput(binding.input_id.clone())
        })?;
        definition.validate_value(&binding.value).map_err(|error| {
            AuthoredStudyRealizationError::InvalidStudyInput {
                input_id: binding.input_id.clone(),
                detail: error.to_string(),
            }
        })?;
        values.insert(binding.input_id.clone(), binding.value.clone());
    }
    for definition in &study.inputs {
        if values.contains_key(&definition.input_id) {
            continue;
        }
        let value = definition.default.clone().ok_or_else(|| {
            AuthoredStudyRealizationError::MissingStudyInput(definition.input_id.clone())
        })?;
        values.insert(definition.input_id.clone(), value);
    }
    Ok(values)
}

fn bind_model_inputs(
    study: &AuthoredStudy,
    model: &AuthoredModel,
    study_inputs: &BTreeMap<String, AuthoredInputValue>,
) -> Result<BTreeMap<String, AuthoredInputValue>, AuthoredStudyRealizationError> {
    let arguments = study
        .model
        .arguments
        .iter()
        .map(|argument| (argument.input_id.as_str(), &argument.value))
        .collect::<BTreeMap<_, _>>();
    let mut values = BTreeMap::new();
    for definition in &model.inputs {
        let value = match arguments.get(definition.input_id.as_str()) {
            Some(AuthoredModelArgumentValue::Constant(value)) => (*value).clone(),
            Some(AuthoredModelArgumentValue::StudyInput(reference)) => study_inputs
                .get(&reference.input_id)
                .cloned()
                .ok_or_else(|| {
                    AuthoredStudyRealizationError::MissingStudyInput(reference.input_id.clone())
                })?,
            None => definition.default.clone().ok_or_else(|| {
                AuthoredStudyRealizationError::MissingModelInput(definition.input_id.clone())
            })?,
        };
        definition.validate_value(&value).map_err(|error| {
            AuthoredStudyRealizationError::InvalidModelInput {
                input_id: definition.input_id.clone(),
                detail: error.to_string(),
            }
        })?;
        values.insert(definition.input_id.clone(), value);
    }
    Ok(values)
}

fn substitute_model_inputs(
    model: &mut AuthoredModel,
    inputs: &BTreeMap<String, AuthoredInputValue>,
) -> Result<(), AuthoredStudyRealizationError> {
    for assignment in &mut model.material_assignments {
        substitute_material(&mut assignment.material, inputs)?;
        substitute_property_target(&mut assignment.target, inputs)?;
    }
    for layup in &mut model.layups {
        for ply in &mut layup.plies {
            substitute_material(&mut ply.material, inputs)?;
        }
    }
    for assignment in &mut model.property_assignments {
        match assignment {
            AuthoredPropertyAssignment::Section { target, .. }
            | AuthoredPropertyAssignment::Orientation { target, .. } => {
                substitute_property_target(target, inputs)?
            }
            AuthoredPropertyAssignment::Layup { target, .. } => substitute_region(target, inputs)?,
        }
    }
    for constraint in &mut model.constraints {
        let mut error = None;
        constraint.kind.visit_targets_mut(|target| {
            if error.is_none() {
                error = substitute_region(&mut target.region, inputs).err();
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
    }
    for boundary in &mut model.boundaries {
        substitute_region(&mut boundary.target, inputs)?;
        boundary
            .kind
            .realize_model_inputs(inputs)
            .map_err(
                |error| AuthoredStudyRealizationError::ParameterizedBoundary {
                    boundary_id: boundary.boundary_id.clone(),
                    detail: error.to_string(),
                },
            )?;
    }
    for connector in &mut model.connectors {
        let mut error = None;
        connector.kind.visit_targets_mut(|target| {
            if error.is_none() {
                error = substitute_region(target, inputs).err();
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
    }
    for interface in &mut model.interfaces {
        substitute_region(&mut interface.primary, inputs)?;
        substitute_region(&mut interface.secondary, inputs)?;
    }
    for excitation in &mut model.excitations {
        if let crate::AuthoredExcitationTarget::Region(target) = &mut excitation.target {
            substitute_region(target, inputs)?;
        }
    }
    for condition in &mut model.initial_conditions {
        if let AuthoredInitialConditionTarget::Region(target) = &mut condition.target {
            substitute_region(target, inputs)?;
        }
    }
    Ok(())
}

fn substitute_output_regions(
    expression: &mut crate::AuthoredOutputExpression,
    inputs: &BTreeMap<String, AuthoredInputValue>,
) -> Result<(), AuthoredStudyRealizationError> {
    match expression {
        crate::AuthoredOutputExpression::Projection { source, .. }
        | crate::AuthoredOutputExpression::Recovery { source, .. }
        | crate::AuthoredOutputExpression::Interpolation { source, .. }
        | crate::AuthoredOutputExpression::Sampling { source, .. }
        | crate::AuthoredOutputExpression::SeriesReduction { source, .. } => {
            substitute_output_regions(source, inputs)
        }
        crate::AuthoredOutputExpression::SpatialReduction { source, target, .. }
        | crate::AuthoredOutputExpression::SpatialSelection { source, target } => {
            substitute_output_regions(source, inputs)?;
            substitute_region(target, inputs)
        }
        crate::AuthoredOutputExpression::PortReduction {
            source,
            first,
            second,
            ..
        } => {
            substitute_output_regions(source, inputs)?;
            substitute_region(first, inputs)?;
            substitute_region(second, inputs)
        }
        crate::AuthoredOutputExpression::AnalysisField { .. }
        | crate::AuthoredOutputExpression::OutputReference { .. } => Ok(()),
    }
}

fn substitute_material(
    source: &mut AuthoredMaterialSource,
    inputs: &BTreeMap<String, AuthoredInputValue>,
) -> Result<(), AuthoredStudyRealizationError> {
    let AuthoredMaterialSource::ModelInput(reference) = source else {
        return Ok(());
    };
    let value = required_input(inputs, &reference.input_id)?;
    let AuthoredInputValue::Material(material) = value else {
        return Err(AuthoredStudyRealizationError::WrongBoundInputKind(
            reference.input_id.clone(),
        ));
    };
    *source = AuthoredMaterialSource::Declaration(material.clone());
    Ok(())
}

fn substitute_property_target(
    target: &mut AuthoredPropertyTarget,
    inputs: &BTreeMap<String, AuthoredInputValue>,
) -> Result<(), AuthoredStudyRealizationError> {
    match target {
        AuthoredPropertyTarget::Beam(source)
        | AuthoredPropertyTarget::Shell(source)
        | AuthoredPropertyTarget::Solid(source)
        | AuthoredPropertyTarget::Cell(source) => substitute_region(source, inputs),
    }
}

fn substitute_region(
    source: &mut AuthoredRegionSource,
    inputs: &BTreeMap<String, AuthoredInputValue>,
) -> Result<(), AuthoredStudyRealizationError> {
    let AuthoredRegionSource::ModelInput(reference) = source else {
        return Ok(());
    };
    let value = required_input(inputs, &reference.input_id)?;
    let AuthoredInputValue::Region(region) = value else {
        return Err(AuthoredStudyRealizationError::WrongBoundInputKind(
            reference.input_id.clone(),
        ));
    };
    *source = AuthoredRegionSource::Geometry(region.clone());
    Ok(())
}

fn required_input<'a>(
    inputs: &'a BTreeMap<String, AuthoredInputValue>,
    input_id: &str,
) -> Result<&'a AuthoredInputValue, AuthoredStudyRealizationError> {
    inputs
        .get(input_id)
        .ok_or_else(|| AuthoredStudyRealizationError::MissingModelInput(input_id.to_string()))
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredStudyRealizationError {
    #[error("authored model `{0:?}` is absent from the realization catalog")]
    MissingModel(AuthoredModelRef),
    #[error("authored model catalog entry does not match its key `{0:?}`")]
    WrongCatalogIdentity(AuthoredModelRef),
    #[error("authored model composition contains an include cycle through `{0:?}`")]
    ModelIncludeCycle(AuthoredModelRef),
    #[error("included authored model is invalid: {0}")]
    InvalidIncludedModel(AuthoredModelError),
    #[error("included authored model `{0:?}` declares inputs and cannot be implicitly bound")]
    ParameterizedIncludedModel(AuthoredModelRef),
    #[error("authored study is invalid: {0}")]
    InvalidStudy(AuthoredStudyError),
    #[error("geometry argument `{0}` cannot be materialized")]
    InvalidGeometryArgument(String),
    #[error("study input `{0}` is duplicated")]
    DuplicateStudyInput(String),
    #[error("study input `{0}` is unknown")]
    UnknownStudyInput(String),
    #[error("required study input `{0}` has no value")]
    MissingStudyInput(String),
    #[error("study input `{input_id}` is invalid: {detail}")]
    InvalidStudyInput { input_id: String, detail: String },
    #[error("required model input `{0}` has no value")]
    MissingModelInput(String),
    #[error("model input `{input_id}` is invalid: {detail}")]
    InvalidModelInput { input_id: String, detail: String },
    #[error("bound model input `{0}` has the wrong declaration kind")]
    WrongBoundInputKind(String),
    #[error("parameterized signal `{signal_id}` could not be realized: {detail}")]
    ParameterizedSignal { signal_id: String, detail: String },
    #[error("boundary `{boundary_id}` coefficient could not be realized: {detail}")]
    ParameterizedBoundary { boundary_id: String, detail: String },
    #[error("realized authored model is invalid: {0}")]
    InvalidRealizedModel(AuthoredModelError),
}
