use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    authored_output_types, AuthoredInputDefinition, AuthoredInputValue, AuthoredOutputType,
    AuthoredOutputValue, AuthoredStudy,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredStudyRef {
    pub module_path: String,
    pub study_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredSweep {
    pub module_path: String,
    pub sweep_id: String,
    pub study: AuthoredStudyRef,
    pub definition: AuthoredSweepDefinition,
    pub outputs: Vec<AuthoredSweepOutput>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredSweepDefinition {
    Axes {
        combination: SweepAxisCombination,
        axes: Vec<AuthoredSweepAxis>,
    },
    Cases {
        cases: Vec<AuthoredSweepCase>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SweepAxisCombination {
    Product,
    Zip,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredSweepAxis {
    pub input_id: String,
    pub values: Vec<AuthoredSweepAxisValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredSweepAxisValue {
    pub value_id: String,
    pub value: AuthoredInputValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredSweepCase {
    pub case_id: String,
    pub bindings: Vec<AuthoredSweepBinding>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredSweepBinding {
    pub input_id: String,
    pub value: AuthoredInputValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredSweepOutput {
    pub output_id: String,
    pub expression: AuthoredSweepOutputExpression,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredSweepOutputExpression {
    Collect {
        source: AuthoredCaseOutputRef,
    },
    RetainEach {
        source: AuthoredCaseOutputRef,
    },
    Extremum {
        reduction: SweepExtremum,
        source: AuthoredCaseOutputRef,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredCaseOutputRef {
    pub output_id: String,
    pub projection: CaseOutputProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseOutputProjection {
    Whole,
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SweepExtremum {
    Minimum,
    Maximum,
}

impl AuthoredSweep {
    pub fn validate(&self, study: &AuthoredStudy) -> Result<(), AuthoredSweepError> {
        if self.module_path.trim().is_empty() || self.sweep_id.trim().is_empty() {
            return Err(AuthoredSweepError::EmptyIdentity);
        }
        if self.study.module_path != study.module_path || self.study.study_id != study.study_id {
            return Err(AuthoredSweepError::WrongStudy);
        }
        let inputs = study
            .inputs
            .iter()
            .map(|input| (input.input_id.as_str(), input))
            .collect::<BTreeMap<_, _>>();
        match &self.definition {
            AuthoredSweepDefinition::Axes { combination, axes } => {
                validate_axes(*combination, axes, &inputs)
            }
            AuthoredSweepDefinition::Cases { cases } => validate_cases(cases, &inputs),
        }?;
        validate_outputs(&self.outputs, study)
    }

    pub fn expand_cases(
        &self,
        study: &AuthoredStudy,
    ) -> Result<Vec<AuthoredSweepCase>, AuthoredSweepError> {
        self.validate(study)?;
        match &self.definition {
            AuthoredSweepDefinition::Cases { cases } => Ok(cases.clone()),
            AuthoredSweepDefinition::Axes {
                combination: SweepAxisCombination::Zip,
                axes,
            } => expand_zip(axes),
            AuthoredSweepDefinition::Axes {
                combination: SweepAxisCombination::Product,
                axes,
            } => Ok(expand_product(axes)),
        }
    }
}

fn validate_outputs(
    outputs: &[AuthoredSweepOutput],
    study: &AuthoredStudy,
) -> Result<(), AuthoredSweepError> {
    let study_outputs = authored_output_types(&study.outputs, &study.sequence.steps)
        .map_err(|error| AuthoredSweepError::InvalidStudyOutput(error.to_string()))?;
    let mut ids = BTreeSet::new();
    for output in outputs {
        if output.output_id.trim().is_empty() || !ids.insert(output.output_id.as_str()) {
            return Err(AuthoredSweepError::DuplicateOutput(
                output.output_id.clone(),
            ));
        }
        let (source, requires_field) = match &output.expression {
            AuthoredSweepOutputExpression::Collect { source }
            | AuthoredSweepOutputExpression::Extremum { source, .. } => (source, false),
            AuthoredSweepOutputExpression::RetainEach { source } => (source, true),
        };
        let source_type = study_outputs
            .get(&source.output_id)
            .ok_or_else(|| AuthoredSweepError::UnknownStudyOutput(source.output_id.clone()))?;
        match (source.projection, source_type, requires_field) {
            (CaseOutputProjection::Value, AuthoredOutputType::SpatialExtremum { .. }, false)
            | (CaseOutputProjection::Value, AuthoredOutputType::SeriesExtremum { .. }, false)
            | (CaseOutputProjection::Whole, AuthoredOutputType::Reduced { .. }, false)
            | (CaseOutputProjection::Whole, AuthoredOutputType::Probe { .. }, false)
            | (CaseOutputProjection::Whole, AuthoredOutputType::Field { .. }, true) => {}
            _ => return Err(AuthoredSweepError::InvalidCaseOutputProjection),
        }
    }
    Ok(())
}

impl AuthoredCaseOutputRef {
    pub fn value(&self, study: &AuthoredStudy) -> Result<AuthoredOutputValue, AuthoredSweepError> {
        let outputs = authored_output_types(&study.outputs, &study.sequence.steps)
            .map_err(|error| AuthoredSweepError::InvalidStudyOutput(error.to_string()))?;
        match (self.projection, outputs.get(&self.output_id)) {
            (CaseOutputProjection::Value, Some(AuthoredOutputType::SpatialExtremum { value }))
            | (CaseOutputProjection::Value, Some(AuthoredOutputType::SeriesExtremum { value }))
            | (CaseOutputProjection::Whole, Some(AuthoredOutputType::Reduced { value, .. }))
            | (CaseOutputProjection::Whole, Some(AuthoredOutputType::Probe { value, .. }))
            | (CaseOutputProjection::Whole, Some(AuthoredOutputType::Field { value, .. })) => {
                Ok(*value)
            }
            _ => Err(AuthoredSweepError::InvalidCaseOutputProjection),
        }
    }
}

fn validate_axes(
    combination: SweepAxisCombination,
    axes: &[AuthoredSweepAxis],
    inputs: &BTreeMap<&str, &AuthoredInputDefinition>,
) -> Result<(), AuthoredSweepError> {
    if axes.is_empty() {
        return Err(AuthoredSweepError::EmptyDefinition);
    }
    let mut axis_ids = BTreeSet::new();
    let mut zip_labels: Option<BTreeSet<&str>> = None;
    for axis in axes {
        if axis.input_id.trim().is_empty() || !axis_ids.insert(axis.input_id.as_str()) {
            return Err(AuthoredSweepError::DuplicateInput(axis.input_id.clone()));
        }
        let input = inputs
            .get(axis.input_id.as_str())
            .ok_or_else(|| AuthoredSweepError::UnknownInput(axis.input_id.clone()))?;
        if axis.values.is_empty() {
            return Err(AuthoredSweepError::EmptyAxis(axis.input_id.clone()));
        }
        let mut labels = BTreeSet::new();
        for value in &axis.values {
            if value.value_id.trim().is_empty() || !labels.insert(value.value_id.as_str()) {
                return Err(AuthoredSweepError::DuplicateValue {
                    input_id: axis.input_id.clone(),
                    value_id: value.value_id.clone(),
                });
            }
            input.validate_value(&value.value).map_err(|error| {
                AuthoredSweepError::InvalidValue {
                    input_id: axis.input_id.clone(),
                    reason: error.to_string(),
                }
            })?;
        }
        if combination == SweepAxisCombination::Zip {
            if zip_labels
                .as_ref()
                .is_some_and(|expected| expected != &labels)
            {
                return Err(AuthoredSweepError::MismatchedZipLabels);
            }
            zip_labels = Some(labels);
        }
    }
    require_runnable_inputs(inputs, &axis_ids)
}

fn validate_cases(
    cases: &[AuthoredSweepCase],
    inputs: &BTreeMap<&str, &AuthoredInputDefinition>,
) -> Result<(), AuthoredSweepError> {
    if cases.is_empty() {
        return Err(AuthoredSweepError::EmptyDefinition);
    }
    let mut case_ids = BTreeSet::new();
    for case in cases {
        if case.case_id.trim().is_empty() || !case_ids.insert(case.case_id.as_str()) {
            return Err(AuthoredSweepError::DuplicateCase(case.case_id.clone()));
        }
        let mut bound = BTreeSet::new();
        for binding in &case.bindings {
            if !bound.insert(binding.input_id.as_str()) {
                return Err(AuthoredSweepError::DuplicateInput(binding.input_id.clone()));
            }
            let input = inputs
                .get(binding.input_id.as_str())
                .ok_or_else(|| AuthoredSweepError::UnknownInput(binding.input_id.clone()))?;
            input.validate_value(&binding.value).map_err(|error| {
                AuthoredSweepError::InvalidValue {
                    input_id: binding.input_id.clone(),
                    reason: error.to_string(),
                }
            })?;
        }
        require_runnable_inputs(inputs, &bound)?;
    }
    Ok(())
}

fn require_runnable_inputs(
    inputs: &BTreeMap<&str, &AuthoredInputDefinition>,
    bound: &BTreeSet<&str>,
) -> Result<(), AuthoredSweepError> {
    for input in inputs.values() {
        if input.is_required() && !bound.contains(input.input_id.as_str()) {
            return Err(AuthoredSweepError::MissingRequiredInput(
                input.input_id.clone(),
            ));
        }
    }
    Ok(())
}

fn expand_zip(axes: &[AuthoredSweepAxis]) -> Result<Vec<AuthoredSweepCase>, AuthoredSweepError> {
    let labels = axes
        .first()
        .expect("validated axes are nonempty")
        .values
        .iter()
        .map(|value| value.value_id.as_str())
        .collect::<BTreeSet<_>>();
    labels
        .into_iter()
        .map(|label| {
            let bindings = axes
                .iter()
                .map(|axis| {
                    let value = axis
                        .values
                        .iter()
                        .find(|value| value.value_id == label)
                        .ok_or(AuthoredSweepError::MismatchedZipLabels)?;
                    Ok(AuthoredSweepBinding {
                        input_id: axis.input_id.clone(),
                        value: value.value.clone(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(AuthoredSweepCase {
                case_id: label.to_string(),
                bindings,
            })
        })
        .collect()
}

fn expand_product(axes: &[AuthoredSweepAxis]) -> Vec<AuthoredSweepCase> {
    let mut axes = axes.iter().collect::<Vec<_>>();
    axes.sort_by(|left, right| left.input_id.cmp(&right.input_id));
    let mut cases = vec![AuthoredSweepCase {
        case_id: String::new(),
        bindings: Vec::new(),
    }];
    for axis in axes {
        let mut values = axis.values.iter().collect::<Vec<_>>();
        values.sort_by(|left, right| left.value_id.cmp(&right.value_id));
        cases = cases
            .into_iter()
            .flat_map(|case| {
                values.iter().map(move |value| {
                    let mut expanded = case.clone();
                    if !expanded.case_id.is_empty() {
                        expanded.case_id.push_str("__");
                    }
                    expanded.case_id.push_str(&axis.input_id);
                    expanded.case_id.push('=');
                    expanded.case_id.push_str(&value.value_id);
                    expanded.bindings.push(AuthoredSweepBinding {
                        input_id: axis.input_id.clone(),
                        value: value.value.clone(),
                    });
                    expanded
                })
            })
            .collect();
    }
    cases
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredSweepError {
    #[error("sweep identity must not be empty")]
    EmptyIdentity,
    #[error("sweep references a different study")]
    WrongStudy,
    #[error("sweep must declare at least one axis or case")]
    EmptyDefinition,
    #[error("sweep input `{0}` is duplicated")]
    DuplicateInput(String),
    #[error("sweep references unknown study input `{0}`")]
    UnknownInput(String),
    #[error("sweep axis `{0}` has no values")]
    EmptyAxis(String),
    #[error("sweep axis `{input_id}` repeats value id `{value_id}`")]
    DuplicateValue { input_id: String, value_id: String },
    #[error("sweep value for `{input_id}` is invalid: {reason}")]
    InvalidValue { input_id: String, reason: String },
    #[error("zipped sweep axes must use the same stable value ids")]
    MismatchedZipLabels,
    #[error("sweep case id `{0}` is duplicated")]
    DuplicateCase(String),
    #[error("sweep does not bind required study input `{0}`")]
    MissingRequiredInput(String),
    #[error("sweep output id `{0}` is empty or duplicated")]
    DuplicateOutput(String),
    #[error("sweep references unknown study output `{0}`")]
    UnknownStudyOutput(String),
    #[error("sweep case-output projection is incompatible with the selected operation")]
    InvalidCaseOutputProjection,
    #[error("sweep source study has invalid outputs: {0}")]
    InvalidStudyOutput(String),
}
