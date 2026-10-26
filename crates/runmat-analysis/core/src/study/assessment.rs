//! Engineering acceptance over explicit scalar outputs. Independent of solver placement.
use super::{AuthoredOutputType, AuthoredOutputValue};
use runmat_types::PhysicalDimension;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentStatus {
    Passed,
    Failed,
    Indeterminate,
}

/// Stable API alias of the shared checked quantity representation. SI dimensions,
/// nominal angle powers and known semantic roles travel together.
pub type AssessmentQuantity = runmat_types::physical::DimensionedScalar;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum AssessmentRule {
    Function {
        function: AssessmentFunctionReference,
        parameters: Vec<AssessmentQuantity>,
    },
    Maximum {
        bound: AssessmentQuantity,
    },
    Minimum {
        bound: AssessmentQuantity,
    },
    Range {
        minimum: AssessmentQuantity,
        maximum: AssessmentQuantity,
    },
}

/// A checked named function revision, resolved by the ordinary project module graph.
/// The output value and parameters cross the numerical boundary in canonical SI units.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssessmentFunctionReference {
    pub module_path: String,
    pub symbol: String,
    pub source_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssessmentFunctionOutcome {
    pub evaluated: bool,
    pub accepted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssessmentFunctionEvaluation {
    NotExecuted,
    Failed,
    Completed(AssessmentFunctionOutcome),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentEvidenceRequirement {
    /// Compare the declared output at its authored sample coordinates; no continuum certification.
    AuthoredSamples,
    /// A certified error bound must establish the comparison, including between samples.
    CertifiedBound,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredAssessment {
    pub assessment_id: String,
    pub output_id: String,
    pub rule: AssessmentRule,
    pub evidence: AssessmentEvidenceRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentIssue {
    InvalidDefinition,
    MissingOutput,
    IncompleteOutput,
    NonfiniteValue,
    DimensionMismatch,
    NumericalEvidenceUnavailable,
    CertifiedBoundUnavailable,
    ComparisonUncertain,
    Cancelled,
    ResourceLimit,
    FunctionFailed,
    FunctionExecutionUnavailable,
    FunctionEvidenceUnavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssessmentResult {
    pub assessment_id: String,
    pub output_id: String,
    pub status: AssessmentStatus,
    pub measured: Option<AssessmentQuantity>,
    pub issue: Option<AssessmentIssue>,
}

#[derive(Debug, Clone, Copy)]
pub struct AssessmentMeasurement {
    pub quantity: AssessmentQuantity,
    pub complete: bool,
    pub numerically_acceptable: bool,
    /// Absolute error bound in the same canonical units; absence is not zero error.
    pub absolute_error_bound: Option<f64>,
}

impl AssessmentRule {
    pub fn bounds(&self) -> (Option<AssessmentQuantity>, Option<AssessmentQuantity>) {
        match *self {
            Self::Function { .. } => (None, None),
            Self::Maximum { bound } => (None, Some(bound)),
            Self::Minimum { bound } => (Some(bound), None),
            Self::Range { minimum, maximum } => (Some(minimum), Some(maximum)),
        }
    }
}

pub fn validate_assessments(
    assessments: &[AuthoredAssessment],
    outputs: &BTreeMap<String, AuthoredOutputType>,
) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for assessment in assessments {
        if assessment.assessment_id.trim().is_empty() || !ids.insert(&assessment.assessment_id) {
            return Err("assessment ids must be nonempty and unique".into());
        }
        let dimension = assessment_output_descriptor(outputs.get(&assessment.output_id))
            .ok_or_else(|| {
                format!(
                    "assessment `{}` requires a declared real scalar output",
                    assessment.assessment_id
                )
            })?;
        if let AssessmentRule::Function {
            function,
            parameters,
        } = &assessment.rule
        {
            if function.module_path.trim().is_empty()
                || function.symbol.trim().is_empty()
                || function.source_digest == [0; 32]
                || parameters.iter().any(|p| !p.canonical_value().is_finite())
            {
                return Err(format!(
                    "assessment `{}` has an invalid function revision or parameters",
                    assessment.assessment_id
                ));
            }
        }
        let (minimum, maximum) = assessment.rule.bounds();
        for bound in [minimum, maximum].into_iter().flatten() {
            if !bound.canonical_value().is_finite()
                || !bound.descriptor().is_compatible_with(dimension)
            {
                return Err(format!(
                    "assessment `{}` has an invalid or dimensionally incompatible bound",
                    assessment.assessment_id
                ));
            }
        }
        if minimum
            .zip(maximum)
            .is_some_and(|(a, b)| a.canonical_value() > b.canonical_value())
        {
            return Err("assessment minimum exceeds maximum".into());
        }
    }
    Ok(())
}

pub fn assessment_output_dimension(
    output: Option<&AuthoredOutputType>,
) -> Option<PhysicalDimension> {
    Some(assessment_output_descriptor(output)?.dimension())
}

pub fn assessment_output_descriptor(
    output: Option<&AuthoredOutputType>,
) -> Option<runmat_types::physical::DynamicDimension> {
    let value = match output? {
        AuthoredOutputType::SpatialExtremum { value }
        | AuthoredOutputType::SeriesExtremum { value }
        | AuthoredOutputType::Reduced {
            value,
            shape: crate::ObservationValueShape::Scalar,
        } => *value,
        _ => return None,
    };
    Some(match value {
        AuthoredOutputValue::Physical(quantity) => quantity.quantity_descriptor()?,
        AuthoredOutputValue::Derived(dimension) => dimension,
        AuthoredOutputValue::Dimensionless => PhysicalDimension::DIMENSIONLESS.into(),
    })
}

pub fn evaluate_assessment(
    assessment: &AuthoredAssessment,
    measurement: Option<AssessmentMeasurement>,
) -> AssessmentResult {
    evaluate_with_outcome(
        assessment,
        measurement,
        AssessmentFunctionEvaluation::NotExecuted,
    )
}

fn evaluate_with_outcome(
    assessment: &AuthoredAssessment,
    measurement: Option<AssessmentMeasurement>,
    outcome: AssessmentFunctionEvaluation,
) -> AssessmentResult {
    let mut result = AssessmentResult {
        assessment_id: assessment.assessment_id.clone(),
        output_id: assessment.output_id.clone(),
        status: AssessmentStatus::Indeterminate,
        measured: measurement
            .filter(|value| value.quantity.canonical_value().is_finite())
            .map(|value| value.quantity),
        issue: None,
    };
    let Some(measurement) = measurement else {
        result.issue = Some(AssessmentIssue::MissingOutput);
        return result;
    };
    let (minimum, maximum) = assessment.rule.bounds();
    if minimum
        .zip(maximum)
        .is_some_and(|(a, b)| a.canonical_value() > b.canonical_value())
    {
        result.issue = Some(AssessmentIssue::InvalidDefinition);
        return result;
    }
    let issue = if !measurement.complete {
        Some(AssessmentIssue::IncompleteOutput)
    } else if !measurement.quantity.canonical_value().is_finite()
        || [minimum, maximum]
            .into_iter()
            .flatten()
            .any(|b| !b.canonical_value().is_finite())
    {
        Some(AssessmentIssue::NonfiniteValue)
    } else if [minimum, maximum].into_iter().flatten().any(|b| {
        !b.descriptor()
            .is_compatible_with(measurement.quantity.descriptor())
    }) {
        Some(AssessmentIssue::DimensionMismatch)
    } else if !measurement.numerically_acceptable {
        Some(AssessmentIssue::NumericalEvidenceUnavailable)
    } else {
        None
    };
    if let Some(issue) = issue {
        result.issue = Some(issue);
        return result;
    }
    // Imported functions must go through callable admission and bounded execution. They must
    // never fall through to a comparison with no bounds and accidentally report success.
    if matches!(assessment.rule, AssessmentRule::Function { .. }) {
        if assessment.evidence == AssessmentEvidenceRequirement::CertifiedBound {
            result.issue = Some(AssessmentIssue::CertifiedBoundUnavailable);
        } else {
            match outcome {
                AssessmentFunctionEvaluation::Completed(outcome) if outcome.evaluated => {
                    result.status = if outcome.accepted {
                        AssessmentStatus::Passed
                    } else {
                        AssessmentStatus::Failed
                    };
                }
                AssessmentFunctionEvaluation::Completed(_) => {
                    result.issue = Some(AssessmentIssue::FunctionEvidenceUnavailable)
                }
                AssessmentFunctionEvaluation::Failed => {
                    result.issue = Some(AssessmentIssue::FunctionFailed)
                }
                AssessmentFunctionEvaluation::NotExecuted => {
                    result.issue = Some(AssessmentIssue::FunctionExecutionUnavailable)
                }
            }
        }
        return result;
    }
    let error = match assessment.evidence {
        AssessmentEvidenceRequirement::AuthoredSamples => 0.0,
        AssessmentEvidenceRequirement::CertifiedBound => match measurement.absolute_error_bound {
            Some(error) if error.is_finite() && error >= 0.0 => error,
            _ => {
                result.issue = Some(AssessmentIssue::CertifiedBoundUnavailable);
                return result;
            }
        },
    };
    let low = measurement.quantity.canonical_value() - error;
    let high = measurement.quantity.canonical_value() + error;
    if !low.is_finite() || !high.is_finite() {
        result.issue = Some(AssessmentIssue::ComparisonUncertain);
        return result;
    }
    let passed = minimum.is_none_or(|b| low >= b.canonical_value())
        && maximum.is_none_or(|b| high <= b.canonical_value());
    let failed = minimum.is_some_and(|b| high < b.canonical_value())
        || maximum.is_some_and(|b| low > b.canonical_value());
    result.status = if passed {
        AssessmentStatus::Passed
    } else if failed {
        AssessmentStatus::Failed
    } else {
        AssessmentStatus::Indeterminate
    };
    if !passed && !failed {
        result.issue = Some(AssessmentIssue::ComparisonUncertain);
    }
    result
}

pub fn evaluate_function_assessment(
    assessment: &AuthoredAssessment,
    measurement: Option<AssessmentMeasurement>,
    outcome: AssessmentFunctionEvaluation,
) -> AssessmentResult {
    let mut result = evaluate_with_outcome(assessment, measurement, outcome);
    if !matches!(assessment.rule, AssessmentRule::Function { .. }) {
        result.issue = Some(AssessmentIssue::InvalidDefinition);
        result.status = AssessmentStatus::Indeterminate;
        return result;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assessment(evidence: AssessmentEvidenceRequirement) -> AuthoredAssessment {
        AuthoredAssessment {
            assessment_id: "displacement_limit".into(),
            output_id: "peak".into(),
            rule: AssessmentRule::Maximum {
                bound: AssessmentQuantity::try_from_descriptor(
                    (PhysicalDimension::LENGTH).into(),
                    0.001,
                )
                .expect("checked finite assessment quantity"),
            },
            evidence,
        }
    }
    fn measure(value_si: f64) -> AssessmentMeasurement {
        AssessmentMeasurement {
            quantity: AssessmentQuantity::try_from_descriptor(
                (PhysicalDimension::LENGTH).into(),
                value_si,
            )
            .expect("checked finite assessment quantity"),
            complete: true,
            numerically_acceptable: true,
            absolute_error_bound: None,
        }
    }
    #[test]
    fn sampled_comparison_distinguishes_pass_fail_and_missing() {
        let definition = assessment(AssessmentEvidenceRequirement::AuthoredSamples);
        assert_eq!(
            evaluate_assessment(&definition, Some(measure(0.001))).status,
            AssessmentStatus::Passed
        );
        assert_eq!(
            evaluate_assessment(&definition, Some(measure(0.002))).status,
            AssessmentStatus::Failed
        );
        assert_eq!(
            evaluate_assessment(&definition, None).issue,
            Some(AssessmentIssue::MissingOutput)
        );
        let mut incomplete = measure(0.0005);
        incomplete.complete = false;
        assert_eq!(
            evaluate_assessment(&definition, Some(incomplete)).status,
            AssessmentStatus::Indeterminate
        );
    }
    #[test]
    fn unconverged_numerical_evidence_is_indeterminate_whatever_its_value() {
        let definition = assessment(AssessmentEvidenceRequirement::AuthoredSamples);
        // A value inside and a value outside the criterion: neither may pass or fail when the
        // solve that produced it did not meet its numerical acceptance gates.
        for value in [0.0005, 0.002] {
            let mut unconverged = measure(value);
            unconverged.numerically_acceptable = false;
            let result = evaluate_assessment(&definition, Some(unconverged));
            assert_eq!(result.status, AssessmentStatus::Indeterminate);
            assert_eq!(
                result.issue,
                Some(AssessmentIssue::NumericalEvidenceUnavailable)
            );
        }
    }
    #[test]
    fn function_result_requires_admitted_measurement_and_evaluated_evidence() {
        let mut definition = assessment(AssessmentEvidenceRequirement::AuthoredSamples);
        definition.rule = AssessmentRule::Function {
            function: AssessmentFunctionReference {
                module_path: "checks.m".into(),
                symbol: "checks".into(),
                source_digest: [1; 32],
            },
            parameters: Vec::new(),
        };
        for accepted in [false, true] {
            let result = evaluate_function_assessment(
                &definition,
                Some(measure(0.001)),
                AssessmentFunctionEvaluation::Completed(AssessmentFunctionOutcome {
                    evaluated: true,
                    accepted,
                }),
            );
            assert_eq!(
                result.status,
                if accepted {
                    AssessmentStatus::Passed
                } else {
                    AssessmentStatus::Failed
                }
            );
            assert_eq!(result.issue, None);
        }
        assert_eq!(
            evaluate_function_assessment(
                &definition,
                Some(measure(0.001)),
                AssessmentFunctionEvaluation::Completed(AssessmentFunctionOutcome {
                    evaluated: false,
                    accepted: true
                })
            )
            .issue,
            Some(AssessmentIssue::FunctionEvidenceUnavailable)
        );
        assert_eq!(
            evaluate_function_assessment(
                &definition,
                None,
                AssessmentFunctionEvaluation::Completed(AssessmentFunctionOutcome {
                    evaluated: true,
                    accepted: true
                })
            )
            .issue,
            Some(AssessmentIssue::MissingOutput)
        );
        assert_eq!(
            evaluate_assessment(&definition, Some(measure(0.001))).issue,
            Some(AssessmentIssue::FunctionExecutionUnavailable)
        );
        definition.evidence = AssessmentEvidenceRequirement::CertifiedBound;
        assert_eq!(
            evaluate_function_assessment(
                &definition,
                Some(measure(0.001)),
                AssessmentFunctionEvaluation::Completed(AssessmentFunctionOutcome {
                    evaluated: true,
                    accepted: true
                })
            )
            .issue,
            Some(AssessmentIssue::CertifiedBoundUnavailable)
        );
    }
    #[test]
    fn certified_bounds_never_treat_absent_or_crossing_error_as_zero() {
        let definition = assessment(AssessmentEvidenceRequirement::CertifiedBound);
        assert_eq!(
            evaluate_assessment(&definition, Some(measure(0.0009))).issue,
            Some(AssessmentIssue::CertifiedBoundUnavailable)
        );
        let mut uncertain = measure(0.0009);
        uncertain.absolute_error_bound = Some(0.0002);
        assert_eq!(
            evaluate_assessment(&definition, Some(uncertain)).issue,
            Some(AssessmentIssue::ComparisonUncertain)
        );
        uncertain.absolute_error_bound = Some(0.00005);
        assert_eq!(
            evaluate_assessment(&definition, Some(uncertain)).status,
            AssessmentStatus::Passed
        );
    }
    #[test]
    fn incompatible_threshold_is_rejected() {
        let outputs = BTreeMap::from([(
            "peak".into(),
            AuthoredOutputType::SpatialExtremum {
                value: AuthoredOutputValue::Physical(crate::ObservationQuantity::Displacement),
            },
        )]);
        let mut definition = assessment(AssessmentEvidenceRequirement::AuthoredSamples);
        validate_assessments(&[definition.clone()], &outputs).unwrap();
        definition.rule = AssessmentRule::Maximum {
            bound: AssessmentQuantity::try_from_descriptor(
                (PhysicalDimension::DIMENSIONLESS).into(),
                1.0,
            )
            .expect("checked finite assessment quantity"),
        };
        assert!(validate_assessments(&[definition], &outputs).is_err());
    }

    #[test]
    fn invalid_values_and_overflow_remain_serializable_indeterminate_results() {
        assert!(
            AssessmentQuantity::try_from_dimension(PhysicalDimension::LENGTH, None, f64::NAN)
                .is_err()
        );
        assert!(AssessmentQuantity::try_from_dimension(
            PhysicalDimension::LENGTH,
            None,
            f64::INFINITY
        )
        .is_err());
        let mut overflow = measure(f64::MAX);
        overflow.absolute_error_bound = Some(f64::MAX);
        assert_eq!(
            evaluate_assessment(
                &assessment(AssessmentEvidenceRequirement::CertifiedBound),
                Some(overflow)
            )
            .issue,
            Some(AssessmentIssue::ComparisonUncertain)
        );
    }
}
