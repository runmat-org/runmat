use std::collections::{HashMap, HashSet};

use crate::{
    AccuracyLimitAction, AccuracyPlan, AccuracyTarget, AccuracyTolerance, AnalysisValidationError,
    ErrorContributionKind, NumericalEquivalencePolicy, ObservationRequest, ProcedureStep,
    RequiredAccuracyEvidence,
};

pub(crate) fn validate_accuracy(step: &ProcedureStep) -> Result<(), AnalysisValidationError> {
    let plan = &step.accuracy;
    let plan_id = plan.accuracy_plan_id.0.as_str();
    if plan_id.trim().is_empty() {
        return accuracy_error(plan, "accuracy plan id must be non-empty");
    }
    if plan.criteria.is_empty() {
        return accuracy_error(plan, "accuracy plan must contain at least one criterion");
    }
    if !plan.criteria.iter().any(|criterion| criterion.mandatory) {
        return accuracy_error(
            plan,
            "accuracy plan must contain at least one mandatory criterion",
        );
    }

    let observations: HashMap<_, _> = step
        .observations
        .iter()
        .map(|request| (request.observation_id().0.as_str(), request))
        .collect();
    let mut criterion_ids = HashSet::new();
    for criterion in &plan.criteria {
        if criterion.criterion_id.trim().is_empty() {
            return accuracy_error(plan, "criterion id must be non-empty");
        }
        if !criterion_ids.insert(criterion.criterion_id.as_str()) {
            return accuracy_error(plan, "criterion ids must be unique within an accuracy plan");
        }

        let target_observation = match &criterion.target {
            AccuracyTarget::Observation(id) => {
                let Some(request) = observations.get(id.0.as_str()) else {
                    return accuracy_error(
                        plan,
                        format!(
                            "criterion `{}` references an observation outside its step",
                            criterion.criterion_id
                        ),
                    );
                };
                let Some(spec) = request.spec() else {
                    return accuracy_error(plan, format!("criterion `{}` must target a numeric observation, not a diagnostic or monitor", criterion.criterion_id));
                };
                Some(spec)
            }
            _ => None,
        };
        if let AccuracyTarget::Quantity(quantity) = &criterion.target {
            if !crate::validate_outputs::spec::quantity_supported(step, *quantity) {
                return accuracy_error(
                    plan,
                    format!(
                        "criterion `{}` targets a quantity outside the step's physics family",
                        criterion.criterion_id
                    ),
                );
            }
        }

        match criterion.tolerance {
            AccuracyTolerance::Relative(value) if !positive_finite(value) => {
                return accuracy_error(
                    plan,
                    format!(
                        "criterion `{}` has an invalid relative tolerance",
                        criterion.criterion_id
                    ),
                );
            }
            AccuracyTolerance::Absolute {
                quantity,
                canonical_value,
            } => {
                if !positive_finite(canonical_value) {
                    return accuracy_error(
                        plan,
                        format!(
                            "criterion `{}` has an invalid absolute tolerance",
                            criterion.criterion_id
                        ),
                    );
                }
                if let AccuracyTarget::Quantity(target_quantity) = &criterion.target {
                    if quantity != *target_quantity {
                        return accuracy_error(plan, format!("criterion `{}` absolute tolerance quantity does not match its target", criterion.criterion_id));
                    }
                }
                if target_observation.is_some_and(|spec| spec.quantity != quantity) {
                    return accuracy_error(plan, format!("criterion `{}` absolute tolerance quantity does not match its observation", criterion.criterion_id));
                }
            }
            AccuracyTolerance::Relative(_) => {}
        }

        if criterion.target == AccuracyTarget::NumericalResult
            && !matches!(criterion.tolerance, AccuracyTolerance::Relative(_))
        {
            return accuracy_error(
                plan,
                format!(
                    "criterion `{}` targets a whole numerical result and requires a dimensionless relative tolerance",
                    criterion.criterion_id
                ),
            );
        }

        validate_target_contribution(plan, criterion)?;
        validate_evidence(plan, criterion)?;
    }

    validate_equivalence(plan)?;
    validate_resources(plan, step)?;
    validate_limit_actions(plan)?;
    Ok(())
}

fn validate_target_contribution(
    plan: &AccuracyPlan,
    criterion: &crate::AccuracyCriterion,
) -> Result<(), AnalysisValidationError> {
    let compatible = match &criterion.target {
        AccuracyTarget::ExecutionEquivalence => {
            matches!(
                criterion.contribution,
                ErrorContributionKind::ExecutionEquivalence
            )
        }
        AccuracyTarget::GlobalEquilibrium => matches!(
            criterion.contribution,
            ErrorContributionKind::Algebraic
                | ErrorContributionKind::Nonlinear
                | ErrorContributionKind::Coupling
        ),
        AccuracyTarget::InterfaceClosure => matches!(
            criterion.contribution,
            ErrorContributionKind::Coupling
                | ErrorContributionKind::InterfaceProjection
                | ErrorContributionKind::FieldTransfer
        ),
        AccuracyTarget::NumericalResult => !matches!(
            criterion.contribution,
            ErrorContributionKind::ObservationSampling
                | ErrorContributionKind::ExecutionEquivalence
        ),
        AccuracyTarget::Observation(_) | AccuracyTarget::Quantity(_) => !matches!(
            criterion.contribution,
            ErrorContributionKind::ExecutionEquivalence
        ),
    };
    if compatible {
        Ok(())
    } else {
        accuracy_error(
            plan,
            format!(
                "criterion `{}` contribution is incompatible with its target",
                criterion.criterion_id
            ),
        )
    }
}

fn validate_evidence(
    plan: &AccuracyPlan,
    criterion: &crate::AccuracyCriterion,
) -> Result<(), AnalysisValidationError> {
    use ErrorContributionKind as C;
    use RequiredAccuracyEvidence as E;
    let compatible = match criterion.contribution {
        C::GeometryAdmission | C::BoundaryApproximation | C::SpatialDiscretization => matches!(
            criterion.evidence,
            E::APosterioriEstimator | E::ConvergenceStudy | E::ReferenceComparison
        ),
        C::Algebraic | C::Nonlinear => matches!(
            criterion.evidence,
            E::ResidualBound
                | E::BackwardErrorBound
                | E::ConvergenceStudy
                | E::ConservationBalance
                | E::ReferenceComparison
        ),
        C::Temporal | C::FrequencySampling | C::ModalTruncation | C::ObservationSampling => {
            matches!(
                criterion.evidence,
                E::ConvergenceStudy | E::ReferenceComparison
            )
        }
        C::Stochastic => matches!(
            criterion.evidence,
            E::StatisticalConfidence | E::ReferenceComparison
        ),
        C::Coupling => matches!(
            criterion.evidence,
            E::ConservationBalance
                | E::TransferBound
                | E::ConvergenceStudy
                | E::ReferenceComparison
        ),
        C::Recovery => matches!(
            criterion.evidence,
            E::APosterioriEstimator
                | E::TransferBound
                | E::ConvergenceStudy
                | E::ReferenceComparison
        ),
        C::InterfaceProjection | C::FieldTransfer | C::SignalInterpolation => {
            matches!(
                criterion.evidence,
                E::TransferBound | E::ConvergenceStudy | E::ReferenceComparison
            )
        }
        C::Precision | C::ExecutionEquivalence => {
            matches!(
                criterion.evidence,
                E::ReferenceComparison | E::ResidualBound | E::BackwardErrorBound
            )
        }
    };
    if compatible {
        Ok(())
    } else {
        accuracy_error(
            plan,
            format!(
                "criterion `{}` evidence does not establish its error contribution",
                criterion.criterion_id
            ),
        )
    }
}

fn validate_equivalence(plan: &AccuracyPlan) -> Result<(), AnalysisValidationError> {
    let valid = match plan.equivalence {
        NumericalEquivalencePolicy::Deterministic => true,
        NumericalEquivalencePolicy::RelativeTolerance(value)
        | NumericalEquivalencePolicy::AbsoluteTolerance(value) => positive_finite(value),
    };
    if valid {
        Ok(())
    } else {
        accuracy_error(
            plan,
            "numerical-equivalence tolerance must be finite and positive",
        )
    }
}

fn validate_resources(
    plan: &AccuracyPlan,
    step: &ProcedureStep,
) -> Result<(), AnalysisValidationError> {
    let resources = plan.resources;
    if resources.maximum_samples == 0
        || resources.maximum_wall_time_milliseconds == 0
        || resources.resident_limit() == Some(0)
    {
        return accuracy_error(
            plan,
            "sample, wall-time, and resident-memory resource ceilings must be nonzero",
        );
    }
    for request in &step.observations {
        let (ObservationRequest::Field(spec)
        | ObservationRequest::History(spec)
        | ObservationRequest::Probe(spec)
        | ObservationRequest::Reduction(spec)) = request
        else {
            continue;
        };
        if spec
            .cost
            .upper_bound()
            .is_some_and(|cost| !resources.admits_resident_bytes(cost.peak_resident_bytes))
        {
            return accuracy_error(
                plan,
                format!(
                    "observation `{}` exceeds the accuracy plan's resident-memory ceiling",
                    spec.observation_id.0
                ),
            );
        }
        let requested_samples = match &spec.sampling {
            crate::SamplingPolicy::TimeSeconds(values)
            | crate::SamplingPolicy::FrequencyHertz(values)
            | crate::SamplingPolicy::LoadFactors(values)
            | crate::SamplingPolicy::StudyParameterValues(values) => Some(values.len()),
            crate::SamplingPolicy::ModeIndices(values) => Some(values.len()),
            crate::SamplingPolicy::TimeIntervalSeconds {
                start,
                end,
                interval,
            }
            | crate::SamplingPolicy::FrequencyIntervalHertz {
                start,
                end,
                interval,
            }
            | crate::SamplingPolicy::LoadFactorInterval {
                start,
                end,
                interval,
            } => interval_sample_count(*start, *end, *interval),
            crate::SamplingPolicy::Adaptive {
                maximum_samples, ..
            } => Some(*maximum_samples),
            crate::SamplingPolicy::Final => Some(1),
            crate::SamplingPolicy::EveryAcceptedIncrement | crate::SamplingPolicy::Event(_) => None,
        };
        if requested_samples.is_some_and(|count| count > resources.maximum_samples) {
            return accuracy_error(
                plan,
                format!(
                    "observation `{}` exceeds the accuracy plan's sample ceiling",
                    spec.observation_id.0
                ),
            );
        }
    }
    Ok(())
}

fn validate_limit_actions(plan: &AccuracyPlan) -> Result<(), AnalysisValidationError> {
    let resources = plan.resources;
    let valid = match plan.exceeded_limit {
        AccuracyLimitAction::Refine => resources.maximum_refinement_cycles > 0,
        AccuracyLimitAction::TightenSolver => resources.maximum_solver_tightenings > 0,
        AccuracyLimitAction::ExtendSampling => resources.maximum_samples > 0,
        AccuracyLimitAction::Fail => true,
    };
    if valid {
        Ok(())
    } else {
        accuracy_error(
            plan,
            "accuracy-limit action lacks the required resource allowance",
        )
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn interval_sample_count(start: f64, end: f64, interval: f64) -> Option<usize> {
    if !start.is_finite()
        || !end.is_finite()
        || !interval.is_finite()
        || interval <= 0.0
        || end < start
    {
        return None;
    }
    let intervals = ((end - start) / interval).floor();
    if intervals > (usize::MAX - 1) as f64 {
        Some(usize::MAX)
    } else {
        Some(intervals as usize + 1)
    }
}

fn accuracy_error<T>(
    plan: &AccuracyPlan,
    detail: impl Into<String>,
) -> Result<T, AnalysisValidationError> {
    Err(AnalysisValidationError::InvalidAccuracyPlan {
        plan_id: plan.accuracy_plan_id.0.clone(),
        detail: detail.into(),
    })
}

#[cfg(test)]
mod tests {
    use crate::{AccuracyTolerance, AnalysisValidationError, ObservationId, ProcedureStep};

    use super::validate_accuracy;

    #[test]
    fn a_posteriori_recovery_evidence_is_valid_without_relabeling_transfer() {
        use crate::{ErrorContributionKind as C, RequiredAccuracyEvidence as E};
        let mut step = ProcedureStep::structural_static("recovery");
        step.accuracy.criteria[0].evidence = E::APosterioriEstimator;
        step.accuracy.criteria[0].contribution = C::Recovery;
        step.accuracy.criteria[0].target = crate::AccuracyTarget::NumericalResult;
        validate_accuracy(&step).unwrap();
        for contribution in [
            C::InterfaceProjection,
            C::FieldTransfer,
            C::SignalInterpolation,
        ] {
            step.accuracy.criteria[0].contribution = contribution;
            assert!(matches!(
                validate_accuracy(&step),
                Err(AnalysisValidationError::InvalidAccuracyPlan { .. })
            ));
        }
    }

    #[test]
    fn baseline_is_valid() {
        validate_accuracy(&ProcedureStep::structural_static("structural"))
            .expect("constructor must produce a valid accuracy plan");
    }

    #[test]
    fn rejects_nonpositive_tolerance() {
        let mut step = ProcedureStep::structural_static("structural");
        step.accuracy.criteria[0].tolerance = AccuracyTolerance::Relative(0.0);
        assert!(matches!(
            validate_accuracy(&step),
            Err(AnalysisValidationError::InvalidAccuracyPlan { .. })
        ));
    }

    #[test]
    fn resource_ceiling_decode_and_validation_share_the_wire_bound() {
        use crate::AccuracyResourceCeiling;
        let max = runmat_types::wire_integer::MAX_WIRE_INTEGER;
        let decode = |samples: &str, bytes: &str| {
            serde_json::from_str::<AccuracyResourceCeiling>(&format!(
                r#"{{"maximum_refinement_cycles":0,"maximum_solver_tightenings":0,"maximum_samples":{samples},"maximum_wall_time_milliseconds":1000{bytes}}}"#
            ))
        };
        // Null is "no authored limit"; omission is rejected rather than read as null.
        let unbounded = decode("10", r#","maximum_resident_bytes":null"#).unwrap();
        assert_eq!(unbounded.resident_limit(), None);
        assert!(unbounded.admits_resident_bytes(u64::MAX));
        assert!(decode("10", "").is_err());
        assert_eq!(
            decode("10", r#","maximum_resident_bytes":4096"#)
                .unwrap()
                .resident_limit(),
            Some(4096)
        );
        assert_eq!(
            decode("10", &format!(r#","maximum_resident_bytes":{max}"#))
                .unwrap()
                .resident_limit(),
            Some(max)
        );
        for rejected in [
            (max + 1).to_string(),
            u64::MAX.to_string(),
            "-1".into(),
            "1.5".into(),
        ] {
            assert!(
                decode("10", &format!(r#","maximum_resident_bytes":{rejected}"#)).is_err(),
                "decoded resident bytes {rejected}"
            );
        }
        assert_eq!(
            decode(&max.to_string(), r#","maximum_resident_bytes":null"#)
                .unwrap()
                .maximum_samples,
            usize::try_from(max).unwrap()
        );
        assert!(decode(&(max + 1).to_string(), r#","maximum_resident_bytes":null"#).is_err());
        // Encoding always writes the field (null when unbounded) and refuses values above the bound.
        let encoded = serde_json::to_value(unbounded).unwrap();
        assert_eq!(encoded["maximum_resident_bytes"], serde_json::Value::Null);
        let mut wide = unbounded;
        wide.maximum_resident_bytes = Some(max + 1);
        assert!(serde_json::to_string(&wide).is_err());
        // Validation keeps each field's lower bound: zero bytes or samples are prohibited,
        // zero refinement cycles and solver tightenings are allowed.
        let mut step = ProcedureStep::structural_static("structural");
        for (bytes, valid) in [
            (None, true),
            (Some(max), true),
            (Some(1), true),
            (Some(0), false),
        ] {
            step.accuracy.resources.maximum_resident_bytes = bytes;
            assert_eq!(
                validate_accuracy(&step).is_ok(),
                valid,
                "resident bytes {bytes:?}"
            );
        }
        step.accuracy.resources.maximum_resident_bytes = None;
        step.accuracy.resources.maximum_refinement_cycles = 0;
        step.accuracy.resources.maximum_solver_tightenings = 0;
        validate_accuracy(&step).unwrap();
        step.accuracy.resources.maximum_samples = 0;
        assert!(validate_accuracy(&step).is_err());
    }

    #[test]
    fn rejects_unresolved_observation_target() {
        let mut step = ProcedureStep::structural_static("structural");
        step.accuracy.criteria[0].target =
            crate::AccuracyTarget::Observation(ObservationId("missing".to_string()));
        assert!(matches!(
            validate_accuracy(&step),
            Err(AnalysisValidationError::InvalidAccuracyPlan { .. })
        ));
    }
}
