use crate::{
    AnalysisProcedure, AnalysisValidationError, CompressionPolicy, IndependentDomain,
    ObservationEvent, ObservationPurpose, ObservationRequest, ObservationSpec, PartialAccessPolicy,
    ProcedureStep, RetentionPolicy, SamplingPolicy, SeriesReduction, SolverObservationEvent,
    SpatialReduction,
};

use super::observation_error;

pub(super) fn validate_sampling(
    step: &ProcedureStep,
    request: &ObservationRequest,
    sampling: &SamplingPolicy,
) -> Result<(), AnalysisValidationError> {
    let invalid = match sampling {
        SamplingPolicy::Final => false,
        SamplingPolicy::EveryAcceptedIncrement => {
            matches!(
                step.procedure,
                AnalysisProcedure::Static
                    | AnalysisProcedure::ModalExtraction
                    | AnalysisProcedure::PrestressedModalExtraction
                    | AnalysisProcedure::EigenvalueBuckling
            ) || !observation_sampling_matches_domain(&step.independent_domain, sampling)
        }
        SamplingPolicy::TimeSeconds(_)
        | SamplingPolicy::TimeIntervalSeconds { .. }
        | SamplingPolicy::FrequencyHertz(_)
        | SamplingPolicy::FrequencyIntervalHertz { .. }
        | SamplingPolicy::LoadFactors(_)
        | SamplingPolicy::LoadFactorInterval { .. }
        | SamplingPolicy::StudyParameterValues(_)
        | SamplingPolicy::ModeIndices(_) => {
            !observation_sampling_matches_domain(&step.independent_domain, sampling)
        }
        SamplingPolicy::Event(_) => {
            !observation_sampling_matches_domain(&step.independent_domain, sampling)
        }
        SamplingPolicy::Adaptive {
            minimum_interval,
            maximum_samples,
            relative_change,
        } => {
            !observation_sampling_matches_domain(&step.independent_domain, sampling)
                || !minimum_interval.is_finite()
                || *minimum_interval <= 0.0
                || *maximum_samples == 0
                || !relative_change.is_finite()
                || *relative_change <= 0.0
        }
    };
    if invalid {
        observation_error(
            request,
            "sampling policy is invalid for the step's independent domain",
        )
    } else {
        Ok(())
    }
}

pub fn observation_sampling_matches_domain(
    domain: &IndependentDomain,
    sampling: &SamplingPolicy,
) -> bool {
    match (domain, sampling) {
        (_, SamplingPolicy::Final) => true,
        (
            IndependentDomain::TimeSeconds { .. }
            | IndependentDomain::FrequencyHertz(_)
            | IndependentDomain::LoadFactor { .. }
            | IndependentDomain::ResponseSpectra { .. },
            SamplingPolicy::EveryAcceptedIncrement | SamplingPolicy::Adaptive { .. },
        ) => true,
        (
            IndependentDomain::TimeSeconds { .. } | IndependentDomain::LoadFactor { .. },
            SamplingPolicy::Event(ObservationEvent::Solver(
                SolverObservationEvent::FirstYield | SolverObservationEvent::ContactStatusChange,
            )),
        ) => true,
        (
            domain,
            SamplingPolicy::Event(
                ObservationEvent::Solver(SolverObservationEvent::ConvergenceFailure)
                | ObservationEvent::Series(_)
                | ObservationEvent::Accuracy(_),
            ),
        ) => !matches!(domain, IndependentDomain::Static),
        (
            IndependentDomain::TimeSeconds { start, end, .. },
            SamplingPolicy::TimeSeconds(values),
        ) => {
            strictly_increasing_finite(values, true)
                && values.first().is_some_and(|value| value >= start)
                && values.last().is_some_and(|value| value <= end)
        }
        (
            IndependentDomain::TimeSeconds { .. },
            SamplingPolicy::TimeIntervalSeconds {
                start,
                end,
                interval,
            },
        ) => {
            valid_interval(*start, *end, *interval, true)
                && interval_within_domain(domain, *start, *end)
        }
        (
            IndependentDomain::FrequencyHertz(domain_values),
            SamplingPolicy::FrequencyHertz(values),
        ) => {
            strictly_increasing_finite(values, true)
                && values.iter().all(|value| domain_values.contains(value))
        }
        (
            IndependentDomain::FrequencyHertz(_),
            SamplingPolicy::FrequencyIntervalHertz {
                start,
                end,
                interval,
            },
        ) => {
            valid_interval(*start, *end, *interval, true)
                && interval_within_domain(domain, *start, *end)
        }
        (IndependentDomain::LoadFactor { .. }, SamplingPolicy::LoadFactors(values)) => {
            strictly_increasing_finite(values, false)
                && values
                    .first()
                    .zip(values.last())
                    .is_some_and(|(first, last)| interval_within_domain(domain, *first, *last))
        }
        (
            IndependentDomain::LoadFactor { .. },
            SamplingPolicy::LoadFactorInterval {
                start,
                end,
                interval,
            },
        ) => {
            valid_interval(*start, *end, *interval, false)
                && interval_within_domain(domain, *start, *end)
        }
        (
            IndependentDomain::Eigenvalue { requested_count },
            SamplingPolicy::ModeIndices(values),
        ) => {
            !values.is_empty()
                && !values.contains(&0)
                && values.windows(2).all(|pair| pair[0] < pair[1])
                && values.last().is_some_and(|value| value <= requested_count)
        }
        (
            IndependentDomain::StudyParameter {
                canonical_values, ..
            },
            SamplingPolicy::StudyParameterValues(values),
        ) => {
            strictly_increasing_finite(values, false)
                && values.iter().all(|value| canonical_values.contains(value))
        }
        _ => false,
    }
}

fn strictly_increasing_finite(values: &[f64], nonnegative: bool) -> bool {
    !values.is_empty()
        && values
            .iter()
            .all(|value| value.is_finite() && (!nonnegative || *value >= 0.0))
        && values.windows(2).all(|pair| pair[0] < pair[1])
}

fn valid_interval(start: f64, end: f64, interval: f64, nonnegative: bool) -> bool {
    start.is_finite()
        && end.is_finite()
        && interval.is_finite()
        && interval > 0.0
        && end >= start
        && (!nonnegative || start >= 0.0)
}

fn interval_within_domain(domain: &IndependentDomain, start: f64, end: f64) -> bool {
    match domain {
        IndependentDomain::TimeSeconds {
            start: domain_start,
            end: domain_end,
            ..
        }
        | IndependentDomain::LoadFactor {
            start: domain_start,
            end: domain_end,
        } => start >= *domain_start && end <= *domain_end,
        IndependentDomain::FrequencyHertz(values) => values
            .first()
            .zip(values.last())
            .is_some_and(|(domain_start, domain_end)| start >= *domain_start && end <= *domain_end),
        _ => false,
    }
}

pub(super) fn validate_reduction(
    step: &ProcedureStep,
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    if spec.reduction.as_ref().is_some_and(|value| {
        matches!(
            value.spatial,
            SpatialReduction::Moment { about } if !about.is_finite()
        )
    }) {
        return observation_error(request, "moment reference position must be finite");
    }
    let has_reduction = spec.reduction.as_ref().is_some_and(|value| {
        !matches!(value.spatial, SpatialReduction::None)
            || !matches!(value.series, SeriesReduction::None)
    });
    if spec
        .reduction
        .as_ref()
        .and_then(|value| value.series.weighting())
        == Some(crate::SeriesWeighting::TrapezoidalIndependentDomain)
        && !matches!(
            step.independent_domain,
            IndependentDomain::TimeSeconds { .. }
                | IndependentDomain::LoadFactor { .. }
                | IndependentDomain::FrequencyHertz(_)
                | IndependentDomain::StudyParameter { .. }
        )
    {
        return observation_error(
            request,
            "trapezoidal series weighting requires a continuous independent domain",
        );
    }
    if matches!(spec.sampling, SamplingPolicy::Event(_))
        && spec
            .reduction
            .as_ref()
            .is_some_and(|value| value.series != SeriesReduction::None)
    {
        return observation_error(
            request,
            "series reductions require a history with a guaranteed nonempty sample inventory",
        );
    }
    match request {
        ObservationRequest::Reduction(_) if !has_reduction => observation_error(
            request,
            "a reduction request must specify a spatial or series reduction",
        ),
        ObservationRequest::Field(_) if spec.reduction.is_some() => {
            observation_error(request, "field requests cannot apply reductions")
        }
        ObservationRequest::Field(_)
        | ObservationRequest::Probe(_)
        | ObservationRequest::Reduction(_)
            if spec
                .reduction
                .as_ref()
                .is_some_and(|value| value.series != SeriesReduction::None) =>
        {
            observation_error(request, "series reductions require a history request")
        }
        _ => Ok(()),
    }
}

pub(super) fn validate_storage(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    validate_retention(request, &spec.storage.retention)?;
    if let CompressionPolicy::ErrorBounded {
        maximum_relative_error,
    } = spec.storage.compression
    {
        if !maximum_relative_error.is_finite()
            || maximum_relative_error <= 0.0
            || maximum_relative_error >= 1.0
        {
            return observation_error(
                request,
                "compression error bound must be finite, positive, and less than one",
            );
        }
        if matches!(
            spec.purpose,
            ObservationPurpose::SolverControl | ObservationPurpose::DownstreamState
        ) {
            return observation_error(
                request,
                "error-bounded compression cannot feed solver control or downstream state",
            );
        }
    }
    if matches!(
        spec.storage.partial_access,
        PartialAccessPolicy::ManifestAndChunks
    ) && spec
        .cost
        .upper_bound()
        .is_some_and(|cost| cost.stored_bytes == 0)
    {
        return observation_error(
            request,
            "chunk-addressable output must estimate nonzero stored bytes",
        );
    }
    if spec
        .cost
        .upper_bound()
        .is_some_and(|cost| cost.stored_bytes == 0)
        && matches!(
            &spec.storage.retention,
            RetentionPolicy::Canonical | RetentionPolicy::UntilDependentsComplete
        )
    {
        return observation_error(
            request,
            "retained output must estimate nonzero stored bytes",
        );
    }
    Ok(())
}

pub(super) fn validate_retention(
    request: &ObservationRequest,
    retention: &RetentionPolicy,
) -> Result<(), AnalysisValidationError> {
    let valid = match retention {
        RetentionPolicy::LatestSamples { count } => *count > 0,
        RetentionPolicy::ExplicitSamples { maximum_count } => *maximum_count > 0,
        RetentionPolicy::Canonical | RetentionPolicy::UntilDependentsComplete => true,
    };
    if valid {
        Ok(())
    } else {
        observation_error(request, "retention count must be nonzero")
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        IndependentDomain, ObservationEvent, SamplingPolicy, SeriesObservationEvent,
        SolverObservationEvent,
    };

    use super::{observation_sampling_matches_domain, valid_interval};

    #[test]
    fn intervals_require_finite_ordered_positive_bounds() {
        assert!(valid_interval(0.0, 1.0, 0.1, true));
        assert!(valid_interval(1.0, 1.0, 0.1, true));
        assert!(!valid_interval(-1.0, 1.0, 0.1, true));
        assert!(!valid_interval(1.0, 0.0, 0.1, false));
        assert!(!valid_interval(0.0, 1.0, 0.0, true));
        assert!(!valid_interval(0.0, f64::INFINITY, 0.1, true));
    }

    #[test]
    fn explicit_samples_must_remain_inside_the_independent_domain() {
        let time = IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 1.0,
            interval_count: 10,
        };
        assert!(observation_sampling_matches_domain(
            &time,
            &SamplingPolicy::TimeSeconds(vec![0.0, 0.5, 1.0]),
        ));
        assert!(!observation_sampling_matches_domain(
            &time,
            &SamplingPolicy::TimeSeconds(vec![0.0, 1.5]),
        ));
        let modes = IndependentDomain::Eigenvalue { requested_count: 3 };
        assert!(!observation_sampling_matches_domain(
            &modes,
            &SamplingPolicy::ModeIndices(vec![1, 4]),
        ));
        let parameter = IndependentDomain::StudyParameter {
            parameter_id: "damping_ratio".into(),
            canonical_values: vec![0.01, 0.02, 0.05],
        };
        assert!(observation_sampling_matches_domain(
            &parameter,
            &SamplingPolicy::StudyParameterValues(vec![0.01, 0.05]),
        ));
        assert!(!observation_sampling_matches_domain(
            &parameter,
            &SamplingPolicy::StudyParameterValues(vec![0.01, 0.03]),
        ));
    }

    #[test]
    fn solver_transition_events_require_a_physical_progress_domain() {
        let first_yield =
            SamplingPolicy::Event(ObservationEvent::Solver(SolverObservationEvent::FirstYield));
        let time = IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 1.0,
            interval_count: 10,
        };
        assert!(observation_sampling_matches_domain(&time, &first_yield));
        assert!(observation_sampling_matches_domain(
            &IndependentDomain::LoadFactor {
                start: 0.0,
                end: 1.0,
            },
            &first_yield,
        ));
        assert!(!observation_sampling_matches_domain(
            &IndependentDomain::FrequencyHertz(vec![10.0]),
            &first_yield,
        ));
        assert!(!observation_sampling_matches_domain(
            &IndependentDomain::Static,
            &SamplingPolicy::Event(ObservationEvent::Series(
                SeriesObservationEvent::PeakResponse,
            )),
        ));
    }
}
