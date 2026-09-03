use crate::{
    AnalysisProcedure, AnalysisValidationError, CompressionPolicy, IndependentDomain,
    ObservationPurpose, ObservationRequest, ObservationSpec, PartialAccessPolicy, ProcedureStep,
    RetentionPolicy, SamplingPolicy, SeriesReduction, SpatialReduction,
};

use super::observation_error;

pub(super) fn validate_sampling(
    step: &ProcedureStep,
    request: &ObservationRequest,
    sampling: &SamplingPolicy,
) -> Result<(), AnalysisValidationError> {
    let invalid = match sampling {
        SamplingPolicy::Final => false,
        SamplingPolicy::EveryAcceptedIncrement => matches!(
            step.procedure,
            AnalysisProcedure::Static
                | AnalysisProcedure::ModalExtraction
                | AnalysisProcedure::PrestressedModalExtraction
                | AnalysisProcedure::EigenvalueBuckling
        ),
        SamplingPolicy::TimeSeconds(values) => {
            !matches!(
                &step.independent_domain,
                IndependentDomain::TimeSeconds { .. }
            ) || !strictly_increasing_finite(values, true)
        }
        SamplingPolicy::TimeIntervalSeconds {
            start,
            end,
            interval,
        } => {
            !matches!(
                &step.independent_domain,
                IndependentDomain::TimeSeconds { .. }
            ) || !valid_interval(*start, *end, *interval, true)
        }
        SamplingPolicy::FrequencyHertz(values) => {
            !matches!(
                &step.independent_domain,
                IndependentDomain::FrequencyHertz(_)
            ) || !strictly_increasing_finite(values, true)
        }
        SamplingPolicy::FrequencyIntervalHertz {
            start,
            end,
            interval,
        } => {
            !matches!(
                &step.independent_domain,
                IndependentDomain::FrequencyHertz(_)
            ) || !valid_interval(*start, *end, *interval, true)
        }
        SamplingPolicy::LoadFactors(values) => {
            !matches!(
                &step.independent_domain,
                IndependentDomain::LoadFactor { .. }
            ) || !strictly_increasing_finite(values, false)
        }
        SamplingPolicy::LoadFactorInterval {
            start,
            end,
            interval,
        } => {
            !matches!(
                &step.independent_domain,
                IndependentDomain::LoadFactor { .. }
            ) || !valid_interval(*start, *end, *interval, false)
        }
        SamplingPolicy::ModeIndices(values) => {
            !matches!(
                &step.independent_domain,
                IndependentDomain::Eigenvalue { .. }
            ) || values.is_empty()
                || values.contains(&0)
                || values.windows(2).any(|pair| pair[0] >= pair[1])
        }
        SamplingPolicy::Event(_) => matches!(&step.independent_domain, IndependentDomain::Static),
        SamplingPolicy::Adaptive {
            minimum_interval,
            maximum_samples,
            relative_change,
        } => {
            !minimum_interval.is_finite()
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

pub(super) fn validate_reduction(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), AnalysisValidationError> {
    let has_reduction = spec.reduction.as_ref().is_some_and(|value| {
        !matches!(value.spatial, SpatialReduction::None)
            || !matches!(value.series, SeriesReduction::None)
    });
    match request {
        ObservationRequest::Reduction(_) if !has_reduction => observation_error(
            request,
            "a reduction request must specify a spatial or series reduction",
        ),
        ObservationRequest::Field(_) if spec.reduction.is_some() => {
            observation_error(request, "field requests cannot apply reductions")
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
    ) && spec.cost.stored_bytes == 0
    {
        return observation_error(
            request,
            "chunk-addressable output must estimate nonzero stored bytes",
        );
    }
    if spec.cost.stored_bytes == 0
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
    use super::valid_interval;

    #[test]
    fn intervals_require_finite_ordered_positive_bounds() {
        assert!(valid_interval(0.0, 1.0, 0.1, true));
        assert!(valid_interval(1.0, 1.0, 0.1, true));
        assert!(!valid_interval(-1.0, 1.0, 0.1, true));
        assert!(!valid_interval(1.0, 0.0, 0.1, false));
        assert!(!valid_interval(0.0, 1.0, 0.0, true));
        assert!(!valid_interval(0.0, f64::INFINITY, 0.1, true));
    }
}
