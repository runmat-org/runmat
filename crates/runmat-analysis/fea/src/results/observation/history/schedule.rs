use runmat_analysis_core::{IndependentDomain, RetentionPolicy, SamplingPolicy};

use crate::results::field::RecoveredFieldSample;

use super::super::contract::{ObservationError, ObservationSampleBounds};

pub(in crate::results::observation) fn resolve_history_sample_bounds(
    sampling: &SamplingPolicy,
    retention: &RetentionPolicy,
    maximum_dynamic_samples: usize,
) -> Result<ObservationSampleBounds, ObservationError> {
    validate_history_policy(sampling, retention)?;
    let scheduled = scheduled_sample_count(sampling)?;
    if maximum_dynamic_samples == 0 {
        return Err(ObservationError::Limit("history dynamic sample count"));
    }
    if scheduled.is_some_and(|scheduled| scheduled > maximum_dynamic_samples) {
        return Err(ObservationError::Limit("history produced sample count"));
    }
    let maximum_produced_samples = scheduled.unwrap_or(maximum_dynamic_samples);
    let maximum_retained_samples = match retention {
        RetentionPolicy::LatestSamples { count } => maximum_produced_samples.min(*count),
        RetentionPolicy::ExplicitSamples { maximum_count } => {
            let scheduled = scheduled.ok_or(ObservationError::Unsupported(
                "explicit retention requires an explicitly bounded sampling schedule",
            ))?;
            if scheduled > *maximum_count {
                return Err(ObservationError::Limit(
                    "explicit history retention sample count",
                ));
            }
            scheduled
        }
        RetentionPolicy::Canonical | RetentionPolicy::UntilDependentsComplete => {
            maximum_produced_samples
        }
    };
    Ok(ObservationSampleBounds {
        maximum_produced_samples,
        maximum_retained_samples,
    })
}

pub(super) fn scheduled_sample_count(
    sampling: &SamplingPolicy,
) -> Result<Option<usize>, ObservationError> {
    let count = match sampling {
        SamplingPolicy::Final => {
            return Err(ObservationError::Invalid(
                "final sampling is not a history".into(),
            ));
        }
        SamplingPolicy::EveryAcceptedIncrement | SamplingPolicy::Event(_) => None,
        SamplingPolicy::TimeSeconds(values)
        | SamplingPolicy::FrequencyHertz(values)
        | SamplingPolicy::LoadFactors(values) => Some(values.len()),
        SamplingPolicy::ModeIndices(values) => Some(values.len()),
        SamplingPolicy::TimeIntervalSeconds {
            start,
            end,
            interval,
        }
        | SamplingPolicy::FrequencyIntervalHertz {
            start,
            end,
            interval,
        }
        | SamplingPolicy::LoadFactorInterval {
            start,
            end,
            interval,
        } => Some(interval_sample_count(*start, *end, *interval)?),
        SamplingPolicy::Adaptive {
            maximum_samples, ..
        } => Some(*maximum_samples),
    };
    if count == Some(0) {
        return Err(ObservationError::Invalid(
            "history sampling schedule must not be empty".into(),
        ));
    }
    Ok(count)
}

pub(super) fn exact_scheduled_sample_count(
    sampling: &SamplingPolicy,
) -> Result<Option<usize>, ObservationError> {
    match sampling {
        SamplingPolicy::Adaptive { .. }
        | SamplingPolicy::EveryAcceptedIncrement
        | SamplingPolicy::Event(_) => Ok(None),
        _ => scheduled_sample_count(sampling),
    }
}

fn interval_sample_count(start: f64, end: f64, interval: f64) -> Result<usize, ObservationError> {
    if !start.is_finite()
        || !end.is_finite()
        || !interval.is_finite()
        || interval <= 0.0
        || end < start
    {
        return Err(ObservationError::Invalid(
            "history interval is nonfinite, reversed, or nonpositive".into(),
        ));
    }
    let ratio = (end - start) / interval;
    let nearest = ratio.round();
    let scale = ratio.abs().max(1.0);
    let steps = if (ratio - nearest).abs() <= 16.0 * f64::EPSILON * scale {
        nearest
    } else {
        ratio.floor()
    };
    if !steps.is_finite() || steps > (usize::MAX - 1) as f64 {
        return Err(ObservationError::Limit("history interval sample count"));
    }
    Ok(steps as usize + 1)
}

pub(super) fn validate_history_policy(
    sampling: &SamplingPolicy,
    retention: &RetentionPolicy,
) -> Result<(), ObservationError> {
    match retention {
        RetentionPolicy::LatestSamples { count } if *count == 0 => Err(ObservationError::Invalid(
            "latest-sample history retention requires a nonzero bound".into(),
        )),
        RetentionPolicy::ExplicitSamples { maximum_count } if *maximum_count == 0 => Err(
            ObservationError::Invalid("explicit history retention requires a nonzero bound".into()),
        ),
        _ => {
            scheduled_sample_count(sampling)?;
            Ok(())
        }
    }
}

pub(in crate::results::observation) fn sample_matches_policy(
    sampling: &SamplingPolicy,
    coordinate: RecoveredFieldSample,
) -> Result<bool, ObservationError> {
    validate_coordinate(coordinate)?;
    Ok(match (sampling, coordinate) {
        (SamplingPolicy::EveryAcceptedIncrement, RecoveredFieldSample::TimeSeconds(_))
        | (SamplingPolicy::EveryAcceptedIncrement, RecoveredFieldSample::LoadFactor(_))
        | (SamplingPolicy::EveryAcceptedIncrement, RecoveredFieldSample::FrequencyHertz(_)) => true,
        (SamplingPolicy::TimeSeconds(values), RecoveredFieldSample::TimeSeconds(value))
        | (SamplingPolicy::FrequencyHertz(values), RecoveredFieldSample::FrequencyHertz(value))
        | (SamplingPolicy::LoadFactors(values), RecoveredFieldSample::LoadFactor(value)) => {
            values.contains(&value)
        }
        (
            SamplingPolicy::TimeIntervalSeconds {
                start,
                end,
                interval,
            },
            RecoveredFieldSample::TimeSeconds(value),
        )
        | (
            SamplingPolicy::FrequencyIntervalHertz {
                start,
                end,
                interval,
            },
            RecoveredFieldSample::FrequencyHertz(value),
        )
        | (
            SamplingPolicy::LoadFactorInterval {
                start,
                end,
                interval,
            },
            RecoveredFieldSample::LoadFactor(value),
        ) => interval_contains(*start, *end, *interval, value),
        (SamplingPolicy::ModeIndices(values), RecoveredFieldSample::ModeIndex(value)) => {
            values.contains(&value)
        }
        (SamplingPolicy::Adaptive { .. }, RecoveredFieldSample::TimeSeconds(_))
        | (SamplingPolicy::Adaptive { .. }, RecoveredFieldSample::LoadFactor(_))
        | (SamplingPolicy::Adaptive { .. }, RecoveredFieldSample::FrequencyHertz(_)) => true,
        (SamplingPolicy::Event(_), _) => {
            return Err(ObservationError::Unsupported(
                "event histories require an attributed event coordinate",
            ));
        }
        _ => false,
    })
}

fn interval_contains(start: f64, end: f64, interval: f64, value: f64) -> bool {
    if value < start || value > end {
        return false;
    }
    let index = ((value - start) / interval).round();
    let expected = start + index * interval;
    let scale = expected.abs().max(value.abs()).max(1.0);
    (expected - value).abs() <= 16.0 * f64::EPSILON * scale
}

pub(in crate::results::observation) fn sample_matches_domain(
    domain: &IndependentDomain,
    coordinate: RecoveredFieldSample,
) -> bool {
    match (domain, coordinate) {
        (
            IndependentDomain::TimeSeconds { start, end, .. },
            RecoveredFieldSample::TimeSeconds(value),
        ) => value >= *start && value <= *end,
        (
            IndependentDomain::FrequencyHertz(values),
            RecoveredFieldSample::FrequencyHertz(value),
        ) => values.contains(&value),
        (IndependentDomain::LoadFactor { start, end }, RecoveredFieldSample::LoadFactor(value)) => {
            value >= *start && value <= *end
        }
        (
            IndependentDomain::Eigenvalue { requested_count },
            RecoveredFieldSample::ModeIndex(index),
        ) => index > 0 && index <= *requested_count,
        _ => false,
    }
}

pub(super) fn coordinate_follows(left: RecoveredFieldSample, right: RecoveredFieldSample) -> bool {
    match (left, right) {
        (RecoveredFieldSample::TimeSeconds(left), RecoveredFieldSample::TimeSeconds(right))
        | (
            RecoveredFieldSample::FrequencyHertz(left),
            RecoveredFieldSample::FrequencyHertz(right),
        )
        | (RecoveredFieldSample::LoadFactor(left), RecoveredFieldSample::LoadFactor(right)) => {
            left < right
        }
        (RecoveredFieldSample::ModeIndex(left), RecoveredFieldSample::ModeIndex(right)) => {
            left < right
        }
        _ => false,
    }
}

pub(super) fn validate_coordinate(
    coordinate: RecoveredFieldSample,
) -> Result<(), ObservationError> {
    match coordinate {
        RecoveredFieldSample::TimeSeconds(value)
        | RecoveredFieldSample::FrequencyHertz(value)
        | RecoveredFieldSample::LoadFactor(value)
            if value.is_finite() && value.to_bits() != (-0.0_f64).to_bits() =>
        {
            Ok(())
        }
        RecoveredFieldSample::ModeIndex(index) if index > 0 => Ok(()),
        _ => Err(ObservationError::Invalid(
            "history sample requires a finite non-final coordinate".into(),
        )),
    }
}
