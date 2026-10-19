use runmat_analysis_core::{IndependentDomain, RetentionPolicy, SamplingPolicy};

use crate::results::field::RecoveredFieldSample;

use super::super::contract::{ObservationError, ObservationSampleBounds};
use super::attribution::ObservationEventAttribution;

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
        | SamplingPolicy::LoadFactors(values)
        | SamplingPolicy::StudyParameterValues(values) => Some(values.len()),
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

pub(crate) fn exact_scheduled_sample_count(
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

pub(crate) fn sample_matches_policy(
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
        | (SamplingPolicy::LoadFactors(values), RecoveredFieldSample::LoadFactor(value))
        | (
            SamplingPolicy::StudyParameterValues(values),
            RecoveredFieldSample::StudyParameterValue(value),
        ) => values.contains(&value),
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

pub(crate) fn sample_matches_history_policy(
    sampling: &SamplingPolicy,
    coordinate: RecoveredFieldSample,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<bool, ObservationError> {
    match (sampling, event_attribution) {
        (SamplingPolicy::Event(expected), Some(attribution)) => {
            attribution.validate()?;
            Ok(attribution.sample() == coordinate && attribution.event() == *expected)
        }
        (SamplingPolicy::Event(_), None) | (_, Some(_)) => Ok(false),
        (_, None) => sample_matches_policy(sampling, coordinate),
    }
}

pub(crate) fn exact_scheduled_coordinate_at(
    sampling: &SamplingPolicy,
    ordinal: usize,
) -> Result<Option<RecoveredFieldSample>, ObservationError> {
    let Some(sample_count) = exact_scheduled_sample_count(sampling)? else {
        return Ok(None);
    };
    if ordinal >= sample_count {
        return Err(ObservationError::Limit("history produced sample count"));
    }
    let coordinate = match sampling {
        SamplingPolicy::TimeSeconds(values) => {
            RecoveredFieldSample::TimeSeconds(normalize_zero(values[ordinal]))
        }
        SamplingPolicy::FrequencyHertz(values) => {
            RecoveredFieldSample::FrequencyHertz(normalize_zero(values[ordinal]))
        }
        SamplingPolicy::LoadFactors(values) => {
            RecoveredFieldSample::LoadFactor(normalize_zero(values[ordinal]))
        }
        SamplingPolicy::StudyParameterValues(values) => {
            RecoveredFieldSample::StudyParameterValue(normalize_zero(values[ordinal]))
        }
        SamplingPolicy::ModeIndices(values) => RecoveredFieldSample::ModeIndex(values[ordinal]),
        SamplingPolicy::TimeIntervalSeconds {
            start, interval, ..
        } => RecoveredFieldSample::TimeSeconds(normalize_zero(*start + ordinal as f64 * *interval)),
        SamplingPolicy::FrequencyIntervalHertz {
            start, interval, ..
        } => RecoveredFieldSample::FrequencyHertz(normalize_zero(
            *start + ordinal as f64 * *interval,
        )),
        SamplingPolicy::LoadFactorInterval {
            start, interval, ..
        } => RecoveredFieldSample::LoadFactor(normalize_zero(*start + ordinal as f64 * *interval)),
        SamplingPolicy::Final
        | SamplingPolicy::EveryAcceptedIncrement
        | SamplingPolicy::Event(_)
        | SamplingPolicy::Adaptive { .. } => {
            return Err(ObservationError::Invalid(
                "history schedule classification is inconsistent".into(),
            ));
        }
    };
    validate_coordinate(coordinate)?;
    Ok(Some(coordinate))
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
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
        (
            IndependentDomain::StudyParameter {
                canonical_values, ..
            },
            RecoveredFieldSample::StudyParameterValue(value),
        ) => canonical_values.contains(&value),
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
        | (RecoveredFieldSample::LoadFactor(left), RecoveredFieldSample::LoadFactor(right))
        | (
            RecoveredFieldSample::StudyParameterValue(left),
            RecoveredFieldSample::StudyParameterValue(right),
        ) => left < right,
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
        | RecoveredFieldSample::StudyParameterValue(value)
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

#[cfg(test)]
mod tests {
    use super::*;
    fn exact_scheduled_coordinates(
        sampling: &SamplingPolicy,
        maximum: usize,
    ) -> Result<Option<Vec<RecoveredFieldSample>>, ObservationError> {
        let Some(count) = exact_scheduled_sample_count(sampling)? else {
            return Ok(None);
        };
        if count > maximum {
            return Err(ObservationError::Limit("history produced sample count"));
        }
        (0..count)
            .map(|i| exact_scheduled_coordinate_at(sampling, i).map(|v| v.unwrap()))
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    #[test]
    fn exact_interval_schedule_uses_only_authored_lattice_points() {
        let coordinates = exact_scheduled_coordinates(
            &SamplingPolicy::TimeIntervalSeconds {
                start: -0.0,
                end: 1.0,
                interval: 0.3,
            },
            4,
        )
        .unwrap()
        .unwrap();
        let values = coordinates
            .iter()
            .map(|coordinate| match coordinate {
                RecoveredFieldSample::TimeSeconds(value) => *value,
                _ => panic!("time interval must retain its coordinate domain"),
            })
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0.0, 0.3, 0.6, 3.0 * 0.3]);
        assert!(values[3] < 1.0);
        assert!(matches!(
            exact_scheduled_coordinates(
                &SamplingPolicy::TimeIntervalSeconds {
                    start: 0.0,
                    end: 1.0,
                    interval: 0.3,
                },
                3,
            ),
            Err(ObservationError::Limit("history produced sample count"))
        ));
    }

    #[test]
    fn exact_schedule_preserves_coordinate_domains() {
        assert_eq!(
            exact_scheduled_coordinates(&SamplingPolicy::ModeIndices(vec![1, 3]), 2).unwrap(),
            Some(vec![
                RecoveredFieldSample::ModeIndex(1),
                RecoveredFieldSample::ModeIndex(3),
            ])
        );
        assert_eq!(
            exact_scheduled_coordinates(&SamplingPolicy::LoadFactors(vec![0.25, 1.0]), 2).unwrap(),
            Some(vec![
                RecoveredFieldSample::LoadFactor(0.25),
                RecoveredFieldSample::LoadFactor(1.0),
            ])
        );
    }
}
