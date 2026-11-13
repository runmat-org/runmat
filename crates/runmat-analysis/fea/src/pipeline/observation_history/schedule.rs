//! Counted original history schedules; interval declarations expand only after heap admission.
use crate::results::{
    field::RecoveredFieldSample,
    observation::{exact_scheduled_coordinate_at, exact_scheduled_sample_count, ObservationError},
};
use runmat_analysis_core::{ObservationRequest, ProcedureStep, SamplingPolicy};

pub(crate) fn required_time_history_samples(
    step: &ProcedureStep,
) -> Result<Vec<f64>, ObservationError> {
    required_scalar_history_samples(
        step,
        ScalarHistoryDomain::TimeSeconds,
        None,
        step.accuracy.resources.resident_byte_budget(),
    )
}

pub(crate) fn required_bounded_time_history_samples(
    step: &ProcedureStep,
    maximum: usize,
) -> Result<Vec<f64>, ObservationError> {
    required_bounded_time_history_samples_admitted(
        step,
        maximum,
        step.accuracy.resources.resident_byte_budget(),
    )
}

pub(crate) fn required_bounded_time_history_samples_admitted(
    step: &ProcedureStep,
    maximum: usize,
    maximum_resident_bytes: u64,
) -> Result<Vec<f64>, ObservationError> {
    required_scalar_history_samples(
        step,
        ScalarHistoryDomain::TimeSeconds,
        Some(maximum),
        maximum_resident_bytes,
    )
}

pub(crate) fn required_load_factor_history_samples(
    step: &ProcedureStep,
) -> Result<Vec<f64>, ObservationError> {
    required_scalar_history_samples(
        step,
        ScalarHistoryDomain::LoadFactor,
        None,
        step.accuracy.resources.resident_byte_budget(),
    )
}

pub(crate) fn required_mode_history_indices(
    step: &ProcedureStep,
) -> Result<Vec<usize>, ObservationError> {
    let coordinates = required_history_coordinates(step)?;
    coordinates
        .into_iter()
        .map(|coordinate| match coordinate {
            RecoveredFieldSample::ModeIndex(index) => Ok(index),
            _ => Err(ObservationError::Invalid(
                "modal history sampling differs from the eigenvalue domain".into(),
            )),
        })
        .collect()
}

#[derive(Clone, Copy)]
enum ScalarHistoryDomain {
    TimeSeconds,
    LoadFactor,
}

fn required_scalar_history_samples(
    step: &ProcedureStep,
    domain: ScalarHistoryDomain,
    maximum: Option<usize>,
    maximum_resident_bytes: u64,
) -> Result<Vec<f64>, ObservationError> {
    let coordinates = required_history_coordinates_bounded(step, maximum, maximum_resident_bytes)?;
    let mut samples = crate::topology::storage::reserve(coordinates.len())
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    for coordinate in coordinates {
        let value = match (domain, coordinate) {
            (ScalarHistoryDomain::TimeSeconds, RecoveredFieldSample::TimeSeconds(value))
            | (ScalarHistoryDomain::LoadFactor, RecoveredFieldSample::LoadFactor(value)) => {
                Ok(value)
            }
            (ScalarHistoryDomain::TimeSeconds, _) => Err(ObservationError::Invalid(
                "transient history sampling differs from the time domain".into(),
            )),
            (ScalarHistoryDomain::LoadFactor, _) => Err(ObservationError::Invalid(
                "nonlinear history sampling differs from the load-factor domain".into(),
            )),
        }?;
        samples.push(value);
    }
    samples.sort_by(f64::total_cmp);
    samples.dedup_by(|left, right| left.to_bits() == right.to_bits());
    Ok(samples)
}

fn required_history_coordinates(
    step: &ProcedureStep,
) -> Result<Vec<RecoveredFieldSample>, ObservationError> {
    required_history_coordinates_bounded(step, None, step.accuracy.resources.resident_byte_budget())
}

fn required_history_coordinates_bounded(
    step: &ProcedureStep,
    maximum: Option<usize>,
    maximum_resident_bytes: u64,
) -> Result<Vec<RecoveredFieldSample>, ObservationError> {
    let count = coordinate_count(step)?;
    let bytes = coordinate_resident_bytes(count)?;
    if bytes > maximum_resident_bytes {
        return Err(ObservationError::Limit("history schedule resident bytes"));
    }
    let mut samples = crate::topology::storage::reserve(count)
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    for request in &step.observations {
        let ObservationRequest::History(spec) = request else {
            continue;
        };
        let count = history_sample_count(&spec.sampling)?;
        for ordinal in 0..count {
            if ordinal % 1024 == 0 && crate::is_fea_cancelled() {
                return Err(ObservationError::Limit("history schedule cancelled"));
            }
            let coordinate =
                exact_scheduled_coordinate_at(&spec.sampling, ordinal)?.ok_or_else(|| {
                    ObservationError::Invalid("exact history lacks coordinate".into())
                })?;
            samples.push(coordinate);
        }
    }
    samples.sort_unstable_by(compare_coordinates);
    samples.dedup();
    if maximum.is_some_and(|maximum| samples.len() > maximum) {
        return Err(ObservationError::Invalid(
            "combined history sample schedule exceeds the solve sample ceiling".into(),
        ));
    }
    Ok(samples)
}
fn history_sample_count(sampling: &SamplingPolicy) -> Result<usize, ObservationError> {
    match sampling {
        SamplingPolicy::EveryAcceptedIncrement | SamplingPolicy::Event(_) => Ok(0),
        SamplingPolicy::Adaptive { .. } => Err(ObservationError::Unsupported(
            "adaptive history sampling requires response-aware sample selection",
        )),
        SamplingPolicy::Final => Err(ObservationError::Invalid(
            "history requests require a sampled series rather than one final state".into(),
        )),
        _ => exact_scheduled_sample_count(sampling)?.ok_or_else(|| {
            ObservationError::Invalid("exact history did not produce an exact schedule".into())
        }),
    }
}
fn coordinate_count(step: &ProcedureStep) -> Result<usize, ObservationError> {
    let mut count = 0usize;
    for request in &step.observations {
        let ObservationRequest::History(spec) = request else {
            continue;
        };
        let samples = history_sample_count(&spec.sampling)?;
        if samples > step.accuracy.resources.maximum_samples {
            return Err(ObservationError::Limit("history produced sample count"));
        }
        count = count
            .checked_add(samples)
            .ok_or(ObservationError::Limit("history schedule count"))?;
    }
    Ok(count)
}
/// Conservative requested heap for original tagged coordinates and scalar conversion.
pub(crate) fn history_schedule_construction_resident_bytes(
    step: &ProcedureStep,
) -> Result<u64, ObservationError> {
    coordinate_resident_bytes(coordinate_count(step)?)
}

fn coordinate_resident_bytes(count: usize) -> Result<u64, ObservationError> {
    use crate::topology::storage::{add, slots};
    // Original tagged coordinates coexist with the scalar output during conversion.
    slots::<RecoveredFieldSample>(count)
        .and_then(|a| slots::<f64>(count).and_then(|b| add(a, b)))
        .map_err(|error| ObservationError::Invalid(error.to_string()))
}

fn compare_coordinates(
    left: &RecoveredFieldSample,
    right: &RecoveredFieldSample,
) -> std::cmp::Ordering {
    match (*left, *right) {
        (RecoveredFieldSample::TimeSeconds(left), RecoveredFieldSample::TimeSeconds(right))
        | (
            RecoveredFieldSample::FrequencyHertz(left),
            RecoveredFieldSample::FrequencyHertz(right),
        )
        | (RecoveredFieldSample::LoadFactor(left), RecoveredFieldSample::LoadFactor(right))
        | (
            RecoveredFieldSample::StudyParameterValue(left),
            RecoveredFieldSample::StudyParameterValue(right),
        ) => left.total_cmp(&right),
        (RecoveredFieldSample::ModeIndex(left), RecoveredFieldSample::ModeIndex(right)) => {
            left.cmp(&right)
        }
        (left, right) => coordinate_rank(left).cmp(&coordinate_rank(right)),
    }
}

fn coordinate_rank(coordinate: RecoveredFieldSample) -> u8 {
    match coordinate {
        RecoveredFieldSample::Final => 0,
        RecoveredFieldSample::TimeSeconds(_) => 1,
        RecoveredFieldSample::FrequencyHertz(_) => 2,
        RecoveredFieldSample::LoadFactor(_) => 3,
        RecoveredFieldSample::StudyParameterValue(_) => 4,
        RecoveredFieldSample::ModeIndex(_) => 5,
    }
}

#[cfg(test)]
mod tests;
