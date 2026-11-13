use runmat_analysis_core::SignalSampleApproach;
use thiserror::Error;

use crate::execution::{
    StructuralTransientContinuationApproach, StructuralTransientSchedule,
    StructuralTransientScheduleArrival, StructuralTransientSchedulePoint,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransientStepBoundaryKind {
    RequiredSample,
    SignalDiscontinuity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TransientStepBoundary {
    time_s: runmat_types::physical::Time,
    kind: TransientStepBoundaryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum TransientStepBoundaryError {
    #[error("transient time domain is nonfinite or reversed")]
    InvalidDomain,
    #[error("transient step boundary lies outside the finite time domain")]
    OutsideDomain,
}

impl TransientStepBoundary {
    fn approach(self) -> SignalSampleApproach {
        match self.kind {
            TransientStepBoundaryKind::RequiredSample => SignalSampleApproach::At,
            TransientStepBoundaryKind::SignalDiscontinuity => SignalSampleApproach::LeftLimit,
        }
    }
}

pub(crate) fn canonical_transient_step_boundaries(
    signal_discontinuities_s: &[f64],
    required_samples_s: &[f64],
    start_time_s: f64,
    end_time_s: f64,
) -> Result<Vec<TransientStepBoundary>, TransientStepBoundaryError> {
    if !start_time_s.is_finite() || !end_time_s.is_finite() || end_time_s < start_time_s {
        return Err(TransientStepBoundaryError::InvalidDomain);
    }
    let mut boundaries = Vec::with_capacity(
        signal_discontinuities_s
            .len()
            .saturating_add(required_samples_s.len()),
    );
    for (times, kind) in [
        (
            signal_discontinuities_s,
            TransientStepBoundaryKind::SignalDiscontinuity,
        ),
        (
            required_samples_s,
            TransientStepBoundaryKind::RequiredSample,
        ),
    ] {
        for &time_s in times {
            if !time_s.is_finite() || time_s < start_time_s || time_s > end_time_s {
                return Err(TransientStepBoundaryError::OutsideDomain);
            }
            boundaries.push(TransientStepBoundary {
                time_s: runmat_types::physical::Time::new(time_s),
                kind,
            });
        }
    }
    boundaries.sort_by(|left, right| {
        left.time_s
            .canonical_value()
            .total_cmp(&right.time_s.canonical_value())
    });
    let mut canonical: Vec<TransientStepBoundary> = Vec::with_capacity(boundaries.len());
    for boundary in boundaries {
        if let Some(previous) = canonical.last_mut().filter(|previous| {
            previous.time_s.canonical_value().to_bits()
                == boundary.time_s.canonical_value().to_bits()
        }) {
            if boundary.kind == TransientStepBoundaryKind::SignalDiscontinuity {
                previous.kind = boundary.kind;
            }
        } else {
            canonical.push(boundary);
        }
    }
    Ok(canonical)
}

pub(super) fn next_step_boundary(
    current_time_s: f64,
    proposed_dt_s: f64,
    final_time_s: f64,
    step_boundaries: &[TransientStepBoundary],
) -> (f64, SignalSampleApproach) {
    let proposed_end = current_time_s + proposed_dt_s;
    // Repeated additions of a nominal interval accumulate a few ulps over long fixed grids.
    // Snap only within a small roundoff envelope of the final endpoint; authored boundaries
    // remain exact and are still selected below.
    let endpoint_tolerance = 128.0
        * f64::EPSILON
        * final_time_s
            .abs()
            .max(current_time_s.abs())
            .max(proposed_dt_s.abs())
            .max(1.0);
    let proposed_end = if final_time_s - proposed_end <= endpoint_tolerance {
        final_time_s
    } else {
        proposed_end.min(final_time_s)
    };
    match step_boundaries.iter().copied().find(|boundary| {
        boundary.time_s.canonical_value() > current_time_s
            && boundary.time_s.canonical_value() <= proposed_end + endpoint_tolerance
    }) {
        Some(boundary) => (boundary.time_s.canonical_value(), boundary.approach()),
        None => (proposed_end, SignalSampleApproach::At),
    }
}

pub(crate) fn canonical_transient_schedule(
    interval_count: usize,
    start_time_s: f64,
    end_time_s: f64,
    step_boundaries: &[TransientStepBoundary],
) -> Result<StructuralTransientSchedule, String> {
    if interval_count == 0
        || !start_time_s.is_finite()
        || !end_time_s.is_finite()
        || end_time_s <= start_time_s
    {
        return Err("a transient integration schedule requires a finite positive time span".into());
    }
    let nominal_time_step_s = (end_time_s - start_time_s) / interval_count as f64;
    let mut points = vec![StructuralTransientSchedulePoint {
        time_s: runmat_types::physical::Time::new(start_time_s),
        arrival: StructuralTransientScheduleArrival::Initial,
        continuation: StructuralTransientContinuationApproach::At,
    }];
    let mut current_time_s = start_time_s;
    while current_time_s < end_time_s {
        let proposed_dt_s = nominal_time_step_s.min(end_time_s - current_time_s);
        let (next_time_s, approach) =
            next_step_boundary(current_time_s, proposed_dt_s, end_time_s, step_boundaries);
        let (arrival, continuation) = match approach {
            SignalSampleApproach::LeftLimit => (
                StructuralTransientScheduleArrival::LeftLimit,
                StructuralTransientContinuationApproach::RightLimit,
            ),
            SignalSampleApproach::At => (
                StructuralTransientScheduleArrival::Continuous,
                StructuralTransientContinuationApproach::At,
            ),
            SignalSampleApproach::RightLimit => {
                return Err(
                    "a positive-duration transient schedule cannot arrive from the right limit"
                        .into(),
                )
            }
        };
        points.push(StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Time::new(next_time_s),
            arrival,
            continuation,
        });
        current_time_s = next_time_s;
    }
    StructuralTransientSchedule::new(points).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_substeps_preserve_the_nominal_horizon_and_never_cross_a_jump() {
        let start = 0.0;
        let final_time = 1.2;
        let nominal_dt = 0.6;
        let events = canonical_transient_step_boundaries(&[0.5], &[], start, final_time).unwrap();
        let mut time = start;
        let mut steps = Vec::new();
        while time < final_time {
            let (next, _) = next_step_boundary(time, nominal_dt, final_time, &events);
            steps.push((time, next));
            time = next;
        }
        assert_eq!(time, final_time);
        assert_eq!(steps, vec![(0.0, 0.5), (0.5, 1.1), (1.1, 1.2)]);
        assert!(steps
            .iter()
            .all(|(left, right)| !(left < &0.5 && right > &0.5)));
    }

    #[test]
    fn required_samples_split_steps_without_changing_signal_continuity() {
        let boundaries = canonical_transient_step_boundaries(&[], &[0.25], 0.0, 1.0).unwrap();
        assert_eq!(
            next_step_boundary(0.0, 0.5, 1.0, &boundaries),
            (0.25, SignalSampleApproach::At)
        );
        let coincident = canonical_transient_step_boundaries(&[0.25], &[0.25], 0.0, 1.0).unwrap();
        assert_eq!(
            next_step_boundary(0.0, 0.5, 1.0, &coincident),
            (0.25, SignalSampleApproach::LeftLimit)
        );
    }

    #[test]
    fn endpoint_roundoff_does_not_create_a_spurious_extra_increment() {
        let final_time = 2.0 * std::f64::consts::PI / 317.0;
        let nominal_dt = final_time / 16.0;
        let mut time = 0.0;
        let mut step_count = 0;
        while time < final_time {
            (time, _) = next_step_boundary(time, nominal_dt, final_time, &[]);
            step_count += 1;
        }

        assert_eq!(time, final_time);
        assert_eq!(step_count, 16);

        let final_time = 1.0;
        let nominal_dt = final_time / 160.0;
        let mut time = 0.0;
        let mut step_count = 0;
        while time < final_time {
            (time, _) = next_step_boundary(time, nominal_dt, final_time, &[]);
            step_count += 1;
        }
        assert_eq!(time, final_time);
        assert_eq!(step_count, 160);
    }

    #[test]
    fn canonical_schedule_retains_exact_ordinals_and_jump_continuation_side() {
        let boundaries = canonical_transient_step_boundaries(&[0.5], &[0.75], 0.0, 1.0).unwrap();
        let schedule = canonical_transient_schedule(2, 0.0, 1.0, &boundaries).unwrap();
        assert_eq!(
            schedule
                .points
                .iter()
                .map(|point| (
                    point.time_s.canonical_value(),
                    point.arrival,
                    point.continuation
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    0.0,
                    StructuralTransientScheduleArrival::Initial,
                    StructuralTransientContinuationApproach::At
                ),
                (
                    0.5,
                    StructuralTransientScheduleArrival::LeftLimit,
                    StructuralTransientContinuationApproach::RightLimit
                ),
                (
                    0.75,
                    StructuralTransientScheduleArrival::Continuous,
                    StructuralTransientContinuationApproach::At
                ),
                (
                    1.0,
                    StructuralTransientScheduleArrival::Continuous,
                    StructuralTransientContinuationApproach::At
                ),
            ]
        );
    }
}
