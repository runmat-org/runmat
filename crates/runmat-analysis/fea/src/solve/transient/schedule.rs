use runmat_analysis_core::SignalSampleApproach;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransientStepBoundaryKind {
    RequiredSample,
    SignalDiscontinuity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TransientStepBoundary {
    time_s: f64,
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
            boundaries.push(TransientStepBoundary { time_s, kind });
        }
    }
    boundaries.sort_by(|left, right| left.time_s.total_cmp(&right.time_s));
    let mut canonical: Vec<TransientStepBoundary> = Vec::with_capacity(boundaries.len());
    for boundary in boundaries {
        if let Some(previous) = canonical
            .last_mut()
            .filter(|previous| previous.time_s.to_bits() == boundary.time_s.to_bits())
        {
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
    let endpoint_tolerance = 8.0
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
    match step_boundaries
        .iter()
        .copied()
        .find(|boundary| boundary.time_s > current_time_s && boundary.time_s <= proposed_end)
    {
        Some(boundary) => (boundary.time_s, boundary.approach()),
        None => (proposed_end, SignalSampleApproach::At),
    }
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
    }
}
