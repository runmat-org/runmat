//! Exact authored load-factor boundaries, independent of the physical Newton driver.
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LoadIncrementBoundaryKind {
    RequiredSample,
    SignalDiscontinuity,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadIncrementBoundary {
    factor: f64,
    kind: LoadIncrementBoundaryKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoadIncrementSchedule {
    boundaries: Vec<LoadIncrementBoundary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum LoadIncrementScheduleError {
    #[error("nonlinear load-factor domain is nonfinite or reversed")]
    InvalidDomain,
    #[error("nonlinear load-increment boundary lies outside the finite load-factor domain")]
    OutsideDomain,
}

impl LoadIncrementSchedule {
    pub(crate) fn canonical(
        signal_discontinuities: &[f64],
        required_samples: &[f64],
        start: f64,
        end: f64,
    ) -> Result<Self, LoadIncrementScheduleError> {
        if !start.is_finite() || !end.is_finite() || end < start {
            return Err(LoadIncrementScheduleError::InvalidDomain);
        }
        let mut boundaries = Vec::with_capacity(
            signal_discontinuities
                .len()
                .saturating_add(required_samples.len()),
        );
        for (factors, kind) in [
            (
                signal_discontinuities,
                LoadIncrementBoundaryKind::SignalDiscontinuity,
            ),
            (required_samples, LoadIncrementBoundaryKind::RequiredSample),
        ] {
            for &factor in factors {
                if !factor.is_finite() || factor < start || factor > end {
                    return Err(LoadIncrementScheduleError::OutsideDomain);
                }
                boundaries.push(LoadIncrementBoundary { factor, kind });
            }
        }
        boundaries.sort_by(|left, right| left.factor.total_cmp(&right.factor));
        let mut canonical: Vec<LoadIncrementBoundary> = Vec::with_capacity(boundaries.len());
        for boundary in boundaries {
            if let Some(previous) = canonical
                .last_mut()
                .filter(|previous| previous.factor.to_bits() == boundary.factor.to_bits())
            {
                if boundary.kind == LoadIncrementBoundaryKind::SignalDiscontinuity {
                    previous.kind = boundary.kind;
                }
            } else {
                canonical.push(boundary);
            }
        }
        Ok(Self {
            boundaries: canonical,
        })
    }

    pub(super) fn next_factor(&self, current: f64, proposed_increment: f64, end: f64) -> f64 {
        let proposed_end = (current + proposed_increment).min(end);
        self.boundaries
            .iter()
            .find(|boundary| boundary.factor > current && boundary.factor <= proposed_end)
            .map_or(proposed_end, |boundary| boundary.factor)
    }

    pub(super) fn validate_for(
        &self,
        start: f64,
        end: f64,
    ) -> Result<(), LoadIncrementScheduleError> {
        if !start.is_finite() || !end.is_finite() || end < start {
            return Err(LoadIncrementScheduleError::InvalidDomain);
        }
        if self
            .boundaries
            .iter()
            .any(|b| !b.factor.is_finite() || b.factor < start || b.factor > end)
            || self
                .boundaries
                .windows(2)
                .any(|p| p[0].factor.total_cmp(&p[1].factor) != std::cmp::Ordering::Less)
        {
            return Err(LoadIncrementScheduleError::OutsideDomain);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_splits_nominal_increments_at_required_factors() {
        let schedule = LoadIncrementSchedule::canonical(&[], &[0.0, 0.3, 1.0], 0.0, 1.0)
            .expect("load-factor samples should form a valid schedule");
        assert_eq!(schedule.next_factor(0.0, 0.5, 1.0), 0.3);
        assert_eq!(schedule.next_factor(0.3, 0.5, 1.0), 0.8);
        assert_eq!(schedule.next_factor(0.8, 0.5, 1.0), 1.0);
    }

    #[test]
    fn schedule_rejects_boundaries_outside_the_load_factor_domain() {
        assert_eq!(
            LoadIncrementSchedule::canonical(&[], &[1.1], 0.0, 1.0),
            Err(LoadIncrementScheduleError::OutsideDomain)
        );
    }
}
