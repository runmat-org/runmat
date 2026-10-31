mod schedule;
use runmat_analysis_core::{IndependentDomain, ProcedureStep};
pub(crate) use schedule::{ProcedureTimeSchedule, TimeScheduleError};

/// A numerical sampling grid derived from the physical time domain authored on a procedure step.
///
/// This type deliberately does not accept a caller-provided time increment. The procedure step is
/// the sole authority for physical start and end time; numerical controls only choose how that
/// interval is discretized.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProcedureTimeGrid {
    points_s: Vec<f64>,
}

impl ProcedureTimeGrid {
    /// Builds `sample_count` points including both authored endpoints.
    pub(crate) fn from_sample_count(
        step: &ProcedureStep,
        sample_count: usize,
    ) -> Result<Self, String> {
        if sample_count < 2 {
            return Err("a time-domain sample grid requires at least two samples".to_string());
        }
        let (start_s, end_s) = time_bounds(step)?;
        Self::uniform(start_s, end_s, sample_count)
    }

    /// Builds `interval_count` uniform intervals including both authored endpoints.
    pub(crate) fn from_interval_count(
        step: &ProcedureStep,
        interval_count: usize,
    ) -> Result<Self, String> {
        if interval_count == 0 {
            return Err("a time-domain interval grid requires at least one interval".to_string());
        }
        let (start_s, end_s) = time_bounds(step)?;
        let sample_count = interval_count
            .checked_add(1)
            .ok_or_else(|| "time-domain sample count overflow".to_string())?;
        Self::uniform(start_s, end_s, sample_count)
    }

    pub(crate) fn points_s(&self) -> &[f64] {
        &self.points_s
    }

    pub(crate) fn into_points_s(self) -> Vec<f64> {
        self.points_s
    }

    /// The grid owner's coordinate formula without allocating the full grid. Endpoint identity
    /// is authored; interior coordinates use the same fraction as `uniform` below.
    pub(crate) fn uniform_coordinate_at(
        start_s: f64,
        end_s: f64,
        interval_count: usize,
        index: usize,
    ) -> Result<f64, String> {
        if !start_s.is_finite()
            || !end_s.is_finite()
            || end_s <= start_s
            || interval_count == 0
            || index > interval_count
        {
            return Err("invalid time-grid coordinate index or authored bounds".into());
        }
        let value = if index == 0 {
            start_s
        } else if index == interval_count {
            end_s
        } else {
            start_s + (end_s - start_s) * (index as f64 / interval_count as f64)
        };
        if !value.is_finite() {
            return Err("time-grid coordinate is not finite".into());
        }
        Ok(value)
    }

    pub(crate) fn start_s(&self) -> f64 {
        self.points_s[0]
    }

    pub(crate) fn end_s(&self) -> f64 {
        self.points_s[self.points_s.len() - 1]
    }

    pub(crate) fn nominal_interval_s(&self) -> f64 {
        (self.end_s() - self.start_s()) / self.interval_count() as f64
    }

    pub(crate) fn interval_count(&self) -> usize {
        self.points_s.len() - 1
    }

    fn uniform(start_s: f64, end_s: f64, sample_count: usize) -> Result<Self, String> {
        let interval_count = sample_count - 1;
        let duration_s = end_s - start_s;
        let interval_s = duration_s / interval_count as f64;
        if !duration_s.is_finite()
            || !interval_s.is_finite()
            || interval_s <= 0.0
            || (sample_count > 2
                && (start_s + interval_s <= start_s || end_s - interval_s >= end_s))
        {
            return Err(
                "time-domain grid intervals are not representable as finite distinct coordinates"
                    .into(),
            );
        }
        let mut points_s = Vec::new();
        points_s
            .try_reserve_exact(sample_count)
            .map_err(|_| "time-domain grid exceeds addressable memory".to_string())?;
        for index in 0..sample_count {
            points_s.push(Self::uniform_coordinate_at(
                start_s,
                end_s,
                interval_count,
                index,
            )?);
        }
        if points_s.iter().any(|point| !point.is_finite())
            || points_s.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "time-domain grid coordinates must be finite and strictly increasing".into(),
            );
        }
        Ok(Self { points_s })
    }
}

fn time_bounds(step: &ProcedureStep) -> Result<(f64, f64), String> {
    let IndependentDomain::TimeSeconds {
        origin, start, end, ..
    } = step.independent_domain
    else {
        return Err(format!(
            "procedure step '{}' requires a time domain for transient execution",
            step.step_id.as_str()
        ));
    };
    if !origin.is_finite() || !start.is_finite() || !end.is_finite() {
        return Err(format!(
            "procedure step '{}' time origin and bounds must be finite",
            step.step_id.as_str()
        ));
    }
    if end <= start {
        return Err(format!(
            "procedure step '{}' time end must be greater than its start",
            step.step_id.as_str()
        ));
    }
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisProcedure, IndependentDomain, PhysicsFamily, ProcedureStep, SolutionSpace,
    };

    use super::ProcedureTimeGrid;

    fn time_step(start: f64, end: f64) -> ProcedureStep {
        ProcedureStep::new(
            "transient",
            PhysicsFamily::Structural,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start,
                end,
                interval_count: 10,
            },
            Vec::new(),
        )
    }

    #[test]
    fn sample_grid_includes_authored_endpoints() {
        let grid = ProcedureTimeGrid::from_sample_count(&time_step(2.0, 5.0), 4).unwrap();
        assert_eq!(grid.points_s(), &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(grid.interval_count(), 3);
        assert_eq!(grid.nominal_interval_s(), 1.0);
    }

    #[test]
    fn interval_grid_distinguishes_intervals_from_samples() {
        let grid = ProcedureTimeGrid::from_interval_count(&time_step(-1.0, 1.0), 4).unwrap();
        assert_eq!(grid.points_s(), &[-1.0, -0.5, 0.0, 0.5, 1.0]);
        assert_eq!(grid.interval_count(), 4);
    }

    #[test]
    fn invalid_domains_fail_closed() {
        let error = ProcedureTimeGrid::from_sample_count(&time_step(1.0, 1.0), 2).unwrap_err();
        assert!(error.contains("end must be greater"));
    }

    #[test]
    fn overflowing_and_unrepresentable_grids_fail_before_allocation() {
        assert!(ProcedureTimeGrid::from_interval_count(&time_step(0.0, 1.0), usize::MAX).is_err());
        assert!(ProcedureTimeGrid::from_sample_count(&time_step(-f64::MAX, f64::MAX), 2).is_err());
        assert!(ProcedureTimeGrid::from_sample_count(&time_step(1e16, 1e16 + 2.0), 3).is_err());
        assert!(ProcedureTimeGrid::from_sample_count(&time_step(0.0, 1.0), usize::MAX).is_err());
    }

    #[test]
    fn large_finite_horizon_does_not_overflow_intermediate_multiplication() {
        let grid = ProcedureTimeGrid::from_sample_count(&time_step(0.0, f64::MAX), 4).unwrap();
        assert!(grid.points_s().iter().all(|point| point.is_finite()));
        assert_eq!(grid.end_s(), f64::MAX);
    }
}
