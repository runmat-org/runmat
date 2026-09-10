use runmat_analysis_core::{IndependentDomain, ProcedureStep};

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
        Ok(Self::uniform(start_s, end_s, sample_count))
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
        Ok(Self::uniform(start_s, end_s, interval_count + 1))
    }

    pub(crate) fn points_s(&self) -> &[f64] {
        &self.points_s
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

    fn uniform(start_s: f64, end_s: f64, sample_count: usize) -> Self {
        let interval_count = sample_count - 1;
        let duration_s = end_s - start_s;
        let mut points_s = (0..sample_count)
            .map(|index| start_s + duration_s * index as f64 / interval_count as f64)
            .collect::<Vec<_>>();
        // Avoid accumulated arithmetic changing the identity of the authored endpoint.
        points_s[0] = start_s;
        points_s[interval_count] = end_s;
        Self { points_s }
    }
}

fn time_bounds(step: &ProcedureStep) -> Result<(f64, f64), String> {
    let IndependentDomain::TimeSeconds { origin, start, end } = step.independent_domain else {
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
        ProcedureStep::experimental(
            "transient",
            PhysicsFamily::Structural,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start,
                end,
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
}
