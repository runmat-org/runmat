//! Shared thermal sample inventory for numerical execution and resource preflight.
use runmat_analysis_core::{AnalysisModel, ProcedureStep, SignalSampleApproach};

use crate::{
    excitation::{time_breakpoints, ProcedureTimeGrid},
    pipeline::required_bounded_time_history_samples,
    planning::resource::ResourcePlanningError,
    FeaRunError, ThermalProcedureControl,
};

/// The exact sorted union of numerical grid, active signal events and observation samples.
/// Fields are private: a caller cannot replace an admitted inventory with arbitrary coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct ThermalSolveSchedule {
    points_s: Vec<f64>,
    events_s: Vec<f64>,
    transient: bool,
}

impl ThermalSolveSchedule {
    pub(crate) fn derive(
        model: &AnalysisModel,
        step: &ProcedureStep,
        procedure: ThermalProcedureControl,
    ) -> Result<Self, FeaRunError> {
        crate::pipeline::thermal::validate_step(step, procedure)?;
        let maximum = step.accuracy.resources.maximum_samples;
        let requested = match procedure {
            ThermalProcedureControl::Steady => 1,
            ThermalProcedureControl::BackwardEuler { snapshot_count } => snapshot_count,
        };
        check_sample_count(requested, maximum)?;
        let required = required_bounded_time_history_samples(step, maximum)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let ThermalProcedureControl::BackwardEuler { snapshot_count } = procedure else {
            return Ok(Self {
                points_s: vec![0.0],
                events_s: Vec::new(),
                transient: false,
            });
        };
        let grid = ProcedureTimeGrid::from_sample_count(step, snapshot_count)
            .map_err(FeaRunError::InvalidModel)?;
        let events = time_breakpoints(model, step, grid.start_s(), grid.end_s())
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        Self::merge(grid.points_s(), &events, &required, maximum)
    }

    fn merge(
        grid: &[f64],
        events: &[f64],
        required: &[f64],
        maximum: usize,
    ) -> Result<Self, FeaRunError> {
        let start = grid[0];
        let end = grid[grid.len() - 1];
        let canonical_time = |value: f64| if value == 0.0 { 0.0 } else { value };
        let mut points = grid.iter().copied().map(canonical_time).collect::<Vec<_>>();
        for source in [events, required] {
            for &raw in source {
                if !raw.is_finite() || raw < start || raw > end {
                    return Err(FeaRunError::InvalidModel(
                        "thermal schedule sample lies outside the finite authored time interval"
                            .into(),
                    ));
                }
            }
            // Both sources are sorted by their shared realizers. Merge linearly, with a bound
            // on the unique union; repeated observation requests must not consume sample budget.
            let mut left = points.into_iter().peekable();
            let mut right = source.iter().copied().map(canonical_time).peekable();
            let mut merged = Vec::new();
            while left.peek().is_some() || right.peek().is_some() {
                let value = match (left.peek(), right.peek()) {
                    (Some(a), Some(b)) if a <= b => left.next().unwrap(),
                    (_, Some(_)) => right.next().unwrap(),
                    _ => left.next().unwrap(),
                };
                if merged.last().is_none_or(|last| *last != value) {
                    check_sample_count(
                        merged
                            .len()
                            .checked_add(1)
                            .ok_or(ResourcePlanningError::Overflow("thermal sample count"))?,
                        maximum,
                    )?;
                    merged.push(value);
                }
            }
            points = merged;
        }
        let mut events_s = events
            .iter()
            .copied()
            .map(canonical_time)
            .collect::<Vec<_>>();
        events_s.sort_by(f64::total_cmp);
        events_s.dedup();
        Ok(Self {
            points_s: points,
            events_s,
            transient: true,
        })
    }

    pub fn points_s(&self) -> &[f64] {
        &self.points_s
    }
    pub fn sample_count(&self) -> usize {
        self.points_s.len()
    }
    pub fn solve_count(&self) -> usize {
        if self.transient {
            self.points_s.len() - 1
        } else {
            1
        }
    }
    pub(crate) fn approach(&self, index: usize) -> SignalSampleApproach {
        if self
            .events_s
            .binary_search_by(|event| event.total_cmp(&self.points_s[index]))
            .is_ok()
        {
            SignalSampleApproach::LeftLimit
        } else {
            SignalSampleApproach::At
        }
    }
}

fn check_sample_count(required: usize, maximum: usize) -> Result<(), ResourcePlanningError> {
    if required > maximum {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "thermal solve samples",
            required: u64::try_from(required)
                .map_err(|_| ResourcePlanningError::Overflow("thermal sample count"))?,
            maximum: u64::try_from(maximum)
                .map_err(|_| ResourcePlanningError::Overflow("thermal maximum samples"))?,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
