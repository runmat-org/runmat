//! Shared thermal sample inventory for numerical execution and resource preflight.
use runmat_analysis_core::{AnalysisModel, ProcedureStep, SignalSampleApproach};

use crate::{
    excitation::{time_breakpoints, ProcedureTimeGrid, ProcedureTimeSchedule, TimeScheduleError},
    pipeline::required_bounded_time_history_samples,
    planning::resource::ResourcePlanningError,
    FeaRunError, ThermalProcedureControl,
};

/// The exact sorted union of numerical grid, active signal events and observation samples.
/// Fields are private: a caller cannot replace an admitted inventory with arbitrary coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct ThermalSolveSchedule {
    steady_point_s: [f64; 1],
    schedule: Option<ProcedureTimeSchedule>,
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
                steady_point_s: [0.0],
                schedule: None,
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
        let schedule =
            ProcedureTimeSchedule::merge(grid, events, required, maximum).map_err(|error| {
                match error {
                    TimeScheduleError::Invalid(message) => {
                        FeaRunError::InvalidModel(message.into())
                    }
                    TimeScheduleError::Ceiling { required, maximum } => {
                        ResourcePlanningError::AccuracyCeiling {
                            resource: "thermal solve samples",
                            required: required as u64,
                            maximum: maximum as u64,
                        }
                        .into()
                    }
                }
            })?;
        Ok(Self {
            steady_point_s: [0.0],
            schedule: Some(schedule),
            transient: true,
        })
    }

    pub fn points_s(&self) -> &[f64] {
        self.schedule
            .as_ref()
            .map_or(&self.steady_point_s, |schedule| schedule.points_s())
    }
    pub fn sample_count(&self) -> usize {
        self.points_s().len()
    }
    pub fn solve_count(&self) -> usize {
        if self.transient {
            self.points_s().len() - 1
        } else {
            1
        }
    }
    /// One primary solve per interval, plus at most one energy-driven refinement.
    /// This is an execution/evidence bound, not another physical field sample.
    pub(crate) fn maximum_linear_solve_count(&self) -> Result<usize, ResourcePlanningError> {
        self.solve_count()
            .checked_mul(if self.transient { 2 } else { 1 })
            .ok_or(ResourcePlanningError::Overflow(
                "thermal linear solve count",
            ))
    }
    pub(crate) fn approach(&self, index: usize) -> SignalSampleApproach {
        self.schedule
            .as_ref()
            .map_or(SignalSampleApproach::At, |schedule| schedule.arrival(index))
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
