//! Shared thermal sample inventory for numerical execution and resource preflight.
use runmat_analysis_core::{AnalysisModel, ProcedureStep, SignalSampleApproach};

use crate::{
    excitation::{
        time_breakpoints_profiled, ProcedureTimeGrid, ProcedureTimeSchedule, TimeScheduleError,
    },
    pipeline::{
        history_schedule_construction_resident_bytes,
        required_bounded_time_history_samples_admitted,
    },
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
    construction_resident_bytes: u64,
}

impl ThermalSolveSchedule {
    pub(crate) fn derive(
        model: &AnalysisModel,
        step: &ProcedureStep,
        procedure: ThermalProcedureControl,
    ) -> Result<Self, FeaRunError> {
        Self::derive_admitted(
            model,
            step,
            procedure,
            step.accuracy.resources.resident_byte_budget(),
        )
    }

    pub(crate) fn derive_admitted(
        model: &AnalysisModel,
        step: &ProcedureStep,
        procedure: ThermalProcedureControl,
        maximum_resident_bytes: u64,
    ) -> Result<Self, FeaRunError> {
        crate::pipeline::thermal::validate_step(step, procedure)?;
        let maximum = step.accuracy.resources.maximum_samples;
        let requested = match procedure {
            ThermalProcedureControl::Steady => 1,
            ThermalProcedureControl::BackwardEuler { snapshot_count } => snapshot_count,
        };
        check_sample_count(requested, maximum)?;
        let ceiling = step
            .accuracy
            .resources
            .resident_byte_budget()
            .min(maximum_resident_bytes);
        let history_bytes = history_schedule_construction_resident_bytes(step)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let required = required_bounded_time_history_samples_admitted(step, maximum, ceiling)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let ThermalProcedureControl::BackwardEuler { snapshot_count } = procedure else {
            return Ok(Self {
                steady_point_s: [0.0],
                schedule: None,
                transient: false,
                construction_resident_bytes: history_bytes,
            });
        };
        let required_bytes = crate::topology::storage::slots::<f64>(required.capacity())
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        let grid_bytes = crate::topology::storage::slots::<f64>(snapshot_count)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        let before_events = remaining_bytes(ceiling, &[required_bytes, grid_bytes])?;
        let grid = ProcedureTimeGrid::from_sample_count(step, snapshot_count)
            .map_err(FeaRunError::InvalidModel)?;
        let (events, event_construction_bytes) =
            time_breakpoints_profiled(model, step, grid.start_s(), grid.end_s(), before_events)
                .map_err(|error| error.into_fea_error())?;
        let event_bytes = crate::topology::storage::slots::<f64>(events.capacity())
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        let remaining = remaining_bytes(before_events, &[event_bytes])?;
        let live_bytes = ceiling - remaining;
        let merged_bytes = ProcedureTimeSchedule::construction_resident_bytes(
            grid.points_s(),
            &events,
            &required,
            maximum,
        )
        .map_err(schedule_error)?;
        // Admit the complete merge phase before either retained union buffer is built.
        remaining_bytes(remaining, &[merged_bytes])?;
        let mut result =
            Self::merge_admitted(grid.points_s(), &events, &required, maximum, remaining)?;
        let event_phase = required_bytes
            .checked_add(grid_bytes)
            .and_then(|v| v.checked_add(event_construction_bytes))
            .ok_or(ResourcePlanningError::Overflow(
                "thermal event construction workspace",
            ))?;
        let merge_phase =
            live_bytes
                .checked_add(merged_bytes)
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal union construction workspace",
                ))?;
        result.construction_resident_bytes = history_bytes.max(event_phase).max(merge_phase);
        Ok(result)
    }

    #[cfg(test)]
    fn merge(
        grid: &[f64],
        events: &[f64],
        required: &[f64],
        maximum: usize,
    ) -> Result<Self, FeaRunError> {
        Self::merge_admitted(grid, events, required, maximum, u64::MAX)
    }
    fn merge_admitted(
        grid: &[f64],
        events: &[f64],
        required: &[f64],
        maximum: usize,
        resident: u64,
    ) -> Result<Self, FeaRunError> {
        let schedule =
            ProcedureTimeSchedule::merge_admitted(grid, events, required, maximum, resident)
                .map_err(schedule_error)?;
        let construction_resident_bytes =
            ProcedureTimeSchedule::construction_resident_bytes(grid, events, required, maximum)
                .map_err(schedule_error)?;
        Ok(Self {
            steady_point_s: [0.0],
            schedule: Some(schedule),
            transient: true,
            construction_resident_bytes,
        })
    }

    /// Requested heap construction bound, including input-buffer coexistence; excludes
    /// borrowed model storage, canonical validation scratch, allocator overhead and RSS.
    pub fn construction_resident_bytes(&self) -> u64 {
        self.construction_resident_bytes
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

fn schedule_error(error: TimeScheduleError) -> FeaRunError {
    match error {
        TimeScheduleError::Invalid(message) => FeaRunError::InvalidModel(message.into()),
        TimeScheduleError::Ceiling { required, maximum } => {
            ResourcePlanningError::AccuracyCeiling {
                resource: "thermal solve samples",
                required: required as u64,
                maximum: maximum as u64,
            }
            .into()
        }
    }
}

fn remaining_bytes(ceiling: u64, live: &[u64]) -> Result<u64, ResourcePlanningError> {
    live.iter().try_fold(ceiling, |remaining, bytes| {
        remaining
            .checked_sub(*bytes)
            .ok_or(ResourcePlanningError::AccuracyCeiling {
                resource: "thermal schedule resident bytes",
                required: *bytes,
                maximum: remaining,
            })
    })
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
