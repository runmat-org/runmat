//! Authored time coordinates, events and observations; no numerical integration here.
use super::{invalid, AnalysisExecutionResult};
use crate::excitation::{
    time_breakpoints, ProcedureTimeGrid, ProcedureTimeSchedule, TimeScheduleError,
};
use runmat_analysis_core::{
    AnalysisModel, AnalysisProcedure, IndependentDomain, ObservationRequest, ProcedureStep,
    SignalSampleApproach, SolutionSpace,
};

#[derive(Clone, Debug, PartialEq)]
pub struct FluidTransientSchedule {
    pub(super) schedule: ProcedureTimeSchedule,
}
impl FluidTransientSchedule {
    pub fn derive(model: &AnalysisModel, step: &ProcedureStep) -> AnalysisExecutionResult<Self> {
        Self::derive_with_ceiling(model, step, step.accuracy.resources.resident_byte_budget())
    }
    pub(super) fn derive_with_ceiling(
        model: &AnalysisModel,
        step: &ProcedureStep,
        ceiling: u64,
    ) -> AnalysisExecutionResult<Self> {
        let mut selected = model
            .sequence
            .steps
            .iter()
            .filter(|s| s.step_id == step.step_id);
        if selected.next() != Some(step) || selected.next().is_some() {
            return Err(invalid(
                "fluid schedule step differs from original authored sequence",
            ));
        }
        if !crate::study::is_flow_step(step)
            || step.procedure != AnalysisProcedure::TransientResponse
            || step.solution_space != SolutionSpace::Direct
        {
            return Err(invalid(
                "fluid time schedule requires a direct transient fluid procedure",
            ));
        }
        let IndependentDomain::TimeSeconds { interval_count, .. } = step.independent_domain else {
            return Err(invalid("fluid transient requires physical time"));
        };
        let maximum = step.accuracy.resources.maximum_samples;
        if interval_count
            .checked_add(1)
            .is_none_or(|count| count > maximum)
        {
            return Err(invalid("fluid time grid exceeds authored sample ceiling"));
        }
        // Use the shared observation cardinality law before expanding interval histories.
        // Raw duplicates still require merge scratch even though unique samples count once.
        let mut raw = interval_count
            .checked_add(1)
            .ok_or_else(|| invalid("fluid grid count overflow"))?;
        for request in &step.observations {
            if let ObservationRequest::History(spec) = request {
                let count =
                    crate::results::observation::exact_scheduled_sample_count(&spec.sampling)
                        .map_err(invalid)?
                        .unwrap_or(0);
                if count > maximum {
                    return Err(invalid("fluid observation schedule exceeds sample ceiling"));
                }
                raw = raw
                    .checked_add(count)
                    .ok_or_else(|| invalid("fluid schedule scratch count overflow"))?;
            }
        }
        let metadata = runmat_canonical_codec::profile_serialized(model, super::LIMITS)?;
        let requested = (raw as u64)
            .checked_mul(64)
            .and_then(|v| {
                metadata
                    .materialization_resident_bytes_estimate
                    .checked_mul(2)
                    .and_then(|m| v.checked_add(m))
            })
            .ok_or_else(|| invalid("fluid schedule scratch bytes overflow"))?;
        if requested > ceiling || requested > isize::MAX as u64 {
            return Err(invalid(
                "fluid time schedule scratch exceeds resident ceiling",
            ));
        }
        let grid = ProcedureTimeGrid::from_interval_count(step, interval_count).map_err(invalid)?;
        let events =
            time_breakpoints(model, step, grid.start_s(), grid.end_s()).map_err(invalid)?;
        let required = crate::pipeline::required_bounded_time_history_samples(step, maximum)
            .map_err(invalid)?;
        let schedule = ProcedureTimeSchedule::merge(grid.points_s(), &events, &required, maximum)
            .map_err(|e| match e {
            TimeScheduleError::Invalid(message) => invalid(message),
            TimeScheduleError::Ceiling { required, maximum } => invalid(format!(
                "fluid time union {required} exceeds authored ceiling {maximum}"
            )),
        })?;
        Ok(Self { schedule })
    }
    pub fn points_s(&self) -> &[f64] {
        self.schedule.points_s()
    }
    pub fn arrival(&self, ordinal: usize) -> Option<SignalSampleApproach> {
        (ordinal < self.points_s().len()).then(|| self.schedule.arrival(ordinal))
    }
    pub fn departure(&self, ordinal: usize) -> Option<SignalSampleApproach> {
        (ordinal < self.points_s().len()).then(|| self.schedule.departure(ordinal))
    }
}
