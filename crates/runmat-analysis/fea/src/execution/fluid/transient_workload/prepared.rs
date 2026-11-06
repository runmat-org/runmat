use super::*;
use crate::{admission::AnalysisRunIdentity, physics::cfd::canonical::CanonicalFluidInitialState};
pub struct PreparedFluidTransient<'a> {
    pub(crate) workload: &'a FluidTransientWorkload,
    pub(crate) source: std::sync::Arc<super::super::source::PreparedFluidSource>,
    pub(crate) initial: CanonicalFluidInitialState,
    pub(crate) formulation: AnalysisRunIdentity,
}
impl PreparedFluidTransient<'_> {
    pub fn initial_state(&self) -> &CanonicalFluidInitialState {
        &self.initial
    }
    pub fn formulation(&self) -> &AnalysisRunIdentity {
        &self.formulation
    }
    pub(super) fn block(&self) -> AnalysisExecutionResult<super::super::FluidTransientBlock<'_>> {
        let mut policy = self.workload.policy;
        policy.set_resident_byte_budget(
            self.workload
                .resident_ceiling()?
                .checked_sub(self.source.preparation_resident_bytes)
                .ok_or_else(|| invalid("transient source exhausts block ceiling"))?,
        );
        let schedule = super::super::FluidTransientSchedule::derive(
            &self.workload.model,
            self.workload.selected_step()?,
        )?;
        super::super::FluidTransientBlock::prepare(
            &self.workload.model,
            &self.workload.step_id,
            self.source.input(&self.workload.model, policy),
            self.initial.original_fields(),
            0..schedule.points_s().len() - 1,
        )
    }
    pub fn execute(&self) -> AnalysisExecutionResult<FluidTransientSolverResult> {
        let profile = self.result_encoding_profile()?;
        let block = self.block()?;
        let required = block
            .reservation()
            .resident_bytes_upper_bound
            .checked_add(profile.materialization_resident_bytes_estimate)
            .ok_or_else(|| invalid("transient full payload bound overflow"))
            .and_then(|later| self.source.resident_with(later))?;
        super::super::source::require_bytes(required, self.workload.resident_ceiling()?)?;
        FluidTransientSolverResult::from_original(self, block.execute()?)
    }
}
