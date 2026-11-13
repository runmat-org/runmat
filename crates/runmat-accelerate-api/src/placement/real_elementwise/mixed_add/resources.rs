//! Exact requested aggregate of all retained owned results and uniforms.
//! Existing input storage, driver/cache residency and fences are separate.
use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ProviderMixedRealAddResources {
    pub retained_output_buffer_bytes: u64,
    pub uniform_allocations: u32,
    pub uniform_buffer_bytes: u64,
    pub dispatches: u32,
    pub queue_submissions: u32,
}
impl ProviderMixedRealAddResources {
    pub fn requested_incremental_peak_bytes(&self) -> Option<u64> {
        self.retained_output_buffer_bytes
            .checked_add(self.uniform_buffer_bytes)
    }
    pub(super) fn planned<'a>(
        steps: impl IntoIterator<Item = &'a ProviderRealElementwisePhysicalProposal>,
    ) -> Option<Self> {
        let mut result = Self::default();
        for step in steps {
            if step.output_route != ProviderRealElementwiseOutputRoute::FreshAllocation {
                return None;
            }
            result.append(
                step.output_buffer_bytes,
                step.uniform_allocations,
                step.uniform_buffer_bytes,
                step.dispatches,
                step.queue_submissions,
            )?;
        }
        result.requested_incremental_peak_bytes()?;
        Some(result)
    }
    fn append(
        &mut self,
        output: u64,
        uniforms: u32,
        uniform_bytes: u64,
        dispatches: u32,
        submissions: u32,
    ) -> Option<()> {
        self.retained_output_buffer_bytes =
            self.retained_output_buffer_bytes.checked_add(output)?;
        self.uniform_allocations = self.uniform_allocations.checked_add(uniforms)?;
        self.uniform_buffer_bytes = self.uniform_buffer_bytes.checked_add(uniform_bytes)?;
        self.dispatches = self.dispatches.checked_add(dispatches)?;
        self.queue_submissions = self.queue_submissions.checked_add(submissions)?;
        Some(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderMixedRealAddStepObservations {
    WidenAddNarrow([ProviderRealElementwiseObserved; 3]),
    AddNarrow([ProviderRealElementwiseObserved; 2]),
}
impl std::ops::Deref for ProviderMixedRealAddStepObservations {
    type Target = [ProviderRealElementwiseObserved];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::WidenAddNarrow(steps) => steps,
            Self::AddNarrow(steps) => steps,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderMixedRealAddObserved {
    /// Canonical optional widening, native Double Add, narrowing order; roles remain
    /// privately minted and are never interpreted as selected source ValueIDs.
    pub steps: ProviderMixedRealAddStepObservations,
    pub resources: ProviderMixedRealAddResources,
}
impl ProviderMixedRealAddObserved {
    pub(super) fn from_steps(
        receipts: Vec<ProviderRealElementwiseObserved>,
        roles: &[Step],
    ) -> Option<Self> {
        let steps = match roles {
            [Step::Widen, Step::Add, Step::Narrow] => {
                ProviderMixedRealAddStepObservations::WidenAddNarrow(receipts.try_into().ok()?)
            }
            [Step::Add, Step::Narrow] => {
                ProviderMixedRealAddStepObservations::AddNarrow(receipts.try_into().ok()?)
            }
            _ => return None,
        };
        let mut resources = ProviderMixedRealAddResources::default();
        for step in steps.iter() {
            resources.append(
                step.output_buffer_bytes?,
                step.uniform_allocations,
                step.uniform_buffer_bytes,
                step.dispatches,
                step.queue_submissions,
            )?;
        }
        resources.requested_incremental_peak_bytes()?;
        Some(Self { steps, resources })
    }
}
