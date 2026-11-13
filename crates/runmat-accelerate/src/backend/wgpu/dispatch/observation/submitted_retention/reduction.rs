//! Exact typed-reduction binding owners. The producer consumes its EXISTING
//! preflight vector; no second vector, owner Box or payload copy is created.
use super::{SubmittedAllocationRetention, SubmittedResources};
use runmat_accelerate_api::{allocation::AllocationLease, GpuTensorHandle, WgpuBufferOwner};
use runmat_execution::resource::{ProviderHostMetadataAdmission, RetainedHostAllocationClaim};
use std::sync::Arc;

/// The original one input vector and its exact backing claim travel together.
/// Field order frees the Vec allocation before refunding its host reservation.
pub(crate) struct SubmittedReductionInputs {
    values: Vec<SubmittedReductionInput>,
    _backing: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

impl SubmittedReductionInputs {
    pub(crate) fn reserve(
        count: usize,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> anyhow::Result<Self> {
        let bytes = std::alloc::Layout::array::<SubmittedReductionInput>(count)?.size();
        let claim = crate::wgsl::host_buffers::claim_bytes(metadata, bytes)?;
        let values = runmat_value::exact_capacity_vec(count)?;
        Ok(Self {
            values,
            _backing: claim,
        })
    }

    pub(crate) fn push(&mut self, value: SubmittedReductionInput) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.values.len() < self.values.capacity(),
            "typed reduction input vector exceeded its admitted capacity"
        );
        self.values.push(value);
        Ok(())
    }

    fn empty() -> Self {
        Self {
            values: Vec::new(),
            _backing: None,
        }
    }
}

impl std::ops::Deref for SubmittedReductionInputs {
    type Target = [SubmittedReductionInput];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

pub(crate) struct SubmittedReductionInput {
    pub(crate) buffer: WgpuBufferOwner,
    _lease: AllocationLease,
}

impl SubmittedReductionInput {
    pub(crate) fn new(handle: &GpuTensorHandle, buffer: WgpuBufferOwner) -> anyhow::Result<Self> {
        handle.allocation_lease().validate_live()?;
        Ok(Self {
            buffer,
            _lease: handle.allocation_lease().clone(),
        })
    }
}

pub(crate) struct SubmittedReductionResources {
    inputs: SubmittedReductionInputs,
    partial_input: Option<WgpuBufferOwner>,
    output: WgpuBufferOwner,
    _output_lease: Option<AllocationLease>,
    completion_claim: crate::wgsl::host_buffers::Claim,
}

impl SubmittedReductionResources {
    pub(crate) fn producer(
        inputs: SubmittedReductionInputs,
        output: &WgpuBufferOwner,
        published: Option<&GpuTensorHandle>,
    ) -> anyhow::Result<Self> {
        if let Some(handle) = published {
            handle.allocation_lease().validate_live()?;
        }
        Ok(Self {
            inputs,
            partial_input: None,
            output: output.clone(),
            _output_lease: published.map(|handle| handle.allocation_lease().clone()),
            completion_claim: None,
        })
    }

    pub(crate) fn final_stage(
        partial: &WgpuBufferOwner,
        output: &WgpuBufferOwner,
        published: &GpuTensorHandle,
    ) -> anyhow::Result<Self> {
        let mut owner = Self::producer(SubmittedReductionInputs::empty(), output, Some(published))?;
        owner.partial_input = Some(partial.clone());
        Ok(owner)
    }

    pub(crate) fn input_count(&self) -> usize {
        self.partial_input.as_ref().map_or(self.inputs.len(), |_| 1)
    }

    pub(crate) fn input_buffer(&self, slot: usize) -> anyhow::Result<&WgpuBufferOwner> {
        if let Some(partial) = &self.partial_input {
            anyhow::ensure!(slot == 0, "typed reduction partial binding out of range");
            Ok(partial)
        } else {
            self.inputs
                .get(slot)
                .map(|input| &input.buffer)
                .ok_or_else(|| anyhow::anyhow!("typed reduction producer binding out of range"))
        }
    }

    pub(crate) fn output_buffer(&self) -> &WgpuBufferOwner {
        &self.output
    }

    pub(crate) fn retain_completion_claim(&mut self, claim: crate::wgsl::host_buffers::Claim) {
        self.completion_claim = claim;
    }

    pub(super) fn take_completion_claim(&mut self) -> crate::wgsl::host_buffers::Claim {
        self.completion_claim.take()
    }

    pub(crate) fn into_retention(self) -> SubmittedAllocationRetention {
        SubmittedAllocationRetention {
            owner: None,
            resources: Some(SubmittedResources::Reduction(self)),
            retirement: None,
        }
    }
}
