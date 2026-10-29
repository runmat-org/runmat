//! Exact typed-reduction binding owners. The producer consumes its EXISTING
//! preflight vector; no second vector, owner Box or payload copy is created.
use super::{SubmittedAllocationRetention, SubmittedResources};
use runmat_accelerate_api::{allocation::AllocationLease, GpuTensorHandle, WgpuBufferOwner};

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
    inputs: Vec<SubmittedReductionInput>,
    partial_input: Option<WgpuBufferOwner>,
    output: WgpuBufferOwner,
    _output_lease: Option<AllocationLease>,
}

impl SubmittedReductionResources {
    pub(crate) fn producer(
        inputs: Vec<SubmittedReductionInput>,
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
        })
    }

    pub(crate) fn final_stage(
        partial: &WgpuBufferOwner,
        output: &WgpuBufferOwner,
        published: &GpuTensorHandle,
    ) -> anyhow::Result<Self> {
        let mut owner = Self::producer(Vec::new(), output, Some(published))?;
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

    pub(crate) fn into_retention(self) -> SubmittedAllocationRetention {
        SubmittedAllocationRetention {
            owner: None,
            resources: Some(SubmittedResources::Reduction(self)),
            retirement: None,
        }
    }
}
