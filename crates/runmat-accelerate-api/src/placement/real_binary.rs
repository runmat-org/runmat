//! Provider-owned physical recipe for one ordinary real Sub or Div.
//! Source-level numerical admission and selected-program authority are separate.
use crate::{
    AccelProvider, GpuTensorHandle, GpuTensorLayout, GpuTensorStorage, NumericElementType,
    ProviderElementType, ProviderExecutionCompletionError, ProviderExecutionError,
    ProviderExecutionResult, ProviderLayout, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ProviderRealBinaryOperation {
    Sub,
    Div,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ProviderRealBinaryInput {
    pub representation: ProviderRepresentation,
    pub buffer_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ProviderRealBinaryRequest {
    pub operation: ProviderRealBinaryOperation,
    pub lhs: ProviderRealBinaryInput,
    pub rhs: ProviderRealBinaryInput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ProviderRealBinaryRoute {
    SameShape,
    Broadcast,
}

/// Effect-free provider proposal. It is not executable without a privately
/// minted recipe and exact live operand/lease revalidation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ProviderRealBinaryPhysicalProposal {
    pub route: ProviderRealBinaryRoute,
    pub output: ProviderRepresentation,
    pub output_buffer_bytes: u64,
    pub workgroup_size: u32,
    pub chunk_capacity: u64,
    pub uniform_allocations: u32,
    pub uniform_buffer_bytes: u64,
    pub dispatches: u32,
    pub pre_output_submissions: u32,
    pub queue_submissions: u32,
}

impl ProviderRealBinaryPhysicalProposal {
    pub fn requested_incremental_peak_bytes(&self) -> Option<u64> {
        self.output_buffer_bytes
            .checked_add(self.uniform_buffer_bytes)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderRealBinaryObserved {
    pub output_buffer_bytes: Option<u64>,
    pub uniform_allocations: u32,
    pub uniform_buffer_bytes: u64,
    pub dispatches: u32,
    pub pre_output_submissions: u32,
    pub queue_submissions: u32,
}

pub struct ProviderExecutedRealBinary {
    pub output: GpuTensorHandle,
    pub observed: ProviderRealBinaryObserved,
}

#[derive(Clone)]
pub struct ProviderRealBinaryRecipe {
    owner: &'static dyn AccelProvider,
    request: ProviderRealBinaryRequest,
    physical: ProviderRealBinaryPhysicalProposal,
}

impl ProviderRealBinaryRecipe {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        request: ProviderRealBinaryRequest,
    ) -> ProviderExecutionResult<Option<Self>> {
        if !native_input(&request.lhs)
            || !native_input(&request.rhs)
            || request.lhs.representation.element_type != request.rhs.representation.element_type
        {
            return Ok(None);
        }
        let Some(physical) = owner.propose_real_binary(&request)? else {
            return Ok(None);
        };
        if physical.output.element_type != request.lhs.representation.element_type
            || physical.output.storage != ProviderStorage::DenseReal
            || physical.output.layout != ProviderLayout::ColumnMajorContiguous
            || physical.output.residency != ProviderResidency::Device
            || physical
                .output
                .checked_element_count()
                .is_none_or(|n| n == 0)
            || physical.workgroup_size == 0
            || physical.chunk_capacity == 0
            || physical
                .output
                .checked_byte_len()
                .is_none_or(|n| n > physical.output_buffer_bytes)
            || physical.requested_incremental_peak_bytes().is_none()
            || physical
                .pre_output_submissions
                .checked_add(physical.dispatches)
                != Some(physical.queue_submissions)
        {
            return Ok(None);
        }
        Ok(Some(Self {
            owner,
            request,
            physical,
        }))
    }

    pub fn request(&self) -> &ProviderRealBinaryRequest {
        &self.request
    }
    pub fn physical(&self) -> &ProviderRealBinaryPhysicalProposal {
        &self.physical
    }
    pub fn is_owned_by(&self, owner: &dyn AccelProvider) -> bool {
        crate::same_provider_instance(self.owner, owner)
    }
    pub fn identity(&self) -> anyhow::Result<runmat_execution::Digest> {
        Ok(runmat_execution::Digest::sha256(serde_json::to_vec(&(
            "ordinary-real-binary-v1",
            &self.request,
            &self.physical,
        ))?))
    }

    pub async fn execute(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> ProviderExecutionResult<ProviderExecutedRealBinary> {
        self.execute_protected(lhs, rhs, &[]).await
    }

    pub async fn execute_protected(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        borrowed: &[&GpuTensorHandle],
    ) -> ProviderExecutionResult<ProviderExecutedRealBinary> {
        self.execute_protected_admitted(lhs, rhs, borrowed, None)
            .await
    }

    pub async fn execute_protected_admitted(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        borrowed: &[&GpuTensorHandle],
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedRealBinary> {
        let protected = std::iter::once(lhs)
            .chain(std::iter::once(rhs))
            .chain(borrowed.iter().copied())
            .collect::<Vec<_>>();
        for handle in &protected {
            handle
                .view()
                .validate(handle.physical())
                .map_err(ProviderExecutionError::from_source)?;
            handle
                .allocation_lease()
                .validate_live()
                .map_err(ProviderExecutionError::from_source)?;
        }
        if !live_matches(self.owner, lhs, &self.request.lhs)
            || !live_matches(self.owner, rhs, &self.request.rhs)
            || self.owner.propose_real_binary(&self.request)?.as_ref() != Some(&self.physical)
        {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "real binary operands or physical recipe changed"
            )));
        }
        let guards = protected
            .iter()
            .copied()
            .map(crate::ResidencyRestoreGuard::new)
            .collect::<Vec<_>>();
        let result = self
            .owner
            .execute_real_binary_recipe(lhs, rhs, self, admission)
            .await;
        for guard in &guards {
            guard.restore();
        }
        drop(guards);
        let result = result?;
        let observed = &result.observed;
        if !live_output_matches(self.owner, &result.output, &self.physical)
            || !live_matches(self.owner, lhs, &self.request.lhs)
            || !live_matches(self.owner, rhs, &self.request.rhs)
            || protected
                .iter()
                .any(|handle| handle.allocation_lease().validate_live().is_err())
            || protected
                .iter()
                .any(|handle| result.output.shares_allocation_with(handle))
            || observed.output_buffer_bytes != Some(self.physical.output_buffer_bytes)
            || observed.uniform_allocations != self.physical.uniform_allocations
            || observed.uniform_buffer_bytes != self.physical.uniform_buffer_bytes
            || observed.dispatches != self.physical.dispatches
            || observed.pre_output_submissions != self.physical.pre_output_submissions
            || observed.queue_submissions != self.physical.queue_submissions
        {
            let error = ProviderExecutionError::from(anyhow::anyhow!(
                "real binary observed execution differs from recipe"
            ))
            .after_effect();
            if protected
                .iter()
                .any(|handle| result.output.shares_allocation_with(handle))
            {
                drop(result.output);
                return Err(error);
            }
            let cleanup = result
                .output
                .release()
                .err()
                .into_iter()
                .collect::<Vec<_>>();
            return if cleanup.is_empty() {
                Err(error)
            } else {
                Err(
                    ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                        Some(error),
                        cleanup,
                    ))
                    .after_effect(),
                )
            };
        }
        Ok(result)
    }
}

fn native_input(input: &ProviderRealBinaryInput) -> bool {
    let source = &input.representation;
    source.storage == ProviderStorage::DenseReal
        && source.layout == ProviderLayout::ColumnMajorContiguous
        && source.residency == ProviderResidency::Device
        && matches!(
            source.element_type,
            ProviderElementType::F32 | ProviderElementType::F64
        )
        && source.checked_element_count().is_some_and(|n| n > 0)
        && source
            .checked_byte_len()
            .is_some_and(|n| n <= input.buffer_bytes)
}

fn live_matches(
    owner: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    input: &ProviderRealBinaryInput,
) -> bool {
    live_representation_matches(owner, handle, &input.representation, input.buffer_bytes)
}

fn live_output_matches(
    owner: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    output: &ProviderRealBinaryPhysicalProposal,
) -> bool {
    live_representation_matches(owner, handle, &output.output, output.output_buffer_bytes)
}

fn live_representation_matches(
    owner: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    representation: &ProviderRepresentation,
    bytes: u64,
) -> bool {
    let expected = match representation.element_type {
        ProviderElementType::F32 => NumericElementType::F32,
        ProviderElementType::F64 => NumericElementType::F64,
        _ => return false,
    };
    handle.allocation_lease().validate_live().is_ok()
        && handle.view().validate(handle.physical()).is_ok()
        && crate::provider_for_handle(handle)
            .is_some_and(|actual| crate::same_provider_instance(actual, owner))
        && handle.device_id() == owner.device_id()
        && handle.physical().element_type() == expected
        && handle.physical().storage() == GpuTensorStorage::Real
        && !handle.view().is_logical()
        && handle.view().layout() == GpuTensorLayout::Contiguous
        && handle.physical().requested_bytes() == bytes
        && handle
            .shape()
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok()
            .as_ref()
            == Some(&representation.shape)
}

#[cfg(test)]
#[path = "real_binary/tests.rs"]
mod tests;
