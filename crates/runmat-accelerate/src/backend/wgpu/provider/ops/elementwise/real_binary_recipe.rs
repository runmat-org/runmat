//! One provider-owned physical route for ordinary real Sub/Div. The ordinary
//! binary executor remains the only implementation of the arithmetic kernel.
use super::*;
use crate::backend::wgpu::pipelines::FrozenLinearLaunch;
use runmat_accelerate_api::{
    ProviderElementType, ProviderExecutedRealBinary, ProviderExecutionError,
    ProviderExecutionResult, ProviderLayout, ProviderRealBinaryInput, ProviderRealBinaryObserved,
    ProviderRealBinaryOperation, ProviderRealBinaryPhysicalProposal, ProviderRealBinaryRecipe,
    ProviderRealBinaryRequest, ProviderRealBinaryRoute, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};

#[derive(Default)]
pub(super) struct RealBinaryObserved {
    effect: bool,
    resources: ProviderRealBinaryObserved,
}

pub(super) enum RealBinarySubmission {
    PreOutput,
    Dispatch,
}

impl RealBinaryObserved {
    #[cfg(test)]
    pub(super) fn into_resources(self) -> ProviderRealBinaryObserved {
        self.resources
    }

    pub(super) fn begin_effect(&mut self) {
        self.effect = true;
    }
    pub(super) fn output(&mut self, buffer: &runmat_accelerate_api::WgpuBufferOwner) {
        self.effect = true;
        self.resources.output_buffer_bytes = Some(buffer.size());
    }
    pub(super) fn uniform(
        &mut self,
        buffer: &runmat_accelerate_api::WgpuBufferOwner,
    ) -> Result<()> {
        self.effect = true;
        self.resources.uniform_allocations = self
            .resources
            .uniform_allocations
            .checked_add(1)
            .ok_or_else(|| anyhow!("real binary uniform count overflow"))?;
        self.resources.uniform_buffer_bytes = self
            .resources
            .uniform_buffer_bytes
            .checked_add(buffer.size())
            .ok_or_else(|| anyhow!("real binary uniform extent overflow"))?;
        Ok(())
    }
    pub(super) fn submitted(&mut self, kind: RealBinarySubmission) -> Result<()> {
        self.effect = true;
        self.resources.queue_submissions = self
            .resources
            .queue_submissions
            .checked_add(1)
            .ok_or_else(|| anyhow!("real binary submission count overflow"))?;
        match kind {
            RealBinarySubmission::Dispatch => {
                self.resources.dispatches = self
                    .resources
                    .dispatches
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("real binary dispatch count overflow"))?;
            }
            RealBinarySubmission::PreOutput => {
                self.resources.pre_output_submissions = self
                    .resources
                    .pre_output_submissions
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("real binary pre-output count overflow"))?;
            }
        }
        Ok(())
    }
}

fn native_input(input: &ProviderRealBinaryInput) -> Option<NumericPrecision> {
    let source = &input.representation;
    if source.storage != ProviderStorage::DenseReal
        || source.layout != ProviderLayout::ColumnMajorContiguous
        || source.residency != ProviderResidency::Device
        || source
            .checked_element_count()
            .is_none_or(|n| n == 0 || n > u64::from(u32::MAX))
        || source
            .checked_byte_len()
            .is_none_or(|n| n > input.buffer_bytes)
    {
        return None;
    }
    match source.element_type {
        ProviderElementType::F32 => Some(NumericPrecision::F32),
        ProviderElementType::F64 => Some(NumericPrecision::F64),
        _ => None,
    }
}

impl WgpuProvider {
    pub(super) fn binary_linear_launch(&self) -> Result<FrozenLinearLaunch> {
        let launch = self
            .pipelines
            .binary
            .linear_launch()
            .ok_or_else(|| anyhow!("selected real binary launch is not frozen"))?;
        let workgroup = launch.workgroup();
        ensure!(
            workgroup <= self.adapter_limits.max_compute_workgroup_size_x
                && workgroup <= self.adapter_limits.max_compute_invocations_per_workgroup,
            "selected real binary workgroup exceeds enabled device limits"
        );
        FrozenLinearLaunch::new(
            workgroup,
            self.adapter_limits.max_compute_workgroups_per_dimension,
        )
        .ok_or_else(|| anyhow!("selected real binary dispatch geometry unavailable"))
    }

    pub(crate) fn propose_real_binary_exec(
        &self,
        request: &ProviderRealBinaryRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealBinaryPhysicalProposal>> {
        if std::env::var("RUNMAT_DISABLE_BINARY").is_ok() {
            return Ok(None);
        }
        let Some(precision) = native_input(&request.lhs) else {
            return Ok(None);
        };
        if native_input(&request.rhs) != Some(precision)
            || (precision == NumericPrecision::F64 && self.precision != NumericPrecision::F64)
            || self.adapter_limits.max_storage_buffers_per_shader_stage < 3
            || self.adapter_limits.max_uniform_buffers_per_shader_stage < 1
        {
            return Ok(None);
        }
        for input in [&request.lhs, &request.rhs] {
            if input.buffer_bytes > self.adapter_limits.max_buffer_size
                || input.buffer_bytes
                    > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            {
                return Ok(None);
            }
        }
        let Some(launch) = self.binary_linear_launch().ok() else {
            return Ok(None);
        };
        let Some(chunk_capacity) = launch.chunk_capacity() else {
            return Ok(None);
        };
        let op = match request.operation {
            ProviderRealBinaryOperation::Sub => crate::backend::wgpu::types::BinaryOpCode::Sub,
            ProviderRealBinaryOperation::Div => crate::backend::wgpu::types::BinaryOpCode::Div,
        };
        let lhs = request
            .lhs
            .representation
            .shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok();
        let rhs = request
            .rhs
            .representation
            .shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok();
        let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
            return Ok(None);
        };
        let route = if lhs == rhs {
            ProviderRealBinaryRoute::SameShape
        } else {
            ProviderRealBinaryRoute::Broadcast
        };
        let element_bytes = u64::from(match precision {
            NumericPrecision::F32 => 4_u32,
            NumericPrecision::F64 => 8_u32,
        });
        let (shape, elements) = match route {
            ProviderRealBinaryRoute::SameShape => {
                let Some(elements) = request.lhs.representation.checked_element_count() else {
                    return Ok(None);
                };
                (lhs, elements)
            }
            ProviderRealBinaryRoute::Broadcast => {
                let Ok(plan) = broadcast_plan::BroadcastRoute::with_launch(
                    &lhs,
                    &rhs,
                    crate::backend::wgpu::params::BCAST_MAX_RANK,
                    element_bytes,
                    launch,
                ) else {
                    return Ok(None);
                };
                (plan.out_shape, plan.len as u64)
            }
        };
        if elements == 0 || elements > u64::from(u32::MAX) {
            return Ok(None);
        }
        let Some(output_shape) = shape
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok()
        else {
            return Ok(None);
        };
        let output = ProviderRepresentation {
            element_type: request.lhs.representation.element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            residency: ProviderResidency::Device,
            shape: output_shape,
        };
        let Some(output_buffer_bytes) = self.plan_dense_real_output_buffer_bytes_exec(&output)?
        else {
            return Ok(None);
        };
        let pipeline = match self.real_binary_pipeline(
            precision,
            matches!(route, ProviderRealBinaryRoute::Broadcast),
            op,
        ) {
            Ok(pipeline) => pipeline,
            Err(_) => return Ok(None),
        };
        pipeline.preflight().map_err(ProviderExecutionError::from)?;
        let Ok(dispatches) = u32::try_from(elements.div_ceil(chunk_capacity as u64)) else {
            return Ok(None);
        };
        let (uniform_allocations, uniform_param_bytes, pre_output_submissions) = match route {
            ProviderRealBinaryRoute::SameShape => (
                dispatches,
                std::mem::size_of::<crate::backend::wgpu::params::LenOpParams>() as u64,
                2_u32,
            ),
            ProviderRealBinaryRoute::Broadcast => (
                1_u32,
                std::mem::size_of::<crate::backend::wgpu::params::BinaryBroadcastParams>() as u64,
                0_u32,
            ),
        };
        let Some(uniform_allocation_bytes) =
            uniform_param_bytes.max(1).div_ceil(256).checked_mul(256)
        else {
            return Ok(None);
        };
        if uniform_allocation_bytes > u64::from(self.adapter_limits.max_uniform_buffer_binding_size)
        {
            return Ok(None);
        }
        let Some(uniform_buffer_bytes) =
            uniform_allocation_bytes.checked_mul(u64::from(uniform_allocations))
        else {
            return Ok(None);
        };
        let Some(queue_submissions) = pre_output_submissions.checked_add(dispatches) else {
            return Ok(None);
        };
        Ok(Some(ProviderRealBinaryPhysicalProposal {
            route,
            output,
            output_buffer_bytes,
            workgroup_size: launch.workgroup(),
            chunk_capacity: chunk_capacity as u64,
            uniform_allocations,
            uniform_buffer_bytes,
            dispatches,
            pre_output_submissions,
            queue_submissions,
        }))
    }

    pub(crate) fn execute_real_binary_recipe_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        recipe: &ProviderRealBinaryRecipe,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedRealBinary> {
        if !recipe.is_owned_by(self) {
            return Err(ProviderExecutionError::from(anyhow!(
                "real binary recipe belongs to another provider instance"
            )));
        }
        for (handle, input) in [(lhs, &recipe.request().lhs), (rhs, &recipe.request().rhs)] {
            if !runmat_accelerate_api::provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, self))
                || handle.device_id() != runmat_accelerate_api::AccelProvider::device_id(self)
            {
                return Err(ProviderExecutionError::from(anyhow!(
                    "real binary source belongs to another provider instance"
                )));
            }
            handle
                .allocation_lease()
                .validate_live()
                .map_err(ProviderExecutionError::from_source)?;
            let source = real_recipe::source_representation(handle).ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("real binary source is not contiguous real"))
            })?;
            self.real_floating_entry(handle)
                .map_err(ProviderExecutionError::from)?;
            if source != input.representation
                || handle.physical().requested_bytes() != input.buffer_bytes
            {
                return Err(ProviderExecutionError::from(anyhow!(
                    "real binary recipe does not match actual source"
                )));
            }
        }
        if self.propose_real_binary_exec(recipe.request())?.as_ref() != Some(recipe.physical()) {
            return Err(ProviderExecutionError::from(anyhow!(
                "real binary physical recipe changed before execution"
            )));
        }
        let observed = RefCell::new(RealBinaryObserved::default());
        let op = match recipe.request().operation {
            ProviderRealBinaryOperation::Sub => crate::backend::wgpu::types::BinaryOpCode::Sub,
            ProviderRealBinaryOperation::Div => crate::backend::wgpu::types::BinaryOpCode::Div,
        };
        let result = self
            .binary_op_exec_recorded(op, lhs, rhs, Some(&observed), None, admission)
            .map_err(|cause| {
                let error = ProviderExecutionError::from(cause);
                if observed.borrow().effect {
                    error.after_effect()
                } else {
                    error
                }
            })?;
        Ok(ProviderExecutedRealBinary {
            output: result,
            observed: observed.into_inner().resources,
        })
    }
}

#[cfg(test)]
#[path = "real_binary_recipe_tests.rs"]
mod tests;
