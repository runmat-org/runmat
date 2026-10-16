//! Target-typed ordinary cast recipe. All effects stay in floating_cast.rs.
use super::*;
use runmat_accelerate_api::{ProviderPrecision, ProviderRealElementwiseOutputRoute as Route};

impl WgpuProvider {
    pub(super) fn propose_real_floating_cast(
        &self,
        request: &ProviderRealElementwiseRequest,
        target: ProviderPrecision,
    ) -> ProviderExecutionResult<Option<Proposal>> {
        let source = &request.source;
        if source.storage != ProviderStorage::DenseReal
            || source.layout != ProviderLayout::ColumnMajorContiguous
            || source.residency != ProviderResidency::Device
            || !matches!(
                source.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
            || source
                .checked_element_count()
                .is_none_or(|count| count == 0 || count > u64::from(u32::MAX))
            || source
                .checked_byte_len()
                .is_none_or(|bytes| bytes > request.source_buffer_bytes)
        {
            return Ok(None);
        }
        let target_type = match target {
            ProviderPrecision::F32 => ProviderElementType::F32,
            ProviderPrecision::F64 => ProviderElementType::F64,
        };
        if (source.element_type == ProviderElementType::F64 || target == ProviderPrecision::F64)
            && (self.precision != NumericPrecision::F64
                || !self.enabled_features.contains(wgpu::Features::SHADER_F64))
        {
            return Ok(None);
        }
        let Some(work) = crate::placement::floating_cast_work(self, source, target_type) else {
            return Ok(None);
        };
        let Some(capacity) =
            crate::backend::wgpu::config::elementwise_chunk_capacity(work.workgroup)
        else {
            return Ok(None);
        };
        let same = source.element_type == target_type;
        let Some(output_bytes) = (if same {
            Some(request.source_buffer_bytes)
        } else {
            work.estimate.output_bytes
        }) else {
            return Ok(None);
        };
        let route = if same {
            Route::SameClassFloatingCastAlias
        } else {
            Route::FreshAllocation
        };
        let uniform_size = std::mem::size_of::<super::super::super::floating_cast::Params>() as u64;
        if !limits_allow(
            route,
            &self.adapter_limits,
            request.source_buffer_bytes,
            output_bytes,
            work.workgroup,
            capacity,
            uniform_size,
        ) {
            return Ok(None);
        }
        let Some(dispatches) = work.estimate.dispatches else {
            return Ok(None);
        };
        if uniform_size.checked_mul(u64::from(dispatches)) != Some(work.uniform_bytes) {
            return Ok(None);
        }
        let pipeline_key = if same {
            None
        } else {
            Some(
                super::super::super::pipeline_recipe::PipelineRecipe::floating_cast(
                    precision(work.source),
                    precision(work.target),
                    work.workgroup,
                )
                .key(),
            )
        };
        Ok(Some(Proposal {
            output_route: route,
            output_buffer_bytes: output_bytes,
            workgroup_size: work.workgroup,
            chunk_capacity: capacity,
            uniform_allocations: dispatches,
            uniform_buffer_bytes: work.uniform_bytes,
            dispatches,
            queue_submissions: dispatches,
            pipeline_key,
        }))
    }

    pub(super) async fn execute_real_floating_cast(
        &self,
        input: &GpuTensorHandle,
        target: ProviderPrecision,
        observed: &RefCell<RealObserved>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.floating_cast_exec_recorded(input, precision(target), admission, |event| {
            use super::super::super::floating_cast::CastResourceEvent;
            match event {
                CastResourceEvent::Output(bytes) => {
                    let mut ledger = observed.borrow_mut();
                    ledger.effect = true;
                    ledger.resources.output_buffer_bytes = Some(bytes);
                    Ok(())
                }
                CastResourceEvent::Alias(bytes) => {
                    observed.borrow_mut().resources.output_buffer_bytes = Some(bytes);
                    Ok(())
                }
                CastResourceEvent::Uniform(bytes) => observed.borrow_mut().uniform_bytes(bytes),
                CastResourceEvent::Submitted { dispatch } => {
                    observed.borrow_mut().submitted(dispatch)
                }
            }
        })
        .await
    }
}

/// Pure projection of device-satisfied limits, not a hardware capability claim.
pub(super) fn limits_allow(
    route: Route,
    limits: &wgpu::Limits,
    source_bytes: u64,
    output_bytes: u64,
    workgroup: u32,
    capacity: u64,
    uniform_size: u64,
) -> bool {
    if [source_bytes, output_bytes].into_iter().any(|bytes| {
        bytes > limits.max_buffer_size || bytes > u64::from(limits.max_storage_buffer_binding_size)
    }) {
        return false;
    }
    match route {
        Route::SameClassFloatingCastAlias => true,
        Route::FreshAllocation => {
            workgroup != 0
                && workgroup <= limits.max_compute_workgroup_size_x
                && workgroup <= limits.max_compute_invocations_per_workgroup
                && capacity != 0
                && capacity.div_ceil(u64::from(workgroup))
                    <= u64::from(limits.max_compute_workgroups_per_dimension)
                && limits.max_storage_buffers_per_shader_stage >= 2
                && limits.max_uniform_buffers_per_shader_stage >= 1
                && uniform_size <= u64::from(limits.max_uniform_buffer_binding_size)
        }
    }
}

fn precision(value: ProviderPrecision) -> NumericPrecision {
    match value {
        ProviderPrecision::F32 => NumericPrecision::F32,
        ProviderPrecision::F64 => NumericPrecision::F64,
    }
}
