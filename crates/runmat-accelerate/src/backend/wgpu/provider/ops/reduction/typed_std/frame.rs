use super::super::super::backend_types::BufferEntry;
use super::super::super::pipeline_recipe::{FusionPipelineLayout, PipelineRecipe};
use super::super::super::validated_fusion::{
    ReductionExtremaPipelineKind, StdPrimitivePipelineKind,
};
use super::*;
use crate::backend::wgpu::types::{NumericPrecision, UnaryOpCode};
use runmat_accelerate_api::{unsupported_provider_operation, GpuTensorStorage};

pub(super) struct StdFrame {
    entry: BufferEntry,
    pub(super) dim: Option<usize>,
    pub(super) output_shape: Vec<usize>,
    pub(super) sample_count: usize,
}

fn output_geometry(
    shape: &[usize],
    len: usize,
    dim: Option<usize>,
) -> Result<(usize, Vec<usize>, usize)> {
    if let Some(dim) = dim {
        if shape.len() != 2 || dim > 1 {
            return Err(unsupported_provider_operation(
                "std dimensional kernel requires rank two and axis zero or one",
            ));
        }
        let (rows, cols) = (shape[0], shape[1]);
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| unsupported_provider_operation("std shape overflows"))?;
        if total != len || total > u32::MAX as usize {
            return Err(unsupported_provider_operation(
                "std dimensional addressing exceeds u32",
            ));
        }
        if dim == 0 {
            Ok((rows, vec![1, cols], cols))
        } else {
            Ok((cols, vec![rows, 1], rows))
        }
    } else {
        Ok((len, vec![1; shape.len().max(2)], 1))
    }
}

fn admit_fill_route(
    output_len: usize,
    element_bytes: u64,
    workgroup: u32,
    limits: &wgpu::Limits,
) -> Result<()> {
    let elements = u64::try_from(output_len)
        .map_err(|_| unsupported_provider_operation("std fill length overflows"))?;
    if elements > u32::MAX as u64 || workgroup == 0 {
        return Err(unsupported_provider_operation(
            "std fill length or frozen workgroup is unsupported",
        ));
    }
    if limits.max_storage_buffers_per_shader_stage < 1
        || limits.max_uniform_buffers_per_shader_stage < 1
    {
        return Err(unsupported_provider_operation(
            "std fill requires storage and uniform bindings",
        ));
    }
    let uniform_bytes = match element_bytes {
        4 => std::mem::size_of::<crate::backend::wgpu::params::FillParamsF32>() as u64,
        8 => std::mem::size_of::<crate::backend::wgpu::params::FillParamsF64>() as u64,
        _ => {
            return Err(unsupported_provider_operation(
                "std fill element width unsupported",
            ))
        }
    };
    if uniform_bytes > u64::from(limits.max_uniform_buffer_binding_size)
        || uniform_bytes > limits.max_buffer_size
    {
        return Err(unsupported_provider_operation(
            "std fill uniform exceeds physical limits",
        ));
    }
    let allocation_bytes = elements
        .max(1)
        .checked_mul(element_bytes)
        .ok_or_else(|| unsupported_provider_operation("std fill output extent overflows"))?;
    if allocation_bytes > limits.max_buffer_size {
        return Err(unsupported_provider_operation(
            "std fill output exceeds buffer limit",
        ));
    }
    let max_elements = u64::from(crate::backend::wgpu_geometry::MAX_DISPATCH_WORKGROUPS)
        .checked_mul(u64::from(workgroup))
        .ok_or_else(|| unsupported_provider_operation("std fill chunk extent overflows"))?;
    let chunk_elements = elements.min(max_elements);
    let chunk_bytes = chunk_elements
        .checked_mul(element_bytes)
        .ok_or_else(|| unsupported_provider_operation("std fill chunk bytes overflow"))?;
    let groups = chunk_elements.div_ceil(u64::from(workgroup));
    if chunk_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || groups > u64::from(limits.max_compute_workgroups_per_dimension)
    {
        return Err(unsupported_provider_operation(
            "std fill chunk binding or launch exceeds adapter limits",
        ));
    }
    Ok(())
}

impl StdFrame {
    pub(super) fn admit(
        provider: &WgpuProvider,
        input: &GpuTensorHandle,
        dim: Option<usize>,
        nan_mode: ProviderNanMode,
    ) -> Result<Self> {
        if matches!(nan_mode, ProviderNanMode::Omit) {
            return Err(unsupported_provider_operation(
                "std omitnan is not supported by this WGPU kernel",
            ));
        }
        let entry = provider.get_entry_raw(input)?;
        if entry.storage != GpuTensorStorage::Real
            || input.view().is_logical()
            || entry.integer_type().is_some()
            || entry.precision != provider.precision
        {
            return Err(unsupported_provider_operation(
                "std requires an exact native real floating input",
            ));
        }
        provider.get_entry(input)?;
        let (sample_count, output_shape, output_len) =
            output_geometry(&entry.shape, entry.len, dim)?;
        let element_bytes = match provider.precision {
            NumericPrecision::F32 => 4u64,
            NumericPrecision::F64 => 8u64,
        };
        let physical_limit = u64::from(provider.adapter_limits.max_storage_buffer_binding_size)
            .min(provider.adapter_limits.max_buffer_size);
        let source_bytes = u64::try_from(entry.len)
            .ok()
            .and_then(|len| len.checked_mul(element_bytes))
            .ok_or_else(|| unsupported_provider_operation("std source extent overflows"))?;
        let output_bytes = u64::try_from(output_len)
            .ok()
            .and_then(|len| len.checked_mul(element_bytes))
            .ok_or_else(|| unsupported_provider_operation("std output extent overflows"))?;
        let output_allocation = output_bytes.max(element_bytes);
        admit_fill_route(
            output_len,
            element_bytes,
            provider.workgroup_config.scalar,
            &provider.adapter_limits,
        )?;
        if sample_count != 0 {
            if entry.len > u32::MAX as usize
                || entry.buffer.size() < source_bytes
                || entry.buffer.size() > physical_limit
                || output_allocation > physical_limit
                || provider.adapter_limits.max_storage_buffers_per_shader_stage < 3
                || provider.adapter_limits.max_uniform_buffers_per_shader_stage < 1
            {
                return Err(unsupported_provider_operation(
                    "std source or intermediate exceeds physical limits",
                ));
            }
            provider.require_ieee_words(entry.precision)?;
            provider.preflight_std_binary_input(input)?;
            for op in [UnaryOpCode::Abs, UnaryOpCode::Sqrt] {
                let shader = crate::backend::wgpu::shaders::elementwise::real_unary_shader(
                    op,
                    provider.precision,
                );
                provider.plan_ordinary_single_fusion(
                    &shader,
                    std::slice::from_ref(input),
                    &entry.shape,
                    entry.len,
                    match entry.precision {
                        NumericPrecision::F32 => runmat_accelerate_api::ProviderElementType::F32,
                        NumericPrecision::F64 => runmat_accelerate_api::ProviderElementType::F64,
                    },
                )?;
            }
            if dim.is_some() {
                provider.pipelines.reduce_dim_sum_mean.preflight()?;
                let (_, shader_wg) = provider
                    .pipelines
                    .reduce_dim_sum_mean
                    .frozen_reduction_source()
                    .ok_or_else(|| {
                        unsupported_provider_operation(
                            "std dimensional shader workgroup is not frozen",
                        )
                    })?;
                if shader_wg == 0 {
                    return Err(unsupported_provider_operation(
                        "std dimensional workgroup is zero",
                    ));
                }
                let groups = output_len.div_ceil(shader_wg as usize);
                if groups > provider.adapter_limits.max_compute_workgroups_per_dimension as usize {
                    return Err(unsupported_provider_operation(
                        "std dimensional launch exceeds adapter limit",
                    ));
                }
            } else {
                provider.pipelines.reduce_global.preflight()?;
                super::super::global_resources::admit(
                    entry.len,
                    entry.buffer.size(),
                    provider.element_size,
                    provider.global_reduce_launch.elements_per_group(),
                    provider.global_reduce_launch.max_input_per_pass(),
                    u64::from(provider.adapter_limits.max_storage_buffer_binding_size),
                    provider.adapter_limits.max_buffer_size,
                )?;
            }
        }
        provider.pipelines.fill.preflight()?;
        let fill_len = u64::try_from(output_len)
            .map_err(|_| unsupported_provider_operation("std fill length overflows"))?;
        if provider.fill_buffer_demand(fill_len).is_none() {
            return Err(unsupported_provider_operation(
                "std fill resource demand overflows",
            ));
        }
        Ok(Self {
            entry,
            dim,
            output_shape,
            sample_count,
        })
    }

    pub(super) fn revalidate(
        &self,
        provider: &WgpuProvider,
        input: &GpuTensorHandle,
    ) -> Result<()> {
        let current = provider.get_entry_raw(input)?;
        ensure!(
            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&current.buffer, &self.entry.buffer)
                && current.shape == self.entry.shape
                && current.len == self.entry.len
                && current.element_type == self.entry.element_type
                && current.allocated_bytes == self.entry.allocated_bytes,
            "std input changed during scoped preparation"
        );
        input.allocation_lease().validate_live()?;
        Ok(())
    }
}

impl StdSelected {
    pub(super) async fn prepare(
        provider: &WgpuProvider,
        frame: &StdFrame,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<Self> {
        let fill_kind = StdPrimitivePipelineKind::Fill;
        let fill_recipe = fill_kind.recipe(provider);
        let fill = provider
            .prepare_validated_std_primitive_pipeline(&fill_recipe, fill_kind)
            .await?;
        fill.require_std_primitive(&fill_recipe, fill_kind)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        if frame.sample_count == 0 {
            return Ok(Self {
                sum: None,
                binary: None,
                fill,
                abs: None,
                sqrt: None,
                temporaries: RefCell::new(StdCleanupOwner::new(input.allocation_identity())),
            });
        }
        let sum = if frame.dim.is_some() {
            let kind = StdPrimitivePipelineKind::DimensionalSum;
            let recipe = kind.recipe(provider);
            let selected = provider
                .prepare_validated_std_primitive_pipeline(&recipe, kind)
                .await?;
            selected
                .require_std_primitive(&recipe, kind)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            selected
        } else {
            let kind = ReductionExtremaPipelineKind::Global;
            let recipe = kind.recipe(provider);
            let selected = provider
                .prepare_validated_reduction_extrema_pipeline(&recipe, kind)
                .await?;
            selected
                .require_reduction_extrema(&recipe, kind)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            selected
        };
        let binary_kind = StdPrimitivePipelineKind::Binary;
        let binary_recipe = binary_kind.recipe(provider);
        let binary = provider
            .prepare_validated_std_primitive_pipeline(&binary_recipe, binary_kind)
            .await?;
        binary
            .require_std_primitive(&binary_recipe, binary_kind)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let unary = |op| {
            PipelineRecipe::fusion(
                &crate::backend::wgpu::shaders::elementwise::real_unary_shader(
                    op,
                    provider.precision,
                ),
                1,
                FusionPipelineLayout::Single,
                provider.linear_launch.workgroup(),
            )
        };
        let abs_recipe = unary(UnaryOpCode::Abs);
        let abs = provider
            .prepare_validated_fusion_pipeline(&abs_recipe, 1, FusionPipelineLayout::Single)
            .await?;
        abs.require_recipe(&abs_recipe, 1, FusionPipelineLayout::Single)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let sqrt_recipe = unary(UnaryOpCode::Sqrt);
        let sqrt = provider
            .prepare_validated_fusion_pipeline(&sqrt_recipe, 1, FusionPipelineLayout::Single)
            .await?;
        sqrt.require_recipe(&sqrt_recipe, 1, FusionPipelineLayout::Single)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        Ok(Self {
            sum: Some(sum),
            binary: Some(binary),
            fill,
            abs: Some(abs),
            sqrt: Some(sqrt),
            temporaries: RefCell::new(StdCleanupOwner::new(input.allocation_identity())),
        })
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn exact_rank_and_empty_geometry_precedes_effects() {
        assert_eq!(
            output_geometry(&[2, 2, 2], 8, None).unwrap(),
            (8, vec![1, 1, 1], 1)
        );
        assert_eq!(
            output_geometry(&[0, 3], 0, Some(0)).unwrap(),
            (0, vec![1, 3], 3)
        );
        assert_eq!(
            output_geometry(&[0, 3], 0, Some(1)).unwrap(),
            (3, vec![0, 1], 0)
        );
        assert_eq!(output_geometry(&[], 1, None).unwrap(), (1, vec![1, 1], 1));
        assert!(output_geometry(&[2, 2, 2], 8, Some(0)).is_err());
        assert!(output_geometry(&[2, 2], 4, Some(2)).is_err());
        assert!(output_geometry(&[2, 2], 3, Some(0)).is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn empty_std_fill_admission_proves_chunk_binding_and_launch() {
        let mut limits = wgpu::Limits {
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: u32::MAX,
            ..wgpu::Limits::default()
        };
        assert!(admit_fill_route(3, 4, 256, &limits).is_ok());
        if let Ok(oversized) = usize::try_from(u64::from(u32::MAX) + 1) {
            let (_, shape, len) = output_geometry(&[0, oversized], 0, Some(0))
                .expect("zero-source geometry can still have an oversized output");
            assert_eq!(shape, vec![1, oversized]);
            assert!(admit_fill_route(len, 4, 256, &limits).is_err());
        }
        limits.max_storage_buffer_binding_size = 1024;
        assert!(admit_fill_route(1000, 4, 256, &limits).is_err());
        limits.max_storage_buffer_binding_size = u32::MAX;
        limits.max_compute_workgroups_per_dimension = 1;
        assert!(admit_fill_route(257, 4, 256, &limits).is_err());
        limits.max_compute_workgroups_per_dimension = 65_535;
        limits.max_uniform_buffers_per_shader_stage = 0;
        assert!(admit_fill_route(0, 4, 256, &limits).is_err());
        limits.max_uniform_buffers_per_shader_stage = 1;
        limits.max_storage_buffers_per_shader_stage = 0;
        assert!(admit_fill_route(0, 4, 256, &limits).is_err());
    }
}
