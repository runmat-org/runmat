//! Pre-effect facts shared by scoped pipeline creation and ordinary emission.
use super::*;
use crate::backend::wgpu::residency::BufferResidency;

/// Internal physical-frame input shared by legacy and explicit typed hooks.
/// Legacy compatibility may use Simple; it never invents a public typed ABI.
struct FrameRequest<'a> {
    shader: &'a str,
    inputs: &'a [GpuTensorHandle],
    output_shape: &'a [usize],
    len: usize,
    output_classes: &'a [runmat_accelerate_api::ProviderElementType],
    precision: runmat_accelerate_api::ProviderPrecision,
    uniform_abi: FusionUniformAbi,
}
impl<'a> FrameRequest<'a> {
    fn typed(request: runmat_accelerate_api::ProviderFusedElementwiseRequest<'a>) -> Self {
        Self {
            shader: request.shader,
            inputs: request.inputs,
            output_shape: request.output_shape,
            len: request.len,
            output_classes: request.output_classes,
            precision: request.abi.precision(),
            uniform_abi: match request.abi {
                runmat_accelerate_api::ProviderFusedElementwiseAbi::BroadcastReal { .. } => {
                    FusionUniformAbi::Broadcast
                }
            },
        }
    }
}

pub(in crate::backend::wgpu::provider::backend) struct SingleFrame {
    pub(in crate::backend::wgpu::provider::backend) precision: NumericPrecision,
    pub(in crate::backend::wgpu::provider::backend) recipe: FrozenFusionRecipe,
    pub(in crate::backend::wgpu::provider::backend) entries:
        Vec<super::super::backend_types::BufferEntry>,
    pub(in crate::backend::wgpu::provider::backend) storage_bindings: usize,
    pub(in crate::backend::wgpu::provider::backend) total_bindings: usize,
    pub(in crate::backend::wgpu::provider::backend) layouts: Vec<(Vec<u32>, Vec<u32>)>,
    pub(in crate::backend::wgpu::provider::backend) logical_output: bool,
    pub(in crate::backend::wgpu::provider::backend) output_class:
        runmat_accelerate_api::ProviderElementType,
    pub(in crate::backend::wgpu::provider::backend) shape: Vec<usize>,
    pub(in crate::backend::wgpu::provider::backend) len: usize,
}

pub(in crate::backend::wgpu::provider::backend) struct MultiFrame {
    pub(in crate::backend::wgpu::provider::backend) precision: NumericPrecision,
    pub(in crate::backend::wgpu::provider::backend) recipe:
        super::super::pipeline_recipe::PipelineRecipe,
    pub(in crate::backend::wgpu::provider::backend) entries:
        Vec<super::super::backend_types::BufferEntry>,
    pub(in crate::backend::wgpu::provider::backend) layouts: Vec<(Vec<u32>, Vec<u32>)>,
    pub(in crate::backend::wgpu::provider::backend) chunk_capacity: usize,
    pub(in crate::backend::wgpu::provider::backend) storage_bindings: usize,
    pub(in crate::backend::wgpu::provider::backend) total_bindings: usize,
    pub(in crate::backend::wgpu::provider::backend) output_classes:
        Vec<runmat_accelerate_api::ProviderElementType>,
    pub(in crate::backend::wgpu::provider::backend) shape: Vec<usize>,
    pub(in crate::backend::wgpu::provider::backend) len: usize,
}

impl SingleFrame {
    pub(in crate::backend::wgpu::provider::backend) fn require_execution(
        &self,
        validated: &super::super::validated_fusion::ValidatedFusionPipeline,
        inputs: usize,
        output_shape: &[usize],
        len: usize,
        representation: &FusedElementwiseRepresentation,
    ) -> Result<()> {
        validated.require_recipe(
            self.recipe.pipeline(),
            inputs,
            super::super::pipeline_recipe::FusionPipelineLayout::Single,
        )?;
        let output_class = if representation.logical_output {
            runmat_accelerate_api::ProviderElementType::Logical
        } else {
            match representation.output_precision {
                NumericPrecision::F32 => runmat_accelerate_api::ProviderElementType::F32,
                NumericPrecision::F64 => runmat_accelerate_api::ProviderElementType::F64,
            }
        };
        ensure!(
            self.logical_output == representation.logical_output
                && self.precision == representation.input_precision
                && self.precision == representation.output_precision
                && self.output_class == output_class
                && self.shape.as_slice() == output_shape
                && self.len == len,
            "validated fused request changed before emission"
        );
        Ok(())
    }
}

impl MultiFrame {
    pub(in crate::backend::wgpu::provider::backend) fn require_execution(
        &self,
        validated: &super::super::validated_fusion::ValidatedFusionPipeline,
        inputs: usize,
        output_shape: &[usize],
        len: usize,
        output_classes: &[runmat_accelerate_api::ProviderElementType],
    ) -> Result<()> {
        validated.require_recipe(
            &self.recipe,
            inputs,
            super::super::pipeline_recipe::FusionPipelineLayout::Multiple(output_classes.len()),
        )?;
        ensure!(
            self.output_classes.as_slice() == output_classes
                && self.shape.as_slice() == output_shape
                && self.len == len,
            "validated fused multi request changed before emission"
        );
        Ok(())
    }
}

impl WgpuProvider {
    pub(in crate::backend::wgpu::provider::backend) fn fusion_entry_for_precision(
        &self,
        handle: &GpuTensorHandle,
        precision: NumericPrecision,
    ) -> Result<super::super::backend_types::BufferEntry> {
        let entry = if precision == self.precision {
            self.get_entry(handle)?
        } else {
            self.real_floating_entry(handle)?
        };
        ensure!(
            entry.precision == precision && entry.storage == GpuTensorStorage::Real,
            "fused input does not match frozen physical precision"
        );
        Ok(entry)
    }
    pub(in crate::backend::wgpu::provider::backend) fn plan_ordinary_single_fusion(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        output_class: runmat_accelerate_api::ProviderElementType,
    ) -> Result<SingleFrame> {
        let abi = if shader.contains("out_shape") || shader.contains("a_shape") {
            FusionUniformAbi::Broadcast
        } else {
            FusionUniformAbi::Simple
        };
        self.plan_single_frame(FrameRequest {
            shader,
            inputs,
            output_shape,
            len,
            output_classes: &[output_class],
            precision: self.provider_precision_exec(),
            uniform_abi: abi,
        })
    }

    pub(in crate::backend::wgpu::provider::backend) fn plan_single_fusion(
        &self,
        request: runmat_accelerate_api::ProviderFusedElementwiseRequest<'_>,
    ) -> Result<SingleFrame> {
        self.plan_single_frame(FrameRequest::typed(request))
    }
    fn plan_single_frame(&self, request: FrameRequest<'_>) -> Result<SingleFrame> {
        let FrameRequest {
            shader,
            inputs,
            output_shape,
            len,
            output_classes,
            precision: data_precision,
            uniform_abi: abi,
        } = request;
        let [output_class] = output_classes else {
            return Err(anyhow!("single fusion requires one output"));
        };
        let output_class = *output_class;
        let precision = match data_precision {
            runmat_accelerate_api::ProviderPrecision::F32 => NumericPrecision::F32,
            runmat_accelerate_api::ProviderPrecision::F64 => NumericPrecision::F64,
        };
        runmat_accelerate_api::validate_fused_output_classes(data_precision, &[output_class])?;
        // Preserve the existing public rejection order before the recipe's
        // shape/output-word check and any scoped driver creation.
        ensure!(!inputs.is_empty(), "fused_elementwise: no inputs");
        ensure!(
            len <= u32::MAX as usize,
            "fused_elementwise: tensor too large"
        );
        let logical_output = output_class == runmat_accelerate_api::ProviderElementType::Logical;
        let representation = FusedElementwiseRepresentation {
            input_precision: precision,
            output_precision: precision,
            output_storage: GpuTensorStorage::Real,
            logical_output,
        };
        let work = FusedElementwiseWork {
            output_shape,
            len,
            invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
            representation,
            uniform_abi: Some(abi),
        };
        let recipe = FrozenFusionRecipe::plan(
            shader,
            inputs.len(),
            &work,
            self.linear_launch,
            &self.adapter_limits,
        )?;
        let storage_bindings = checked_binding_count("fused_elementwise", inputs.len(), 1)?;
        let total_bindings = checked_binding_count("fused_elementwise", storage_bindings, 1)?;
        let entries = inputs
            .iter()
            .map(|handle| self.fusion_entry_for_precision(handle, precision))
            .collect::<Result<Vec<_>>>()?;
        let layouts = match abi {
            FusionUniformAbi::Broadcast => inputs
                .iter()
                .zip(&entries)
                .map(|(handle, entry)| {
                    super::super::backend_shared::fusion_operand_layout(handle, entry, output_shape)
                })
                .collect::<Result<Vec<_>>>()?,
            FusionUniformAbi::Simple => {
                if inputs
                    .iter()
                    .any(runmat_accelerate_api::handle_is_transposed)
                {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "linear fused shader does not support a lazy transpose",
                    ));
                }
                Vec::new()
            }
        };
        Ok(SingleFrame {
            precision,
            recipe,
            entries,
            storage_bindings,
            total_bindings,
            layouts,
            logical_output,
            output_class,
            shape: output_shape.to_vec(),
            len,
        })
    }

    pub(in crate::backend::wgpu::provider::backend) fn plan_ordinary_multi_fusion(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        output_classes: &[runmat_accelerate_api::ProviderElementType],
    ) -> Result<MultiFrame> {
        self.plan_multi_frame(FrameRequest {
            shader,
            inputs,
            output_shape,
            len,
            output_classes,
            precision: self.provider_precision_exec(),
            uniform_abi: FusionUniformAbi::Broadcast,
        })
    }

    pub(in crate::backend::wgpu::provider::backend) fn plan_multi_fusion(
        &self,
        request: runmat_accelerate_api::ProviderFusedElementwiseRequest<'_>,
    ) -> Result<MultiFrame> {
        self.plan_multi_frame(FrameRequest::typed(request))
    }
    fn plan_multi_frame(&self, request: FrameRequest<'_>) -> Result<MultiFrame> {
        let FrameRequest {
            shader,
            inputs,
            output_shape,
            len,
            output_classes,
            precision: data_precision,
            uniform_abi,
        } = request;
        ensure!(
            matches!(uniform_abi, FusionUniformAbi::Broadcast),
            "multi fusion requires broadcast indexing"
        );
        let precision = match data_precision {
            runmat_accelerate_api::ProviderPrecision::F32 => NumericPrecision::F32,
            runmat_accelerate_api::ProviderPrecision::F64 => NumericPrecision::F64,
        };
        runmat_accelerate_api::validate_fused_output_classes(data_precision, output_classes)?;
        ensure!(!inputs.is_empty(), "fused_elementwise_multi: no inputs");
        ensure!(
            !output_classes.is_empty(),
            "fused_elementwise_multi: num_outputs is zero"
        );
        ensure!(
            len <= u32::MAX as usize,
            "fused_elementwise_multi: tensor too large"
        );
        fusion_recipe::validate_output_words(
            output_shape,
            len,
            &FusedElementwiseRepresentation {
                input_precision: precision,
                output_precision: precision,
                output_storage: GpuTensorStorage::Real,
                logical_output: false,
            },
        )?;
        let workgroup_size = self.workgroup_config.scalar;
        let chunk_capacity =
            crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup_size)
                .ok_or_else(|| anyhow!("fused_elementwise_multi: workgroup size is zero"))?
                as usize;
        let storage_bindings = checked_binding_count(
            "fused_elementwise_multi",
            inputs.len(),
            output_classes.len(),
        )?;
        let total_bindings = checked_binding_count("fused_elementwise_multi", storage_bindings, 1)?;
        validate_compute_binding_counts(
            "fused_elementwise_multi",
            storage_bindings,
            total_bindings,
            &self.adapter_limits,
        )?;
        ensure!(
            self.adapter_limits.max_uniform_buffers_per_shader_stage >= 1,
            "fused_elementwise_multi: uniform binding unavailable"
        );
        ensure!(
            workgroup_size <= self.adapter_limits.max_compute_workgroup_size_x
                && workgroup_size <= self.adapter_limits.max_compute_invocations_per_workgroup,
            "fused_elementwise_multi: workgroup exceeds enabled limits"
        );
        let uniform_bytes = crate::backend::wgpu_geometry::fusion_broadcast_uniform_bytes(
            inputs.len(),
            crate::backend::wgpu::params::BCAST_MAX_RANK,
        )
        .ok_or_else(|| anyhow!("fused_elementwise_multi: uniform extent overflows"))?;
        ensure!(
            uniform_bytes <= u64::from(self.adapter_limits.max_uniform_buffer_binding_size)
                && uniform_bytes <= self.adapter_limits.max_buffer_size,
            "fused_elementwise_multi: uniform exceeds enabled limits"
        );
        let element_size = match precision {
            NumericPrecision::F32 => 4,
            NumericPrecision::F64 => 8,
        };
        let output_bytes = BufferResidency::requested_bytes(len, element_size)
            .ok_or_else(|| anyhow!("fused_elementwise_multi: output extent overflows"))?;
        ensure!(
            output_bytes <= self.adapter_limits.max_buffer_size
                && output_bytes <= u64::from(self.adapter_limits.max_storage_buffer_binding_size),
            "fused_elementwise_multi: output exceeds enabled limits"
        );
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            len.min(chunk_capacity) as u32,
            workgroup_size,
        );
        ensure!(
            groups <= self.adapter_limits.max_compute_workgroups_per_dimension,
            "fused_elementwise_multi: dispatch exceeds enabled limits"
        );
        let entries = inputs
            .iter()
            .map(|handle| self.fusion_entry_for_precision(handle, precision))
            .collect::<Result<Vec<_>>>()?;
        let layouts = inputs
            .iter()
            .zip(&entries)
            .map(|(handle, entry)| {
                super::super::backend_shared::fusion_operand_layout(handle, entry, output_shape)
            })
            .collect::<Result<Vec<_>>>()?;
        let recipe = super::super::pipeline_recipe::PipelineRecipe::fusion(
            shader,
            inputs.len(),
            super::super::pipeline_recipe::FusionPipelineLayout::Multiple(output_classes.len()),
            workgroup_size,
        );
        Ok(MultiFrame {
            precision,
            recipe,
            entries,
            layouts,
            chunk_capacity,
            storage_bindings,
            total_bindings,
            output_classes: output_classes.to_vec(),
            shape: output_shape.to_vec(),
            len,
        })
    }
}
