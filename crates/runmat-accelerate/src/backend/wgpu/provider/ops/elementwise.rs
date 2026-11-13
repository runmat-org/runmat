use anyhow::{anyhow, ensure, Result};
use bytemuck::{bytes_of, Pod, Zeroable};
use runmat_accelerate_api::{unsupported_provider_operation, GpuTensorHandle, GpuTensorStorage};
use runmat_time::Instant;
use std::sync::Arc;

use super::backend_shared::{
    checked_binding_count, gpu_dispatch_length_limit_error, validate_compute_binding_counts,
};
use super::backend_types::WgpuProvider;
use crate::backend::wgpu::residency::BufferUsageClass;
use crate::backend::wgpu::resources::UniformBufferKey;
use crate::backend::wgpu::shaders::classification::classification_shader;
use crate::backend::wgpu::shaders::elementwise::{
    complex_from_real_imag_shader, complex_from_real_shader, complex_unary_shader,
    real_unary_shader, round_digits_shader, ComplexUnaryOp,
};
use crate::backend::wgpu::shaders::logical::{
    ELEM_EQ_SHADER_F32, ELEM_EQ_SHADER_F64, ELEM_GE_SHADER_F32, ELEM_GE_SHADER_F64,
    ELEM_GT_SHADER_F32, ELEM_GT_SHADER_F64, ELEM_LE_SHADER_F32, ELEM_LE_SHADER_F64,
    ELEM_LT_SHADER_F32, ELEM_LT_SHADER_F64, ELEM_NE_SHADER_F32, ELEM_NE_SHADER_F64,
    LOGICAL_AND_SHADER_F32, LOGICAL_AND_SHADER_F64, LOGICAL_NOT_SHADER_F32, LOGICAL_NOT_SHADER_F64,
    LOGICAL_OR_SHADER_F32, LOGICAL_OR_SHADER_F64, LOGICAL_XOR_SHADER_F32, LOGICAL_XOR_SHADER_F64,
};
use crate::backend::wgpu::types::NumericPrecision;

#[path = "elementwise/broadcast_plan.rs"]
mod broadcast_plan;
#[path = "elementwise/complex.rs"]
mod complex;
#[path = "elementwise/complex_broadcast.rs"]
mod complex_broadcast;
#[path = "elementwise/fusion_recipe.rs"]
mod fusion_recipe;
#[path = "elementwise_indexed.rs"]
mod indexed;
#[path = "elementwise/numeric_output.rs"]
mod numeric_output;
#[path = "elementwise/scoped_generic.rs"]
mod scoped_generic;
pub(super) use scoped_generic::check_kernel_limits;
#[path = "elementwise/validated_request.rs"]
mod validated_request;
pub(super) use fusion_recipe::{FrozenFusionRecipe, FusionUniformAbi};
pub(in crate::backend::wgpu::provider::backend) use validated_request::{MultiFrame, SingleFrame};
#[cfg(test)]
#[path = "elementwise/binary_max_tests.rs"]
mod binary_max_tests;
#[path = "elementwise/real_binary_recipe.rs"]
mod real_binary_recipe;
#[cfg(test)]
#[path = "elementwise/real_floating_tests.rs"]
mod real_floating_tests;
#[path = "elementwise/real_recipe.rs"]
mod real_recipe;
#[cfg(test)]
#[path = "elementwise/required_geometry_tests.rs"]
mod required_geometry_tests;
#[path = "elementwise/scalar_control.rs"]
mod scalar_control;
#[cfg(test)]
#[path = "elementwise/scalar_max_tests.rs"]
mod scalar_max_tests;
use indexed::{
    validate_indexed_layout, FusionMultiRequest, IndexedFusionObserved, MultiFusionLayout,
};
pub(super) use indexed::{IndexedFusionMultiRequest, IndexedFusionOperand};
use real_binary_recipe::{RealBinaryObserved, RealBinarySubmission};
use real_recipe::RealObserved;
use std::cell::{Cell, RefCell};

/// The selected standard binary path carries its pipeline and publication
/// effect marker together. Ordinary/recipe executions carry neither.
#[derive(Clone, Copy)]
struct SelectedBinaryExecution<'a> {
    pipeline: &'a super::validated_fusion::ValidatedFusionPipeline,
    effect: &'a Cell<bool>,
}

/// Internal shader ABI: input words can differ from the native output words.
/// The owning shader must declare both precisions explicitly.
#[derive(Clone, Copy)]
pub(super) struct FusedElementwiseRepresentation {
    pub(super) input_precision: NumericPrecision,
    pub(super) output_precision: NumericPrecision,
    pub(super) output_storage: GpuTensorStorage,
    pub(super) logical_output: bool,
}

/// Requested output geometry and shader ABI for one fused execution.
/// Allocation callbacks and resource observation remain separate effects.
#[derive(Clone, Copy)]
pub(super) struct FusedElementwiseWork<'a> {
    pub(super) output_shape: &'a [usize],
    pub(super) len: usize,
    pub(super) invocation_geometry: ElementwiseInvocationGeometry,
    pub(super) representation: FusedElementwiseRepresentation,
    /// None is the explicitly unprepared arbitrary-shader compatibility path.
    pub(super) uniform_abi: Option<FusionUniformAbi>,
}

#[derive(Clone, Copy)]
pub(super) enum ElementwiseInvocationGeometry {
    OutputLanes,
    ComplexElements {
        logical_elements: usize,
        uniform_abi: FusionUniformAbi,
    },
}

impl ElementwiseInvocationGeometry {
    fn complex_elements(
        logical_elements: usize,
        precision: NumericPrecision,
        uniform_abi: FusionUniformAbi,
    ) -> Result<Self> {
        let output_lanes = logical_elements
            .checked_mul(2)
            .ok_or_else(|| anyhow!("complex division output extent overflows"))?;
        let words_per_lane = match precision {
            NumericPrecision::F32 => 1,
            NumericPrecision::F64 => 2,
        };
        let word_count = output_lanes
            .checked_mul(words_per_lane)
            .ok_or_else(|| anyhow!("complex division word extent overflows"))?;
        ensure!(
            word_count <= u32::MAX as usize,
            "complex division raw-word address exceeds shader index range"
        );
        Ok(Self::ComplexElements {
            logical_elements,
            uniform_abi,
        })
    }

    fn invocation_len(
        self,
        output_lanes: usize,
        representation: &FusedElementwiseRepresentation,
        prepared_or_validated: bool,
    ) -> Result<usize> {
        match self {
            Self::OutputLanes => Ok(output_lanes),
            Self::ComplexElements {
                logical_elements, ..
            } => {
                ensure!(
                    !prepared_or_validated
                        && logical_elements.checked_mul(2) == Some(output_lanes)
                        && representation.output_storage == GpuTensorStorage::ComplexInterleaved
                        && !representation.logical_output,
                    "complex-element invocation geometry does not match physical output"
                );
                Ok(logical_elements)
            }
        }
    }
}

enum SingleFusionEvidence<'a> {
    Legacy {
        observed: Option<&'a RefCell<RealObserved>>,
        prepared: Option<&'a FrozenFusionRecipe>,
    },
    Validated {
        frame: &'a SingleFrame,
        pipeline: &'a super::validated_fusion::ValidatedFusionPipeline,
    },
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IntegerComparisonParams {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
    integer_type: u32,
    rank: u32,
    _pad0: u32,
    _pad1: u32,
    out_shape:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
    a_shape:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
    b_shape:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
    a_strides:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
    b_strides:
        [crate::backend::wgpu::params::AlignedU32; crate::backend::wgpu::params::BCAST_MAX_RANK],
}

impl WgpuProvider {
    async fn integer_comparison_exec(
        &self,
        operation: u32,
        operation_name: &str,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        use runmat_accelerate_api::ProviderExecutionError;
        let frame = scoped_generic::IntegerComparisonFrame::plan(self, operation_name, a, b)?;
        let len = frame.broadcast.len;
        if len == 0 {
            let allocated = Cell::new(false);
            let out = self
                .with_emission(|emission, _| {
                    let out =
                        self.create_storage_buffer(emission, 0, "runmat-integer-compare-empty")?;
                    allocated.set(true);
                    Ok::<_, anyhow::Error>(out)
                })?
                .map_err(|error| {
                    let typed = ProviderExecutionError::from(error);
                    if allocated.get() {
                        typed.after_effect()
                    } else {
                        typed
                    }
                })?;
            let handle = self
                .register_logical_buffer(
                    out,
                    frame.broadcast.output_shape,
                    0,
                    BufferUsageClass::Generic,
                )
                .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
            return Ok(handle);
        }
        let validated = self
            .prepare_validated_integer_comparison_pipeline(&frame.recipe)
            .await?;
        validated
            .require_integer_comparison(&frame.recipe)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        frame
            .require_live_inputs(self, a, b)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        let entry_a = &frame.lhs;
        let entry_b = &frame.rhs;
        let broadcast = &frame.broadcast;
        let integer_type = frame.integer_type;
        let allocated = Cell::new(false);
        let out_buffer = self
            .with_emission(|emission, pending| -> Result<_> {
                frame.require_live_inputs(self, a, b)?;
                let bind_group_layout = validated.layout.as_ref();
                let pipeline = validated.pipeline.as_ref();
                let out_buffer = self.create_storage_buffer_checked(
                    emission,
                    len,
                    "runmat-integer-compare-out",
                )?;
                allocated.set(true);
                let mut offset = 0usize;
                while offset < len {
                    let chunk_len = (len - offset).min(frame.chunk_capacity);
                    let params = IntegerComparisonParams {
                        len: chunk_len as u32,
                        op: operation,
                        offset: offset as u32,
                        total: len as u32,
                        integer_type: super::integer::integer_type_code(integer_type),
                        rank: broadcast.rank,
                        _pad0: 0,
                        _pad1: 0,
                        out_shape: broadcast.out_shape,
                        a_shape: broadcast.a_shape,
                        b_shape: broadcast.b_shape,
                        a_strides: broadcast.a_strides,
                        b_strides: broadcast.b_strides,
                    };
                    let params_buffer =
                        self.uniform_buffer(emission, &params, "runmat-integer-compare-params")?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-integer-compare-bg"),
                                layout: bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: entry_a.buffer.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: entry_b.buffer.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: out_buffer.as_ref().as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: params_buffer.as_entire_binding(),
                                    },
                                ],
                            });
                    let groups = frame.launch.dispatch_groups(chunk_len).ok_or_else(|| {
                        anyhow!("integer comparison: selected launch exceeds enabled limits")
                    })?;
                    crate::backend::wgpu::dispatch::elementwise::run(
                        emission,
                        pending,
                        pipeline,
                        &bind_group,
                        groups,
                    )?;
                    offset += chunk_len;
                }
                Ok(out_buffer)
            })?
            .map_err(|error| {
                let typed = ProviderExecutionError::from(error);
                if allocated.get() {
                    typed.after_effect()
                } else {
                    typed.with_unproven_effects()
                }
            })?;
        let handle = self
            .register_logical_buffer(
                out_buffer,
                broadcast.output_shape.clone(),
                len,
                BufferUsageClass::Generic,
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        Ok(handle)
    }

    fn effective_storage_for_entry(
        &self,
        handle: &GpuTensorHandle,
        entry: &super::backend_types::BufferEntry,
    ) -> runmat_accelerate_api::GpuTensorStorage {
        if runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            || entry.storage == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        } else {
            runmat_accelerate_api::GpuTensorStorage::Real
        }
    }

    pub(crate) fn logical_islogical_exec(&self, a: &GpuTensorHandle) -> Result<bool> {
        let _ = self.get_entry(a)?;
        Ok(runmat_accelerate_api::handle_is_logical(a))
    }

    pub(crate) fn logical_isreal_exec(&self, a: &GpuTensorHandle) -> Result<bool> {
        let entry = self.get_entry(a)?;
        Ok(self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
    }

    pub(crate) fn complex_from_real_exec(&self, real: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(real)?;
        ensure!(
            runmat_accelerate_api::handle_storage(real)
                != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            "complex_from_real requires a real-valued input"
        );
        let out_len = entry
            .len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("complex_from_real: output length overflow"))?;
        let handle = if out_len == 0 {
            let buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-complex-from-real-empty")
            })??;
            self.register_existing_buffer_with_storage(
                buffer,
                entry.shape,
                out_len,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            )?
        } else {
            let shader = complex_from_real_shader(self.precision);
            self.fused_elementwise_with_telemetry_and_storage_exec(
                &shader,
                std::slice::from_ref(real),
                &entry.shape,
                out_len,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            )?
        };
        Ok(handle)
    }

    pub(crate) fn complex_from_real_imag_exec(
        &self,
        real: &GpuTensorHandle,
        imag: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_real = self.get_entry(real)?;
        let entry_imag = self.get_entry(imag)?;
        ensure!(
            runmat_accelerate_api::handle_storage(real)
                != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                && runmat_accelerate_api::handle_storage(imag)
                    != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            "complex_from_real_imag requires real-valued inputs"
        );

        let real_scalar = entry_real.len == 1;
        let imag_scalar = entry_imag.len == 1;
        let out_shape = if entry_real.shape == entry_imag.shape {
            entry_real.shape.clone()
        } else if real_scalar {
            entry_imag.shape.clone()
        } else if imag_scalar {
            entry_real.shape.clone()
        } else {
            return Err(anyhow!(
                "complex_from_real_imag: shape mismatch between inputs"
            ));
        };
        let logical_len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("complex_from_real_imag: output shape overflow"))?;
        let out_len = logical_len
            .checked_mul(2)
            .ok_or_else(|| anyhow!("complex_from_real_imag: output length overflow"))?;
        let handle = if out_len == 0 {
            let buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-complex-from-real-imag-empty")
            })??;
            self.register_existing_buffer_with_storage(
                buffer,
                out_shape,
                out_len,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            )?
        } else {
            let shader = complex_from_real_imag_shader(self.precision, real_scalar, imag_scalar);
            self.fused_elementwise_with_telemetry_and_storage_exec(
                &shader,
                &[real.clone(), imag.clone()],
                &out_shape,
                out_len,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            )?
        };
        Ok(handle)
    }

    pub(crate) fn unary_sin_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sin, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Sin, true)
    }

    pub(crate) fn unary_sinc_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sinc, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Sinc, true)
    }

    pub(crate) fn unary_cos_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cos, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Cos, true)
    }

    pub(crate) fn unary_sinh_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sinh, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Sinh, true)
    }

    pub(crate) fn unary_cosh_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cosh, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Cosh, true)
    }

    pub(crate) fn unary_tan_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Tan, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Tan, true)
    }

    pub(crate) fn unary_sign_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sign, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Sign, true)
    }

    pub(crate) fn unary_real_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return Ok(a.clone());
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Real, false)
    }

    pub(crate) fn unary_imag_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.fill_exec(&entry.shape, 0.0);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Imag, false)
    }

    pub(crate) fn unary_abs_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Abs, false)
    }

    pub(crate) fn unary_conj_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Conj, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Conj, true)
    }

    pub(crate) fn unary_angle_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if self.effective_storage_for_entry(a, &entry)
            != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            return self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Angle, a);
        }
        self.complex_unary_exec(a, ComplexUnaryOp::Angle, false)
    }

    fn complex_unary_exec(
        &self,
        a: &GpuTensorHandle,
        op: ComplexUnaryOp,
        output_complex: bool,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        ensure!(
            self.effective_storage_for_entry(a, &entry)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            "complex unary operation requires complex-interleaved input"
        );
        ensure!(
            entry.len % 2 == 0,
            "complex unary operation requires even interleaved buffer length"
        );
        let len = if output_complex {
            entry.len
        } else {
            entry.len / 2
        };
        if len != 0
            && crate::backend::wgpu::shaders::elementwise::complex_unary_requires_ieee_words(op)
        {
            self.require_ieee_words(entry.precision)?;
        }
        let shader = complex_unary_shader(op, entry.precision);
        let output_storage = if output_complex {
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        } else {
            runmat_accelerate_api::GpuTensorStorage::Real
        };
        let out = self.fused_elementwise_with_representation_telemetry_exec(
            &shader,
            std::slice::from_ref(a),
            &entry.shape,
            len,
            FusedElementwiseRepresentation {
                input_precision: entry.precision,
                output_precision: entry.precision,
                output_storage,
                logical_output: false,
            },
        )?;
        self.inherit_elementwise_layout(out, a)
    }

    pub(crate) fn unary_pow2_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let out = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, a)?;
        // Record squared->base mapping for later reduction fusion (moments reuse).
        self.remember_square(&out, a)?;
        Ok(out)
    }

    pub(crate) async fn round_digits_exec(
        &self,
        a: &GpuTensorHandle,
        digits: i32,
        significant: bool,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        use runmat_accelerate_api::ProviderExecutionError;
        let frame = scoped_generic::RoundDigitsFrame::plan(self, a, digits, significant)?;
        let len = frame.scalar_len;
        if len != 0 {
            let validated = self
                .prepare_validated_round_digits_pipeline(&frame.recipe, significant)
                .await?;
            validated
                .require_round_digits(&frame.recipe, frame.significant)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            frame
                .require_live_input(self, a)
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
            return self.round_digits_prepared_exec(a, digits, frame, validated);
        }
        let out_buffer = self.with_emission(|emission, _| {
            self.create_storage_buffer_checked(emission, 0, "runmat-round-digits-empty")
        })??;
        let out = self
            .register_numeric_buffer(
                out_buffer,
                super::backend_types::NumericBufferRegistration {
                    shape: frame.input.shape,
                    len: frame.input.len,
                    physical_element_type: frame.input.element_type,
                    storage: frame.storage,
                    allocated_bytes: frame.allocated_bytes,
                    usage: BufferUsageClass::Generic,
                },
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        self.inherit_elementwise_layout(out, a)
            .map_err(|error| ProviderExecutionError::from(error).after_effect())
    }

    fn round_digits_prepared_exec(
        &self,
        a: &GpuTensorHandle,
        digits: i32,
        frame: scoped_generic::RoundDigitsFrame,
        validated: Arc<super::validated_fusion::ValidatedFusionPipeline>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        use runmat_accelerate_api::ProviderExecutionError;
        let len = frame.scalar_len;
        let allocated = Cell::new(false);
        let (out_buffer, start) = self
            .with_emission(|emission, pending| -> Result<_> {
                frame.require_live_input(self, a)?;
                let out_buffer =
                    self.create_storage_buffer_checked(emission, len, "runmat-round-digits-out")?;
                allocated.set(true);
                let layout = validated.layout.as_ref();
                let pipeline = validated.pipeline.as_ref();

                let params_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-round-digits-params"),
                        size: std::mem::size_of::<crate::backend::wgpu::params::RoundDigitsParams>()
                            as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-round-digits-bind"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: frame.input.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_ref().as_entire_binding(),
                            },
                        ],
                    });

                let mut offset = 0usize;
                let start = Instant::now();
                while offset < len {
                    let chunk_len = (len - offset).min(frame.chunk_capacity);
                    let params = crate::backend::wgpu::params::RoundDigitsParams {
                        len: chunk_len as u32,
                        offset: offset as u32,
                        total: len as u32,
                        digits,
                    };
                    emission.write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
                    let groups = frame
                        .launch
                        .dispatch_groups(chunk_len)
                        .ok_or_else(|| anyhow!("round_digits launch exceeds enabled limits"))?;
                    crate::backend::wgpu::dispatch::elementwise::run(
                        emission,
                        pending,
                        pipeline,
                        &bind_group,
                        groups,
                    )?;
                    offset += chunk_len;
                }

                Ok((out_buffer, start))
            })?
            .map_err(|error| {
                let typed = ProviderExecutionError::from(error);
                if allocated.get() {
                    typed.after_effect()
                } else {
                    typed.with_unproven_effects()
                }
            })?;
        let out = self
            .register_numeric_buffer(
                out_buffer,
                super::backend_types::NumericBufferRegistration {
                    shape: frame.input.shape,
                    len: frame.input.len,
                    physical_element_type: frame.input.element_type,
                    storage: frame.storage,
                    allocated_bytes: frame.allocated_bytes,
                    usage: BufferUsageClass::Generic,
                },
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        self.inherit_elementwise_layout(out, a)
            .map_err(|error| ProviderExecutionError::from(error).after_effect())
    }

    pub(crate) fn pow2_scale_exec(
        &self,
        mantissa: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            mantissa.shape() == exponent.shape(),
            "pow2_scale requires matching shapes"
        );
        // This internal synchronous product is floating-only. Check the
        // mantissa before creating the floating Pow2 temporary so a native
        // integer view cannot reach a cold binary pipeline after that effect.
        let _ = self.get_entry(mantissa)?;
        let _ = self.get_entry(exponent)?;
        let pow = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, exponent)?;
        let result = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            mantissa,
            &pow,
        );
        drop(pow);
        result
    }

    pub(crate) async fn elem_eq_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.get_entry_raw(a)?.integer_type().is_some()
            || self.get_entry_raw(b)?.integer_type().is_some()
        {
            return self.integer_comparison_exec(0, "elem_eq", a, b).await;
        }
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_eq: shape mismatch between inputs").into());
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-elem-eq-empty")
                })?
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
                })?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_EQ_SHADER_F64,
                NumericPrecision::F32 => ELEM_EQ_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?
        };
        Ok(handle)
    }

    pub(crate) async fn elem_ne_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.get_entry_raw(a)?.integer_type().is_some()
            || self.get_entry_raw(b)?.integer_type().is_some()
        {
            return self.integer_comparison_exec(1, "elem_ne", a, b).await;
        }
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_ne: shape mismatch between inputs").into());
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-elem-ne-empty")
                })?
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
                })?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_NE_SHADER_F64,
                NumericPrecision::F32 => ELEM_NE_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?
        };
        Ok(handle)
    }
    pub(crate) async fn elem_lt_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.get_entry_raw(a)?.integer_type().is_some()
            || self.get_entry_raw(b)?.integer_type().is_some()
        {
            return self.integer_comparison_exec(2, "elem_lt", a, b).await;
        }
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_lt: shape mismatch between inputs").into());
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-elem-lt-empty")
                })?
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
                })?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_LT_SHADER_F64,
                NumericPrecision::F32 => ELEM_LT_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?
        };
        Ok(handle)
    }

    pub(crate) async fn elem_le_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.get_entry_raw(a)?.integer_type().is_some()
            || self.get_entry_raw(b)?.integer_type().is_some()
        {
            return self.integer_comparison_exec(3, "elem_le", a, b).await;
        }
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_le: shape mismatch between inputs").into());
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-elem-le-empty")
                })?
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
                })?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_LE_SHADER_F64,
                NumericPrecision::F32 => ELEM_LE_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?
        };
        Ok(handle)
    }

    pub(crate) async fn elem_gt_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.get_entry_raw(a)?.integer_type().is_some()
            || self.get_entry_raw(b)?.integer_type().is_some()
        {
            return self.integer_comparison_exec(4, "elem_gt", a, b).await;
        }
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_gt: shape mismatch between inputs").into());
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-elem-gt-empty")
                })?
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
                })?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_GT_SHADER_F64,
                NumericPrecision::F32 => ELEM_GT_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?
        };
        Ok(handle)
    }

    pub(crate) async fn elem_ge_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.get_entry_raw(a)?.integer_type().is_some()
            || self.get_entry_raw(b)?.integer_type().is_some()
        {
            return self.integer_comparison_exec(5, "elem_ge", a, b).await;
        }
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_ge: shape mismatch between inputs").into());
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self
                .with_emission(|emission, _| {
                    self.create_storage_buffer(emission, 0, "runmat-elem-ge-empty")
                })?
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
                })?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_GE_SHADER_F64,
                NumericPrecision::F32 => ELEM_GE_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?
        };
        Ok(handle)
    }

    pub(crate) fn logical_and_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.storage != GpuTensorStorage::Real || entry_b.storage != GpuTensorStorage::Real {
            return Err(unsupported_provider_operation(
                "logical kernel requires real storage",
            ));
        }
        if entry_a.shape != entry_b.shape {
            super::backend_shared::matlab_broadcast_shapes(
                "logical",
                &entry_a.shape,
                &entry_b.shape,
                entry_a.shape.len().max(entry_b.shape.len()),
            )?;
            return Err(unsupported_provider_operation(
                "logical_and: broadcast kernel unavailable",
            ));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-logical-and-empty")
            })??;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_AND_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_AND_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
        };
        Ok(handle)
    }

    pub(crate) fn logical_or_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.storage != GpuTensorStorage::Real || entry_b.storage != GpuTensorStorage::Real {
            return Err(unsupported_provider_operation(
                "logical kernel requires real storage",
            ));
        }
        if entry_a.shape != entry_b.shape {
            super::backend_shared::matlab_broadcast_shapes(
                "logical",
                &entry_a.shape,
                &entry_b.shape,
                entry_a.shape.len().max(entry_b.shape.len()),
            )?;
            return Err(unsupported_provider_operation(
                "logical_or: broadcast kernel unavailable",
            ));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-logical-or-empty")
            })??;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_OR_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_OR_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
        };
        Ok(handle)
    }
    pub(crate) fn logical_xor_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.storage != GpuTensorStorage::Real || entry_b.storage != GpuTensorStorage::Real {
            return Err(unsupported_provider_operation(
                "logical kernel requires real storage",
            ));
        }
        if entry_a.shape != entry_b.shape {
            super::backend_shared::matlab_broadcast_shapes(
                "logical",
                &entry_a.shape,
                &entry_b.shape,
                entry_a.shape.len().max(entry_b.shape.len()),
            )?;
            return Err(unsupported_provider_operation(
                "logical_xor: broadcast kernel unavailable",
            ));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-logical-xor-empty")
            })??;
            self.register_logical_buffer(out, entry_a.shape, 0, BufferUsageClass::Generic)?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_XOR_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_XOR_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                &[a.clone(), b.clone()],
                &entry_a.shape,
                len,
            )?
        };
        Ok(handle)
    }

    pub(crate) fn logical_not_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.storage != GpuTensorStorage::Real {
            return Err(unsupported_provider_operation(
                "logical kernel requires real storage",
            ));
        }
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-logical-not-empty")
            })??;
            self.register_logical_buffer(out, entry.shape, 0, BufferUsageClass::Generic)?
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_NOT_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_NOT_SHADER_F32,
            };
            self.fused_logical_with_telemetry_exec(
                shader,
                std::slice::from_ref(a),
                &entry.shape,
                len,
            )?
        };
        Ok(handle)
    }

    pub(crate) fn logical_isfinite_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-logical-isfinite-empty")
            })??;
            self.register_logical_buffer(out, entry.shape, 0, BufferUsageClass::Generic)?
        } else {
            let shader = classification_shader(
                self.precision,
                runmat_builtins::NumericClassificationPredicate::Finite,
            );
            self.fused_logical_with_telemetry_exec(
                &shader,
                std::slice::from_ref(a),
                &entry.shape,
                len,
            )?
        };
        Ok(handle)
    }

    pub(crate) fn logical_isnan_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-logical-isnan-empty")
            })??;
            self.register_logical_buffer(out, entry.shape, 0, BufferUsageClass::Generic)?
        } else {
            let shader = classification_shader(
                self.precision,
                runmat_builtins::NumericClassificationPredicate::Nan,
            );
            self.fused_logical_with_telemetry_exec(
                &shader,
                std::slice::from_ref(a),
                &entry.shape,
                len,
            )?
        };
        Ok(handle)
    }

    pub(crate) fn logical_isinf_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "runmat-logical-isinf-empty")
            })??;
            self.register_logical_buffer(out, entry.shape, 0, BufferUsageClass::Generic)?
        } else {
            let shader = classification_shader(
                self.precision,
                runmat_builtins::NumericClassificationPredicate::Infinite,
            );
            self.fused_logical_with_telemetry_exec(
                &shader,
                std::slice::from_ref(a),
                &entry.shape,
                len,
            )?
        };
        Ok(handle)
    }
    pub(crate) fn unary_op_exec(
        &self,
        op: crate::backend::wgpu::types::UnaryOpCode,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.unary_op_exec_recorded(op, a, None, None)
    }

    pub(super) fn unary_std_selected_exec(
        &self,
        op: crate::backend::wgpu::types::UnaryOpCode,
        a: &GpuTensorHandle,
        validated: &super::validated_fusion::ValidatedFusionPipeline,
        effect: &std::cell::Cell<bool>,
    ) -> Result<GpuTensorHandle> {
        if std::env::var("RUNMAT_DISABLE_UNARY").is_ok() {
            return Err(anyhow!("unary ops disabled via RUNMAT_DISABLE_UNARY"));
        }
        ensure!(
            matches!(
                op,
                crate::backend::wgpu::types::UnaryOpCode::Abs
                    | crate::backend::wgpu::types::UnaryOpCode::Sqrt
            ),
            "std selected unary operation is not admitted"
        );
        let entry = self.get_entry(a)?;
        let shader = real_unary_shader(op, entry.precision);
        let frame = self.plan_ordinary_single_fusion(
            &shader,
            std::slice::from_ref(a),
            &entry.shape,
            entry.len,
            match entry.precision {
                NumericPrecision::F32 => runmat_accelerate_api::ProviderElementType::F32,
                NumericPrecision::F64 => runmat_accelerate_api::ProviderElementType::F64,
            },
        )?;
        self.fused_elementwise_typed_exec_recorded(
            &shader,
            std::slice::from_ref(a),
            FusedElementwiseWork {
                output_shape: &entry.shape,
                len: entry.len,
                invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                representation: FusedElementwiseRepresentation {
                    input_precision: entry.precision,
                    output_precision: entry.precision,
                    output_storage: GpuTensorStorage::Real,
                    logical_output: false,
                },
                uniform_abi: None,
            },
            |_| {
                effect.set(true);
                Ok(())
            },
            SingleFusionEvidence::Validated {
                frame: &frame,
                pipeline: validated,
            },
            None,
        )
    }

    fn unary_op_exec_recorded(
        &self,
        op: crate::backend::wgpu::types::UnaryOpCode,
        a: &GpuTensorHandle,
        observed: Option<&RefCell<RealObserved>>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        if std::env::var("RUNMAT_DISABLE_UNARY").is_ok() {
            return Err(anyhow!("unary ops disabled via RUNMAT_DISABLE_UNARY"));
        }
        let entry_a = self.real_floating_entry(a)?;
        ensure!(
            self.effective_storage_for_entry(a, &entry_a)
                != runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            "real unary operation does not support complex-interleaved buffers"
        );
        let len = entry_a.len;
        if len != 0
            && crate::backend::wgpu::shaders::elementwise::real_unary_requires_ieee_words(op)
        {
            self.require_ieee_words(entry_a.precision)?;
        }
        let shader = real_unary_shader(op, entry_a.precision);
        let start = Instant::now();
        let result = self.fused_elementwise_typed_exec_recorded(
            &shader,
            std::slice::from_ref(a),
            FusedElementwiseWork {
                output_shape: &entry_a.shape,
                len,
                invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                uniform_abi: Some(FusionUniformAbi::Simple),
                representation: FusedElementwiseRepresentation {
                    input_precision: entry_a.precision,
                    output_precision: entry_a.precision,
                    output_storage: GpuTensorStorage::Real,
                    logical_output: false,
                },
            },
            |_| Ok(()),
            SingleFusionEvidence::Legacy {
                observed,
                prepared: None,
            },
            admission,
        );
        if result.is_ok() {
            self.telemetry
                .record_fused_elementwise_duration(start.elapsed());
        }
        result
    }
    /// Exact upper bound on the explicitly allocated output and per-chunk
    /// scalar uniforms. The buffer pool may reuse storage, which only lowers
    /// new allocation demand; pending submissions can retain every chunk.
    pub(crate) fn scalar_op_buffer_demand(&self, len: u64) -> Option<u64> {
        self.scalar_op_buffer_demand_for_precision(len, self.precision)
    }

    fn scalar_op_buffer_demand_for_precision(
        &self,
        len: u64,
        precision: NumericPrecision,
    ) -> Option<u64> {
        let element_size = match precision {
            NumericPrecision::F32 => 4,
            NumericPrecision::F64 => 8,
        };
        let output = len.max(1).checked_mul(element_size)?;
        if len == 0 {
            return Some(output);
        }
        let launch = self.real_scalar_pipeline(precision).ok()?.scalar_launch()?;
        let capacity = launch.chunk_capacity()?;
        let uniform = launch.uniform_requested_bytes();
        output.checked_add(len.div_ceil(capacity).checked_mul(uniform)?)
    }

    pub(crate) fn binary_same_shape_buffer_demand(&self, len: u64) -> Option<u64> {
        self.binary_same_shape_buffer_demand_for_precision(len, self.precision)
    }

    fn binary_same_shape_buffer_demand_for_precision(
        &self,
        len: u64,
        precision: NumericPrecision,
    ) -> Option<u64> {
        let element_bytes = match precision {
            NumericPrecision::F32 => 4,
            NumericPrecision::F64 => 8,
        };
        let output = len.max(1).checked_mul(element_bytes)?;
        if len == 0 {
            return Some(output);
        }
        let capacity = u64::try_from(self.binary_linear_launch().ok()?.chunk_capacity()?).ok()?;
        let uniform = (std::mem::size_of::<crate::backend::wgpu::params::LenOpParams>() as u64)
            .max(1)
            .div_ceil(256)
            .checked_mul(256)?;
        output.checked_add(len.div_ceil(capacity).checked_mul(uniform)?)
    }

    pub(crate) fn binary_broadcast_buffer_demand(&self, len: u64) -> Option<u64> {
        self.binary_broadcast_buffer_demand_for_precision(len, self.precision)
    }

    fn binary_broadcast_buffer_demand_for_precision(
        &self,
        len: u64,
        precision: NumericPrecision,
    ) -> Option<u64> {
        let element_bytes = match precision {
            NumericPrecision::F32 => 4,
            NumericPrecision::F64 => 8,
        };
        broadcast_plan::BroadcastRoute::requested_buffer_bound(len, element_bytes)
    }

    pub(crate) fn scalar_op_exec(
        &self,
        op: crate::backend::wgpu::types::ScalarOpCode,
        a: &GpuTensorHandle,
        scalar: f64,
    ) -> Result<GpuTensorHandle> {
        self.scalar_op_exec_recorded(op, a, scalar, None, None)
    }

    fn scalar_op_exec_recorded(
        &self,
        op: crate::backend::wgpu::types::ScalarOpCode,
        a: &GpuTensorHandle,
        scalar: f64,
        observed: Option<&RefCell<RealObserved>>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        let entry_a = if runmat_accelerate_api::handle_storage(a)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            self.get_entry(a)?
        } else {
            self.real_floating_entry(a)?
        };
        let selected_precision = entry_a.precision;
        let scalar_precision = match selected_precision {
            NumericPrecision::F32 => runmat_accelerate_api::ProviderPrecision::F32,
            NumericPrecision::F64 => runmat_accelerate_api::ProviderPrecision::F64,
        };
        scalar_control::admit(op, scalar_precision, scalar)?;
        if runmat_accelerate_api::handle_storage(a)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            let scalar_handle = self.fill_exec(&entry_a.shape, scalar)?;
            let binary_op = match op {
                crate::backend::wgpu::types::ScalarOpCode::Add => {
                    crate::backend::wgpu::types::BinaryOpCode::Add
                }
                crate::backend::wgpu::types::ScalarOpCode::Sub => {
                    crate::backend::wgpu::types::BinaryOpCode::Sub
                }
                crate::backend::wgpu::types::ScalarOpCode::Mul => {
                    crate::backend::wgpu::types::BinaryOpCode::Mul
                }
                crate::backend::wgpu::types::ScalarOpCode::Div => {
                    crate::backend::wgpu::types::BinaryOpCode::Div
                }
                crate::backend::wgpu::types::ScalarOpCode::RSub => {
                    let result = self.binary_op_exec(
                        crate::backend::wgpu::types::BinaryOpCode::Sub,
                        &scalar_handle,
                        a,
                    );
                    drop(scalar_handle);
                    return result;
                }
                crate::backend::wgpu::types::ScalarOpCode::RDiv => {
                    let result = self.binary_op_exec(
                        crate::backend::wgpu::types::BinaryOpCode::Div,
                        &scalar_handle,
                        a,
                    );
                    drop(scalar_handle);
                    return result;
                }
                _ => {
                    drop(scalar_handle);
                    return Err(anyhow!(
                        "complex scalar operation is not supported for this operator"
                    ));
                }
            };
            let result = self.binary_op_exec(binary_op, a, &scalar_handle);
            drop(scalar_handle);
            return result;
        }
        let len = entry_a.len;
        let pipeline = self.real_scalar_pipeline_for_operation(op, selected_precision)?;
        let launch = pipeline
            .scalar_launch()
            .ok_or_else(|| anyhow!("scalar pipeline has no frozen launch recipe"))?;
        let chunk_capacity = usize::try_from(
            launch
                .chunk_capacity()
                .ok_or_else(|| anyhow!("scalar launch capacity overflow"))?,
        )?;
        self.validate_real_floating_output_extent(len, selected_precision)?;
        self.scalar_op_buffer_demand_for_precision(u64::try_from(len)?, selected_precision)
            .ok_or_else(|| anyhow!("selected real scalar resource extent overflow"))?;
        let (out_buffer, start) = self.with_emission(|emission, pending| -> Result<_> {
            let out_buffer = self.create_storage_buffer_checked_with_element_size(
                emission,
                len,
                "runmat-scalar-out",
                BufferUsageClass::Generic,
                entry_a.element_type.element_size(),
            )?;
            if let Some(observed) = observed {
                observed.borrow_mut().output(&out_buffer);
            }
            if len == 0 {
                return Ok((out_buffer, None));
            }
            if len > (u32::MAX as usize) {
                return Err(gpu_dispatch_length_limit_error("scalar_op", len));
            }
            let mut offset = 0usize;
            let start = Instant::now();
            while offset < len {
                let remaining = len - offset;
                let chunk_len = remaining.min(chunk_capacity);
                let params_buffer = match selected_precision {
                    NumericPrecision::F64 => {
                        let params = crate::backend::wgpu::params::ScalarParamsF64 {
                            len: chunk_len as u32,
                            op: op as u32,
                            offset: offset as u32,
                            total: len as u32,
                            scalar,
                            _pad_scalar: 0.0,
                            _pad_tail: 0.0,
                            _pad_tail2: 0.0,
                            _pad_tail3: 0.0,
                            _pad_tail4: 0.0,
                        };
                        let buf = self.kernel_resources.uniform_buffer(
                            emission,
                            UniformBufferKey::ScalarParamsF64,
                            std::mem::size_of::<crate::backend::wgpu::params::ScalarParamsF64>()
                                as u64,
                            "runmat-scalar-params-f64",
                        )?;
                        emission.write_buffer(buf.as_ref(), 0, bytes_of(&params));
                        buf
                    }
                    NumericPrecision::F32 => {
                        let params = crate::backend::wgpu::params::ScalarParamsF32 {
                            len: chunk_len as u32,
                            op: op as u32,
                            offset: offset as u32,
                            total: len as u32,
                            scalar: scalar as f32,
                            _pad_scalar: [0.0; 3],
                            _pad_tail: [0.0; 4],
                            _pad_tail2: [0.0; 4],
                        };
                        let buf = self.kernel_resources.uniform_buffer(
                            emission,
                            UniformBufferKey::ScalarParamsF32,
                            std::mem::size_of::<crate::backend::wgpu::params::ScalarParamsF32>()
                                as u64,
                            "runmat-scalar-params-f32",
                        )?;
                        emission.write_buffer(buf.as_ref(), 0, bytes_of(&params));
                        buf
                    }
                };
                if let Some(observed) = observed {
                    observed.borrow_mut().uniform(&params_buffer)?;
                }
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-scalar-bind"),
                        layout: &pipeline.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry_a.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    launch.workgroup(),
                );
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    &pipeline.get(emission)?.pipeline,
                    &bind_group,
                    groups,
                )?;
                if let Some(observed) = observed {
                    observed.borrow_mut().submitted(groups > 0)?;
                }
                offset += chunk_len;
            }
            Ok((out_buffer, Some(start)))
        })??;
        let handle = self.register_real_floating_output_admitted(
            out_buffer,
            entry_a.shape,
            len,
            selected_precision,
            admission,
        )?;
        if let Some(start) = start {
            self.telemetry
                .record_fused_elementwise_duration(start.elapsed());
        }
        self.inherit_elementwise_layout(handle, a)
    }

    pub(crate) async fn binary_op_public_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        operation_name: &str,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        use crate::backend::wgpu::types::BinaryOpCode;
        use runmat_accelerate_api::ProviderExecutionError;
        if std::env::var("RUNMAT_DISABLE_BINARY").is_ok() {
            return Err(ProviderExecutionError::from(anyhow!(
                "binary ops disabled via RUNMAT_DISABLE_BINARY"
            )));
        }
        let integer = self.get_entry_raw(a)?.integer_type().is_some()
            || self.get_entry_raw(b)?.integer_type().is_some();
        if integer {
            return match op {
                BinaryOpCode::Min | BinaryOpCode::Max => {
                    self.integer_minmax_exec(op, operation_name, a, b).await
                }
                BinaryOpCode::Add | BinaryOpCode::Sub | BinaryOpCode::Mul
                | BinaryOpCode::Div | BinaryOpCode::Pow => {
                    self.integer_arithmetic_exec(op, operation_name, a, b).await
                }
                _ => Err(ProviderExecutionError::from(anyhow!(
                    "native integer gpuArray arithmetic requires a dedicated integer provider kernel"
                ))),
            };
        }
        // Admit the operation's physical precision before effects. The native
        // entry gate remains strict for complex kernels, while real kernels
        // can use their selected F32 pipelines on an F64 provider.
        let raw_a = self.get_entry_raw(a)?;
        let raw_b = self.get_entry_raw(b)?;
        let complex = self.effective_storage_for_entry(a, &raw_a)
            == GpuTensorStorage::ComplexInterleaved
            || self.effective_storage_for_entry(b, &raw_b) == GpuTensorStorage::ComplexInterleaved;
        if complex && op == BinaryOpCode::Div {
            let (entry_a, entry_b) = self.complex_division_inputs(a, b)?;
            if entry_a.len != 0 && entry_b.len != 0 {
                self.require_ieee_words(entry_a.precision)
                    .map_err(ProviderExecutionError::from)?;
            }
            if entry_a.shape != entry_b.shape {
                complex_broadcast::admit_shapes(&entry_a.shape, &entry_b.shape)?;
            }
        } else if complex {
            let (entry_a, entry_b) = (self.get_entry(a)?, self.get_entry(b)?);
            if entry_a.shape != entry_b.shape {
                // Pure shape admission retains its typed Unsupported cause.
                // Discovering the same refusal inside the unphased executor
                // cannot authorize replay because later work may have effects.
                complex_broadcast::admit_shapes(&entry_a.shape, &entry_b.shape)?;
            }
        } else {
            let (entry_a, entry_b) = self.real_floating_binary_inputs(a, b)?;
            let broadcast = entry_a.shape != entry_b.shape;
            // Select the exact physical pipeline before entering the legacy
            // executor. In particular Max's IEEE feature gate also applies to
            // empty outputs, which still allocate a physical buffer.
            let pipeline = self.real_binary_pipeline(entry_a.precision, broadcast, op)?;
            let len = if broadcast {
                broadcast_plan::BroadcastRoute::with_launch(
                    &entry_a.shape,
                    &entry_b.shape,
                    crate::backend::wgpu::params::BCAST_MAX_RANK,
                    entry_a.element_type.element_size() as u64,
                    self.binary_linear_launch()?,
                )?
                .len
            } else {
                entry_a.len
            };
            self.validate_real_floating_output_extent(len, entry_a.precision)?;
            if len != 0 || op == BinaryOpCode::Max {
                pipeline.preflight()?;
            }
        }
        // This inherited synchronous floating route can allocate or submit;
        // its unphased errors are never certified pre-effect replay causes.
        self.binary_op_exec(op, a, b)
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())
    }

    pub(crate) fn binary_op_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.binary_op_exec_recorded(op, a, b, None, None, None)
    }

    pub(super) fn binary_std_selected_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        validated: &super::validated_fusion::ValidatedFusionPipeline,
        effect: &std::cell::Cell<bool>,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            matches!(
                op,
                crate::backend::wgpu::types::BinaryOpCode::Add
                    | crate::backend::wgpu::types::BinaryOpCode::Sub
                    | crate::backend::wgpu::types::BinaryOpCode::Mul
                    | crate::backend::wgpu::types::BinaryOpCode::Div
            ),
            "std selected binary operation is not admitted"
        );
        let left = self.get_entry(a)?;
        let right = self.get_entry(b)?;
        ensure!(
            left.shape == right.shape
                && left.precision == self.precision
                && right.precision == self.precision
                && left.storage == GpuTensorStorage::Real
                && right.storage == GpuTensorStorage::Real
                && left.integer_type().is_none()
                && right.integer_type().is_none(),
            "std selected binary operands are not same-shape native real floating tensors"
        );
        self.binary_op_exec_recorded(
            op,
            a,
            b,
            None,
            Some(SelectedBinaryExecution {
                pipeline: validated,
                effect,
            }),
            None,
        )
    }

    pub(super) fn preflight_std_binary_input(&self, input: &GpuTensorHandle) -> Result<()> {
        let entry = self.get_entry(input)?;
        let selected = self.real_binary_pipeline(
            entry.precision,
            false,
            crate::backend::wgpu::types::BinaryOpCode::Mul,
        )?;
        selected.preflight()?;
        self.validate_real_floating_output_extent(entry.len, entry.precision)?;
        self.binary_linear_launch()?
            .chunk_capacity()
            .ok_or_else(|| anyhow!("std binary selected launch capacity overflows"))?;
        self.binary_same_shape_buffer_demand(u64::try_from(entry.len)?)
            .ok_or_else(|| anyhow!("std binary physical demand overflows"))?;
        Ok(())
    }

    fn binary_op_exec_recorded(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        observed: Option<&RefCell<RealBinaryObserved>>,
        selected: Option<SelectedBinaryExecution<'_>>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        let validated = selected.map(|frame| frame.pipeline);
        let effect = selected.map(|frame| frame.effect);
        if std::env::var("RUNMAT_DISABLE_BINARY").is_ok() {
            return Err(anyhow!("binary ops disabled via RUNMAT_DISABLE_BINARY"));
        }
        // The ordinary public integer hooks prepare asynchronously before effects.
        // Internal synchronous users of this routine admit floating storage via
        // get_entry; they cannot create an unvalidated cold integer pipeline.
        let raw_a = self.get_entry_raw(a)?;
        let raw_b = self.get_entry_raw(b)?;
        let storage_a = self.effective_storage_for_entry(a, &raw_a);
        let storage_b = self.effective_storage_for_entry(b, &raw_b);
        if storage_a == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            || storage_b == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            let (entry_a, entry_b) = if op == crate::backend::wgpu::types::BinaryOpCode::Div {
                self.complex_division_inputs(a, b)?
            } else {
                (self.get_entry(a)?, self.get_entry(b)?)
            };
            if entry_a.shape != entry_b.shape {
                return self.complex_binary_broadcast_exec(op, a, b, &entry_a, &entry_b);
            }
            return self.complex_binary_op_exec(op, a, b, &entry_a, &entry_b);
        }
        let (entry_a, entry_b) = self.real_floating_binary_inputs(a, b)?;
        if validated.is_some() {
            ensure!(
                entry_a.precision == self.precision && entry_a.shape == entry_b.shape,
                "std selected binary frame changed physical precision or shape"
            );
        }
        let selected_precision = entry_a.precision;
        if entry_a.shape != entry_b.shape {
            // Attempt general N-D broadcasted binary op
            return self.binary_op_broadcast_exec_recorded(op, a, b, observed, admission);
        }
        let len = entry_a.len;
        let selected_max = if op == crate::backend::wgpu::types::BinaryOpCode::Max {
            let pipeline = self.real_binary_pipeline(selected_precision, false, op)?;
            pipeline.preflight()?;
            Some(pipeline)
        } else {
            None
        };
        if len == 0 {
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked_with_element_size(
                    emission,
                    0,
                    "runmat-binary-out",
                    BufferUsageClass::Generic,
                    entry_a.element_type.element_size(),
                )
            })??;
            if let Some(effect) = effect {
                effect.set(true);
            }
            if let Some(observed) = observed {
                observed.borrow_mut().output(&out_buffer);
            }
            return self.register_real_floating_output_admitted(
                out_buffer,
                entry_a.shape,
                entry_a.len,
                selected_precision,
                admission,
            );
        }
        if len > (u32::MAX as usize) {
            return Err(gpu_dispatch_length_limit_error("binary_op", len));
        }
        let pipeline = match selected_max {
            Some(pipeline) => pipeline,
            None => {
                let pipeline = self.real_binary_pipeline(selected_precision, false, op)?;
                pipeline.preflight()?;
                pipeline
            }
        };
        self.validate_real_floating_output_extent(len, selected_precision)?;
        self.binary_same_shape_buffer_demand_for_precision(u64::try_from(len)?, selected_precision)
            .ok_or_else(|| anyhow!("selected real binary resource extent overflow"))?;
        let start = Instant::now();
        self.with_emission(|emission, pending| -> Result<()> {
            if let Some(observed) = observed {
                observed.borrow_mut().begin_effect();
            }
            let mut enc = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-binary-noop"),
                },
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-binary-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(match validated {
                Some(selected) => selected.pipeline.as_ref(),
                None => &pipeline.get(emission)?.pipeline,
            });
            drop(pass);
            self.submit(emission, pending, enc)?;
            if let Some(effect) = effect {
                effect.set(true);
            }
            if let Some(observed) = observed {
                observed
                    .borrow_mut()
                    .submitted(RealBinarySubmission::PreOutput)?;
            }
            Ok(())
        })??;
        self.access.poll(wgpu::Maintain::Poll);
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-binary-flush-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
                if let Some(observed) = observed {
                    observed
                        .borrow_mut()
                        .submitted(RealBinarySubmission::PreOutput)?;
                }
            }
            let out_buffer = self.create_storage_buffer_checked_with_element_size(
                emission,
                len,
                "runmat-binary-out",
                BufferUsageClass::Generic,
                entry_a.element_type.element_size(),
            )?;
            if let Some(effect) = effect {
                effect.set(true);
            }
            if let Some(observed) = observed {
                observed.borrow_mut().output(&out_buffer);
            }
            let chunk_capacity = self
                .binary_linear_launch()?
                .chunk_capacity()
                .ok_or_else(|| anyhow!("selected real binary dispatch capacity overflow"))?;
            let mut offset = 0usize;
            while offset < len {
                let remaining = len - offset;
                let chunk_len = remaining.min(chunk_capacity);
                let params = crate::backend::wgpu::params::LenOpParams {
                    len: chunk_len as u32,
                    op: op as u32,
                    offset: offset as u32,
                    total: len as u32,
                };
                let params_buffer = self.kernel_resources.uniform_buffer(
                    emission,
                    UniformBufferKey::LenOpParams,
                    std::mem::size_of::<crate::backend::wgpu::params::LenOpParams>() as u64,
                    "runmat-binary-params",
                )?;
                if let Some(observed) = observed {
                    observed.borrow_mut().uniform(&params_buffer)?;
                }
                emission.write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-binary-bind"),
                        layout: match validated {
                            Some(selected) => selected.layout.as_ref(),
                            None => &pipeline.get(emission)?.layout,
                        },
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry_a.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: entry_b.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                    });
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    self.binary_linear_launch()?.workgroup(),
                );
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    match validated {
                        Some(selected) => selected.pipeline.as_ref(),
                        None => &pipeline.get(emission)?.pipeline,
                    },
                    &bind_group,
                    groups,
                )?;
                if let Some(observed) = observed {
                    observed
                        .borrow_mut()
                        .submitted(RealBinarySubmission::Dispatch)?;
                }
                offset += chunk_len;
            }
            Ok(out_buffer)
        })??;
        let handle = self.register_real_floating_output_admitted(
            out_buffer,
            entry_a.shape,
            len,
            selected_precision,
            admission,
        )?;
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        self.inherit_elementwise_layout(handle, a)
    }

    fn binary_op_broadcast_exec_recorded(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        observed: Option<&RefCell<RealBinaryObserved>>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        use crate::backend::wgpu::params::{AlignedU32, BinaryBroadcastParams, BCAST_MAX_RANK};
        let (entry_a, entry_b) = self.real_floating_binary_inputs(a, b)?;
        let selected_precision = entry_a.precision;
        let pipeline = self.real_binary_pipeline(selected_precision, true, op)?;
        // Compute broadcasted output shape
        let route = broadcast_plan::BroadcastRoute::with_launch(
            &entry_a.shape,
            &entry_b.shape,
            BCAST_MAX_RANK,
            entry_a.element_type.element_size() as u64,
            self.binary_linear_launch()?,
        )?;
        let (shape_a, shape_b, out_shape) = (&route.shape_a, &route.shape_b, &route.out_shape);
        let rank = out_shape.len();
        let len = route.len;
        if len != 0 || op == crate::backend::wgpu::types::BinaryOpCode::Max {
            pipeline.preflight()?;
        }
        self.validate_real_floating_output_extent(len, selected_precision)?;
        self.binary_broadcast_buffer_demand_for_precision(u64::try_from(len)?, selected_precision)
            .ok_or_else(|| anyhow!("selected real broadcast resource extent overflow"))?;
        if len == 0 {
            let out_buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked_with_element_size(
                    emission,
                    0,
                    "runmat-binary-bcast-out",
                    BufferUsageClass::Generic,
                    entry_a.element_type.element_size(),
                )
            })??;
            if let Some(observed) = observed {
                observed.borrow_mut().output(&out_buffer);
            }
            return self.register_real_floating_output_admitted(
                out_buffer,
                out_shape.clone(),
                0,
                selected_precision,
                admission,
            );
        }
        // Compute strides (column-major)
        let mut stride_a: Vec<u32> = vec![0; rank];
        let mut stride_b: Vec<u32> = vec![0; rank];
        let mut s: u64 = 1;
        for i in 0..rank {
            stride_a[i] = if shape_a[i] == 1 { 0 } else { s as u32 };
            s = s.saturating_mul(shape_a[i] as u64);
        }
        s = 1;
        for i in 0..rank {
            stride_b[i] = if shape_b[i] == 1 { 0 } else { s as u32 };
            s = s.saturating_mul(shape_b[i] as u64);
        }

        // Create output buffer
        let (out_buffer, start) = self.with_emission(|emission, pending| -> Result<_> {
            if let Some(observed) = observed {
                observed.borrow_mut().begin_effect();
            }
            let out_buffer = self.create_storage_buffer_checked_with_element_size(
                emission,
                len,
                "runmat-binary-bcast-out",
                BufferUsageClass::Generic,
                entry_a.element_type.element_size(),
            )?;
            if let Some(observed) = observed {
                observed.borrow_mut().output(&out_buffer);
            }
            // Prepare params buffer and bind group once; update params per chunk
            let params_size = std::mem::size_of::<BinaryBroadcastParams>() as u64;
            let params_buffer = self.kernel_resources.uniform_buffer(
                emission,
                UniformBufferKey::BinaryBroadcastParams,
                params_size,
                "runmat-binary-bcast-params",
            )?;
            if let Some(observed) = observed {
                observed.borrow_mut().uniform(&params_buffer)?;
            }
            let bind_group_layout = &pipeline.get(emission)?.layout;
            let bind_group = emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-binary-bcast-bind"),
                    layout: bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_a.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: entry_b.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            // Dispatch in chunks
            let start = Instant::now();
            for chunk in 0..route.submissions {
                let offset = chunk as usize * route.chunk_capacity;
                let remaining = len - offset;
                let chunk_len = remaining.min(route.chunk_capacity);
                // Pack params
                let mut params = BinaryBroadcastParams {
                    len: chunk_len as u32,
                    offset: offset as u32,
                    rank: rank as u32,
                    op: op as u32,
                    out_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                    a_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                    b_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                    a_strides: [AlignedU32::new(0); BCAST_MAX_RANK],
                    b_strides: [AlignedU32::new(0); BCAST_MAX_RANK],
                };
                for i in 0..rank {
                    params.out_shape[i] = AlignedU32::new(out_shape[i] as u32);
                    params.a_shape[i] = AlignedU32::new(shape_a[i] as u32);
                    params.b_shape[i] = AlignedU32::new(shape_b[i] as u32);
                    params.a_strides[i] = AlignedU32::new(stride_a[i]);
                    params.b_strides[i] = AlignedU32::new(stride_b[i]);
                }
                emission.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    self.binary_linear_launch()?.workgroup(),
                );
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    &pipeline.get(emission)?.pipeline,
                    &bind_group,
                    groups,
                )?;
                if let Some(observed) = observed {
                    observed
                        .borrow_mut()
                        .submitted(RealBinarySubmission::Dispatch)?;
                }
            }
            Ok((out_buffer, start))
        })??;
        let handle = self.register_real_floating_output_admitted(
            out_buffer,
            out_shape.clone(),
            len,
            selected_precision,
            admission,
        )?;
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
    }

    pub(crate) fn fused_elementwise_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        output_storage: GpuTensorStorage,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_typed_exec(
            shader,
            inputs,
            output_shape,
            len,
            FusedElementwiseRepresentation {
                input_precision: self.precision,
                output_precision: self.precision,
                output_storage,
                logical_output: false,
            },
        )
    }

    pub(super) fn fused_elementwise_typed_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        representation: FusedElementwiseRepresentation,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_typed_exec_observed(
            shader,
            inputs,
            output_shape,
            len,
            representation,
            |_| Ok(()),
        )
    }

    pub(super) fn fused_elementwise_typed_exec_observed(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        representation: FusedElementwiseRepresentation,
        mut after_allocation: impl FnMut(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_typed_exec_recorded(
            shader,
            inputs,
            FusedElementwiseWork {
                output_shape,
                len,
                invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                representation,
                uniform_abi: None,
            },
            &mut after_allocation,
            SingleFusionEvidence::Legacy {
                observed: None,
                prepared: None,
            },
            None,
        )
    }

    #[cfg(test)]
    pub(super) fn fused_elementwise_prepared_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        work: FusedElementwiseWork<'_>,
        prepared: &FrozenFusionRecipe,
    ) -> Result<(
        GpuTensorHandle,
        runmat_accelerate_api::ProviderRealElementwiseObserved,
    )> {
        let (output, resources, _) = self
            .fused_elementwise_prepared_inner_exec(shader, inputs, work, prepared, false, None)?;
        Ok((output, resources))
    }

    pub(super) fn fused_elementwise_prepared_probe_admitted_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        work: FusedElementwiseWork<'_>,
        prepared: &FrozenFusionRecipe,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<(
        GpuTensorHandle,
        runmat_accelerate_api::ProviderRealElementwiseObserved,
        Vec<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
    )> {
        self.fused_elementwise_prepared_inner_exec(shader, inputs, work, prepared, true, admission)
    }

    fn fused_elementwise_prepared_inner_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        work: FusedElementwiseWork<'_>,
        prepared: &FrozenFusionRecipe,
        capture_probe_tickets: bool,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<(
        GpuTensorHandle,
        runmat_accelerate_api::ProviderRealElementwiseObserved,
        Vec<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
    )> {
        let observed = RefCell::new(RealObserved {
            capture_probe_tickets,
            ..RealObserved::default()
        });
        let output = self.fused_elementwise_typed_exec_recorded(
            shader,
            inputs,
            work,
            |_| Ok(()),
            SingleFusionEvidence::Legacy {
                observed: Some(&observed),
                prepared: Some(prepared),
            },
            admission,
        )?;
        prepared.require(&observed.borrow().resources)?;
        let observed = observed.into_inner();
        Ok((output, observed.resources, observed.probe_tickets))
    }

    pub(super) fn fused_elementwise_validated_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        frame: &SingleFrame,
        validated: &super::validated_fusion::ValidatedFusionPipeline,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_typed_exec_recorded(
            shader,
            inputs,
            FusedElementwiseWork {
                output_shape,
                len,
                invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                representation: FusedElementwiseRepresentation {
                    input_precision: frame.precision,
                    output_precision: frame.precision,
                    output_storage: GpuTensorStorage::Real,
                    logical_output: frame.logical_output,
                },
                uniform_abi: None,
            },
            |_| Ok(()),
            SingleFusionEvidence::Validated {
                frame,
                pipeline: validated,
            },
            None,
        )
    }

    fn fused_elementwise_typed_exec_recorded(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        work: FusedElementwiseWork<'_>,
        mut after_allocation: impl FnMut(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
        evidence: SingleFusionEvidence<'_>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        let (observed, prepared, validated) = match evidence {
            SingleFusionEvidence::Legacy { observed, prepared } => (observed, prepared, None),
            SingleFusionEvidence::Validated { frame, pipeline } => {
                (None, None, Some((frame, pipeline)))
            }
        };
        let FusedElementwiseWork {
            output_shape,
            len,
            invocation_geometry,
            representation,
            uniform_abi,
        } = work;
        let dispatch_len = invocation_geometry.invocation_len(
            len,
            &representation,
            validated.is_some() || prepared.is_some(),
        )?;
        let output_storage = representation.output_storage;
        let validated_frame = validated.map(|(frame, _)| frame);
        if validated_frame.is_none() {
            if inputs.is_empty() {
                return Err(anyhow!("fused_elementwise: no inputs"));
            }
            if len > u32::MAX as usize {
                return Err(anyhow!("fused_elementwise: tensor too large"));
            }
            fusion_recipe::validate_output_words(output_shape, len, &representation)?;
        }
        let derived = uniform_abi
            .filter(|_| validated_frame.is_none())
            .map(|_| {
                FrozenFusionRecipe::plan(
                    shader,
                    inputs.len(),
                    &work,
                    self.linear_launch,
                    &self.adapter_limits,
                )
            })
            .transpose()?;
        if let Some(prepared) = prepared {
            ensure!(
                derived
                    .as_ref()
                    .is_some_and(|derived| prepared.same_work(derived)),
                "fused_elementwise: prepared physical work changed before execution"
            );
        }
        let frozen = validated_frame
            .map(|frame| &frame.recipe)
            .or(prepared)
            .or(derived.as_ref());
        let launch = frozen.map_or(self.linear_launch, FrozenFusionRecipe::launch);
        let chunk_capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("fused_elementwise: launch capacity overflows"))?;
        // Freeze the selector for the shader, cache identity, and dispatch. The
        // generic shader helper must not re-read a different @WG@ selection.
        let legacy_recipe = if frozen.is_none() {
            Some(super::pipeline_recipe::PipelineRecipe::fusion(
                shader,
                inputs.len(),
                super::pipeline_recipe::FusionPipelineLayout::Single,
                launch.workgroup(),
            ))
        } else {
            None
        };
        let recipe = frozen
            .map(FrozenFusionRecipe::pipeline)
            .or(legacy_recipe.as_ref())
            .expect("typed or legacy fusion recipe");
        if let Some((frame, validated)) = validated {
            frame.require_execution(validated, inputs.len(), output_shape, len, &representation)?;
        }
        let shader = recipe.shader.as_str();
        let (storage_bindings, total_bindings) = if let Some(frame) = validated_frame {
            (frame.storage_bindings, frame.total_bindings)
        } else {
            let storage_bindings = checked_binding_count("fused_elementwise", inputs.len(), 1)?;
            let total = checked_binding_count("fused_elementwise", inputs.len(), 2)?;
            validate_compute_binding_counts(
                "fused_elementwise",
                storage_bindings,
                total,
                &self.adapter_limits,
            )?;
            (storage_bindings, total)
        };
        let entries = if let Some(frame) = validated_frame {
            frame.entries.clone()
        } else {
            inputs
                .iter()
                .map(|handle| {
                    if representation.input_precision == self.precision {
                        return self.get_entry(handle);
                    }
                    ensure!(
                        representation.input_precision == NumericPrecision::F32
                            && self.precision == NumericPrecision::F64,
                        "fused input precision cannot be represented exactly"
                    );
                    let entry = self.get_entry_raw(handle)?;
                    ensure!(
                        entry.element_type == runmat_accelerate_api::NumericElementType::F32
                            && entry.precision == NumericPrecision::F32
                            && (matches!(
                                invocation_geometry,
                                ElementwiseInvocationGeometry::ComplexElements { .. }
                            ) || entry.storage == GpuTensorStorage::Real)
                            && handle.descriptor().element_type == Some(entry.element_type)
                            && handle.descriptor().storage == Some(entry.storage)
                            && !runmat_accelerate_api::handle_is_logical(handle),
                        "fused typed input does not match its shader ABI"
                    );
                    Ok(entry)
                })
                .collect::<Result<Vec<_>>>()?
        };
        let abi = match frozen {
            Some(frozen) => frozen.uniform_abi(),
            None => match invocation_geometry {
                ElementwiseInvocationGeometry::ComplexElements { uniform_abi, .. } => uniform_abi,
                // Legacy shader inspection does not mint a physical recipe/receipt.
                ElementwiseInvocationGeometry::OutputLanes => {
                    if shader.contains("out_shape") || shader.contains("a_shape") {
                        FusionUniformAbi::Broadcast
                    } else {
                        FusionUniformAbi::Simple
                    }
                }
            },
        };
        let layouts = if let Some(frame) = validated_frame {
            frame.layouts.clone()
        } else {
            match abi {
                FusionUniformAbi::Broadcast => inputs
                    .iter()
                    .zip(&entries)
                    .map(|(handle, entry)| {
                        super::backend_shared::fusion_operand_layout(handle, entry, output_shape)
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
            }
        };
        let (output_buffer, bind_group_layout, pipeline) =
            self.with_emission(|emission, pending| -> Result<_> {
                if let Some(frame) = validated_frame {
                    for (handle, prepared) in inputs.iter().zip(&frame.entries) {
                        let live = self.fusion_entry_for_precision(handle, frame.precision)?;
                        ensure!(
                            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
                                &live.buffer,
                                &prepared.buffer,
                            ),
                            "validated fused input changed before emission"
                        );
                    }
                }
                let output_element_size = match representation.output_precision {
                    NumericPrecision::F32 => 4,
                    NumericPrecision::F64 => 8,
                };
                let output_buffer = self.create_storage_buffer_checked_with_element_size(
                    emission,
                    len,
                    "runmat-fusion-output",
                    BufferUsageClass::FusionOut,
                    output_element_size,
                )?;
                after_allocation(&output_buffer)?;
                if let Some(observed) = observed {
                    if observed.borrow().capture_probe_tickets {
                        observed.borrow_mut().output_resource_ticket =
                            self.issue_power_probe_resource_ticket(output_buffer.size())?;
                    }
                    observed.borrow_mut().output(&output_buffer);
                }
                let (bind_group_layout, pipeline) = if let Some((_, validated)) = validated {
                    (validated.layout.clone(), validated.pipeline.clone())
                } else {
                    let layout = self.cached_fusion_bind_group_layout(emission, inputs.len())?;
                    let pipeline = self.recipe_pipeline(
                        emission,
                        recipe,
                        layout.as_ref(),
                        "runmat-fusion-pipeline",
                    )?;
                    (layout, pipeline)
                };
                // Warmup once
                let ticket = observed
                    .filter(|observed| observed.borrow().capture_probe_tickets)
                    .map(|_| pending.arm_probe_completion());
                crate::backend::wgpu::dispatch::elementwise::warmup_noop(
                    emission, pending, &pipeline,
                )?;
                if let (Some(observed), Some(ticket)) = (observed, ticket) {
                    observed.borrow_mut().probe_tickets.push(ticket);
                }
                if let Some(observed) = observed {
                    observed.borrow_mut().submitted(false)?;
                }
                Ok((output_buffer, bind_group_layout, pipeline))
            })??;
        self.access.poll(wgpu::Maintain::Poll);
        let emitted = self.with_emission(|emission, pending| -> Result<_> {

        struct BroadcastUniformState {
            buffer: runmat_accelerate_api::WgpuBufferOwner,
            template: Vec<u8>,
        }

        impl BroadcastUniformState {
            fn update(&mut self, emission: &runmat_accelerate_api::WgpuEmission<'_>, len: u32, offset: u32) {
                self.template[..4].copy_from_slice(&len.to_ne_bytes());
                self.template[4..8].copy_from_slice(&offset.to_ne_bytes());
                emission.write_buffer(self.buffer.as_ref(), 0, &self.template);
            }
        }

        enum FusionUniformState {
            Broadcast(BroadcastUniformState),
            Simple { buffer: runmat_accelerate_api::WgpuBufferOwner },
        }

        let uniform_state = match abi {
            FusionUniformAbi::Broadcast => {
            let rank = output_shape.len();
            let max_rank = crate::backend::wgpu::params::BCAST_MAX_RANK;
            let uniform_bytes = usize::try_from(abi.requested_bytes(inputs.len())?)
                .map_err(|_| anyhow!("fused broadcast uniform size overflow"))?;
            ensure!(
                uniform_bytes <= self.adapter_limits.max_uniform_buffer_binding_size as usize,
                "fused broadcast uniform exceeds adapter limit"
            );
            let mut bytes: Vec<u8> = Vec::with_capacity(uniform_bytes);
            let write_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_ne_bytes());
            write_u32(&mut bytes, 0); // len placeholder
            write_u32(&mut bytes, 0); // offset placeholder
            write_u32(&mut bytes, rank as u32);
            write_u32(&mut bytes, 0);
            let write_packed_array = |buf: &mut Vec<u8>, vals: &[u32]| {
                for &val in vals.iter() {
                    write_u32(buf, val);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                }
                for _ in vals.len()..max_rank {
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                }
            };
            let out_shape_u32: Vec<u32> = output_shape.iter().map(|&d| d as u32).collect();
            write_packed_array(&mut bytes, &out_shape_u32);
            for (shape_u32, strides) in &layouts {
                write_packed_array(&mut bytes, shape_u32);
                write_packed_array(&mut bytes, strides);
            }
            let buffer = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-fusion-params"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            emission.write_buffer(buffer.as_ref(), 0, &bytes);
            let uniform_buffer = buffer.clone();
            let state = BroadcastUniformState {
                buffer,
                template: bytes,
            };
            (FusionUniformState::Broadcast(state), uniform_buffer)
            }
            FusionUniformAbi::Simple => {
            let buffer = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-fusion-params"),
                    size: abi.requested_bytes(inputs.len())?,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            (
                FusionUniformState::Simple {
                    buffer: buffer.clone(),
                },
                buffer,
            )
            }
        };

        let (mut uniform_state, uniform_buffer) = uniform_state;
        let mut uniform_resource_ticket = if observed
            .is_some_and(|observed| observed.borrow().capture_probe_tickets)
        {
            self.issue_power_probe_resource_ticket(uniform_buffer.size())?
        } else {
            None
        };
        if let Some(observed) = observed { observed.borrow_mut().uniform(&uniform_buffer)?; }

        let mut bind_entries = Vec::with_capacity(total_bindings);
        for (idx, entry) in entries.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource: entry.buffer.as_ref().as_entire_binding(),
            });
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output_buffer.as_ref().as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: storage_bindings as u32,
            resource: uniform_buffer.as_ref().as_entire_binding(),
        });

        let bind_group =
            self.bind_group_cache
                .get_or_create(emission, bind_group_layout.as_ref(), &bind_entries, |emission| {
                    Arc::new(
                        emission.device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fusion-bind-group"),
                                layout: bind_group_layout.as_ref(),
                                entries: &bind_entries,
                            }),
                    )
                })?;

        let output_ptr = output_buffer.as_ref() as *const wgpu::Buffer as usize;
        let input_ids: Vec<u64> = inputs.iter().map(|h| h.buffer_id()).collect();
        log::trace!(
            "fusion elementwise begin len={} out_ptr=0x{:x} inputs={:?}",
            len,
            output_ptr,
            input_ids
        );
        // Dispatch in chunks bounded by the selected shader workgroup size.
        let mut offset_elems = 0usize;
        let mut chunk_index = 0usize;
        let mut last_submission_id = None;
        while offset_elems < dispatch_len {
            let remaining = dispatch_len - offset_elems;
            let chunk_len = remaining.min(chunk_capacity);

            log::trace!(
                "fusion elementwise chunk start out_ptr=0x{:x} chunk_len={} offset={} chunk_index={}",
                output_ptr,
                chunk_len,
                offset_elems,
                chunk_index
            );
            match &mut uniform_state {
                FusionUniformState::Broadcast(state) => {
                    state.update(emission, chunk_len as u32, offset_elems as u32)
                }
                FusionUniformState::Simple { buffer } => {
                    let params = crate::backend::wgpu::params::FusionParams {
                        len: chunk_len as u32,
                        offset: offset_elems as u32,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    emission.write_buffer(buffer.as_ref(), 0, bytes_of(&params));
                }
            }

            let workgroups = launch
                .dispatch_groups(chunk_len)
                .ok_or_else(|| anyhow!("fused_elementwise: launch exceeds enabled limits"))?;
            let mut enc = pending.profiled_encoder(emission, &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-fusion-elementwise-encoder"),
                    });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-fusion-elementwise-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, bind_group.as_ref(), &[]);
                if workgroups > 0 {
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }
            let ticket = observed
                .filter(|observed| observed.borrow().capture_probe_tickets)
                .map(|_| pending.arm_probe_completion());
            if chunk_len == remaining {
                if let Some(resource) = uniform_resource_ticket.take() {
                    pending.retain_probe_resource_until_next_submit(resource);
                }
            }
            let submission_id = self.submit(emission, pending, enc)?;
            if let (Some(observed), Some(ticket)) = (observed, ticket) {
                observed.borrow_mut().probe_tickets.push(ticket);
            }
            if let Some(observed) = observed { observed.borrow_mut().submitted(workgroups > 0)?; }
            last_submission_id = Some(submission_id);
            log::trace!(
                "fusion elementwise chunk complete out_ptr=0x{:x} chunk_len={} offset={} submission_id={}",
                output_ptr,
                chunk_len,
                offset_elems,
                submission_id
            );
            offset_elems += chunk_len;
            chunk_index += 1;
        }
        Ok((last_submission_id, output_ptr, chunk_index))
        })??;
        let (last_submission_id, output_ptr, chunk_index) = emitted;
        let handle = if representation.logical_output {
            ensure!(
                output_storage == GpuTensorStorage::Real,
                "logical output requires real storage"
            );
            self.register_logical_buffer(
                output_buffer,
                output_shape.to_vec(),
                len,
                BufferUsageClass::FusionOut,
            )?
        } else {
            let element_type = match representation.output_precision {
                NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
            };
            let allocated_bytes = output_buffer.size();
            self.register_numeric_buffer_produced_admitted(
                output_buffer,
                super::backend_types::NumericBufferRegistration {
                    shape: output_shape.to_vec(),
                    len,
                    physical_element_type: element_type,
                    storage: output_storage,
                    allocated_bytes,
                    usage: BufferUsageClass::FusionOut,
                },
                false,
                admission,
            )?
        };
        if let Some(observed) = observed {
            if let Some(ticket) = observed.borrow_mut().output_resource_ticket.take() {
                self.allocations
                    .attach_probe_resource_ticket(&handle, ticket)?;
            }
        }
        log::trace!(
            "fusion elementwise complete buffer_id={} out_ptr=0x{:x} len={} chunks={} last_submission_id={:?}",
            handle.buffer_id(),
            output_ptr,
            len,
            chunk_index,
            last_submission_id
        );
        if let Some(submission_id) = last_submission_id {
            self.record_buffer_submission(handle.buffer_id(), submission_id);
        }
        Ok(handle)
    }
    pub(super) fn fused_elementwise_multi_validated_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        frame: &MultiFrame,
        validated: &super::validated_fusion::ValidatedFusionPipeline,
    ) -> Result<Vec<GpuTensorHandle>> {
        self.fused_elementwise_multi_layout_exec(
            FusionMultiRequest {
                shader,
                inputs,
                output_shape: &frame.shape,
                len: frame.len,
                output_classes: &frame.output_classes,
                layout: MultiFusionLayout::Broadcast,
            },
            Some((frame, validated)),
        )
    }

    pub(super) fn fused_indexed_multi_exec(
        &self,
        request: IndexedFusionMultiRequest<'_>,
    ) -> Result<Vec<GpuTensorHandle>> {
        self.fused_elementwise_multi_layout_exec(
            FusionMultiRequest {
                shader: request.shader,
                inputs: request.inputs,
                output_shape: request.output_shape,
                len: request.len,
                output_classes: request.output_classes,
                layout: MultiFusionLayout::Indexed(request.operands),
            },
            None,
        )
    }

    fn fused_elementwise_multi_layout_exec(
        &self,
        request: FusionMultiRequest<'_>,
        validated: Option<(
            &MultiFrame,
            &super::validated_fusion::ValidatedFusionPipeline,
        )>,
    ) -> Result<Vec<GpuTensorHandle>> {
        let FusionMultiRequest {
            shader,
            inputs,
            output_shape,
            len,
            output_classes,
            layout,
        } = request;
        let validated_frame = validated.map(|(frame, _)| frame);
        let physical_precision = validated_frame.map_or(self.precision, |frame| frame.precision);
        let output_element_size = match physical_precision {
            NumericPrecision::F32 => 4,
            NumericPrecision::F64 => 8,
        };
        if validated_frame.is_none() {
            runmat_accelerate_api::validate_fused_output_classes(
                self.provider_precision_exec(),
                output_classes,
            )?;
        }
        let num_outputs = output_classes.len();
        if validated_frame.is_none() {
            if inputs.is_empty() {
                return Err(anyhow!("fused_elementwise_multi: no inputs"));
            }
            if num_outputs == 0 {
                return Err(anyhow!("fused_elementwise_multi: num_outputs is zero"));
            }
            if len > u32::MAX as usize {
                return Err(anyhow!("fused_elementwise_multi: tensor too large"));
            }
        }
        let workgroup_size = validated_frame
            .map(|frame| frame.recipe.workgroup)
            .unwrap_or(self.workgroup_config.scalar);
        let chunk_capacity = if let Some(frame) = validated_frame {
            frame.chunk_capacity
        } else {
            crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup_size)
                .ok_or_else(|| anyhow!("fused_elementwise_multi: workgroup size is zero"))?
                as usize
        };
        let legacy_recipe = validated_frame.is_none().then(|| {
            super::pipeline_recipe::PipelineRecipe::fusion(
                shader,
                inputs.len(),
                super::pipeline_recipe::FusionPipelineLayout::Multiple(num_outputs),
                workgroup_size,
            )
        });
        let recipe = validated_frame
            .map(|frame| &frame.recipe)
            .or(legacy_recipe.as_ref())
            .expect("validated or legacy multi recipe");
        if let Some((frame, validated)) = validated {
            frame.require_execution(validated, inputs.len(), output_shape, len, output_classes)?;
        }
        let (storage_bindings, total_bindings) = if let Some(frame) = validated_frame {
            (frame.storage_bindings, frame.total_bindings)
        } else {
            let storage_bindings =
                checked_binding_count("fused_elementwise_multi", inputs.len(), num_outputs)?;
            let total = checked_binding_count("fused_elementwise_multi", storage_bindings, 1)?;
            validate_compute_binding_counts(
                "fused_elementwise_multi",
                storage_bindings,
                total,
                &self.adapter_limits,
            )?;
            (storage_bindings, total)
        };

        let entries = if let Some(frame) = validated_frame {
            frame.entries.clone()
        } else {
            inputs
                .iter()
                .map(|h| self.get_entry(h))
                .collect::<Result<Vec<_>>>()?
        };
        let layouts = if let Some(frame) = validated_frame {
            frame.layouts.clone()
        } else {
            match layout {
                MultiFusionLayout::Broadcast => inputs
                    .iter()
                    .zip(&entries)
                    .map(|(handle, entry)| {
                        super::backend_shared::fusion_operand_layout(handle, entry, output_shape)
                    })
                    .collect::<Result<Vec<_>>>()?,
                MultiFusionLayout::Indexed(operands) => {
                    let observed = inputs
                        .iter()
                        .zip(&entries)
                        .map(|(handle, entry)| IndexedFusionObserved {
                            allocation_shape: &entry.shape,
                            view_shape: handle.shape(),
                            lanes: entry.len,
                            storage: entry.storage,
                            view_storage: runmat_accelerate_api::handle_storage(handle),
                            transposed: runmat_accelerate_api::handle_is_transposed(handle),
                        })
                        .collect::<Vec<_>>();
                    validate_indexed_layout(
                        &observed,
                        operands,
                        output_shape,
                        len,
                        crate::backend::wgpu::params::BCAST_MAX_RANK,
                    )?
                }
            }
        };
        let (output_buffers, bind_group_layout, pipeline) =
            self.with_emission(|emission, pending| -> Result<_> {
                if let Some(frame) = validated_frame {
                    for (handle, prepared) in inputs.iter().zip(&frame.entries) {
                        let live = self.fusion_entry_for_precision(handle, frame.precision)?;
                        ensure!(
                            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
                                &live.buffer,
                                &prepared.buffer,
                            ),
                            "validated fused input changed before emission"
                        );
                    }
                }
                let output_buffers: Vec<_> = (0..num_outputs)
                    .map(|_| {
                        self.create_storage_buffer_checked_with_element_size(
                            emission,
                            len,
                            "runmat-fusion-multi-output",
                            BufferUsageClass::FusionOut,
                            output_element_size,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;

                let (bind_group_layout, pipeline) = if let Some((_, validated)) = validated {
                    (validated.layout.clone(), validated.pipeline.clone())
                } else {
                    let bind_group_layout = {
                        let key = format!(
                            "runmat-fusion-multi-layout-{}-{}",
                            inputs.len(),
                            num_outputs
                        );
                        self.cached_bind_group_layout(emission, &key, |emission| {
                            crate::backend::wgpu::bindings::build_fusion_multi_bgl(
                                emission,
                                inputs.len(),
                                num_outputs,
                            )
                        })?
                    };
                    let pipeline = self.recipe_pipeline(
                        emission,
                        recipe,
                        bind_group_layout.as_ref(),
                        "runmat-fusion-multi-pipeline",
                    )?;
                    (bind_group_layout, pipeline)
                };
                crate::backend::wgpu::dispatch::elementwise::warmup_noop(
                    emission, pending, &pipeline,
                )?;
                Ok((output_buffers, bind_group_layout, pipeline))
            })??;
        self.access.poll(wgpu::Maintain::Poll);
        let last_submission_id = self.with_emission(|emission, pending| -> Result<_> {
            struct BroadcastUniformState {
                buffer: runmat_accelerate_api::WgpuBufferOwner,
                template: Vec<u8>,
            }
            impl BroadcastUniformState {
                fn update(
                    &mut self,
                    emission: &runmat_accelerate_api::WgpuEmission<'_>,
                    len: u32,
                    offset: u32,
                ) {
                    self.template[..4].copy_from_slice(&len.to_ne_bytes());
                    self.template[4..8].copy_from_slice(&offset.to_ne_bytes());
                    emission.write_buffer(self.buffer.as_ref(), 0, &self.template);
                }
            }

            let rank = output_shape.len();
            let max_rank = crate::backend::wgpu::params::BCAST_MAX_RANK;
            let mut bytes: Vec<u8> = Vec::with_capacity(4 * 4 + (max_rank * 4 * 4));
            let write_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_ne_bytes());
            write_u32(&mut bytes, 0); // len placeholder
            write_u32(&mut bytes, 0); // offset placeholder
            write_u32(&mut bytes, rank as u32);
            write_u32(&mut bytes, 0);
            let write_packed_array = |buf: &mut Vec<u8>, vals: &[u32]| {
                for &val in vals.iter() {
                    write_u32(buf, val);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                }
                for _ in vals.len()..max_rank {
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                }
            };
            let out_shape_u32: Vec<u32> = output_shape.iter().map(|&d| d as u32).collect();
            write_packed_array(&mut bytes, &out_shape_u32);
            for (shape_u32, strides) in &layouts {
                write_packed_array(&mut bytes, shape_u32);
                write_packed_array(&mut bytes, strides);
            }
            let uniform_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-fusion-multi-params"),
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            emission.write_buffer(uniform_buffer.as_ref(), 0, &bytes);

            let mut uniform_state = BroadcastUniformState {
                buffer: uniform_buffer.clone(),
                template: bytes,
            };

            let mut bind_entries = Vec::with_capacity(total_bindings);
            for (idx, entry) in entries.iter().enumerate() {
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: idx as u32,
                    resource: entry.buffer.as_ref().as_entire_binding(),
                });
            }
            for (k, out_buf) in output_buffers.iter().enumerate() {
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: (inputs.len() + k) as u32,
                    resource: out_buf.as_ref().as_entire_binding(),
                });
            }
            bind_entries.push(wgpu::BindGroupEntry {
                binding: storage_bindings as u32,
                resource: uniform_buffer.as_ref().as_entire_binding(),
            });

            let bind_group = self.bind_group_cache.get_or_create(
                emission,
                bind_group_layout.as_ref(),
                &bind_entries,
                |emission| {
                    Arc::new(
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fusion-multi-bind-group"),
                                layout: bind_group_layout.as_ref(),
                                entries: &bind_entries,
                            }),
                    )
                },
            )?;

            let mut offset_elems = 0usize;
            let mut last_submission_id = None;
            while offset_elems < len {
                let remaining = len - offset_elems;
                let chunk_len = remaining.min(chunk_capacity);
                uniform_state.update(emission, chunk_len as u32, offset_elems as u32);
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    workgroup_size,
                );
                let mut enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-fusion-multi-elementwise-encoder"),
                    },
                );
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("runmat-fusion-multi-elementwise-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, bind_group.as_ref(), &[]);
                    if workgroups > 0 {
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                }
                let submission_id = self.submit(emission, pending, enc)?;
                last_submission_id = Some(submission_id);
                offset_elems += chunk_len;
            }

            Ok(last_submission_id)
        })??;
        let mut handles = Vec::with_capacity(num_outputs);
        for (out_buf, class) in output_buffers.into_iter().zip(output_classes) {
            let handle = if *class == runmat_accelerate_api::ProviderElementType::Logical {
                self.register_logical_buffer(
                    out_buf,
                    output_shape.to_vec(),
                    len,
                    BufferUsageClass::FusionOut,
                )?
            } else {
                let allocated_bytes = out_buf.size();
                self.register_numeric_buffer(
                    out_buf,
                    super::backend_types::NumericBufferRegistration {
                        shape: output_shape.to_vec(),
                        len,
                        physical_element_type: match physical_precision {
                            NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                            NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                        },
                        storage: GpuTensorStorage::Real,
                        allocated_bytes,
                        usage: BufferUsageClass::FusionOut,
                    },
                )?
            };
            if let Some(submission_id) = last_submission_id {
                self.record_buffer_submission(handle.buffer_id(), submission_id);
            }
            handles.push(handle);
        }
        Ok(handles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        AccelProvider, GpuTensorStorage, HostIntegerDataOwned, HostIntegerDataView,
        HostIntegerTensorView, HostNumericDataView, HostNumericTensorView, HostTensorView,
        IntegerElementType,
    };

    async fn complex_pair(
        provider: &'static dyn AccelProvider,
        real: &[f64],
        imag: &[f64],
        shape: &[usize],
    ) -> GpuTensorHandle {
        let hr = provider
            .upload(&HostTensorView { data: real, shape })
            .expect("upload real");
        let hi = provider
            .upload(&HostTensorView { data: imag, shape })
            .expect("upload imag");
        provider
            .complex_from_real_imag(&hr, &hi)
            .await
            .expect("complex_from_real_imag")
    }

    fn assert_interleaved_close(got: &[f64], expected: &[(f64, f64)]) {
        assert_eq!(got.len(), expected.len() * 2);
        const EPS: f64 = 1e-5;
        for (idx, (re, im)) in expected.iter().enumerate() {
            assert!(
                (got[idx * 2] - re).abs() < EPS,
                "real lane {idx}: got {}, expected {re}",
                got[idx * 2]
            );
            assert!(
                (got[idx * 2 + 1] - im).abs() < EPS,
                "imag lane {idx}: got {}, expected {im}",
                got[idx * 2 + 1]
            );
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            if expected.is_nan() {
                assert!(actual.is_nan(), "lane {idx}: expected NaN, got {actual}");
            } else {
                assert!(
                    (actual - expected).abs() <= tol * expected.abs().max(1.0),
                    "lane {idx}: expected {expected}, got {actual}"
                );
            }
        }
    }

    fn assert_close_scaled(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let scale = expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tol * scale,
                "lane {idx}: expected {expected}, got {actual}"
            );
        }
    }

    fn register_wgpu_provider_for_test(
    ) -> Option<&'static crate::backend::wgpu::provider::WgpuProvider> {
        match crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => Some(provider),
            Err(err) if err.to_string() == "wgpu: no compatible adapter found" => None,
            Err(err) => panic!("register wgpu provider failed: {err:?}"),
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_comparisons_preserve_all_classes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [3, 1];
        let cases = [
            (
                HostIntegerDataView::I8(&[i8::MIN, 0, i8::MAX]),
                HostIntegerDataView::I8(&[i8::MIN + 1, 0, i8::MAX - 1]),
            ),
            (
                HostIntegerDataView::I16(&[i16::MIN, 0, i16::MAX]),
                HostIntegerDataView::I16(&[i16::MIN + 1, 0, i16::MAX - 1]),
            ),
            (
                HostIntegerDataView::I32(&[i32::MIN, 0, i32::MAX]),
                HostIntegerDataView::I32(&[i32::MIN + 1, 0, i32::MAX - 1]),
            ),
            (
                HostIntegerDataView::I64(&[i64::MIN, 0, i64::MAX]),
                HostIntegerDataView::I64(&[i64::MIN + 1, 0, i64::MAX - 1]),
            ),
            (
                HostIntegerDataView::U8(&[0, 1, u8::MAX]),
                HostIntegerDataView::U8(&[1, 1, u8::MAX - 1]),
            ),
            (
                HostIntegerDataView::U16(&[0, 1, u16::MAX]),
                HostIntegerDataView::U16(&[1, 1, u16::MAX - 1]),
            ),
            (
                HostIntegerDataView::U32(&[0, 1, u32::MAX]),
                HostIntegerDataView::U32(&[1, 1, u32::MAX - 1]),
            ),
            (
                HostIntegerDataView::U64(&[0, 1, u64::MAX]),
                HostIntegerDataView::U64(&[1, 1, u64::MAX - 1]),
            ),
        ];

        for (left, right) in cases {
            let lhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: left,
                    shape: &shape,
                })
                .expect("upload integer lhs");
            let rhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: right,
                    shape: &shape,
                })
                .expect("upload integer rhs");
            let equal = provider.elem_eq(&lhs, &rhs).await.expect("integer eq");
            let not_equal = provider.elem_ne(&lhs, &rhs).await.expect("integer ne");
            let less = provider.elem_lt(&lhs, &rhs).await.expect("integer lt");
            let less_equal = provider.elem_le(&lhs, &rhs).await.expect("integer le");
            let greater = provider.elem_gt(&lhs, &rhs).await.expect("integer gt");
            let greater_equal = provider.elem_ge(&lhs, &rhs).await.expect("integer ge");
            for handle in [
                &equal,
                &not_equal,
                &less,
                &less_equal,
                &greater,
                &greater_equal,
            ] {
                assert!(runmat_accelerate_api::handle_is_logical(handle));
            }
            assert_eq!(
                provider.download(&equal).await.expect("download eq").data,
                vec![0.0, 1.0, 0.0]
            );
            assert_eq!(
                provider
                    .download(&not_equal)
                    .await
                    .expect("download ne")
                    .data,
                vec![1.0, 0.0, 1.0]
            );
            assert_eq!(
                provider.download(&less).await.expect("download lt").data,
                vec![1.0, 0.0, 0.0]
            );
            assert_eq!(
                provider
                    .download(&less_equal)
                    .await
                    .expect("download le")
                    .data,
                vec![1.0, 1.0, 0.0]
            );
            assert_eq!(
                provider.download(&greater).await.expect("download gt").data,
                vec![0.0, 0.0, 1.0]
            );
            assert_eq!(
                provider
                    .download(&greater_equal)
                    .await
                    .expect("download ge")
                    .data,
                vec![0.0, 1.0, 1.0]
            );
            for handle in [
                lhs,
                rhs,
                equal,
                not_equal,
                less,
                less_equal,
                greater,
                greater_equal,
            ] {
                handle.release().unwrap();
            }
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_comparisons_reject_mixed_classes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [1, 1];
        let signed = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I8(&[1]),
                shape: &shape,
            })
            .expect("upload signed integer");
        let unsigned = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&[1]),
                shape: &shape,
            })
            .expect("upload unsigned integer");
        let error = provider
            .elem_eq(&signed, &unsigned)
            .await
            .expect_err("mixed native integer classes must not use an implicit promotion");
        assert!(error.to_string().contains("same class"));
        signed.release().expect("free signed handle");
        unsigned.release().expect("free unsigned handle");
    }

    #[tokio::test]
    async fn wgpu_native_integer_minmax_preserves_all_classes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [3, 1];
        let cases = [
            (
                HostIntegerDataView::I8(&[i8::MIN, 0, i8::MAX]),
                HostIntegerDataView::I8(&[i8::MIN + 1, 0, i8::MAX - 1]),
                HostIntegerDataOwned::I8(vec![i8::MIN, 0, i8::MAX - 1]),
                HostIntegerDataOwned::I8(vec![i8::MIN + 1, 0, i8::MAX]),
            ),
            (
                HostIntegerDataView::I16(&[i16::MIN, 0, i16::MAX]),
                HostIntegerDataView::I16(&[i16::MIN + 1, 0, i16::MAX - 1]),
                HostIntegerDataOwned::I16(vec![i16::MIN, 0, i16::MAX - 1]),
                HostIntegerDataOwned::I16(vec![i16::MIN + 1, 0, i16::MAX]),
            ),
            (
                HostIntegerDataView::I32(&[i32::MIN, 0, i32::MAX]),
                HostIntegerDataView::I32(&[i32::MIN + 1, 0, i32::MAX - 1]),
                HostIntegerDataOwned::I32(vec![i32::MIN, 0, i32::MAX - 1]),
                HostIntegerDataOwned::I32(vec![i32::MIN + 1, 0, i32::MAX]),
            ),
            (
                HostIntegerDataView::I64(&[i64::MIN, 0, i64::MAX]),
                HostIntegerDataView::I64(&[i64::MIN + 1, 0, i64::MAX - 1]),
                HostIntegerDataOwned::I64(vec![i64::MIN, 0, i64::MAX - 1]),
                HostIntegerDataOwned::I64(vec![i64::MIN + 1, 0, i64::MAX]),
            ),
            (
                HostIntegerDataView::U8(&[0, 1, u8::MAX]),
                HostIntegerDataView::U8(&[1, 1, u8::MAX - 1]),
                HostIntegerDataOwned::U8(vec![0, 1, u8::MAX - 1]),
                HostIntegerDataOwned::U8(vec![1, 1, u8::MAX]),
            ),
            (
                HostIntegerDataView::U16(&[0, 1, u16::MAX]),
                HostIntegerDataView::U16(&[1, 1, u16::MAX - 1]),
                HostIntegerDataOwned::U16(vec![0, 1, u16::MAX - 1]),
                HostIntegerDataOwned::U16(vec![1, 1, u16::MAX]),
            ),
            (
                HostIntegerDataView::U32(&[0, 1, u32::MAX]),
                HostIntegerDataView::U32(&[1, 1, u32::MAX - 1]),
                HostIntegerDataOwned::U32(vec![0, 1, u32::MAX - 1]),
                HostIntegerDataOwned::U32(vec![1, 1, u32::MAX]),
            ),
            (
                HostIntegerDataView::U64(&[0, 1, u64::MAX]),
                HostIntegerDataView::U64(&[1, 1, u64::MAX - 1]),
                HostIntegerDataOwned::U64(vec![0, 1, u64::MAX - 1]),
                HostIntegerDataOwned::U64(vec![1, 1, u64::MAX]),
            ),
        ];

        for (left, right, expected_min, expected_max) in cases {
            let integer_type = left.element_type();
            let lhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: left,
                    shape: &shape,
                })
                .expect("upload integer lhs");
            let rhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: right,
                    shape: &shape,
                })
                .expect("upload integer rhs");
            let minimum = provider.elem_min(&lhs, &rhs).await.expect("integer min");
            let maximum = provider.elem_max(&lhs, &rhs).await.expect("integer max");
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&minimum),
                Some(integer_type)
            );
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&maximum),
                Some(integer_type)
            );
            assert_eq!(
                provider
                    .download_integer(&minimum)
                    .await
                    .expect("download min")
                    .data,
                expected_min
            );
            assert_eq!(
                provider
                    .download_integer(&maximum)
                    .await
                    .expect("download max")
                    .data,
                expected_max
            );
            for handle in [lhs, rhs, minimum, maximum] {
                handle.release().unwrap();
            }
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_arithmetic_preserves_all_classes_and_saturates() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [5, 1];
        let cases = [
            (
                HostIntegerDataView::I8(&[i8::MAX, i8::MIN, 12, -3, i8::MIN]),
                HostIntegerDataView::I8(&[1, 1, -120, 7, -1]),
                HostIntegerDataOwned::I8(vec![i8::MAX, i8::MIN + 1, -108, 4, i8::MIN]),
                HostIntegerDataOwned::I8(vec![i8::MAX - 1, i8::MIN, i8::MAX, -10, i8::MIN + 1]),
                HostIntegerDataOwned::I8(vec![i8::MAX, i8::MIN, i8::MIN, -21, i8::MAX]),
            ),
            (
                HostIntegerDataView::I16(&[i16::MAX, i16::MIN, 12, -3, i16::MIN]),
                HostIntegerDataView::I16(&[1, 1, -32_760, 7, -1]),
                HostIntegerDataOwned::I16(vec![i16::MAX, i16::MIN + 1, -32_748, 4, i16::MIN]),
                HostIntegerDataOwned::I16(vec![
                    i16::MAX - 1,
                    i16::MIN,
                    i16::MAX,
                    -10,
                    i16::MIN + 1,
                ]),
                HostIntegerDataOwned::I16(vec![i16::MAX, i16::MIN, i16::MIN, -21, i16::MAX]),
            ),
            (
                HostIntegerDataView::I32(&[i32::MAX, i32::MIN, 12, -3, i32::MIN]),
                HostIntegerDataView::I32(&[1, 1, -2_147_483_640, 7, -1]),
                HostIntegerDataOwned::I32(vec![i32::MAX, i32::MIN + 1, i32::MIN + 20, 4, i32::MIN]),
                HostIntegerDataOwned::I32(vec![
                    i32::MAX - 1,
                    i32::MIN,
                    i32::MAX,
                    -10,
                    i32::MIN + 1,
                ]),
                HostIntegerDataOwned::I32(vec![i32::MAX, i32::MIN, i32::MIN, -21, i32::MAX]),
            ),
            (
                HostIntegerDataView::I64(&[i64::MAX, i64::MIN, 12, -3, i64::MIN]),
                HostIntegerDataView::I64(&[1, 1, i64::MIN + 8, 7, -1]),
                HostIntegerDataOwned::I64(vec![i64::MAX, i64::MIN + 1, i64::MIN + 20, 4, i64::MIN]),
                HostIntegerDataOwned::I64(vec![
                    i64::MAX - 1,
                    i64::MIN,
                    i64::MAX,
                    -10,
                    i64::MIN + 1,
                ]),
                HostIntegerDataOwned::I64(vec![i64::MAX, i64::MIN, i64::MIN, -21, i64::MAX]),
            ),
            (
                HostIntegerDataView::U8(&[u8::MAX, 0, 10, 3, 0]),
                HostIntegerDataView::U8(&[1, 1, 250, 7, 1]),
                HostIntegerDataOwned::U8(vec![u8::MAX, 1, u8::MAX, 10, 1]),
                HostIntegerDataOwned::U8(vec![u8::MAX - 1, 0, 0, 0, 0]),
                HostIntegerDataOwned::U8(vec![u8::MAX, 0, u8::MAX, 21, 0]),
            ),
            (
                HostIntegerDataView::U16(&[u16::MAX, 0, 10, 3, 0]),
                HostIntegerDataView::U16(&[1, 1, u16::MAX - 5, 7, 1]),
                HostIntegerDataOwned::U16(vec![u16::MAX, 1, u16::MAX, 10, 1]),
                HostIntegerDataOwned::U16(vec![u16::MAX - 1, 0, 0, 0, 0]),
                HostIntegerDataOwned::U16(vec![u16::MAX, 0, u16::MAX, 21, 0]),
            ),
            (
                HostIntegerDataView::U32(&[u32::MAX, 0, 10, 3, 0]),
                HostIntegerDataView::U32(&[1, 1, u32::MAX - 5, 7, 1]),
                HostIntegerDataOwned::U32(vec![u32::MAX, 1, u32::MAX, 10, 1]),
                HostIntegerDataOwned::U32(vec![u32::MAX - 1, 0, 0, 0, 0]),
                HostIntegerDataOwned::U32(vec![u32::MAX, 0, u32::MAX, 21, 0]),
            ),
            (
                HostIntegerDataView::U64(&[u64::MAX, 0, 10, 0x0000_0001_0000_0001, 0]),
                HostIntegerDataView::U64(&[1, 1, u64::MAX - 5, 2, 1]),
                HostIntegerDataOwned::U64(vec![u64::MAX, 1, u64::MAX, 0x0000_0001_0000_0003, 1]),
                HostIntegerDataOwned::U64(vec![u64::MAX - 1, 0, 0, 0x0000_0000_ffff_ffff, 0]),
                HostIntegerDataOwned::U64(vec![u64::MAX, 0, u64::MAX, 0x0000_0002_0000_0002, 0]),
            ),
        ];

        for (left, right, expected_add, expected_subtract, expected_multiply) in cases {
            let integer_type = left.element_type();
            let lhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: left,
                    shape: &shape,
                })
                .expect("upload integer lhs");
            let rhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: right,
                    shape: &shape,
                })
                .expect("upload integer rhs");
            let add = provider.elem_add(&lhs, &rhs).await.expect("integer add");
            let subtract = provider
                .elem_sub(&lhs, &rhs)
                .await
                .expect("integer subtract");
            let multiply = provider
                .elem_mul(&lhs, &rhs)
                .await
                .expect("integer multiply");
            let divide = provider.elem_div(&lhs, &rhs).await.expect("integer divide");
            for handle in [&add, &subtract, &multiply, &divide] {
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(handle),
                    Some(integer_type)
                );
            }
            assert_eq!(
                provider
                    .download_integer(&add)
                    .await
                    .expect("download add")
                    .data,
                expected_add
            );
            assert_eq!(
                provider
                    .download_integer(&subtract)
                    .await
                    .expect("download subtract")
                    .data,
                expected_subtract
            );
            assert_eq!(
                provider
                    .download_integer(&multiply)
                    .await
                    .expect("download multiply")
                    .data,
                expected_multiply
            );
            let expected_divide = match integer_type {
                IntegerElementType::I8 => HostIntegerDataOwned::I8(vec![127, -128, 0, 0, 127]),
                IntegerElementType::I16 => {
                    HostIntegerDataOwned::I16(vec![i16::MAX, i16::MIN, 0, 0, i16::MAX])
                }
                IntegerElementType::I32 => {
                    HostIntegerDataOwned::I32(vec![i32::MAX, i32::MIN, 0, 0, i32::MAX])
                }
                IntegerElementType::I64 => {
                    HostIntegerDataOwned::I64(vec![i64::MAX, i64::MIN, 0, 0, i64::MAX])
                }
                IntegerElementType::U8 => HostIntegerDataOwned::U8(vec![u8::MAX, 0, 0, 0, 0]),
                IntegerElementType::U16 => HostIntegerDataOwned::U16(vec![u16::MAX, 0, 0, 0, 0]),
                IntegerElementType::U32 => HostIntegerDataOwned::U32(vec![u32::MAX, 0, 0, 0, 0]),
                IntegerElementType::U64 => {
                    HostIntegerDataOwned::U64(vec![u64::MAX, 0, 0, 0x0000_0000_8000_0001, 0])
                }
            };
            assert_eq!(
                provider
                    .download_integer(&divide)
                    .await
                    .expect("download divide")
                    .data,
                expected_divide
            );
            for handle in [lhs, rhs, add, subtract, multiply, divide] {
                handle.release().unwrap();
            }
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_arithmetic_rejects_mixed_classes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [1, 1];
        let signed = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I8(&[1]),
                shape: &shape,
            })
            .expect("upload signed integer");
        let unsigned = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&[1]),
                shape: &shape,
            })
            .expect("upload unsigned integer");
        let error = provider
            .elem_mul(&signed, &unsigned)
            .await
            .expect_err("mixed native integer classes must not use an implicit promotion");
        assert!(error.to_string().contains("same class"));
        signed.release().expect("free signed handle");
        unsigned.release().expect("free unsigned handle");
    }

    #[tokio::test]
    async fn wgpu_native_integer_division_rounds_and_saturates_boundaries() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [1, 4];
        let cases = [
            (
                HostIntegerDataView::I8(&[3, -3, 5, i8::MIN]),
                HostIntegerDataView::I8(&[2, 2, 0, -1]),
                HostIntegerDataOwned::I8(vec![2, -2, i8::MAX, i8::MAX]),
            ),
            (
                HostIntegerDataView::I64(&[3, -3, 5, i64::MIN]),
                HostIntegerDataView::I64(&[2, 2, 0, -1]),
                HostIntegerDataOwned::I64(vec![2, -2, i64::MAX, i64::MAX]),
            ),
            (
                HostIntegerDataView::U8(&[3, 5, 0, u8::MAX]),
                HostIntegerDataView::U8(&[2, 0, 0, 0]),
                HostIntegerDataOwned::U8(vec![2, u8::MAX, 0, u8::MAX]),
            ),
            (
                HostIntegerDataView::U64(&[3, 5, 0, u64::MAX]),
                HostIntegerDataView::U64(&[2, 0, 0, 0]),
                HostIntegerDataOwned::U64(vec![2, u64::MAX, 0, u64::MAX]),
            ),
        ];

        for (left, right, expected) in cases {
            let lhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: left,
                    shape: &shape,
                })
                .expect("upload lhs");
            let rhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: right,
                    shape: &shape,
                })
                .expect("upload rhs");
            let quotient = provider
                .elem_div(&lhs, &rhs)
                .await
                .expect("integer division");
            let downloaded = provider
                .download_integer(&quotient)
                .await
                .expect("download integer division");
            assert_eq!(downloaded.shape, vec![1, 4]);
            assert_eq!(downloaded.data, expected);
            for handle in [lhs, rhs, quotient] {
                handle.release().unwrap();
            }
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_power_preserves_all_classes_and_rejects_negative_exponents() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [1, 4];
        let cases = [
            (
                HostIntegerDataView::I8(&[2, -2, 0, -1]),
                HostIntegerDataView::I8(&[7, 7, 0, 3]),
                HostIntegerDataOwned::I8(vec![i8::MAX, i8::MIN, 1, -1]),
            ),
            (
                HostIntegerDataView::I16(&[2, -2, 0, -1]),
                HostIntegerDataView::I16(&[15, 15, 0, 3]),
                HostIntegerDataOwned::I16(vec![i16::MAX, i16::MIN, 1, -1]),
            ),
            (
                HostIntegerDataView::I32(&[2, -2, 0, -1]),
                HostIntegerDataView::I32(&[31, 31, 0, 3]),
                HostIntegerDataOwned::I32(vec![i32::MAX, i32::MIN, 1, -1]),
            ),
            (
                HostIntegerDataView::I64(&[2, -2, 0, -1]),
                HostIntegerDataView::I64(&[63, 63, 0, 3]),
                HostIntegerDataOwned::I64(vec![i64::MAX, i64::MIN, 1, -1]),
            ),
            (
                HostIntegerDataView::U8(&[2, u8::MAX, 0, 1]),
                HostIntegerDataView::U8(&[8, 2, 0, 7]),
                HostIntegerDataOwned::U8(vec![u8::MAX, u8::MAX, 1, 1]),
            ),
            (
                HostIntegerDataView::U16(&[2, u16::MAX, 0, 1]),
                HostIntegerDataView::U16(&[16, 2, 0, 7]),
                HostIntegerDataOwned::U16(vec![u16::MAX, u16::MAX, 1, 1]),
            ),
            (
                HostIntegerDataView::U32(&[2, u32::MAX, 0, 1]),
                HostIntegerDataView::U32(&[32, 2, 0, 7]),
                HostIntegerDataOwned::U32(vec![u32::MAX, u32::MAX, 1, 1]),
            ),
            (
                HostIntegerDataView::U64(&[2, u64::MAX, 0, 1]),
                HostIntegerDataView::U64(&[64, 2, 0, 7]),
                HostIntegerDataOwned::U64(vec![u64::MAX, u64::MAX, 1, 1]),
            ),
        ];
        for (base, exponent, expected) in cases {
            let lhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: base,
                    shape: &shape,
                })
                .expect("upload power base");
            let rhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: exponent,
                    shape: &shape,
                })
                .expect("upload power exponent");
            let value = provider.elem_pow(&lhs, &rhs).await.expect("integer power");
            assert_eq!(
                provider
                    .download_integer(&value)
                    .await
                    .expect("download integer power")
                    .data,
                expected
            );
            for handle in [lhs, rhs, value] {
                handle.release().unwrap();
            }
        }

        let base = [2_i8];
        let exponent = [-1_i8];
        let lhs = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I8(&base),
                shape: &[1, 1],
            })
            .expect("upload negative-exponent base");
        let rhs = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I8(&exponent),
                shape: &[1, 1],
            })
            .expect("upload negative exponent");
        let error = provider
            .elem_pow(&lhs, &rhs)
            .await
            .expect_err("negative integer exponent must reject");
        assert!(error.to_string().contains("must be nonnegative"));
        for handle in [lhs, rhs] {
            handle.release().unwrap();
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_remainder_and_modulus_preserve_all_classes() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [1, 4];
        let signed = [
            (
                HostIntegerDataView::I8(&[-7, 7, -7, 5]),
                HostIntegerDataView::I8(&[4, -4, 0, 0]),
                HostIntegerDataOwned::I8(vec![-3, 3, 0, 0]),
                HostIntegerDataOwned::I8(vec![1, -1, -7, 5]),
            ),
            (
                HostIntegerDataView::I16(&[-7, 7, -7, 5]),
                HostIntegerDataView::I16(&[4, -4, 0, 0]),
                HostIntegerDataOwned::I16(vec![-3, 3, 0, 0]),
                HostIntegerDataOwned::I16(vec![1, -1, -7, 5]),
            ),
            (
                HostIntegerDataView::I32(&[-7, 7, -7, 5]),
                HostIntegerDataView::I32(&[4, -4, 0, 0]),
                HostIntegerDataOwned::I32(vec![-3, 3, 0, 0]),
                HostIntegerDataOwned::I32(vec![1, -1, -7, 5]),
            ),
            (
                HostIntegerDataView::I64(&[-7, 7, -7, 5]),
                HostIntegerDataView::I64(&[4, -4, 0, 0]),
                HostIntegerDataOwned::I64(vec![-3, 3, 0, 0]),
                HostIntegerDataOwned::I64(vec![1, -1, -7, 5]),
            ),
        ];
        let unsigned = [
            (
                HostIntegerDataView::U8(&[7, 8, 7, 0]),
                HostIntegerDataView::U8(&[4, 3, 0, 0]),
                HostIntegerDataOwned::U8(vec![3, 2, 0, 0]),
                HostIntegerDataOwned::U8(vec![3, 2, 7, 0]),
            ),
            (
                HostIntegerDataView::U16(&[7, 8, 7, 0]),
                HostIntegerDataView::U16(&[4, 3, 0, 0]),
                HostIntegerDataOwned::U16(vec![3, 2, 0, 0]),
                HostIntegerDataOwned::U16(vec![3, 2, 7, 0]),
            ),
            (
                HostIntegerDataView::U32(&[7, 8, 7, 0]),
                HostIntegerDataView::U32(&[4, 3, 0, 0]),
                HostIntegerDataOwned::U32(vec![3, 2, 0, 0]),
                HostIntegerDataOwned::U32(vec![3, 2, 7, 0]),
            ),
            (
                HostIntegerDataView::U64(&[7, 8, 7, 0]),
                HostIntegerDataView::U64(&[4, 3, 0, 0]),
                HostIntegerDataOwned::U64(vec![3, 2, 0, 0]),
                HostIntegerDataOwned::U64(vec![3, 2, 7, 0]),
            ),
        ];
        for (left, right, expected_rem, expected_mod) in signed.into_iter().chain(unsigned) {
            let lhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: left,
                    shape: &shape,
                })
                .expect("upload remainder lhs");
            let rhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: right,
                    shape: &shape,
                })
                .expect("upload remainder rhs");
            let remainder = provider.elem_rem(&lhs, &rhs).await.expect("integer rem");
            let modulus = provider.elem_mod(&lhs, &rhs).await.expect("integer mod");
            assert_eq!(
                provider
                    .download_integer(&remainder)
                    .await
                    .expect("download rem")
                    .data,
                expected_rem
            );
            assert_eq!(
                provider
                    .download_integer(&modulus)
                    .await
                    .expect("download mod")
                    .data,
                expected_mod
            );
            for handle in [lhs, rhs, remainder, modulus] {
                handle.release().unwrap();
            }
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_remainder_and_modulus_broadcast_column_major() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let lhs = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[-7, 7]),
                shape: &[2, 1],
            })
            .expect("upload lhs");
        let rhs = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[4, -4, 3]),
                shape: &[1, 3],
            })
            .expect("upload rhs");
        let rem = provider.elem_rem(&lhs, &rhs).await.expect("broadcast rem");
        let modulus = provider.elem_mod(&lhs, &rhs).await.expect("broadcast mod");
        assert_eq!(
            provider
                .download_integer(&rem)
                .await
                .expect("download rem")
                .data,
            HostIntegerDataOwned::I64(vec![-3, 3, -3, 3, -1, 1])
        );
        assert_eq!(
            provider
                .download_integer(&modulus)
                .await
                .expect("download mod")
                .data,
            HostIntegerDataOwned::I64(vec![1, 3, -3, -1, 2, 1])
        );
        for handle in [lhs, rhs, rem, modulus] {
            handle.release().unwrap();
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_arithmetic_broadcasts_all_classes_column_major() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let lhs_shape = [2, 1];
        let rhs_shape = [1, 3];
        let cases = [
            (
                HostIntegerDataView::I8(&[2, -2]),
                HostIntegerDataView::I8(&[1, 2, 3]),
                HostIntegerDataOwned::I8(vec![3, -1, 4, 0, 5, 1]),
                HostIntegerDataOwned::I8(vec![1, -3, 0, -4, -1, -5]),
                HostIntegerDataOwned::I8(vec![2, -2, 4, -4, 6, -6]),
            ),
            (
                HostIntegerDataView::I16(&[2, -2]),
                HostIntegerDataView::I16(&[1, 2, 3]),
                HostIntegerDataOwned::I16(vec![3, -1, 4, 0, 5, 1]),
                HostIntegerDataOwned::I16(vec![1, -3, 0, -4, -1, -5]),
                HostIntegerDataOwned::I16(vec![2, -2, 4, -4, 6, -6]),
            ),
            (
                HostIntegerDataView::I32(&[2, -2]),
                HostIntegerDataView::I32(&[1, 2, 3]),
                HostIntegerDataOwned::I32(vec![3, -1, 4, 0, 5, 1]),
                HostIntegerDataOwned::I32(vec![1, -3, 0, -4, -1, -5]),
                HostIntegerDataOwned::I32(vec![2, -2, 4, -4, 6, -6]),
            ),
            (
                HostIntegerDataView::I64(&[2, -2]),
                HostIntegerDataView::I64(&[1, 2, 3]),
                HostIntegerDataOwned::I64(vec![3, -1, 4, 0, 5, 1]),
                HostIntegerDataOwned::I64(vec![1, -3, 0, -4, -1, -5]),
                HostIntegerDataOwned::I64(vec![2, -2, 4, -4, 6, -6]),
            ),
            (
                HostIntegerDataView::U8(&[2, 4]),
                HostIntegerDataView::U8(&[1, 2, 3]),
                HostIntegerDataOwned::U8(vec![3, 5, 4, 6, 5, 7]),
                HostIntegerDataOwned::U8(vec![1, 3, 0, 2, 0, 1]),
                HostIntegerDataOwned::U8(vec![2, 4, 4, 8, 6, 12]),
            ),
            (
                HostIntegerDataView::U16(&[2, 4]),
                HostIntegerDataView::U16(&[1, 2, 3]),
                HostIntegerDataOwned::U16(vec![3, 5, 4, 6, 5, 7]),
                HostIntegerDataOwned::U16(vec![1, 3, 0, 2, 0, 1]),
                HostIntegerDataOwned::U16(vec![2, 4, 4, 8, 6, 12]),
            ),
            (
                HostIntegerDataView::U32(&[2, 4]),
                HostIntegerDataView::U32(&[1, 2, 3]),
                HostIntegerDataOwned::U32(vec![3, 5, 4, 6, 5, 7]),
                HostIntegerDataOwned::U32(vec![1, 3, 0, 2, 0, 1]),
                HostIntegerDataOwned::U32(vec![2, 4, 4, 8, 6, 12]),
            ),
            (
                HostIntegerDataView::U64(&[2, 4]),
                HostIntegerDataView::U64(&[1, 2, 3]),
                HostIntegerDataOwned::U64(vec![3, 5, 4, 6, 5, 7]),
                HostIntegerDataOwned::U64(vec![1, 3, 0, 2, 0, 1]),
                HostIntegerDataOwned::U64(vec![2, 4, 4, 8, 6, 12]),
            ),
        ];

        for (left, right, expected_add, expected_subtract, expected_multiply) in cases {
            let lhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: left,
                    shape: &lhs_shape,
                })
                .expect("upload lhs");
            let rhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: right,
                    shape: &rhs_shape,
                })
                .expect("upload rhs");
            let add = provider.elem_add(&lhs, &rhs).await.expect("broadcast add");
            let subtract = provider
                .elem_sub(&lhs, &rhs)
                .await
                .expect("broadcast subtract");
            let multiply = provider
                .elem_mul(&lhs, &rhs)
                .await
                .expect("broadcast multiply");
            let divide = provider
                .elem_div(&lhs, &rhs)
                .await
                .expect("broadcast divide");
            let expected_divide = match left.element_type() {
                IntegerElementType::I8 => HostIntegerDataOwned::I8(vec![2, -2, 1, -1, 1, -1]),
                IntegerElementType::I16 => HostIntegerDataOwned::I16(vec![2, -2, 1, -1, 1, -1]),
                IntegerElementType::I32 => HostIntegerDataOwned::I32(vec![2, -2, 1, -1, 1, -1]),
                IntegerElementType::I64 => HostIntegerDataOwned::I64(vec![2, -2, 1, -1, 1, -1]),
                IntegerElementType::U8 => HostIntegerDataOwned::U8(vec![2, 4, 1, 2, 1, 1]),
                IntegerElementType::U16 => HostIntegerDataOwned::U16(vec![2, 4, 1, 2, 1, 1]),
                IntegerElementType::U32 => HostIntegerDataOwned::U32(vec![2, 4, 1, 2, 1, 1]),
                IntegerElementType::U64 => HostIntegerDataOwned::U64(vec![2, 4, 1, 2, 1, 1]),
            };
            for (handle, expected) in [
                (&add, expected_add),
                (&subtract, expected_subtract),
                (&multiply, expected_multiply),
                (&divide, expected_divide),
            ] {
                let downloaded = provider
                    .download_integer(handle)
                    .await
                    .expect("download broadcast result");
                assert_eq!(downloaded.shape, vec![2, 3]);
                assert_eq!(downloaded.data, expected);
            }
            for handle in [lhs, rhs, add, subtract, multiply, divide] {
                handle.release().unwrap();
            }
        }
    }

    #[tokio::test]
    async fn wgpu_native_integer_comparison_and_minmax_broadcast_all_classes_column_major() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let lhs_shape = [2, 1];
        let rhs_shape = [1, 3];
        let cases = [
            (
                HostIntegerDataView::I8(&[2, -2]),
                HostIntegerDataView::I8(&[1, 2, 3]),
                HostIntegerDataOwned::I8(vec![1, -2, 2, -2, 2, -2]),
                HostIntegerDataOwned::I8(vec![2, 1, 2, 2, 3, 3]),
            ),
            (
                HostIntegerDataView::I16(&[2, -2]),
                HostIntegerDataView::I16(&[1, 2, 3]),
                HostIntegerDataOwned::I16(vec![1, -2, 2, -2, 2, -2]),
                HostIntegerDataOwned::I16(vec![2, 1, 2, 2, 3, 3]),
            ),
            (
                HostIntegerDataView::I32(&[2, -2]),
                HostIntegerDataView::I32(&[1, 2, 3]),
                HostIntegerDataOwned::I32(vec![1, -2, 2, -2, 2, -2]),
                HostIntegerDataOwned::I32(vec![2, 1, 2, 2, 3, 3]),
            ),
            (
                HostIntegerDataView::I64(&[2, -2]),
                HostIntegerDataView::I64(&[1, 2, 3]),
                HostIntegerDataOwned::I64(vec![1, -2, 2, -2, 2, -2]),
                HostIntegerDataOwned::I64(vec![2, 1, 2, 2, 3, 3]),
            ),
            (
                HostIntegerDataView::U8(&[2, 0]),
                HostIntegerDataView::U8(&[1, 2, 3]),
                HostIntegerDataOwned::U8(vec![1, 0, 2, 0, 2, 0]),
                HostIntegerDataOwned::U8(vec![2, 1, 2, 2, 3, 3]),
            ),
            (
                HostIntegerDataView::U16(&[2, 0]),
                HostIntegerDataView::U16(&[1, 2, 3]),
                HostIntegerDataOwned::U16(vec![1, 0, 2, 0, 2, 0]),
                HostIntegerDataOwned::U16(vec![2, 1, 2, 2, 3, 3]),
            ),
            (
                HostIntegerDataView::U32(&[2, 0]),
                HostIntegerDataView::U32(&[1, 2, 3]),
                HostIntegerDataOwned::U32(vec![1, 0, 2, 0, 2, 0]),
                HostIntegerDataOwned::U32(vec![2, 1, 2, 2, 3, 3]),
            ),
            (
                HostIntegerDataView::U64(&[2, 0]),
                HostIntegerDataView::U64(&[1, 2, 3]),
                HostIntegerDataOwned::U64(vec![1, 0, 2, 0, 2, 0]),
                HostIntegerDataOwned::U64(vec![2, 1, 2, 2, 3, 3]),
            ),
        ];

        for (left, right, expected_minimum, expected_maximum) in cases {
            let lhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: left,
                    shape: &lhs_shape,
                })
                .expect("upload lhs");
            let rhs = provider
                .upload_integer(&HostIntegerTensorView {
                    data: right,
                    shape: &rhs_shape,
                })
                .expect("upload rhs");
            let equal = provider.elem_eq(&lhs, &rhs).await.expect("broadcast eq");
            let not_equal = provider.elem_ne(&lhs, &rhs).await.expect("broadcast ne");
            let less = provider.elem_lt(&lhs, &rhs).await.expect("broadcast lt");
            let greater_equal = provider.elem_ge(&lhs, &rhs).await.expect("broadcast ge");
            let minimum = provider.elem_min(&lhs, &rhs).await.expect("broadcast min");
            let maximum = provider.elem_max(&lhs, &rhs).await.expect("broadcast max");
            for handle in [&equal, &not_equal, &less, &greater_equal] {
                assert!(runmat_accelerate_api::handle_is_logical(handle));
                let downloaded = provider
                    .download(handle)
                    .await
                    .expect("download comparison");
                assert_eq!(downloaded.shape, vec![2, 3]);
            }
            assert_eq!(
                provider.download(&equal).await.expect("download eq").data,
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            );
            assert_eq!(
                provider
                    .download(&not_equal)
                    .await
                    .expect("download ne")
                    .data,
                vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
            );
            assert_eq!(
                provider.download(&less).await.expect("download lt").data,
                vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
            );
            assert_eq!(
                provider
                    .download(&greater_equal)
                    .await
                    .expect("download ge")
                    .data,
                vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            );
            for (handle, expected) in [(&minimum, expected_minimum), (&maximum, expected_maximum)] {
                let downloaded = provider
                    .download_integer(handle)
                    .await
                    .expect("download minmax");
                assert_eq!(downloaded.shape, vec![2, 3]);
                assert_eq!(downloaded.data, expected);
            }
            for handle in [
                lhs,
                rhs,
                equal,
                not_equal,
                less,
                greater_equal,
                minimum,
                maximum,
            ] {
                handle.release().unwrap();
            }
        }
    }

    #[tokio::test]
    async fn wgpu_provider_parity_samples_real_unary_and_broadcast_binary_paths() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-8,
            runmat_accelerate_api::ProviderPrecision::F32 => 2e-4,
        };

        let unary_input = provider
            .upload(&HostTensorView {
                data: &[0.25, 0.5, 1.0, 1.5],
                shape: &[4, 1],
            })
            .expect("upload unary");
        let erf = provider.unary_erf(&unary_input).await.expect("erf");
        let erf_host = provider.download(&erf).await.expect("download erf");
        let gamma = provider.unary_gamma(&unary_input).await.expect("gamma");
        let gamma_host = provider.download(&gamma).await.expect("download gamma");
        let gammaln = provider.unary_gammaln(&unary_input).await.expect("gammaln");
        let gammaln_host = provider.download(&gammaln).await.expect("download gammaln");
        let realsqrt = provider
            .unary_sqrt(&unary_input)
            .await
            .expect("realsqrt provider sqrt");
        let realsqrt_host = provider
            .download(&realsqrt)
            .await
            .expect("download realsqrt");
        for host in [&erf_host, &gamma_host, &gammaln_host, &realsqrt_host] {
            assert_eq!(host.shape, vec![4, 1]);
            assert_eq!(host.storage, GpuTensorStorage::Real);
        }
        assert_close(
            &erf_host.data,
            &[
                libm::erf(0.25),
                libm::erf(0.5),
                libm::erf(1.0),
                libm::erf(1.5),
            ],
            tol,
        );
        assert_close(
            &gamma_host.data,
            &[
                libm::tgamma(0.25),
                libm::tgamma(0.5),
                libm::tgamma(1.0),
                libm::tgamma(1.5),
            ],
            tol,
        );
        assert_close(
            &gammaln_host.data,
            &[
                libm::lgamma(0.25),
                libm::lgamma(0.5),
                libm::lgamma(1.0),
                libm::lgamma(1.5),
            ],
            tol,
        );
        assert_close(
            &realsqrt_host.data,
            &[0.5, 0.5_f64.sqrt(), 1.0, 1.5_f64.sqrt()],
            tol,
        );

        let lhs = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: &[2, 3],
            })
            .expect("upload lhs");
        let rhs = provider
            .upload(&HostTensorView {
                data: &[10.0, 20.0, 30.0],
                shape: &[1, 3],
            })
            .expect("upload rhs");
        let add = provider.elem_add(&lhs, &rhs).await.expect("add");
        let sub = provider.elem_sub(&lhs, &rhs).await.expect("sub");
        let mul = provider.elem_mul(&lhs, &rhs).await.expect("mul");
        let div = provider.elem_div(&lhs, &rhs).await.expect("div");
        let pow = provider.elem_pow(&lhs, &rhs).await.expect("pow");
        let add_host = provider.download(&add).await.expect("download add");
        let sub_host = provider.download(&sub).await.expect("download sub");
        let mul_host = provider.download(&mul).await.expect("download mul");
        let div_host = provider.download(&div).await.expect("download div");
        let pow_host = provider.download(&pow).await.expect("download pow");
        for host in [&add_host, &sub_host, &mul_host, &div_host, &pow_host] {
            assert_eq!(host.shape, vec![2, 3]);
            assert_eq!(host.storage, GpuTensorStorage::Real);
        }
        assert_close(&add_host.data, &[11.0, 12.0, 23.0, 24.0, 35.0, 36.0], tol);
        assert_close(
            &sub_host.data,
            &[-9.0, -8.0, -17.0, -16.0, -25.0, -24.0],
            tol,
        );
        assert_close(&mul_host.data, &[10.0, 20.0, 60.0, 80.0, 150.0, 180.0], tol);
        assert_close(&div_host.data, &[0.1, 0.2, 0.15, 0.2, 5.0 / 30.0, 0.2], tol);
        assert_close_scaled(
            &pow_host.data,
            &[
                1.0_f64.powf(10.0),
                2.0_f64.powf(10.0),
                3.0_f64.powf(20.0),
                4.0_f64.powf(20.0),
                5.0_f64.powf(30.0),
                6.0_f64.powf(30.0),
            ],
            tol,
        );
    }

    fn sin_complex_host(re: f64, im: f64) -> (f64, f64) {
        (re.sin() * im.cosh(), re.cos() * im.sinh())
    }

    fn cos_complex_host(re: f64, im: f64) -> (f64, f64) {
        (re.cos() * im.cosh(), -re.sin() * im.sinh())
    }

    fn tan_complex_host(re: f64, im: f64) -> (f64, f64) {
        let two_re = 2.0 * re;
        let two_im = 2.0 * im;
        let inv_cosh = 1.0 / two_im.cosh();
        let denom = 1.0 + two_re.cos() * inv_cosh;
        ((two_re.sin() * inv_cosh) / denom, two_im.tanh() / denom)
    }

    #[tokio::test]
    async fn wgpu_erfcinv_matches_erfc_inverse() {
        let Ok(provider) = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let shape = [9, 1];
        let input = [0.0, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0, -0.1, 2.1];
        let handle = provider
            .upload(&HostTensorView {
                data: &input,
                shape: &shape,
            })
            .expect("upload");

        let result = provider.unary_erfcinv(&handle).await.expect("erfcinv");
        assert_eq!(
            runmat_accelerate_api::handle_storage(&result),
            GpuTensorStorage::Real
        );
        let host = provider.download(&result).await.expect("download erfcinv");
        assert_eq!(host.shape, shape.to_vec());
        assert!(host.data[0].is_infinite() && host.data[0].is_sign_positive());
        assert_eq!(host.data[3], 0.0);
        assert!(host.data[6].is_infinite() && host.data[6].is_sign_negative());
        assert!(host.data[7].is_nan());
        assert!(host.data[8].is_nan());

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-8,
            runmat_accelerate_api::ProviderPrecision::F32 => 2e-4,
        };
        for (idx, (&actual, &expected_input)) in
            host.data[1..6].iter().zip(input[1..6].iter()).enumerate()
        {
            let roundtrip = libm::erfc(actual);
            assert!(
                (roundtrip - expected_input).abs() <= tol,
                "lane {}: erfc(erfcinv({expected_input})) = {roundtrip}, value {actual}",
                idx + 1
            );
        }
    }

    #[tokio::test]
    async fn wgpu_round_ops_match_cpu() {
        let Ok(provider) = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let shape = [6, 1];
        let input = [-2.5, -0.2, 0.5, 1.2345, 149.9, 98765.0];
        let handle = provider
            .upload(&HostTensorView {
                data: &input,
                shape: &shape,
            })
            .expect("upload");

        let integer = provider.unary_round(&handle).await.expect("round");
        let decimals = provider
            .round_digits(&handle, -2, false)
            .await
            .expect("round decimals");
        let significant = provider
            .round_digits(&handle, 3, true)
            .await
            .expect("round significant");

        let integer_host = provider.download(&integer).await.expect("download integer");
        let decimals_host = provider
            .download(&decimals)
            .await
            .expect("download decimals");
        let significant_host = provider
            .download(&significant)
            .await
            .expect("download significant");

        assert_eq!(integer_host.data, vec![-3.0, 0.0, 1.0, 1.0, 150.0, 98765.0]);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-8,
            runmat_accelerate_api::ProviderPrecision::F32 => 2e-6,
        };
        assert_close(
            &decimals_host.data,
            &[-0.0, -0.0, 0.0, 0.0, 100.0, 98800.0],
            tol,
        );
        let expected = [-2.5, -0.2, 0.5, 1.23, 150.0, 98800.0];
        assert_close_scaled(&significant_host.data, &expected, tol);
    }

    #[tokio::test]
    #[ignore = "requires an actual compatible WGPU adapter"]
    async fn complex_broadcast_over_shader_rank_rejects_before_effects() {
        let provider = register_wgpu_provider_for_test().expect("required actual WGPU adapter");
        let mut complex_shape = vec![1; crate::backend::wgpu::params::BCAST_MAX_RANK + 1];
        let mut real_shape = complex_shape.clone();
        complex_shape[0] = 2;
        real_shape[1] = 2;
        let complex = complex_pair(provider, &[1.0, 3.0], &[2.0, 4.0], &complex_shape).await;
        let real = provider
            .upload(&HostTensorView {
                data: &[10.0, 20.0],
                shape: &real_shape,
            })
            .expect("real high-rank upload");
        let telemetry = provider.telemetry_snapshot();
        let allocations = provider
            .allocations
            .record_count()
            .expect("allocation records");

        let error = provider
            .elem_add(&complex, &real)
            .await
            .expect_err("shader ABI cannot address this rank");
        assert!(error.is_unsupported());
        assert_eq!(provider.telemetry_snapshot(), telemetry);
        assert_eq!(
            provider
                .allocations
                .record_count()
                .expect("allocation records"),
            allocations
        );
    }

    #[tokio::test]
    async fn wgpu_complex_binary_ops_match_cpu() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [2, 2];
        let a = complex_pair(
            provider,
            &[1.0, -2.0, 3.5, 0.5],
            &[0.5, 4.0, -1.0, -2.5],
            &shape,
        )
        .await;
        let b = complex_pair(
            provider,
            &[2.0, 0.25, -1.0, 4.0],
            &[-1.0, 0.75, 2.0, -0.5],
            &shape,
        )
        .await;

        let add = provider.elem_add(&a, &b).await.expect("complex add");
        let sub = provider.elem_sub(&a, &b).await.expect("complex sub");
        let mul = provider.elem_mul(&a, &b).await.expect("complex mul");
        let div = provider.elem_div(&a, &b).await;
        let ieee_div_available = provider.ieee_words_available(provider.precision);
        if !ieee_div_available {
            let error = div
                .as_ref()
                .expect_err("complex f64 division needs IEEE words");
            assert!(error.is_unsupported());
        }

        for handle in [&add, &sub, &mul] {
            assert_eq!(
                handle.descriptor().storage,
                Some(GpuTensorStorage::ComplexInterleaved)
            );
            assert_eq!(
                runmat_accelerate_api::handle_storage(handle),
                GpuTensorStorage::ComplexInterleaved
            );
        }

        let add_host = provider.download(&add).await.expect("download add");
        let sub_host = provider.download(&sub).await.expect("download sub");
        let mul_host = provider.download(&mul).await.expect("download mul");
        let div_host = if ieee_div_available {
            Some(
                provider
                    .download(&div.expect("complex div"))
                    .await
                    .expect("download div"),
            )
        } else {
            None
        };
        assert_eq!(add_host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(add_host.shape, shape.to_vec());

        let lhs = [(1.0, 0.5), (-2.0, 4.0), (3.5, -1.0), (0.5, -2.5)];
        let rhs = [(2.0, -1.0), (0.25, 0.75), (-1.0, 2.0), (4.0, -0.5)];
        let expected_add: Vec<(f64, f64)> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|((ar, ai), (br, bi))| (ar + br, ai + bi))
            .collect();
        let expected_sub: Vec<(f64, f64)> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|((ar, ai), (br, bi))| (ar - br, ai - bi))
            .collect();
        let expected_mul: Vec<(f64, f64)> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|((ar, ai), (br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
            .collect();
        let expected_div: Vec<(f64, f64)> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|((ar, ai), (br, bi))| {
                let denom = br * br + bi * bi;
                ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
            })
            .collect();

        assert_interleaved_close(&add_host.data, &expected_add);
        assert_interleaved_close(&sub_host.data, &expected_sub);
        assert_interleaved_close(&mul_host.data, &expected_mul);
        if let Some(div_host) = div_host {
            assert_interleaved_close(&div_host.data, &expected_div);
        }

        if provider.precision() == runmat_accelerate_api::ProviderPrecision::F64 {
            let native_f32 = [8.0_f32, f32::from_bits(1)];
            let divisor_f32 = [0.0_f32, 1.0];
            let a32 = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&native_f32),
                    shape: &[1, 1],
                    storage: GpuTensorStorage::ComplexInterleaved,
                })
                .expect("upload native f32 complex lhs on f64 provider");
            let b32 = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&divisor_f32),
                    shape: &[1, 1],
                    storage: GpuTensorStorage::ComplexInterleaved,
                })
                .expect("upload native f32 complex rhs on f64 provider");
            let q32 = provider
                .elem_div(&a32, &b32)
                .await
                .expect("native f32 complex div");
            assert_eq!(
                q32.descriptor().element_type,
                Some(runmat_accelerate_api::NumericElementType::F32)
            );
            let q32_host = provider
                .download(&q32)
                .await
                .expect("download native f32 div");
            assert_eq!(q32_host.data, vec![f32::from_bits(1) as f64, -8.0]);
            let before_mixed = provider.telemetry_snapshot();
            let mixed = provider
                .elem_div(&a32, &b)
                .await
                .expect_err("mixed native precision");
            assert!(mixed.is_unsupported());
            assert_eq!(provider.telemetry_snapshot(), before_mixed);
        }

        if !ieee_div_available {
            return;
        }
        // Upload interleaved words directly: constructing complex values by a
        // prior GPU arithmetic kernel could flush the input subnormal first.
        let (lhs, rhs, expected) = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => {
                let lhs = [f32::MAX, f32::from_bits(1), 1.0, 0.0];
                let rhs = [0.0_f32, 1.0];
                let a = provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&lhs),
                        shape: &[2, 1],
                        storage: GpuTensorStorage::ComplexInterleaved,
                    })
                    .expect("upload complex f32 boundary lhs");
                let b = provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&rhs),
                        shape: &[1, 1],
                        storage: GpuTensorStorage::ComplexInterleaved,
                    })
                    .expect("upload complex f32 boundary rhs");
                (a, b, (f32::from_bits(1) as f64, -(f32::MAX as f64)))
            }
            runmat_accelerate_api::ProviderPrecision::F64 => {
                let lhs = [f64::MAX, f64::from_bits(1), 1.0, 0.0];
                let rhs = [0.0_f64, 1.0];
                let a = provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F64(&lhs),
                        shape: &[2, 1],
                        storage: GpuTensorStorage::ComplexInterleaved,
                    })
                    .expect("upload complex f64 boundary lhs");
                let b = provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F64(&rhs),
                        shape: &[1, 1],
                        storage: GpuTensorStorage::ComplexInterleaved,
                    })
                    .expect("upload complex f64 boundary rhs");
                (a, b, (f64::from_bits(1), -f64::MAX))
            }
        };
        let quotient = provider
            .elem_div(&lhs, &rhs)
            .await
            .expect("boundary broadcast div");
        let boundary = provider
            .download(&quotient)
            .await
            .expect("download boundary div");
        assert_eq!(boundary.shape, vec![2, 1]);
        assert_eq!(boundary.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(boundary.data[0], expected.0, "retained minor component");
        assert_eq!(boundary.data[1], expected.1, "finite major component");
    }

    #[tokio::test]
    async fn wgpu_complex_unary_trig_ops_match_cpu() {
        let Ok(provider) = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let input = [(0.5, 0.75), (2.0, -0.25), (-0.75, 0.5)];
        let real = input.iter().map(|&(re, _)| re).collect::<Vec<_>>();
        let imag = input.iter().map(|&(_, im)| im).collect::<Vec<_>>();
        let handle = complex_pair(provider, &real, &imag, &[3, 1]).await;

        let sin = provider.unary_sin(&handle).await.expect("complex sin");
        let cos = provider.unary_cos(&handle).await.expect("complex cos");
        let tan = provider.unary_tan(&handle).await.expect("complex tan");

        for handle in [&sin, &cos, &tan] {
            assert_eq!(
                handle.descriptor().storage,
                Some(GpuTensorStorage::ComplexInterleaved)
            );
            assert_eq!(
                runmat_accelerate_api::handle_storage(handle),
                GpuTensorStorage::ComplexInterleaved
            );
        }

        let sin_host = provider.download(&sin).await.expect("download sin");
        let cos_host = provider.download(&cos).await.expect("download cos");
        let tan_host = provider.download(&tan).await.expect("download tan");
        assert_eq!(sin_host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(cos_host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(tan_host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(sin_host.shape, vec![3, 1]);
        assert_eq!(cos_host.shape, vec![3, 1]);
        assert_eq!(tan_host.shape, vec![3, 1]);

        assert_interleaved_close(
            &sin_host.data,
            &input.map(|(re, im)| sin_complex_host(re, im)),
        );
        assert_interleaved_close(
            &cos_host.data,
            &input.map(|(re, im)| cos_complex_host(re, im)),
        );
        assert_interleaved_close(
            &tan_host.data,
            &input.map(|(re, im)| tan_complex_host(re, im)),
        );
    }

    #[tokio::test]
    async fn wgpu_complex_unary_trig_large_imag_edges_are_not_nan() {
        let Ok(provider) = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let input = [
            (0.0, 90.0),
            (0.0, -90.0),
            (std::f64::consts::FRAC_PI_2, 90.0),
            (std::f64::consts::FRAC_PI_2, -90.0),
            (0.25, 90.0),
            (-0.25, -90.0),
            (90.0, 0.0),
            (-90.0, 0.0),
            (0.0, 50.0),
        ];
        let real = input.iter().map(|&(re, _)| re).collect::<Vec<_>>();
        let imag = input.iter().map(|&(_, im)| im).collect::<Vec<_>>();
        let shape = [input.len(), 1];
        let handle = complex_pair(provider, &real, &imag, &shape).await;

        let sin = provider.unary_sin(&handle).await.expect("complex sin");
        let sinc = provider.unary_sinc(&handle).await.expect("complex sinc");
        let cos = provider.unary_cos(&handle).await.expect("complex cos");
        let sinh = provider.unary_sinh(&handle).await.expect("complex sinh");
        let cosh = provider.unary_cosh(&handle).await.expect("complex cosh");
        let tan = provider.unary_tan(&handle).await.expect("complex tan");

        let sin_host = provider.download(&sin).await.expect("download sin");
        let sinc_host = provider.download(&sinc).await.expect("download sinc");
        let cos_host = provider.download(&cos).await.expect("download cos");
        let sinh_host = provider.download(&sinh).await.expect("download sinh");
        let cosh_host = provider.download(&cosh).await.expect("download cosh");
        let tan_host = provider.download(&tan).await.expect("download tan");
        assert_eq!(sin_host.shape, vec![9, 1]);
        assert_eq!(sinc_host.shape, vec![9, 1]);
        assert_eq!(cos_host.shape, vec![9, 1]);
        assert_eq!(sinh_host.shape, vec![9, 1]);
        assert_eq!(cosh_host.shape, vec![9, 1]);
        assert_eq!(tan_host.shape, vec![9, 1]);

        assert_eq!(sin_host.data[0], 0.0, "sin(0 + 90i) real lane");
        assert!(
            sin_host.data[1].is_finite(),
            "sin(0 + 90i) imag lane must be finite"
        );
        assert_eq!(sin_host.data[2], 0.0, "sin(0 - 90i) real lane");
        assert!(
            sin_host.data[3].is_finite(),
            "sin(0 - 90i) imag lane must be finite"
        );
        assert!(
            cos_host.data[0].is_finite(),
            "cos(0 + 90i) real lane must be finite"
        );
        assert!(
            cos_host.data[1].abs() < 1e-5,
            "cos(0 + 90i) imag lane got {}",
            cos_host.data[1]
        );
        assert!(
            cos_host.data[2].is_finite(),
            "cos(0 - 90i) real lane must be finite"
        );
        assert!(
            cos_host.data[3].abs() < 1e-5,
            "cos(0 - 90i) imag lane got {}",
            cos_host.data[3]
        );
        assert!(
            tan_host.data[16].abs() < 1e-5,
            "tan(0 + 50i) real lane got {}",
            tan_host.data[16]
        );
        assert!(
            (tan_host.data[17] - 1.0).abs() < 1e-5,
            "tan(0 + 50i) imag lane got {}",
            tan_host.data[17]
        );
        for (idx, lane) in sin_host
            .data
            .iter()
            .chain(sinc_host.data.iter())
            .chain(cos_host.data.iter())
            .chain(sinh_host.data.iter())
            .chain(cosh_host.data.iter())
            .chain(tan_host.data.iter())
            .enumerate()
        {
            assert!(lane.is_finite(), "lane {idx} was not finite");
        }
    }

    #[tokio::test]
    async fn wgpu_complex_real_mix_and_scalar_ops_stay_complex() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [3, 1];
        let complex = complex_pair(provider, &[1.0, -2.0, 4.0], &[0.5, 3.0, -1.5], &shape).await;
        let real = provider
            .upload(&HostTensorView {
                data: &[2.0, -1.0, 0.25],
                shape: &shape,
            })
            .expect("upload real");

        let mixed = provider
            .elem_mul(&complex, &real)
            .await
            .expect("complex .* real");
        let scalar = provider.scalar_rdiv(&complex, 2.0).expect("2 ./ complex");

        assert_eq!(
            runmat_accelerate_api::handle_storage(&mixed),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            runmat_accelerate_api::handle_storage(&scalar),
            GpuTensorStorage::ComplexInterleaved
        );

        let mixed_host = provider.download(&mixed).await.expect("download mixed");
        let scalar_host = provider.download(&scalar).await.expect("download scalar");
        assert_interleaved_close(&mixed_host.data, &[(2.0, 1.0), (2.0, -3.0), (1.0, -0.375)]);
        let expected_scalar = [
            (1.6, -0.8),
            (-0.3076923076923077, -0.46153846153846156),
            (0.4383561643835616, 0.1643835616438356),
        ];
        assert_interleaved_close(&scalar_host.data, &expected_scalar);
    }

    #[tokio::test]
    async fn wgpu_complex_repmat_preserves_interleaved_storage() {
        let Ok(provider) = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let scalar = complex_pair(provider, &[2.0], &[-3.0], &[1, 1]).await;
        let tiled = provider
            .repmat(&scalar, &[3, 1])
            .expect("complex repmat")
            .expect("supported");

        assert_eq!(tiled.shape(), vec![3, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&tiled),
            GpuTensorStorage::ComplexInterleaved
        );
        let host = provider.download(&tiled).await.expect("download tiled");
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_interleaved_close(&host.data, &[(2.0, -3.0), (2.0, -3.0), (2.0, -3.0)]);
    }

    #[tokio::test]
    async fn wgpu_precise_expm1_and_exp_hooks_reject_complex_interleaved_storage() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        let shape = [1, 1];
        let real = provider
            .upload(&HostTensorView {
                data: &[1.0e-7],
                shape: &shape,
            })
            .expect("upload real");
        let output = provider.unary_expm1(&real).await.expect("expm1");
        let host = provider.download(&output).await.expect("download expm1");
        let expected = 1.0e-7_f64.exp_m1();
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-14,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-7,
        };
        assert!((host.data[0] - expected).abs() <= tolerance);

        let complex = complex_pair(provider, &[1.0], &[2.0], &shape).await;
        assert!(provider.unary_exp(&complex).await.is_err());
        assert!(provider.unary_expm1(&complex).await.is_err());
    }

    #[tokio::test]
    async fn wgpu_float_hooks_report_native_precision_mismatch_as_unsupported() {
        let Some(provider) = register_wgpu_provider_for_test() else {
            return;
        };
        if provider.precision() == runmat_accelerate_api::ProviderPrecision::F64 {
            return;
        }
        let values = [0.0_f64, 1.0];
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&values),
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            })
            .expect("upload native f64");

        let before = provider.telemetry_snapshot();
        for error in [
            provider.unary_exp(&input).await.expect_err("exp mismatch"),
            provider
                .unary_expm1(&input)
                .await
                .expect_err("expm1 mismatch"),
        ] {
            assert!(runmat_accelerate_api::is_unsupported_provider_operation(
                &error
            ));
        }
        assert_eq!(provider.telemetry_snapshot(), before);
        assert_eq!(
            input.descriptor().element_type,
            Some(runmat_accelerate_api::NumericElementType::F64)
        );
    }
}
