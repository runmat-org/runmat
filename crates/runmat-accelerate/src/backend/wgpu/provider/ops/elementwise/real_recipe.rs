//! Shared physical recipe for ordinary real scalar and unary operations.
//! This owns allocation/ABI/launch geometry, not source arithmetic equivalence.
use super::*;
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, ProviderElementType, ProviderExecutedRealElementwise,
    ProviderExecutionError, ProviderExecutionResult, ProviderLayout,
    ProviderRealElementwiseObserved, ProviderRealElementwiseOperation as Operation,
    ProviderRealElementwisePhysicalProposal as Proposal, ProviderRealElementwiseRecipe,
    ProviderRealElementwiseRequest, ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use std::cell::RefCell;

#[path = "real_recipe/cast.rs"]
mod cast;

#[derive(Default)]
pub(super) struct RealObserved {
    pub(super) effect: bool,
    pub(super) resources: ProviderRealElementwiseObserved,
    pub(super) capture_probe_tickets: bool,
    pub(super) probe_tickets: Vec<crate::backend::wgpu::dispatch::observation::ProbeFenceTicket>,
    pub(super) output_resource_ticket:
        Option<runmat_accelerate_api::observation::ProviderResourceTicket>,
}
impl RealObserved {
    pub(super) fn output(&mut self, buffer: &runmat_accelerate_api::WgpuBufferOwner) {
        self.effect = true;
        self.resources.output_buffer_bytes = Some(buffer.size());
    }
    pub(super) fn uniform(
        &mut self,
        buffer: &runmat_accelerate_api::WgpuBufferOwner,
    ) -> Result<()> {
        self.uniform_bytes(buffer.size())
    }
    fn uniform_bytes(&mut self, bytes: u64) -> Result<()> {
        self.effect = true;
        self.resources.uniform_allocations = self
            .resources
            .uniform_allocations
            .checked_add(1)
            .ok_or_else(|| anyhow!("real uniform count overflow"))?;
        self.resources.uniform_buffer_bytes = self
            .resources
            .uniform_buffer_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("real uniform extent overflow"))?;
        Ok(())
    }
    pub(super) fn submitted(&mut self, dispatch: bool) -> Result<()> {
        self.effect = true;
        self.resources.queue_submissions = self
            .resources
            .queue_submissions
            .checked_add(1)
            .ok_or_else(|| anyhow!("real submission count overflow"))?;
        if dispatch {
            self.resources.dispatches = self
                .resources
                .dispatches
                .checked_add(1)
                .ok_or_else(|| anyhow!("real dispatch count overflow"))?;
        }
        Ok(())
    }
    pub(super) fn require(&self, plan: &Proposal) -> Result<()> {
        ensure!(
            self.resources.output_buffer_bytes == Some(plan.output_buffer_bytes)
                && self.resources.uniform_allocations == plan.uniform_allocations
                && self.resources.uniform_buffer_bytes == plan.uniform_buffer_bytes
                && self.resources.dispatches == plan.dispatches
                && self.resources.queue_submissions == plan.queue_submissions,
            "ordinary real observed resources differ from its execution recipe"
        );
        Ok(())
    }
}

pub(super) fn source_representation(input: &GpuTensorHandle) -> Option<ProviderRepresentation> {
    if input.physical().storage() != GpuTensorStorage::Real || input.view().is_logical() {
        return None;
    }
    let element_type = match input.physical().element_type() {
        runmat_accelerate_api::NumericElementType::F32 => ProviderElementType::F32,
        runmat_accelerate_api::NumericElementType::F64 => ProviderElementType::F64,
        _ => return None,
    };
    Some(ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: if runmat_accelerate_api::handle_is_transposed(input) {
            ProviderLayout::Strided
        } else {
            ProviderLayout::ColumnMajorContiguous
        },
        residency: ProviderResidency::Device,
        shape: input
            .shape()
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok()?,
    })
}

fn scalar_uniform_available(limits: &wgpu::Limits) -> bool {
    limits.max_uniform_buffers_per_shader_stage >= 1
}

fn unary_opcode(operation: Operation) -> Option<crate::backend::wgpu::types::UnaryOpCode> {
    use crate::backend::wgpu::types::UnaryOpCode as Code;
    use runmat_types::ScalarUnaryMath as Math;
    Some(match operation {
        Operation::Sqrt => Code::Sqrt,
        Operation::Spacing => Code::Spacing,
        Operation::Expm1 => Code::Expm1,
        Operation::UnaryMath(operation) => match operation {
            Math::Exp => Code::Exp,
            Math::Sin => Code::Sin,
            Math::Cos => Code::Cos,
            Math::Tan => Code::Tan,
            Math::Sinh => Code::Sinh,
            Math::Cosh => Code::Cosh,
            Math::Tanh => Code::Tanh,
            Math::Atan => Code::Atan,
            Math::Asinh => Code::Asinh,
            Math::Acosh => Code::Acosh,
            Math::Atanh => Code::Atanh,
            Math::Log => Code::Log,
            Math::Log1p => Code::Log1p,
            Math::Log10 => Code::Log10,
            Math::Log2 => Code::Log2,
            Math::Sqrt => Code::Sqrt,
            Math::Asin => Code::Asin,
            Math::Acos => Code::Acos,
        },
        Operation::ExactSingleBasePower { .. }
        | Operation::ScalarAdd { .. }
        | Operation::ScalarMul { .. }
        | Operation::ScalarMax { .. }
        | Operation::FloatingCast { .. } => return None,
    })
}

fn native_scalar_control(operation: Operation, precision: NumericPrecision) -> bool {
    match operation {
        Operation::ScalarAdd { control, mode } => matches!(
            (precision, control, mode),
            (
                NumericPrecision::F32,
                ScalarLiteral::F32Bits(_),
                runmat_types::RealFloatingAddMode::Single
            ) | (
                NumericPrecision::F64,
                ScalarLiteral::F64Bits(_),
                runmat_types::RealFloatingAddMode::Double
            )
        ),
        Operation::ScalarMul { control } | Operation::ScalarMax { control } => matches!(
            (precision, control),
            (NumericPrecision::F32, ScalarLiteral::F32Bits(_))
                | (NumericPrecision::F64, ScalarLiteral::F64Bits(_))
        ),
        Operation::Sqrt
        | Operation::Spacing
        | Operation::Expm1
        | Operation::UnaryMath(_)
        | Operation::FloatingCast { .. }
        | Operation::ExactSingleBasePower { .. } => false,
    }
}

impl WgpuProvider {
    pub(crate) fn propose_real_elementwise_exec(
        &self,
        request: &ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<Proposal>> {
        if let Operation::FloatingCast { target } = request.operation {
            return self.propose_real_floating_cast(request, target);
        }
        let mut output = request.source.clone();
        output.layout = ProviderLayout::ColumnMajorContiguous;
        let Some(output_buffer_bytes) = self.plan_dense_real_output_buffer_bytes_exec(&output)?
        else {
            return Ok(None);
        };
        let Some(elements) = request.source.checked_element_count() else {
            return Ok(None);
        };
        if elements == 0 {
            if !matches!(request.operation, Operation::ExactSingleBasePower { control }
                if runmat_accelerate_api::ExactSingleBasePower::for_control(control).is_some())
                || request.source.element_type != ProviderElementType::F32
            {
                return Ok(None);
            }
            return Ok(Some(Proposal {
                output_route:
                    runmat_accelerate_api::ProviderRealElementwiseOutputRoute::FreshAllocation,
                output_buffer_bytes,
                workgroup_size: self.linear_launch.workgroup(),
                chunk_capacity: self
                    .linear_launch
                    .chunk_capacity()
                    .and_then(|n| n.try_into().ok())
                    .ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!(
                            "empty POWER launch capacity unavailable"
                        ))
                    })?,
                uniform_allocations: 0,
                uniform_buffer_bytes: 0,
                dispatches: 0,
                queue_submissions: 0,
                pipeline_key: None,
            }));
        }
        if elements > u64::from(u32::MAX)
            || self.adapter_limits.max_storage_buffers_per_shader_stage < 2
        {
            return Ok(None);
        }
        if request
            .source
            .checked_byte_len()
            .is_none_or(|bytes| bytes > request.source_buffer_bytes)
        {
            return Ok(None);
        }
        if request.source_buffer_bytes > self.adapter_limits.max_buffer_size
            || request.source_buffer_bytes
                > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
        {
            return Ok(None);
        }
        let precision = match request.source.element_type {
            ProviderElementType::F32 => NumericPrecision::F32,
            ProviderElementType::F64 => NumericPrecision::F64,
            _ => return Ok(None),
        };
        let (workgroup_size, chunk_capacity, uniform_per_allocation, pipeline_key, warmups) =
            match request.operation {
                Operation::ScalarAdd { .. }
                | Operation::ScalarMul { .. }
                | Operation::ScalarMax { .. } => {
                    if !native_scalar_control(request.operation, precision) {
                        return Ok(None);
                    }
                    if !scalar_uniform_available(&self.adapter_limits) {
                        return Ok(None);
                    }
                    let scalar_op = match request.operation {
                        Operation::ScalarAdd { .. } => {
                            crate::backend::wgpu::types::ScalarOpCode::Add
                        }
                        Operation::ScalarMul { .. } => {
                            crate::backend::wgpu::types::ScalarOpCode::Mul
                        }
                        Operation::ScalarMax { .. } => {
                            crate::backend::wgpu::types::ScalarOpCode::Max
                        }
                        Operation::Sqrt
                        | Operation::Spacing
                        | Operation::Expm1
                        | Operation::UnaryMath(_)
                        | Operation::FloatingCast { .. }
                        | Operation::ExactSingleBasePower { .. } => {
                            unreachable!("matched scalar operation")
                        }
                    };
                    let pipeline = self
                        .real_scalar_pipeline_for_operation(scalar_op, precision)
                        .map_err(ProviderExecutionError::from)?;
                    pipeline.preflight().map_err(ProviderExecutionError::from)?;
                    let Some(launch) = pipeline.scalar_launch() else {
                        return Ok(None);
                    };
                    let Some(chunk_capacity) = launch.chunk_capacity() else {
                        return Ok(None);
                    };
                    (
                        launch.workgroup(),
                        chunk_capacity,
                        launch.uniform_requested_bytes(),
                        launch.key(),
                        0,
                    )
                }
                Operation::Sqrt
                | Operation::Spacing
                | Operation::Expm1
                | Operation::UnaryMath(_)
                | Operation::ExactSingleBasePower { .. } => {
                    if !scalar_uniform_available(&self.adapter_limits) {
                        return Ok(None);
                    }
                    if std::env::var("RUNMAT_DISABLE_UNARY").is_ok() {
                        return Ok(None);
                    }
                    let shader =
                        if let Operation::ExactSingleBasePower { control } = request.operation {
                            if precision != NumericPrecision::F32 {
                                return Ok(None);
                            }
                            let Some(operation) =
                                runmat_accelerate_api::ExactSingleBasePower::for_control(control)
                            else {
                                return Ok(None);
                            };
                            if request.source.layout == ProviderLayout::Strided
                                && request.source.shape.len() != 2
                            {
                                return Ok(None);
                            }
                            crate::backend::wgpu::shaders::exact_power::shader_for_layout(
                                operation,
                                request.source.layout == ProviderLayout::Strided,
                            )
                        } else {
                            real_unary_shader(
                                unary_opcode(request.operation)
                                    .expect("matched ordinary unary operation"),
                                precision,
                            )
                        };
                    let recipe = super::super::pipeline_recipe::PipelineRecipe::fusion(
                        &shader,
                        1,
                        super::super::pipeline_recipe::FusionPipelineLayout::Single,
                        self.linear_launch.workgroup(),
                    );
                    let Some(chunk_capacity) = self
                        .linear_launch
                        .chunk_capacity()
                        .and_then(|capacity| u64::try_from(capacity).ok())
                    else {
                        return Ok(None);
                    };
                    (
                        recipe.workgroup,
                        chunk_capacity,
                        if request.source.layout == ProviderLayout::Strided {
                            FusionUniformAbi::Broadcast
                                .requested_bytes(1)
                                .map_err(ProviderExecutionError::from)?
                        } else {
                            std::mem::size_of::<crate::backend::wgpu::params::FusionParams>() as u64
                        },
                        recipe.key(),
                        1,
                    )
                }
                Operation::FloatingCast { .. } => {
                    unreachable!("cast route was projected separately")
                }
            };
        if workgroup_size == 0
            || workgroup_size > self.adapter_limits.max_compute_workgroup_size_x
            || workgroup_size > self.adapter_limits.max_compute_invocations_per_workgroup
            || uniform_per_allocation
                > u64::from(self.adapter_limits.max_uniform_buffer_binding_size)
        {
            return Ok(None);
        }
        let Ok(dispatches) = u32::try_from(elements.div_ceil(chunk_capacity)) else {
            return Ok(None);
        };
        let uniform_allocations = if matches!(
            request.operation,
            Operation::Sqrt
                | Operation::Spacing
                | Operation::Expm1
                | Operation::UnaryMath(_)
                | Operation::ExactSingleBasePower { .. }
        ) {
            1
        } else {
            dispatches
        };
        let Some(uniform_buffer_bytes) =
            uniform_per_allocation.checked_mul(u64::from(uniform_allocations))
        else {
            return Ok(None);
        };
        let Some(queue_submissions) = dispatches.checked_add(warmups) else {
            return Ok(None);
        };
        Ok(Some(Proposal {
            output_route:
                runmat_accelerate_api::ProviderRealElementwiseOutputRoute::FreshAllocation,
            output_buffer_bytes,
            workgroup_size,
            chunk_capacity,
            uniform_allocations,
            uniform_buffer_bytes,
            dispatches,
            queue_submissions,
            pipeline_key: Some(pipeline_key),
        }))
    }

    pub(crate) async fn execute_real_elementwise_recipe_exec(
        &self,
        input: &GpuTensorHandle,
        recipe: &ProviderRealElementwiseRecipe,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        if !recipe.is_owned_by(self) {
            return Err(ProviderExecutionError::from(anyhow!(
                "real recipe belongs to another provider instance"
            )));
        }
        input
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let source = source_representation(input).ok_or_else(|| {
            ProviderExecutionError::from(anyhow!(
                "real recipe input has an unsupported floating-real view"
            ))
        })?;
        let empty_exact = matches!(
            recipe.request().operation,
            Operation::ExactSingleBasePower { .. }
        ) && source.checked_element_count() == Some(0);
        if empty_exact {
            self.allocations
                .validate_empty_view(input)
                .map_err(ProviderExecutionError::from)?;
        } else {
            self.real_floating_entry(input)
                .map_err(ProviderExecutionError::from)?;
        }
        if source != recipe.request().source
            || input.physical().requested_bytes() != recipe.request().source_buffer_bytes
            || self
                .propose_real_elementwise_exec(recipe.request())?
                .as_ref()
                != Some(recipe.physical())
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "ordinary real recipe does not match actual input"
            )));
        }
        let observed = RefCell::new(RealObserved::default());
        let result = match recipe.request().operation {
            Operation::ScalarAdd { control, .. }
            | Operation::ScalarMul { control }
            | Operation::ScalarMax { control } => {
                let scalar = match control {
                    ScalarLiteral::F32Bits(bits) => f64::from(f32::from_bits(bits)),
                    ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
                };
                let op = match recipe.request().operation {
                    Operation::ScalarAdd { .. } => crate::backend::wgpu::types::ScalarOpCode::Add,
                    Operation::ScalarMul { .. } => crate::backend::wgpu::types::ScalarOpCode::Mul,
                    Operation::ScalarMax { .. } => crate::backend::wgpu::types::ScalarOpCode::Max,
                    Operation::Sqrt
                    | Operation::Spacing
                    | Operation::Expm1
                    | Operation::UnaryMath(_)
                    | Operation::FloatingCast { .. }
                    | Operation::ExactSingleBasePower { .. } => {
                        unreachable!("matched scalar operation")
                    }
                };
                self.scalar_op_exec_recorded(op, input, scalar, Some(&observed), admission)
                    .map_err(|cause| {
                        let error = ProviderExecutionError::from(cause);
                        if observed.borrow().effect {
                            error.after_effect()
                        } else {
                            error
                        }
                    })
            }
            Operation::ExactSingleBasePower { control } => {
                let operation = runmat_accelerate_api::ExactSingleBasePower::for_control(control)
                    .ok_or_else(|| {
                    ProviderExecutionError::from(anyhow!("uncertified POWER control"))
                })?;
                if empty_exact {
                    let buffer = self.with_emission(|emission, _| {
                        self.create_storage_buffer_checked_with_element_size(
                            emission,
                            0,
                            "runmat-empty-exact-power",
                            BufferUsageClass::FusionOut,
                            4,
                        )
                    })??;
                    observed.borrow_mut().output(&buffer);
                    let allocated_bytes = buffer.size();
                    self.register_numeric_buffer_produced_admitted(
                        buffer,
                        super::super::backend_types::NumericBufferRegistration {
                            shape: input.shape().to_vec(),
                            len: 0,
                            physical_element_type: runmat_accelerate_api::NumericElementType::F32,
                            storage: GpuTensorStorage::Real,
                            allocated_bytes,
                            usage: BufferUsageClass::FusionOut,
                        },
                        false,
                        admission,
                    )
                    .map_err(|error| ProviderExecutionError::from(error).after_effect())
                } else {
                    let entry = self.real_floating_entry(input)?;
                    let broadcast = runmat_accelerate_api::handle_is_transposed(input);
                    let shader = crate::backend::wgpu::shaders::exact_power::shader_for_layout(
                        operation, broadcast,
                    );
                    self.fused_elementwise_typed_exec_recorded(
                        &shader,
                        std::slice::from_ref(input),
                        FusedElementwiseWork {
                            output_shape: &entry.shape,
                            len: entry.len,
                            invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                            uniform_abi: Some(if broadcast {
                                FusionUniformAbi::Broadcast
                            } else {
                                FusionUniformAbi::Simple
                            }),
                            representation: FusedElementwiseRepresentation {
                                input_precision: NumericPrecision::F32,
                                output_precision: NumericPrecision::F32,
                                output_storage: GpuTensorStorage::Real,
                                logical_output: false,
                            },
                        },
                        |_| Ok(()),
                        SingleFusionEvidence::Legacy {
                            observed: Some(&observed),
                            prepared: None,
                        },
                        admission,
                    )
                    .map_err(|cause| {
                        let error = ProviderExecutionError::from(cause);
                        if observed.borrow().effect {
                            error.after_effect()
                        } else {
                            error
                        }
                    })
                }
            }
            Operation::Sqrt | Operation::Spacing | Operation::Expm1 | Operation::UnaryMath(_) => {
                self.unary_op_exec_recorded(
                    unary_opcode(recipe.request().operation)
                        .expect("matched ordinary unary operation"),
                    input,
                    Some(&observed),
                    admission,
                )
                .map_err(|cause| {
                    let error = ProviderExecutionError::from(cause);
                    if observed.borrow().effect {
                        error.after_effect()
                    } else {
                        error
                    }
                })
            }
            Operation::FloatingCast { target } => {
                self.execute_real_floating_cast(input, target, &observed, admission)
                    .await
            }
        };
        let result = result?;
        if let Err(cause) = observed.borrow().require(recipe.physical()) {
            let error = ProviderExecutionError::from(cause).after_effect();
            if result.shares_allocation_with(input) {
                drop(result);
                return Err(error);
            }
            let cleanup = result.release().err().into_iter().collect::<Vec<_>>();
            return if cleanup.is_empty() {
                Err(error)
            } else {
                Err(ProviderExecutionError::from_source(
                    runmat_accelerate_api::ProviderExecutionCompletionError::new(
                        Some(error),
                        cleanup,
                    ),
                )
                .after_effect())
            };
        }
        Ok(ProviderExecutedRealElementwise {
            output: result,
            observed: observed.into_inner().resources,
        })
    }
}

#[cfg(test)]
#[path = "real_recipe_tests.rs"]
mod tests;
