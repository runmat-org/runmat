//! Backend implementation of the runtime's already-admitted operand port.
use super::{
    preparation::{ArgumentOperation, BuiltinPreparationContext},
    *,
};
use runmat_builtins::{
    BuiltinPreparationOperation as Operation, BuiltinPreparationProviderRequirement,
    BuiltinSinkPreparation,
};
use runmat_runtime::context::{
    preparation::{
        AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
    },
    RuntimeAccelerationService, RuntimeServiceFuture,
};
use std::rc::Rc;
mod host;
mod native_region;
pub(super) use host::prepare_async;

/// Install the standard adapter only where the embedding host has not supplied one.
/// The existing placement session and every other service remain unchanged.
pub fn install_preparation_service(runtime: RuntimeContext) -> RuntimeContext {
    if runtime.service_ports().acceleration().is_some() {
        return runtime;
    }
    let ports = runtime
        .service_ports()
        .clone()
        .with_acceleration(Rc::new(AdmittedAcceleration));
    runtime.with_service_ports(ports)
}

pub struct AdmittedAcceleration;

impl RuntimeAccelerationService for AdmittedAcceleration {
    fn numeric_region_capability(
        &self,
    ) -> Option<runmat_runtime::numeric_region::service::NumericRegionCapability> {
        (auto_enabled() && runmat_accelerate_api::provider().is_some()).then_some(
            runmat_runtime::numeric_region::service::NumericRegionCapability::RealFloating,
        )
    }

    fn execute_numeric_region(
        &self,
        context: RuntimeContext,
        request: runmat_runtime::numeric_region::service::NativeNumericRegionRequest,
    ) -> Result<
        runmat_runtime::execution::RuntimeContinuationFuture<
            'static,
            runmat_runtime::numeric_region::service::NativeNumericRegionOutcome,
        >,
        RuntimeError,
    > {
        let admission = context.clone();
        runmat_runtime::execution::admit_native_continuation_with(&admission, || {
            native_region::execute(context, request)
        })
    }
    fn prepare_legacy(
        &self,
        context: RuntimeContext,
        name: String,
        args: Vec<Value>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, RuntimeError>> {
        Box::pin(context.clone().scope(async move {
            super::prepare(&name, &args, Some(&context))
                .await
                .map(PreparedBuiltinCall::into_args)
                .map_err(preparation_error)
        }))
    }
    fn prepare_admitted(
        &self,
        context: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        Box::pin(context.clone().scope(async move {
            if request.form().operation == Operation::RuntimeOwned {
                return request
                    .into_operands()
                    .with_unbounded_values(PreparedBuiltinOperands::passthrough)?;
            }
            if let Operation::Sink(mode) = request.form().operation {
                fusion_residency::clear_inputs(
                    request.operands(),
                    request.admission().binding.builtin.name,
                )?;
                return match mode {
                    BuiltinSinkPreparation::Barrier => request
                        .into_operands()
                        .with_unbounded_values(PreparedBuiltinOperands::passthrough)?,
                    BuiltinSinkPreparation::Gather => {
                        let operands = gather_args(request.operands())
                            .await
                            .map_err(preparation_error)?;
                        PreparedBuiltinOperands::passthrough(operands)
                    }
                };
            }
            let prepared = if !auto_enabled() {
                return host::prepare_unobserved_operands(&context, request)
                    .map_err(|error| *error);
            } else if let Some(auto) = global().await {
                prepare_async(&auto, &context, request).await
            } else {
                return host::prepare_unobserved_operands(&context, request)
                    .map_err(|error| *error);
            }
            .map_err(|error| *error)?;
            let (operands, guard) = prepared.into_execution();
            PreparedBuiltinOperands::new(operands, Some(Box::new(guard)))
        }))
    }
}

#[cfg(test)]
pub(super) fn prepare_with(
    auto: &NativeAutoOffload,
    context: &RuntimeContext,
    request: AdmittedBuiltinPreparation,
) -> PreparationResult<PreparedBuiltinCall> {
    // The request owns the original managed graph. Keep an independent
    // root-only lease across private preparation's raw argument handoff,
    // then transfer it to the execution guard until the fallible prepared
    // carrier has synchronously published its own operand roots.
    let input_root =
        runmat_runtime::execution::RootedValueLease::for_values(request.operands().iter())
            .map_err(|error| Box::new(error.into_runtime_error()))?;
    let prepared = prepare_with_inner(auto, context, request, None, None)?;
    Ok(prepared.with_request_input_root(input_root))
}

fn prepare_with_inner(
    auto: &NativeAutoOffload,
    context: &RuntimeContext,
    mut request: AdmittedBuiltinPreparation,
    host: Option<(
        &crate::fusion_exec::host_readback::CpuHostResources,
        runmat_execution::Digest,
    )>,
    compound: Option<&runmat_runtime::context::preparation::AdmittedBuiltinCompound>,
) -> PreparationResult<PreparedBuiltinCall> {
    if request.form().operation == Operation::RuntimeOwned {
        return request
            .into_operands()
            .with_unbounded_values(PreparedBuiltinCall::unobserved)
            .map_err(Box::new);
    }
    let host_instruction = match request.form().operation {
        Operation::ElementwiseBinary => {
            Some(runmat_runtime::context::VmFusionHostInstruction::ElementwiseBinary)
        }
        Operation::Reduction(runmat_builtins::BuiltinPreparationReduction::Min) => {
            Some(runmat_runtime::context::VmFusionHostInstruction::Reduction)
        }
        _ => None,
    };
    if host_instruction.is_some_and(|instruction| {
        context.vm_fusion_host_route().is_some_and(|route| {
            route.allows_host_preparation(context, instruction)
                && request.operands().iter().all(|value| {
                    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32)
                })
        })
    }) {
        return request
            .into_operands()
            .with_unbounded_values(PreparedBuiltinCall::unobserved)
            .map_err(Box::new);
    }
    if !auto.enabled
        || (request.form().provider_requirement == BuiltinPreparationProviderRequirement::F64
            && auto.provider.precision() != ProviderPrecision::F64)
    {
        return host::prepare_passthrough(context, request);
    }
    if !super::owner_admission::matches_selected_owner(auto.provider, request.operands())? {
        // The domain already owns the correct direct/cross-provider behavior.
        // Do not upload other operands or observe a candidate whose retained
        // provider cannot execute the actual borrowed operands.
        return host::prepare_passthrough(context, request);
    }
    let operation = match request.form().operation {
        Operation::Unary => ArgumentOperation::Unary(UnaryOp::Generic),
        Operation::FloatingCast(class) => ArgumentOperation::FloatingCast(match class {
            runmat_types::NumericClass::Single => ProviderPrecision::F32,
            runmat_types::NumericClass::Double => ProviderPrecision::F64,
            _ => {
                return Err(Box::new(preparation_error(
                    anyhow::anyhow!("invalid admitted floating cast target").into(),
                )))
            }
        }),
        Operation::RealUnary(_) => ArgumentOperation::RealUnary(request.numerical_domain()),
        Operation::RoundDigits => ArgumentOperation::RoundDigits(request.numerical_domain()),
        Operation::FloatingRemainder(_) => {
            ArgumentOperation::FloatingRemainder(request.numerical_domain())
        }
        Operation::Transpose => ArgumentOperation::Unary(UnaryOp::Transpose),
        Operation::ElementwiseBinary => ArgumentOperation::Binary(BinaryOp::Elementwise),
        Operation::ElementwiseNary { data_arity } => ArgumentOperation::ElementwiseNary {
            data_arity,
            requested_outputs: request
                .admission()
                .requested_outputs
                .map(u32::try_from)
                .transpose()
                .map_err(|_| {
                    Box::new(preparation_error(
                        anyhow::anyhow!("requested output count exceeds workload representation")
                            .into(),
                    ))
                })?,
        },
        Operation::RealPower => ArgumentOperation::RealPower(request.numerical_domain()),
        Operation::MatrixMultiply => ArgumentOperation::Binary(BinaryOp::MatMul),
        Operation::Reduction(runmat_builtins::BuiltinPreparationReduction::Sum) => {
            let controls = request.sum_reduction().ok_or_else(|| {
                Box::new(preparation_error(
                    anyhow::anyhow!("admitted sum is missing its resolved controls").into(),
                ))
            })?;
            if controls.axes.is_empty() && controls.input_class == controls.publication_class {
                // The domain has proved an identity route with no class change.
                // Do not upload a host array only to return it unchanged.
                return request
                    .into_operands()
                    .with_unbounded_values(PreparedBuiltinCall::unobserved)
                    .map_err(Box::new);
            }
            let axes = controls
                .axes
                .iter()
                .copied()
                .map(u64::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    Box::new(preparation_error(
                        anyhow::anyhow!("admitted sum axis exceeds workload representation").into(),
                    ))
                })?;
            ArgumentOperation::AdmittedSum {
                route: runmat_accelerate_api::ProviderSumWorkload {
                    invocation: controls.invocation,
                    axes,
                    missing: controls.missing,
                    output: controls.output,
                    input_class: controls.input_class,
                    input_domain: controls.input_domain,
                    accumulation_class: controls.accumulation_class,
                    publication_class: controls.publication_class,
                },
                requested_outputs: request
                    .admission()
                    .requested_outputs
                    .map(u32::try_from)
                    .transpose()
                    .map_err(|_| {
                        Box::new(preparation_error(
                            anyhow::anyhow!(
                                "requested output count exceeds workload representation"
                            )
                            .into(),
                        ))
                    })?,
            }
        }
        Operation::Reduction(_) => ArgumentOperation::Reduction,
        Operation::Sink(_) | Operation::RuntimeOwned => {
            unreachable!("runtime-owned and sink forms bypass automatic promotion")
        }
    };
    let completion = request.take_prepared_completion_right();
    let prepared = auto
        .prepare_operation_with_context(
            request.admission().binding.builtin.name,
            request.operands(),
            operation,
            BuiltinPreparationContext {
                runtime: Some(context),
                completion,
                host,
                compound,
            },
        )
        .map_err(preparation_error)?;
    Ok(prepared)
}

fn preparation_error(error: ProviderExecutionError) -> RuntimeError {
    runmat_runtime::build_runtime_error(format!("prepare builtin args: {error}"))
        .with_identifier("RunMat:AccelerationOperationFailed")
        .with_source(error)
        .build()
}

impl BuiltinPreparationObservation for BuiltinCallGuard {
    fn take_host_numerical_output(
        &mut self,
    ) -> Option<runmat_runtime::context::preparation::PreparedHostNumericalOutput> {
        self.host_numerical_output.take()
    }
    fn take_compound(
        &mut self,
    ) -> Option<runmat_runtime::context::preparation::AdmittedBuiltinCompound> {
        self.compound.take()
    }
    fn gpu_gather_retry_policy(&self) -> runmat_runtime::GpuGatherRetry {
        if self.readbacks.is_some()
            || self
                .uploads
                .as_ref()
                .is_some_and(ownership::OwnedUploads::has_uploads)
        {
            runmat_runtime::GpuGatherRetry::Never
        } else {
            runmat_runtime::GpuGatherRetry::Legacy
        }
    }
    fn provider_recorder(
        &self,
    ) -> Option<runmat_accelerate_api::observation::ProviderOperationRecorder> {
        self.observation
            .as_ref()
            .and_then(BuiltinCallObservation::recorder)
    }
    fn finish(self: Box<Self>, result: Result<&Value, &RuntimeError>) -> Result<(), RuntimeError> {
        (*self).finish(result).map_err(|error| *error)
    }
    fn abort(self: Box<Self>) {
        drop(self);
    }
}
