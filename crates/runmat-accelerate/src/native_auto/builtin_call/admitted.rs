//! Backend implementation of the runtime's already-admitted operand port.
use super::{preparation::ArgumentOperation, *};
use runmat_builtins::{
    BuiltinOperandPreparation, BuiltinPreparationOperation as Operation,
    BuiltinPreparationProviderRequirement, BuiltinSinkPreparation,
};
use runmat_runtime::context::{
    preparation::{
        AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
    },
    RuntimeAccelerationService, RuntimeServiceFuture,
};
use std::rc::Rc;

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
    fn prepare_legacy(
        &self,
        context: RuntimeContext,
        name: String,
        args: Vec<Value>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, RuntimeError>> {
        Box::pin(async move {
            super::prepare(&name, &args, Some(&context))
                .await
                .map(PreparedBuiltinCall::into_args)
                .map_err(preparation_error)
        })
    }
    fn supports_operation(&self, operation: &str) -> bool {
        runmat_builtins::builtin_catalog_primary_entry_by_name(operation).is_some_and(|entry| {
            matches!(
                entry.placement.preparation,
                BuiltinOperandPreparation::Forms(_)
            )
        })
    }

    fn prepare_admitted(
        &self,
        context: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        Box::pin(async move {
            if let Operation::Sink(mode) = request.form().operation {
                for value in request.operands() {
                    runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
                        fusion_residency::clear(handle);
                    })?;
                }
                let operands = match mode {
                    BuiltinSinkPreparation::Barrier => request.into_operands(),
                    BuiltinSinkPreparation::Gather => gather_args(request.operands())
                        .await
                        .map_err(|error| RuntimeError::new(error.to_string()))?,
                };
                return Ok(PreparedBuiltinOperands::passthrough(operands));
            }
            if !auto_enabled() {
                return Ok(PreparedBuiltinOperands::passthrough(
                    request.into_operands(),
                ));
            }
            let Some(auto) = global().await else {
                return Ok(PreparedBuiltinOperands::passthrough(
                    request.into_operands(),
                ));
            };
            let prepared = prepare_with(auto, &context, request)?;
            let (operands, guard) = prepared.into_execution();
            Ok(PreparedBuiltinOperands::new(
                operands,
                Some(Box::new(guard)),
            ))
        })
    }
}

pub(super) fn prepare_with(
    auto: &NativeAutoOffload,
    context: &RuntimeContext,
    request: AdmittedBuiltinPreparation,
) -> Result<PreparedBuiltinCall, RuntimeError> {
    if !auto.enabled
        || (request.form().provider_requirement == BuiltinPreparationProviderRequirement::F64
            && auto.provider.precision() != ProviderPrecision::F64)
    {
        return Ok(PreparedBuiltinCall::unobserved(request.into_operands()));
    }
    let operation = match request.form().operation {
        Operation::Unary => ArgumentOperation::Unary(UnaryOp::Generic),
        Operation::Transpose => ArgumentOperation::Unary(UnaryOp::Transpose),
        Operation::ElementwiseBinary => ArgumentOperation::Binary(BinaryOp::Elementwise),
        Operation::MatrixMultiply => ArgumentOperation::Binary(BinaryOp::MatMul),
        Operation::Reduction(_) => ArgumentOperation::Reduction,
        Operation::Sink(_) => unreachable!("sinks handled independently of automatic promotion"),
    };
    let prepared = auto
        .prepare_operation(
            request.admission().binding.builtin.name,
            request.operands(),
            Some(context),
            operation,
        )
        .map_err(preparation_error)?;
    Ok(prepared)
}

fn preparation_error(error: impl std::fmt::Display) -> RuntimeError {
    runmat_runtime::build_runtime_error(format!("prepare builtin args: {error}"))
        .with_identifier("RunMat:AccelerationOperationFailed")
        .build()
}

impl BuiltinPreparationObservation for BuiltinCallGuard {
    fn begin_execution(&mut self) {
        BuiltinCallGuard::begin_execution(self);
    }
    fn finish(self: Box<Self>, result: &Result<Value, RuntimeError>) -> Result<(), RuntimeError> {
        (*self).finish(result)
    }
    fn abort(self: Box<Self>) {
        drop(self);
    }
}
