//! Shared element-wise division semantics.

mod host;
mod provider;

use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
use runmat_value::{NumericDType, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Clone, Copy)]
pub(super) struct DivisionContext {
    pub identity: BuiltinCatalogIdentity,
    pub invalid_input: &'static BuiltinErrorDescriptor,
    pub size_mismatch: &'static BuiltinErrorDescriptor,
    pub internal: &'static BuiltinErrorDescriptor,
}

impl DivisionContext {
    fn error(self, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
        let mut builder = build_runtime_error(descriptor.message).with_builtin(self.identity.name);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    fn error_with_detail(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl AsRef<str>,
    ) -> RuntimeError {
        let mut builder =
            build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
                .with_builtin(self.identity.name);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    fn internal_error(self, detail: impl std::fmt::Display) -> RuntimeError {
        build_runtime_error(format!("{}: {detail}", self.identity.name))
            .with_builtin(self.identity.name)
            .build()
    }
}

pub(super) fn execute_host(
    context: DivisionContext,
    numerator: Value,
    denominator: Value,
) -> BuiltinResult<Value> {
    host::execute(context, numerator, denominator)
}

pub(super) async fn execute(
    context: DivisionContext,
    numerator: Value,
    denominator: Value,
) -> BuiltinResult<Value> {
    validate_integer_admission(context, &numerator, &denominator)?;
    let source = preferred_source(&numerator, &denominator);
    let resident = match (&numerator, &denominator) {
        (Value::GpuTensor(numerator), Value::GpuTensor(denominator)) => {
            provider::try_pair(context, numerator, denominator).await?
        }
        (Value::GpuTensor(numerator), denominator) => {
            provider::try_resident_numerator(context, numerator, denominator).await?
        }
        (numerator, Value::GpuTensor(denominator)) => {
            provider::try_resident_denominator(context, numerator, denominator).await?
        }
        _ => None,
    };
    if let Some(output) = resident {
        return Ok(super::provider_support::resident_output_from_sources(
            output,
            [source_handle(&numerator), source_handle(&denominator)]
                .into_iter()
                .flatten(),
        ));
    }
    let numerator = gather_operand(context, numerator).await?;
    let denominator = gather_operand(context, denominator).await?;
    let host = execute_host(context, numerator, denominator)?;
    match source {
        Some(source) => restore_residency(context, &source, host),
        None => Ok(host),
    }
}

fn integer_class(value: &Value) -> Option<runmat_types::IntegerClass> {
    match value {
        Value::Int(value) => Some(value.integer_class()),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(runmat_value::IntegerStorage::integer_class),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_class(handle),
        _ => None,
    }
}

fn is_scalar_double(value: &Value) -> bool {
    match value {
        Value::Num(_) => true,
        Value::Tensor(value) => {
            tensor::is_scalar_tensor(value) && value.numeric_dtype() == NumericDType::F64
        }
        Value::GpuTensor(value) => {
            value.shape.iter().copied().product::<usize>() <= 1
                && runmat_accelerate_api::handle_integer_type(value).is_none()
                && !runmat_accelerate_api::handle_is_logical(value)
                && runmat_accelerate_api::handle_storage(value)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::handle_precision(value)
                    == Some(runmat_accelerate_api::ProviderPrecision::F64)
        }
        _ => false,
    }
}

fn validate_integer_admission(
    context: DivisionContext,
    numerator: &Value,
    denominator: &Value,
) -> BuiltinResult<()> {
    match (integer_class(numerator), integer_class(denominator)) {
        (None, None) => Ok(()),
        (Some(numerator), Some(denominator)) if numerator == denominator => Ok(()),
        (Some(_), Some(_)) => {
            Err(context.internal_error("integer operands must have the same integer class"))
        }
        (Some(_), None) if is_scalar_double(denominator) => Ok(()),
        (None, Some(_)) if is_scalar_double(numerator) => Ok(()),
        _ => {
            Err(context
                .internal_error("integer arrays can only be combined with scalar double values"))
        }
    }
}

fn source_handle(value: &Value) -> Option<&runmat_accelerate_api::GpuTensorHandle> {
    match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    }
}

fn preferred_source(
    numerator: &Value,
    denominator: &Value,
) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    let handles = [source_handle(numerator), source_handle(denominator)];
    handles
        .iter()
        .flatten()
        .find(|handle| runmat_accelerate_api::handle_is_explicit(handle))
        .or_else(|| handles.iter().flatten().next())
        .map(|handle| (*handle).clone())
}

async fn gather_operand(context: DivisionContext, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, context.identity.name)),
        _ => Ok(value),
    }
}

fn restore_residency(
    context: DivisionContext,
    source: &runmat_accelerate_api::GpuTensorHandle,
    mut host: Value,
) -> BuiltinResult<Value> {
    if matches!(host, Value::Num(_) | Value::Int(_) | Value::Bool(_)) {
        host = Value::Tensor(
            tensor::value_into_tensor_for(context.identity.name, host)
                .map_err(|error| context.internal_error(error))?,
        );
    }
    let restored =
        gpu_helpers::restore_class_preserving_value(source, host, context.identity.name)?;
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(context.internal_error("provider cannot preserve explicit gpuArray output"));
    }
    Ok(restored)
}
