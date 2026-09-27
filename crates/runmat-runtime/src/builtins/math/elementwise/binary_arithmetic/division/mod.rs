//! Shared element-wise division semantics.

mod host;
mod provider;

use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
use runmat_value::Value;

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

    fn provider_error(self, source: anyhow::Error) -> RuntimeError {
        let mut error = self.provider_contract_error(&source);
        error.source = Some(source.into_boxed_dyn_error());
        error
    }

    fn provider_contract_error(self, detail: impl std::fmt::Display) -> RuntimeError {
        let mut error = self.internal_error(detail);
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
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
        return super::provider_support::resident_output_from_sources(
            output,
            [source_handle(&numerator), source_handle(&denominator)]
                .into_iter()
                .flatten(),
        );
    }
    let numerator = gather_operand(context, numerator).await?;
    let denominator = gather_operand(context, denominator).await?;
    let host = execute_host(context, numerator, denominator)?;
    match source {
        Some(source) => restore_residency(context, &source, host),
        None => Ok(host),
    }
}

fn operand_is_scalar(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => tensor::is_scalar_tensor(value),
        Value::GpuTensor(value) => value.shape().iter().all(|&extent| extent == 1),
        Value::Num(_) | Value::Int(_) | Value::Complex(_, _) | Value::Bool(_) => true,
        _ => false,
    }
}

fn validate_integer_admission(
    context: DivisionContext,
    numerator: &Value,
    denominator: &Value,
) -> BuiltinResult<()> {
    crate::builtins::math::elementwise::integer_arithmetic::validate_integer_binary_inputs(
        numerator,
        denominator,
        operand_is_scalar(numerator),
        operand_is_scalar(denominator),
        context.identity.name,
    )
    .map_err(|detail| {
        build_runtime_error(detail)
            .with_builtin(context.identity.name)
            .build()
    })
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn division_scalar_admission_requires_one_element_for_host_and_resident() {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        for (shape, scalar) in [(vec![1, 1], true), (vec![0, 1], false), (vec![2, 1], false)] {
            let host = runmat_value::Tensor::new(vec![0.0; shape.iter().product()], shape.clone())
                .unwrap();
            let resident = owner
                .allocate_zeroed(
                    shape,
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap();
            assert_eq!(operand_is_scalar(&Value::Tensor(host)), scalar);
            assert_eq!(
                operand_is_scalar(&Value::GpuTensor(resident.handle)),
                scalar
            );
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
