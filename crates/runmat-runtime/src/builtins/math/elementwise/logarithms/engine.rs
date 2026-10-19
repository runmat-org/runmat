use super::operation::LogarithmOperation;
use crate::builtins::math::symbolic::symbolic_function;
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    operation: LogarithmOperation,
    value: Value,
) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    if let Some(function) = operation.symbolic_function() {
        if let Some(symbolic) = symbolic_function(&value, function) {
            return RootedValue::new(symbolic).map_err(|cause| cause.into_runtime_error());
        }
    }
    match value {
        Value::Object(object)
            if operation.accepts_tabular()
                && crate::builtins::table::is_tabular_object(&object) =>
        {
            super::table::evaluate(runtime, operation, object).await
        }
        value => execute_non_table(runtime, operation, value).await,
    }
}
pub(super) async fn execute_non_table(
    runtime: Option<&RuntimeContext>,
    operation: LogarithmOperation,
    value: Value,
) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    super::extensions::ensure(operation, &value).await?;
    crate::builtins::common::real_unary_preparation::execute(
        runtime,
        operation.catalog_entry(),
        operation.instruction(),
        operation.fallback_publication(),
        crate::builtins::common::real_unary_preparation::RealUnaryAdapter {
            internal_error: operation.internal_error(),
            before_host: |input: &GpuTensorHandle, classification| {
                if classification == runmat_types::RealUnaryClassification::Outside {
                    super::provider::ensure_explicit_complex_extension(operation, input)?;
                }
                Ok(())
            },
            publish_host: |provider: &'static dyn AccelProvider,
                           input: &GpuTensorHandle,
                           result| {
                super::provider::publish_host_result(operation, provider, input, result)
            },
            publish_real: |provider: &'static dyn AccelProvider,
                           input: &GpuTensorHandle,
                           output| {
                super::provider::contract::validate(operation, provider, input, output)
            },
        },
        value,
        true,
        |value: f32| Ok(operation.complex_f32(value, 0.0)),
        |value: f64| Ok(operation.complex_f64(value, 0.0)),
        |value| async move {
            let output = match value {
                Value::GpuTensor(handle) => super::provider::evaluate(operation, handle).await,
                value => super::host::evaluate(operation, value),
            }?;
            RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}
