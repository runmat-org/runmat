use runmat_accelerate_api::{AccelProvider, ProviderPrecision};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};

pub(super) fn select_provider(
    orientation: SolveOrientation,
    lhs: &Value,
    rhs: &Value,
) -> BuiltinResult<Option<&'static dyn AccelProvider>> {
    let source = gpu_helpers::select_resident_output_source(
        [lhs, rhs].into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        orientation.name(),
    )
    .map_err(|error| errors::map_control_flow(orientation, error))?;
    let Some(source) = source else {
        return Ok(None);
    };
    let provider = gpu_helpers::exact_provider_for_handle(&source).ok_or_else(|| {
        errors::internal(
            orientation,
            format!(
                "{}: resident input has no registered owner",
                orientation.name()
            ),
        )
    })?;
    Ok(Some(provider))
}

pub(super) fn selected_precision(lhs: &Value, rhs: &Value) -> Option<ProviderPrecision> {
    let lhs = value_precision(lhs)?;
    let rhs = value_precision(rhs)?;
    if lhs == ProviderPrecision::F32 || rhs == ProviderPrecision::F32 {
        Some(ProviderPrecision::F32)
    } else {
        Some(ProviderPrecision::F64)
    }
}

fn value_precision(value: &Value) -> Option<ProviderPrecision> {
    match value {
        Value::Num(_) | Value::Bool(_) | Value::LogicalArray(_) => Some(ProviderPrecision::F64),
        Value::Tensor(tensor) => Some(
            if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
                ProviderPrecision::F32
            } else {
                ProviderPrecision::F64
            },
        ),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_precision(handle),
        _ => None,
    }
}

pub(super) fn is_resident(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(_))
}

pub(super) fn is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}
