//! Sorting and set-related array builtins.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, IntegerElementType};
use runmat_builtins::{
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerScalarDoubleRule,
    ComplexTensor, IntegerStorage, Tensor, Value,
};

use crate::builtins::common::{gpu_helpers, tensor};

pub mod argsort;
pub(super) mod float_order;
pub(super) mod integer_order;
pub mod intersect;
pub mod ismember;
pub mod ismembertol;
pub mod issorted;
pub mod issortedrows;
pub mod setdiff;
pub mod setxor;
pub mod sort;
pub mod sortrows;
pub(crate) mod type_resolvers;
pub mod union;
pub mod unique;

pub(super) const BINARY_SET_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A accepts every real integer class; unlike nondouble numeric inputs must use the same class, while double is the documented cross-class exception.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "B accepts every real integer class; unlike nondouble numeric inputs must use the same class, while double is the documented cross-class exception.",
    },
];

pub(super) fn is_unsupported_set_gpu_integer(handle: &GpuTensorHandle) -> bool {
    matches!(
        runmat_accelerate_api::handle_integer_type(handle),
        Some(IntegerElementType::I64 | IntegerElementType::U64)
    )
}

pub(super) fn set_output_provider(a: &Value, b: &Value) -> Option<&'static dyn AccelProvider> {
    output_provider(a).or_else(|| output_provider(b))
}

pub(super) fn output_provider(value: &Value) -> Option<&'static dyn AccelProvider> {
    let Value::GpuTensor(handle) = value else {
        return None;
    };
    runmat_accelerate_api::provider_for_handle(handle).or_else(runmat_accelerate_api::provider)
}

pub(super) fn restore_set_outputs(
    provider: Option<&'static dyn AccelProvider>,
    builtin: &str,
    outputs: Vec<Value>,
    internal_error: fn(String) -> crate::RuntimeError,
) -> crate::BuiltinResult<Vec<Value>> {
    let Some(provider) = provider else {
        return Ok(outputs);
    };
    outputs
        .into_iter()
        .map(|value| upload_set_output(provider, builtin, value, internal_error))
        .collect()
}

fn upload_set_output(
    provider: &'static dyn AccelProvider,
    builtin: &str,
    value: Value,
    internal_error: fn(String) -> crate::RuntimeError,
) -> crate::BuiltinResult<Value> {
    let upload_tensor = |tensor: Tensor, logical: bool| -> crate::BuiltinResult<Value> {
        let handle = gpu_helpers::upload_tensor(provider, &tensor)
            .map_err(|error| internal_error(format!("{builtin}: GPU upload failed: {error}")))?;
        Ok(if logical {
            gpu_helpers::logical_gpu_value(handle)
        } else {
            gpu_helpers::resident_gpu_value(handle)
        })
    };
    match value {
        Value::Tensor(tensor) => upload_tensor(tensor, false),
        Value::Num(number) => upload_tensor(
            Tensor::new(vec![number], vec![1, 1])
                .map_err(|error| internal_error(format!("{builtin}: {error}")))?,
            false,
        ),
        Value::Int(integer) => upload_tensor(
            Tensor::new_integer(IntegerStorage::from_scalar(integer), vec![1, 1])
                .map_err(|error| internal_error(format!("{builtin}: {error}")))?,
            false,
        ),
        Value::Bool(logical) => upload_tensor(
            Tensor::new(vec![f64::from(u8::from(logical))], vec![1, 1])
                .map_err(|error| internal_error(format!("{builtin}: {error}")))?,
            true,
        ),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|error| internal_error(format!("{builtin}: {error}")))?;
            upload_tensor(tensor, true)
        }
        Value::Complex(real, imaginary) => {
            let tensor = ComplexTensor::new(vec![(real, imaginary)], vec![1, 1])
                .map_err(|error| internal_error(format!("{builtin}: {error}")))?;
            let handle =
                gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|error| {
                    internal_error(format!("{builtin}: GPU upload failed: {}", error.message()))
                })?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        Value::ComplexTensor(tensor) => {
            let handle =
                gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|error| {
                    internal_error(format!("{builtin}: GPU upload failed: {}", error.message()))
                })?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        other => Err(internal_error(format!(
            "{builtin}: cannot restore resident output {other:?}"
        ))),
    }
}
