use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::{error, BUILTIN_NAME};

pub(super) fn for_handle(
    handle: &GpuTensorHandle,
) -> BuiltinResult<&'static dyn runmat_accelerate_api::AccelProvider> {
    gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| error::internal("no active GPU provider for rescale input"))
}

pub(super) fn output(
    tensor: Tensor,
    restore: bool,
    source: Option<&GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let value = into_value(tensor);
    if !restore {
        return Ok(value);
    }
    let source = source.ok_or_else(|| error::internal("missing provider source for output"))?;
    gpu_helpers::restore_class_preserving_value(source, value, BUILTIN_NAME)
}

pub(super) fn empty(
    shape: Vec<usize>,
    dtype: NumericDType,
    restore: bool,
    source: Option<&GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let tensor = Tensor::new_with_dtype(Vec::new(), shape, dtype).map_err(error::internal)?;
    output(tensor, restore, source)
}

fn into_value(tensor: Tensor) -> Value {
    if tensor.numeric_dtype() == NumericDType::F64 && tensor.len() == 1 {
        Value::Num(tensor.as_f64_slice().expect("double storage")[0])
    } else {
        Value::Tensor(tensor)
    }
}
