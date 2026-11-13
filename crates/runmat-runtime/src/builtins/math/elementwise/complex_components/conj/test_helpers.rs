use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use super::{host, provider};
use crate::BuiltinResult;

pub(super) fn conj_real(value: Value) -> BuiltinResult<Value> {
    host::execute(value)
}

pub(super) async fn conj_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    provider::execute(handle).await
}
