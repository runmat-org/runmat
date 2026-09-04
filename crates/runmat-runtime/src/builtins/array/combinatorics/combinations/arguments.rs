use runmat_builtins::COMBINATIONS_RESIDENT_INPUT_EXTENSION;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

pub(super) async fn gather(first: Value, rest: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut inputs = Vec::with_capacity(rest.len() + 1);
    inputs.push(first);
    inputs.extend(rest);
    if inputs
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COMBINATIONS_RESIDENT_INPUT_EXTENSION,
            "combinations",
        )?;
    }

    let mut gathered = Vec::with_capacity(inputs.len());
    for input in inputs {
        gathered.push(
            gpu_helpers::gather_value_async(&input)
                .await
                .map_err(super::error::internal)?,
        );
    }
    Ok(gathered)
}
