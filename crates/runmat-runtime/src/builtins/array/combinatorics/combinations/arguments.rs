use runmat_builtins::COMBINATIONS_RESIDENT_INPUT_EXTENSION;
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) async fn gather(first: Value, rest: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut inputs = Vec::with_capacity(rest.len() + 1);
    inputs.push(first);
    inputs.extend(rest);
    let mut gathers_resident = false;
    for input in &inputs {
        crate::dispatcher::try_visit_gathered_gpu_handles(input, &mut |_| gathers_resident = true)
            .map_err(super::error::internal)?;
    }
    if gathers_resident {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COMBINATIONS_RESIDENT_INPUT_EXTENSION,
            "combinations",
        )?;
    }

    crate::dispatcher::gather_values_async(&inputs)
        .await
        .map_err(super::error::internal)
}
