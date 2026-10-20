use runmat_builtins::COMBINATIONS_RESIDENT_INPUT_EXTENSION;
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) async fn gather(
    runtime: Option<&crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let first = crate::execution::RootedValue::in_optional_context(first, runtime)?;
    let inputs =
        crate::builtins::common::mapped_callable::prepend_rooted_argument(&rest, first, runtime)
            .await?;
    let mut gathers_resident = false;
    for input in &inputs {
        crate::dispatcher::try_visit_gathered_gpu_handles(input, &mut |_| gathers_resident = true)
            .map_err(super::error::gathered)?;
    }
    if gathers_resident {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COMBINATIONS_RESIDENT_INPUT_EXTENSION,
            "combinations",
        )?;
    }

    if !gathers_resident {
        return Ok(inputs);
    }
    inputs
        .try_map_preserving_roots_async(inputs.len(), runtime, async |index| {
            crate::gather_if_needed_async(&inputs[index])
                .await
                .map_err(super::error::gathered)
        })
        .await
}
