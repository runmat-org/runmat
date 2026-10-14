//! The scalar/array binary form owns preparation; tabular mapping stays outside.
use super::*;
use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId};

#[cfg(test)]
#[path = "preparation_tests.rs"]
mod tests;

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    y: Value,
    x: Value,
) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values([&y, &x].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    let floating = admission::pair(&y, &x)?;
    let protected: Vec<_> = [&y, &x]
        .into_iter()
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle.clone())
            } else {
                None
            }
        })
        .collect();
    if !floating {
        let output = atan2_admitted(y, x, &protected).await?;
        return RootedValue::new(output).map_err(|cause| cause.into_runtime_error());
    }
    execute_invocation(
        runtime,
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: runmat_builtins::ATAN2_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 2,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![y, x],
        |mut operands| async move {
            let x = operands.pop().expect("two admitted coordinate operands");
            let y = operands.pop().expect("two admitted coordinate operands");
            if !admission::pair(&y, &x)? {
                return Err(atan2_error_with_detail(
                    &ATAN2_ERROR_INTERNAL,
                    "preparation changed admitted floating representation",
                ));
            }
            let output = atan2_admitted(y, x, &protected).await?;
            RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}
