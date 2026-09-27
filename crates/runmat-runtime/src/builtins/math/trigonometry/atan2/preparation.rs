//! The scalar/array binary form owns preparation; tabular mapping stays outside.
use super::*;
use crate::context::preparation::{execute_current_invocation, BuiltinPreparationAdmission};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId};

#[cfg(test)]
#[path = "preparation_tests.rs"]
mod tests;

pub(super) async fn execute(y: Value, x: Value) -> BuiltinResult<Value> {
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
        return atan2_admitted(y, x, &protected).await;
    }
    execute_current_invocation(
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
            atan2_admitted(y, x, &protected).await
        },
    )
    .await
}
