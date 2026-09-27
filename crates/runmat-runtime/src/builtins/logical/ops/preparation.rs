//! Dense logical conversion after the existing NaN/storage admission.
use super::*;
use crate::context::preparation::{execute_current_invocation, BuiltinPreparationAdmission};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId, LOGICAL_CATALOG_ENTRY};

#[cfg(test)]
mod tests;

pub(super) async fn execute(tensor: Tensor) -> BuiltinResult<Value> {
    execute_current_invocation(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: LOGICAL_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![Value::Tensor(tensor)],
        |mut operands| async move {
            match operands.remove(0) {
                Value::Tensor(tensor) => {
                    logical_buffer_to_host(LogicalBuffer::from_real_tensor(&tensor))
                }
                Value::GpuTensor(handle) => logical_from_gpu(handle).await,
                _ => Err(logical_error_with_message(
                    "logical: preparation changed the admitted data kind",
                    &LOGICAL_ERROR_INTERNAL,
                )),
            }
        },
    )
    .await
}
