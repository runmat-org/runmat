//! Dense logical conversion after the existing NaN/storage admission.
use super::*;
use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId, LOGICAL_CATALOG_ENTRY};

#[cfg(test)]
mod tests;

pub(super) async fn execute(
    context: Option<&crate::context::RuntimeContext>,
    tensor: Tensor,
) -> BuiltinResult<crate::execution::RootedValue> {
    execute_invocation(
        context,
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
            let value = match operands.remove(0) {
                Value::Tensor(tensor) => {
                    let value = logical_buffer_to_host(LogicalBuffer::from_real_tensor(&tensor))?;
                    if let Some(context) = context {
                        if let Some(route) = crate::builtins::math::host_route::selected(
                            context,
                            crate::context::VmFusionHostInstruction::ElementwiseCall,
                        ) {
                            route.confirm_host_branch(
                                context,
                                crate::context::VmFusionHostInstruction::ElementwiseCall,
                            );
                        }
                    }
                    Ok(value)
                }
                Value::GpuTensor(handle) => logical_from_gpu(handle).await,
                _ => Err(logical_error_with_message(
                    "logical: preparation changed the admitted data kind",
                    &LOGICAL_ERROR_INTERNAL,
                )),
            }?;
            crate::execution::RootedValue::new(value).map_err(|error| error.into_runtime_error())
        },
    )
    .await
}
