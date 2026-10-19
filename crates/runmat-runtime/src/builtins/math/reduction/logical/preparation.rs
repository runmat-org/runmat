//! The option parser has already admitted controls before dense data preparation.
use super::{arguments::ReductionSpec, gpu, host, route, LogicalReductionConfig};
use crate::builtins::common::spec::ReductionNaN;
use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId};
use runmat_value::{Tensor, Value};

#[cfg(test)]
mod tests;

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    config: &'static LogicalReductionConfig,
    tensor: Tensor,
    input_count: usize,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<RootedValue> {
    if route::select(config.kind, nan_mode) == route::LogicalReductionRoute::HostExact {
        let reduced = host::reduce(config, Value::Tensor(tensor), spec, nan_mode).await?;
        return RootedValue::new(reduced).map_err(|cause| cause.into_runtime_error());
    }
    execute_invocation(
        runtime,
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: config.entry.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            // Signature 6 is the existing variadic option grammar, not another
            // parser or an artificial truncation of the original argument list.
            signature: 6,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![Value::Tensor(tensor)],
        |mut operands| async move {
            match operands.remove(0) {
                Value::GpuTensor(handle) => gpu::reduce(config, handle, spec, nan_mode).await,
                value => host::reduce(config, value, spec, nan_mode).await,
            }
            .and_then(|reduced| {
                RootedValue::new(reduced).map_err(|cause| cause.into_runtime_error())
            })
        },
    )
    .await
}
