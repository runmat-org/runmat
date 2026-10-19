mod specs;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod mapped_tests;
use super::operation::LogarithmOperation;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const OPERATION: LogarithmOperation = LogarithmOperation::Natural;

/// Whether mapped scalar calls may use the canonical whole-array operation.
/// This is a selection probe, not execution or completion evidence. The
/// canonical operation still performs its own admission when selected.
pub(crate) async fn mapped_batch_equivalent(input: &Value) -> BuiltinResult<bool> {
    super::provider::mapped_batch_equivalent(OPERATION, input).await
}

#[runtime_builtin(
    name = "log",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::logarithms::log"
)]
async fn log_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::LOG_DESCRIPTOR,
        &runmat_builtins::LOG_ERROR_TOO_MANY_OUTPUTS,
        "log",
    )?;
    super::engine::execute(runtime.as_ref(), OPERATION, value).await
}
