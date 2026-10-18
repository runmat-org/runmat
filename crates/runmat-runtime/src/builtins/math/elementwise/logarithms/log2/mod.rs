mod dissection;
mod specs;
#[cfg(test)]
mod tests;

use super::operation::LogarithmOperation;
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const OPERATION: LogarithmOperation = LogarithmOperation::Binary;

#[runtime_builtin(
    name = "log2",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::logarithms::log2"
)]
async fn log2_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    match crate::output_count::current_output_count().unwrap_or(1) {
        0 => {
            super::engine::execute(runtime.as_ref(), OPERATION, value).await?;
            RootedValue::new(Value::output_list(Vec::new()))
                .map_err(|cause| cause.into_runtime_error())
        }
        1 => super::engine::execute(runtime.as_ref(), OPERATION, value).await,
        2 => {
            let (fraction, exponent) = dissection::execute(value).await?;
            RootedValue::new(Value::output_list(vec![
                fraction.value().clone(),
                exponent.value().clone(),
            ]))
            .map_err(|cause| cause.into_runtime_error())
        }
        count => Err(super::errors::with_detail(
            OPERATION,
            &runmat_builtins::LOG2_ERROR_TOO_MANY_OUTPUTS,
            format!("requested {count}"),
        )),
    }
}
