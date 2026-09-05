mod dissection;
mod specs;
#[cfg(test)]
mod tests;

use super::operation::LogarithmOperation;
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
    builtin_path = "crate::builtins::math::elementwise::logarithms::log2"
)]
async fn log2_builtin(value: Value) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count().unwrap_or(1) {
        0 => {
            super::engine::execute(OPERATION, value).await?;
            Ok(Value::OutputList(Vec::new()))
        }
        1 => super::engine::execute(OPERATION, value).await,
        2 => {
            let (fraction, exponent) = dissection::execute(value).await?;
            Ok(Value::OutputList(vec![fraction, exponent]))
        }
        count => Err(super::errors::with_detail(
            OPERATION,
            &runmat_builtins::LOG2_ERROR_TOO_MANY_OUTPUTS,
            format!("requested {count}"),
        )),
    }
}
