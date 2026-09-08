mod conversion;
mod error;
mod spec;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[runtime_builtin(
    name = "cellstr",
    builtin_path = "crate::builtins::cells::core::cellstr"
)]
fn cellstr_builtin(value: Value) -> crate::BuiltinResult<Value> {
    conversion::convert(value)
}
