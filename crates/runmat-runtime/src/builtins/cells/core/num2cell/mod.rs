mod conversion;
mod dimension;
mod error;
mod input;
mod plan;
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
    name = "num2cell",
    builtin_path = "crate::builtins::cells::core::num2cell"
)]
async fn num2cell_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(error::invalid_input("expected A or A,dim"));
    }
    let dims = rest
        .first()
        .map(dimension::parse)
        .transpose()?
        .unwrap_or_default();
    let value = input::gather_top_level(value).await?;
    conversion::convert(value, &dims)
}
