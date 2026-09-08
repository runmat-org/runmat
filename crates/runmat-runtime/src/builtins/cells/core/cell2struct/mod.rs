mod assembly;
mod dimension;
mod error;
mod fields;
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
    name = "cell2struct",
    builtin_path = "crate::builtins::cells::core::cell2struct"
)]
fn cell2struct_builtin(
    cells: Value,
    fields: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(error::invalid("expected C, fields, and optional dim"));
    }
    let dimension = dimension::parse(rest.first())?;
    let Value::Cell(cells) = cells else {
        return Err(error::invalid("C must be a cell array"));
    };
    let fields = fields::parse(&fields)?;
    assembly::build(cells, fields, dimension)
}
