mod conversion;
mod dimension;
mod error;
mod input;
mod managed_entry;
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::cells::core::num2cell"
)]
async fn num2cell_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::NUM2CELL_DESCRIPTOR,
        &runmat_builtins::NUM2CELL_ERROR_TOO_MANY_OUTPUTS,
        "num2cell",
    )?;

    if rest.len() > 1 {
        return Err(error::invalid_input("expected A or A,dim"));
    }
    if let Some(context) = managed_entry::selected(runtime.as_ref(), &value)? {
        if let Some(dimensions) = rest.first() {
            dimension::visit(dimensions, |_| Ok(()))?;
        }
        let Value::StringArray(array) = value else {
            unreachable!()
        };
        return conversion::convert_managed_strings(array, rest.first(), context);
    }
    let dims = rest
        .first()
        .map(dimension::parse)
        .transpose()?
        .unwrap_or_default();
    let value = input::gather_top_level(value).await?;
    conversion::convert(value, &dims, runtime.as_ref())
}
