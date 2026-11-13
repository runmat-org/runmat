//! MATLAB-compatible `getfield` builtin with struct-array and object support.

mod arguments;
mod errors;
mod execution;
mod object;
mod spec;
mod special;

#[cfg(test)]
pub(crate) mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(crate) use execution::get_member_value_in_context;

#[cfg(test)]
pub(crate) use errors::is_undefined_function;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[runtime_builtin(
    name = "getfield",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::structs::core::getfield"
)]
async fn getfield_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    base: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let work = async {
        crate::output_count::validate_fixed_outputs(
            &runmat_builtins::GETFIELD_DESCRIPTOR,
            &runmat_builtins::GETFIELD_ERROR_TOO_MANY_OUTPUTS,
            "getfield",
        )?;
        execution::getfield_in_scope(runtime.as_ref(), base, rest, true).await
    };
    match runtime.as_ref() {
        Some(runtime) => runtime.scope(work).await,
        None => work.await,
    }
}

pub(crate) async fn getfield_internal_retained_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    execution::getfield(runtime, base, rest, false).await
}
