//! MATLAB-compatible setfield builtin with struct-array and object support.
//!
//! Direct field replacement preserves resident values. Indexed mutation
//! gathers only the selected resident target and stores the updated value on
//! the host.

mod arguments;
mod assignment;
mod errors;
mod handle;
mod object;
mod selector;
mod spec;
mod struct_array;

#[cfg(test)]
pub(crate) mod tests;

use runmat_builtins::SETFIELD_ERROR_NOT_ENOUGH_INPUTS;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(test)]
pub(crate) use errors::is_undefined_function;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[runtime_builtin(
    name = "setfield",
    builtin_path = "crate::builtins::structs::core::setfield"
)]
async fn setfield_builtin(base: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() < 2 {
        return Err(errors::with_message(
            SETFIELD_ERROR_NOT_ENOUGH_INPUTS.message,
            &SETFIELD_ERROR_NOT_ENOUGH_INPUTS,
        ));
    }
    let mut arguments = rest;
    let value = arguments.pop().expect("input count is validated above");
    let parsed = arguments::parse(arguments)?;
    assignment::assign_value(base, parsed.leading_index, parsed.fields, value).await
}
