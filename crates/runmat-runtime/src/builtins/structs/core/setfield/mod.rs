//! MATLAB-compatible setfield builtin with struct-array and object support.
//!
//! Direct field replacement preserves resident values. Indexed mutation
//! gathers only the selected resident target and stores the updated value on
//! the host.

mod arguments;
mod assignment;
mod entry;
mod errors;
mod handle;
mod object;
mod selector;
mod spec;
mod struct_array;

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
use entry::setfield_builtin;
#[cfg(test)]
pub(crate) use errors::is_undefined_function;

#[cfg(target_arch = "wasm32")]
pub(crate) use entry::__runmat_wasm_register_builtin_setfield_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};
