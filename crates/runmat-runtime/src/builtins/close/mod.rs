//! Canonical `close` builtin dispatcher.
//!
//! This module owns the single runtime registration for `close` and routes
//! requests to plotting or networking close handlers.

mod contract;
mod dispatch;

#[cfg(test)]
mod tests;

pub use contract::{
    CLOSE_DESCRIPTOR, CLOSE_EXTENSIONS, CLOSE_INTEGER_CAPABILITIES, FUSION_SPEC, GPU_SPEC,
};
pub use dispatch::close_builtin;

#[cfg(target_arch = "wasm32")]
pub(crate) use contract::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};
#[cfg(target_arch = "wasm32")]
pub(crate) use dispatch::__runmat_wasm_register_builtin_close_builtin;
