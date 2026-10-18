mod normalization;
mod characteristic;
#[cfg(target_arch = "wasm32")]
pub(crate) use characteristic::__runmat_wasm_register_builtin_poly_builtin;
pub mod polyder;
pub mod polyfit;
pub mod polyint;
pub mod polyval;
pub mod roots;
