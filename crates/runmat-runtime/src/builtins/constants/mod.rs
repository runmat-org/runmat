mod definitions;

#[cfg(target_arch = "wasm32")]
pub(crate) use definitions::{
    __runmat_wasm_register_const_Inf, __runmat_wasm_register_const_NaN,
    __runmat_wasm_register_const_eps, __runmat_wasm_register_const_false,
    __runmat_wasm_register_const_i, __runmat_wasm_register_const_inf,
    __runmat_wasm_register_const_j, __runmat_wasm_register_const_nan,
    __runmat_wasm_register_const_pi, __runmat_wasm_register_const_sqrt2,
    __runmat_wasm_register_const_true,
};

#[cfg(test)]
mod tests;
