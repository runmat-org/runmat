use super::{execute, Function, Spec};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(super) const SPEC: Spec = Spec {
    name: "acotd",
    function: Function::Acot,
    invalid_input: &runmat_builtins::ACOTD_ERROR_INVALID_INPUT,
    internal: &runmat_builtins::ACOTD_ERROR_INTERNAL,
    integer: &runmat_builtins::ACOTD_INTEGER_INPUT_EXTENSION,
    logical: &runmat_builtins::ACOTD_LOGICAL_INPUT_EXTENSION,
    character: &runmat_builtins::ACOTD_CHARACTER_INPUT_EXTENSION,
    gpu_complex: &runmat_builtins::ACOTD_GPU_REAL_COMPLEX_EXTENSION,
};

#[runtime_builtin(
    name = "acotd",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::inverse_degree"
)]
async fn acotd_builtin(value: Value) -> crate::BuiltinResult<Value> {
    execute(&SPEC, value).await
}
