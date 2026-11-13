use super::{execute, Function, Spec};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(super) const SPEC: Spec = Spec {
    name: "asind",
    function: Function::Asin,
    invalid_input: &runmat_builtins::ASIND_ERROR_INVALID_INPUT,
    internal: &runmat_builtins::ASIND_ERROR_INTERNAL,
    integer: &runmat_builtins::ASIND_INTEGER_INPUT_EXTENSION,
    logical: &runmat_builtins::ASIND_LOGICAL_INPUT_EXTENSION,
    character: &runmat_builtins::ASIND_CHARACTER_INPUT_EXTENSION,
    gpu_complex: &runmat_builtins::ASIND_GPU_REAL_COMPLEX_EXTENSION,
};

#[runtime_builtin(
    name = "asind",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::inverse_degree"
)]
async fn asind_builtin(value: Value) -> crate::BuiltinResult<Value> {
    execute(&SPEC, value).await
}
