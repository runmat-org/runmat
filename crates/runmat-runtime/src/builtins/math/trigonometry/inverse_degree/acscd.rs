use super::{execute, Function, Spec};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(super) const SPEC: Spec = Spec {
    name: "acscd",
    function: Function::Acsc,
    invalid_input: &runmat_builtins::ACSCD_ERROR_INVALID_INPUT,
    internal: &runmat_builtins::ACSCD_ERROR_INTERNAL,
    integer: &runmat_builtins::ACSCD_INTEGER_INPUT_EXTENSION,
    logical: &runmat_builtins::ACSCD_LOGICAL_INPUT_EXTENSION,
    character: &runmat_builtins::ACSCD_CHARACTER_INPUT_EXTENSION,
    gpu_complex: &runmat_builtins::ACSCD_GPU_REAL_COMPLEX_EXTENSION,
};

#[runtime_builtin(
    name = "acscd",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::inverse_degree"
)]
async fn acscd_builtin(value: Value) -> crate::BuiltinResult<Value> {
    execute(&SPEC, value).await
}
