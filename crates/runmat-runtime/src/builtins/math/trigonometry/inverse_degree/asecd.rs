use super::{execute, Function, Spec};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(super) const SPEC: Spec = Spec {
    name: "asecd",
    function: Function::Asec,
    invalid_input: &runmat_builtins::ASECD_ERROR_INVALID_INPUT,
    internal: &runmat_builtins::ASECD_ERROR_INTERNAL,
    integer: &runmat_builtins::ASECD_INTEGER_INPUT_EXTENSION,
    logical: &runmat_builtins::ASECD_LOGICAL_INPUT_EXTENSION,
    character: &runmat_builtins::ASECD_CHARACTER_INPUT_EXTENSION,
    gpu_complex: &runmat_builtins::ASECD_GPU_REAL_COMPLEX_EXTENSION,
};

#[runtime_builtin(
    name = "asecd",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::inverse_degree"
)]
async fn asecd_builtin(value: Value) -> crate::BuiltinResult<Value> {
    execute(&SPEC, value).await
}
