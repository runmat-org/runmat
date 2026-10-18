use super::{execute, Function, Spec};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(super) const SPEC: Spec = Spec {
    name: "acosd",
    function: Function::Acos,
    invalid_input: &runmat_builtins::ACOSD_ERROR_INVALID_INPUT,
    internal: &runmat_builtins::ACOSD_ERROR_INTERNAL,
    integer: &runmat_builtins::ACOSD_INTEGER_INPUT_EXTENSION,
    logical: &runmat_builtins::ACOSD_LOGICAL_INPUT_EXTENSION,
    character: &runmat_builtins::ACOSD_CHARACTER_INPUT_EXTENSION,
    gpu_complex: &runmat_builtins::ACOSD_GPU_REAL_COMPLEX_EXTENSION,
};

#[runtime_builtin(
    name = "acosd",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::inverse_degree"
)]
async fn acosd_builtin(value: Value) -> crate::BuiltinResult<Value> {
    execute(&SPEC, value).await
}
