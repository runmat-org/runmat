use super::{execute, Function, Spec};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(super) const SPEC: Spec = Spec {
    name: "atand",
    function: Function::Atan,
    invalid_input: &runmat_builtins::ATAND_ERROR_INVALID_INPUT,
    internal: &runmat_builtins::ATAND_ERROR_INTERNAL,
    integer: &runmat_builtins::ATAND_INTEGER_INPUT_EXTENSION,
    logical: &runmat_builtins::ATAND_LOGICAL_INPUT_EXTENSION,
    character: &runmat_builtins::ATAND_CHARACTER_INPUT_EXTENSION,
    gpu_complex: &runmat_builtins::ATAND_GPU_REAL_COMPLEX_EXTENSION,
};

#[runtime_builtin(
    name = "atand",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::inverse_degree"
)]
async fn atand_builtin(value: Value) -> crate::BuiltinResult<Value> {
    execute(&SPEC, value).await
}
