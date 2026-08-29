use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "codistributor1d",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn codistributor_1d_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::parallel::codistributor::one_dimensional(&arguments)
}

#[runtime_builtin(
    name = "codistributor2dbc",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn codistributor_2dbc_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::parallel::codistributor::two_dimensional(&arguments)
}
