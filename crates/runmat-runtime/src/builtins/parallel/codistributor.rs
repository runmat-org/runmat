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

#[runtime_builtin(
    name = "codistributor",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn codistributor_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::parallel::codistributor::factory(&arguments)
}

#[runtime_builtin(
    name = "isComplete",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn is_complete_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    let [value] = arguments.try_into().map_err(|_| {
        crate::runtime_error::semantic_error(
            "RunMat:parallel:Codistributor",
            "isComplete requires exactly one codistributor object",
        )
    })?;
    crate::parallel::codistributor::is_complete(&value).map(Value::Bool)
}

#[runtime_builtin(
    name = "iscodistributed",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn is_codistributed_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    let [value] = arguments.try_into().map_err(|_| {
        crate::runtime_error::semantic_error(
            "RunMat:parallel:Codistributed",
            "iscodistributed requires exactly one value",
        )
    })?;
    Ok(Value::Bool(matches!(value, Value::Distributed(_))))
}
