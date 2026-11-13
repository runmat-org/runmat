use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "codistributor1d",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn codistributor_1d_builtin(
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::parallel::codistributor::one_dimensional(arguments.view())
}

#[runtime_builtin(
    name = "codistributor2dbc",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn codistributor_2dbc_builtin(
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::parallel::codistributor::two_dimensional(arguments.view())
}

#[runtime_builtin(
    name = "codistributor",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn codistributor_builtin(
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::parallel::codistributor::factory(arguments.view())
}

#[runtime_builtin(
    name = "isComplete",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn is_complete_builtin(
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if arguments.len() != 1 {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:parallel:Codistributor",
            "isComplete requires exactly one codistributor object",
        ));
    }
    crate::parallel::codistributor::is_complete(&arguments[0]).map(Value::Bool)
}

#[runtime_builtin(
    name = "iscodistributed",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::codistributor"
)]
async fn is_codistributed_builtin(
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if arguments.len() != 1 {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:parallel:Codistributed",
            "iscodistributed requires exactly one value",
        ));
    }
    Ok(Value::Bool(matches!(arguments[0], Value::Distributed(_))))
}
