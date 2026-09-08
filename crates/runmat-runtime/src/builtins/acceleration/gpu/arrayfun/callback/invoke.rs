use crate::user_functions;
use runmat_builtins::{ARRAYFUN_ERROR_CALLBACK_FAILED, ARRAYFUN_ERROR_UNDEFINED_FUNCTION};
use runmat_value::Value;

use super::super::error::{arrayfun_error_with_detail, arrayfun_error_with_message};
use super::Callable;

impl Callable {
    pub(in crate::builtins::acceleration::gpu::arrayfun) async fn call(
        &self,
        arguments: &[Value],
    ) -> crate::BuiltinResult<Value> {
        match self {
            Self::Builtin { identity } => builtin(identity.name, arguments).await,
            Self::DynamicName { name } => dynamic(name, arguments).await,
            Self::ExternalName { name } => external(name, arguments).await,
            Self::Closure(closure) => invoke_closure(closure, arguments).await,
        }
    }
}

async fn builtin(name: &'static str, arguments: &[Value]) -> crate::BuiltinResult<Value> {
    let request = user_functions::CallableRequest::resolved(
        runmat_types::CallableIdentity::Builtin(runmat_types::BuiltinId(name.into())),
        runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
        arguments.to_vec(),
        1,
    );
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result;
    }
    crate::call_builtin_async(name, arguments).await
}

async fn dynamic(name: &str, arguments: &[Value]) -> crate::BuiltinResult<Value> {
    let request = user_functions::CallableRequest::resolved(
        runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(name.into())),
        runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
        arguments.to_vec(),
        1,
    );
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result;
    }
    crate::call_builtin_async(name, arguments).await
}

async fn external(name: &str, arguments: &[Value]) -> crate::BuiltinResult<Value> {
    let identity = crate::external_callable_identity_for_name(name);
    let request = user_functions::CallableRequest::resolved(
        identity.clone(),
        runmat_types::CallableFallbackPolicy::ExternalBoundary,
        arguments.to_vec(),
        1,
    );
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result;
    }
    Err(arrayfun_error_with_message(
        format!("Undefined function for callable identity {identity:?}"),
        &ARRAYFUN_ERROR_UNDEFINED_FUNCTION,
    ))
}

async fn invoke_closure(
    closure: &runmat_value::Closure,
    arguments: &[Value],
) -> crate::BuiltinResult<Value> {
    let mut merged = closure.captures.clone();
    merged.extend_from_slice(arguments);
    if let Some(function) = closure.bound_function {
        let request = user_functions::CallableRequest::semantic(function, merged.clone(), 1);
        if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
            return result;
        }
        return Err(arrayfun_error_with_detail(
            &ARRAYFUN_ERROR_CALLBACK_FAILED,
            format!(
                "semantic closure '{}' ({function}) is unavailable",
                closure.function_name
            ),
        ));
    }
    if let Some(function) =
        user_functions::resolve_semantic_function_by_name(&closure.function_name)
    {
        let request = user_functions::CallableRequest::semantic(function, merged.clone(), 1);
        if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
            return result;
        }
    }
    crate::call_builtin_async(&closure.function_name, &merged).await
}
