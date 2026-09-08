use crate::user_functions;
use runmat_value::Value;

use super::{CallableCallError, MappedCallable};

impl MappedCallable {
    pub(crate) async fn invoke(&self, arguments: &[Value]) -> Result<Value, CallableCallError> {
        match self {
            Self::Builtin { identity } => invoke_builtin(identity.name, arguments).await,
            Self::DynamicName { name } => invoke_dynamic(name, arguments).await,
            Self::ExternalName { name } => invoke_external(name, arguments).await,
            Self::Closure(closure) => invoke_closure(closure, arguments).await,
        }
    }
}

async fn invoke_builtin(
    name: &'static str,
    arguments: &[Value],
) -> Result<Value, CallableCallError> {
    invoke_resolved(
        runmat_types::CallableIdentity::Builtin(runmat_types::BuiltinId(name.into())),
        name,
        arguments,
    )
    .await
}

async fn invoke_dynamic(name: &str, arguments: &[Value]) -> Result<Value, CallableCallError> {
    invoke_resolved(
        runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(name.into())),
        name,
        arguments,
    )
    .await
}

async fn invoke_resolved(
    identity: runmat_types::CallableIdentity,
    name: &str,
    arguments: &[Value],
) -> Result<Value, CallableCallError> {
    let request = user_functions::CallableRequest::resolved(
        identity,
        runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
        arguments.to_vec(),
        1,
    );
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result.map_err(CallableCallError::Runtime);
    }
    crate::call_builtin_async(name, arguments)
        .await
        .map_err(CallableCallError::Runtime)
}

async fn invoke_external(name: &str, arguments: &[Value]) -> Result<Value, CallableCallError> {
    let identity = crate::external_callable_identity_for_name(name);
    let request = user_functions::CallableRequest::resolved(
        identity.clone(),
        runmat_types::CallableFallbackPolicy::ExternalBoundary,
        arguments.to_vec(),
        1,
    );
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result.map_err(CallableCallError::Runtime);
    }
    Err(CallableCallError::UndefinedExternal { identity })
}

async fn invoke_closure(
    closure: &runmat_value::Closure,
    arguments: &[Value],
) -> Result<Value, CallableCallError> {
    let mut merged = closure.captures.clone();
    merged.extend_from_slice(arguments);
    if let Some(function) = closure.bound_function {
        let request = user_functions::CallableRequest::semantic(function, merged.clone(), 1);
        if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
            return result.map_err(CallableCallError::Runtime);
        }
        return Err(CallableCallError::SemanticUnavailable {
            function_name: closure.function_name.clone(),
            function: function.to_string(),
        });
    }
    if let Some(function) =
        user_functions::resolve_semantic_function_by_name(&closure.function_name)
    {
        let request = user_functions::CallableRequest::semantic(function, merged.clone(), 1);
        if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
            return result.map_err(CallableCallError::Runtime);
        }
    }
    crate::call_builtin_async(&closure.function_name, &merged)
        .await
        .map_err(CallableCallError::Runtime)
}
