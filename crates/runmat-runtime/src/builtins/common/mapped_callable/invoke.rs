use crate::user_functions;
use runmat_value::Value;

use super::{CallableCallError, MappedCallable};

impl MappedCallable {
    pub(crate) async fn invoke_with_outputs(
        &self,
        arguments: &[Value],
        requested_outputs: usize,
    ) -> Result<Value, CallableCallError> {
        self.invoke_with_outputs_in_context(None, arguments, requested_outputs)
            .await
    }

    pub(crate) async fn invoke_with_outputs_in_context(
        &self,
        context: Option<&crate::context::RuntimeContext>,
        arguments: &[Value],
        requested_outputs: usize,
    ) -> Result<Value, CallableCallError> {
        match self {
            Self::Builtin { identity } => {
                invoke_builtin(context, identity.name, arguments, requested_outputs).await
            }
            Self::DynamicName { name } => {
                invoke_dynamic(context, name, arguments, requested_outputs).await
            }
            Self::ExternalName { name } => {
                invoke_external(name, arguments, requested_outputs).await
            }
            Self::Closure(closure) => {
                invoke_closure(context, closure, arguments, requested_outputs).await
            }
        }
    }
}

async fn invoke_builtin(
    context: Option<&crate::context::RuntimeContext>,
    name: &'static str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<Value, CallableCallError> {
    invoke_resolved(
        context,
        runmat_types::CallableIdentity::Builtin(runmat_types::BuiltinId(name.into())),
        name,
        arguments,
        requested_outputs,
    )
    .await
}

async fn invoke_dynamic(
    context: Option<&crate::context::RuntimeContext>,
    name: &str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<Value, CallableCallError> {
    invoke_resolved(
        context,
        runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(name.into())),
        name,
        arguments,
        requested_outputs,
    )
    .await
}

async fn invoke_resolved(
    context: Option<&crate::context::RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    name: &str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<Value, CallableCallError> {
    let request = user_functions::CallableRequest::resolved(
        identity,
        runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
        arguments.to_vec(),
        requested_outputs,
    );
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result
            .map(crate::call::arguments::project_legacy_builtin_value_abi)
            .map_err(CallableCallError::Runtime);
    }
    invoke_builtin_dispatch(context, name, arguments, requested_outputs).await
}

async fn invoke_external(
    name: &str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<Value, CallableCallError> {
    let identity = crate::external_callable_identity_for_name(name);
    let request = user_functions::CallableRequest::resolved(
        identity.clone(),
        runmat_types::CallableFallbackPolicy::ExternalBoundary,
        arguments.to_vec(),
        requested_outputs,
    );
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result
            .map(crate::call::arguments::project_legacy_builtin_value_abi)
            .map_err(CallableCallError::Runtime);
    }
    Err(CallableCallError::UndefinedExternal { identity })
}

async fn invoke_closure(
    context: Option<&crate::context::RuntimeContext>,
    closure: &runmat_value::Closure,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<Value, CallableCallError> {
    let mut merged = closure.captures.clone();
    merged.extend_from_slice(arguments);
    if let Some(binding) = closure.binding.clone() {
        let function = binding.function;
        let request =
            user_functions::CallableRequest::bound(binding, merged.clone(), requested_outputs);
        if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
            return result
                .map(crate::call::arguments::project_legacy_builtin_value_abi)
                .map_err(CallableCallError::Runtime);
        }
        return Err(CallableCallError::SemanticUnavailable {
            function_name: closure.function_name.clone(),
            function: function.to_string(),
        });
    }
    if let Some(function) =
        user_functions::resolve_semantic_function_by_name(&closure.function_name)
    {
        let request =
            user_functions::CallableRequest::semantic(function, merged.clone(), requested_outputs);
        if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
            return result
                .map(crate::call::arguments::project_legacy_builtin_value_abi)
                .map_err(CallableCallError::Runtime);
        }
    }
    invoke_builtin_dispatch(context, &closure.function_name, &merged, requested_outputs).await
}

async fn invoke_builtin_dispatch(
    context: Option<&crate::context::RuntimeContext>,
    name: &str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<Value, CallableCallError> {
    let result = match context {
        Some(context) => {
            crate::call_builtin_async_with_outputs_in_context(
                context,
                name,
                arguments,
                requested_outputs,
            )
            .await
        }
        None => crate::call_builtin_async_with_outputs(name, arguments, requested_outputs).await,
    };
    result.map_err(CallableCallError::Runtime)
}
