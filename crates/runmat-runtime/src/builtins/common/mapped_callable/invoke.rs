use crate::execution::RootedValue;
use crate::user_functions;
use runmat_value::Value;

use super::{CallableCallError, MappedCallable};

impl MappedCallable {
    pub(crate) fn invoke_with_outputs<'a>(
        &'a self,
        arguments: &'a [Value],
        requested_outputs: usize,
    ) -> impl std::future::Future<Output = Result<RootedValue, CallableCallError>> + 'a {
        self.invoke_with_outputs_in_context(None, arguments, requested_outputs)
    }

    pub(crate) fn invoke_with_outputs_in_context<'a>(
        &'a self,
        context: Option<&'a crate::context::RuntimeContext>,
        arguments: &'a [Value],
        requested_outputs: usize,
    ) -> impl std::future::Future<Output = Result<RootedValue, CallableCallError>> + 'a {
        let closure_roots = match self {
            Self::Closure(closure) => Some(
                crate::execution::RootedValueLease::for_traced(closure)
                    .map_err(|error| CallableCallError::Runtime(error.into_runtime_error())),
            ),
            _ => None,
        };
        let input_roots = crate::execution::RootedValueLease::for_values(arguments.iter())
            .map_err(|error| CallableCallError::Runtime(error.into_runtime_error()));
        async move {
            let _input_roots = input_roots?;
            let _closure_roots = closure_roots.transpose()?;
            match self {
                Self::Builtin { identity } => {
                    invoke_builtin(context, identity.name, arguments, requested_outputs).await
                }
                Self::DynamicName { name } => {
                    invoke_dynamic(context, name, arguments, requested_outputs).await
                }
                Self::ExternalName { name } => {
                    invoke_external(context, name, arguments, requested_outputs).await
                }
                Self::Closure(closure) => {
                    invoke_closure(context, closure, arguments, requested_outputs).await
                }
            }
        }
    }
}

async fn invoke_builtin(
    context: Option<&crate::context::RuntimeContext>,
    name: &'static str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<RootedValue, CallableCallError> {
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
) -> Result<RootedValue, CallableCallError> {
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
) -> Result<RootedValue, CallableCallError> {
    let request = user_functions::CallableRequest::resolved(
        identity,
        runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
        arguments.to_vec(),
        requested_outputs,
    )
    .with_selected_runtime(context);
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result
            .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
            .map_err(CallableCallError::Runtime);
    }
    invoke_builtin_dispatch(context, name, arguments, requested_outputs).await
}

async fn invoke_external(
    context: Option<&crate::context::RuntimeContext>,
    name: &str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<RootedValue, CallableCallError> {
    let identity = crate::external_callable_identity_for_name(name);
    let request = user_functions::CallableRequest::resolved(
        identity.clone(),
        runmat_types::CallableFallbackPolicy::ExternalBoundary,
        arguments.to_vec(),
        requested_outputs,
    )
    .with_selected_runtime(context);
    if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
        return result
            .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
            .map_err(CallableCallError::Runtime);
    }
    Err(CallableCallError::UndefinedExternal { identity })
}

async fn invoke_closure(
    context: Option<&crate::context::RuntimeContext>,
    closure: &runmat_value::Closure,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<RootedValue, CallableCallError> {
    let mut merged = closure.captures.clone();
    merged.extend_from_slice(arguments);
    if let Some(binding) = closure.binding.clone() {
        let function = binding.function;
        let request =
            user_functions::CallableRequest::bound(binding, merged.clone(), requested_outputs)
                .with_lexical_environment(closure.lexical_environment.clone())
                .with_selected_runtime(context);
        if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
            return result
                .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
                .map_err(CallableCallError::Runtime);
        }
        return Err(CallableCallError::SemanticUnavailable {
            function_name: closure.function_name.clone(),
            function: function.to_string(),
        });
    }
    if let Some(result) = user_functions::try_call_semantic_function_by_name_in_context(
        context,
        &closure.function_name,
        &merged,
        requested_outputs,
    )
    .await
    {
        return result
            .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
            .map_err(CallableCallError::Runtime);
    }
    invoke_builtin_dispatch(context, &closure.function_name, &merged, requested_outputs).await
}

async fn invoke_builtin_dispatch(
    context: Option<&crate::context::RuntimeContext>,
    name: &str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<RootedValue, CallableCallError> {
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
