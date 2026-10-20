//! Mapped callbacks enter the canonical retained semantic/descriptor pipeline.
use super::{CallableCallError, MappedCallable};
use crate::execution::RootedValue;
use runmat_value::Value;

impl MappedCallable {
    #[cfg(test)]
    pub(crate) fn invoke_with_outputs<'a>(
        &'a self,
        arguments: &'a [Value],
        requested_outputs: usize,
    ) -> impl std::future::Future<Output = Result<RootedValue, CallableCallError>> + 'a {
        self.invoke_with_outputs_in_context(None, arguments, requested_outputs)
    }

    /// Genuine borrowed embedding ingress; canonical mapping calls retain owners.
    pub(crate) fn invoke_with_outputs_in_context<'a>(
        &'a self,
        context: Option<&'a crate::context::RuntimeContext>,
        arguments: &'a [Value],
        requested_outputs: usize,
    ) -> impl std::future::Future<Output = Result<RootedValue, CallableCallError>> + 'a {
        // Callable roots must precede borrowed argument admission, which can
        // invoke an external allocator/collector before the future is polled.
        let prepared = self
            .prepare_closure_owner(context)
            .and_then(|closure_owner| {
                crate::call::BuiltinArguments::from_borrowed(arguments, context)
                    .map(|arguments| (arguments, closure_owner))
            });
        self.invoke_prepared(context, prepared, requested_outputs)
    }

    pub(crate) fn invoke_retained<'a>(
        &'a self,
        context: Option<&'a crate::context::RuntimeContext>,
        arguments: crate::call::BuiltinArguments,
        requested_outputs: usize,
    ) -> impl std::future::Future<Output = Result<RootedValue, CallableCallError>> + 'a {
        let prepared = self
            .prepare_closure_owner(context)
            .map(|closure_owner| (arguments, closure_owner));
        self.invoke_prepared(context, prepared, requested_outputs)
    }

    fn prepare_closure_owner(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<Option<crate::call::RetainedCallableTarget>, crate::RuntimeError> {
        match self {
            Self::Closure(closure) => closure.retain_for_call(context).map(Some),
            _ => Ok(None),
        }
    }

    async fn invoke_prepared<'a>(
        &'a self,
        context: Option<&'a crate::context::RuntimeContext>,
        prepared: Result<
            (
                crate::call::BuiltinArguments,
                Option<crate::call::RetainedCallableTarget>,
            ),
            crate::RuntimeError,
        >,
        requested_outputs: usize,
    ) -> Result<RootedValue, CallableCallError> {
        let (arguments, closure_owner) = prepared.map_err(CallableCallError::Runtime)?;
        let result = match self {
            Self::Closure(closure) => {
                let owner = closure_owner.expect("selected closure owner");
                crate::call::descriptor::invoke_selected_retained_closure(
                    context,
                    closure,
                    owner,
                    arguments,
                    requested_outputs,
                )
                .await
                .map_err(CallableCallError::Runtime)?
                .ok_or_else(|| CallableCallError::SemanticUnavailable {
                    function_name: closure.as_closure().function_name.clone(),
                    function: closure
                        .as_closure()
                        .binding
                        .as_ref()
                        .expect("only a bound closure can decline")
                        .function
                        .to_string(),
                })?
            }
            Self::ExternalName { name } => {
                let identity = crate::external_callable_identity_for_name(name);
                let request = crate::user_functions::CallableRequest::rooted_resolved(
                    identity.clone(),
                    runmat_types::CallableFallbackPolicy::ExternalBoundary,
                    arguments
                        .into_supplied_function_arguments(context)
                        .map_err(CallableCallError::Runtime)?,
                    requested_outputs,
                )
                .with_selected_runtime(context);
                match crate::user_functions::try_call_semantic_descriptor_owned(request).await {
                    std::ops::ControlFlow::Break(result) => {
                        result.map_err(CallableCallError::Runtime)?
                    }
                    std::ops::ControlFlow::Continue(_) => {
                        return Err(CallableCallError::UndefinedExternal { identity })
                    }
                }
            }
            Self::Builtin { identity } => crate::dispatch_retained_callable_with_policy_in_runtime(
                context,
                runmat_types::CallableIdentity::Builtin(runmat_types::BuiltinId(
                    identity.name.into(),
                )),
                runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
                arguments,
                requested_outputs,
                None,
            )
            .await
            .map_err(CallableCallError::Runtime)?,
            Self::DynamicName { name } => crate::dispatch_retained_callable_with_policy_in_runtime(
                context,
                runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(name.clone())),
                runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
                arguments,
                requested_outputs,
                None,
            )
            .await
            .map_err(CallableCallError::Runtime)?,
        };
        crate::call::arguments::project_rooted_legacy_builtin_value_abi(result)
            .map_err(CallableCallError::Runtime)
    }
}
pub(crate) async fn prepend_rooted_argument(
    arguments: &crate::call::BuiltinArguments,
    prefix: RootedValue,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<crate::call::BuiltinArguments, crate::RuntimeError> {
    let count = arguments.len().checked_add(1).ok_or_else(|| {
        crate::runtime_error::semantic_error(
            "RunMat:Call:Arguments",
            "callback argument count overflow",
        )
    })?;
    let mut prefix = Some(prefix);
    arguments
        .try_map_mixed_async(count, context, async |index| {
            Ok(if index == 0 {
                crate::execution::ArgumentProjection::Rooted(
                    prefix.take().expect("one prefix slot"),
                )
            } else {
                crate::execution::ArgumentProjection::Source(arguments[index - 1].clone())
            })
        })
        .await
}
