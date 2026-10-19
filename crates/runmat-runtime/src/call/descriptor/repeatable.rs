//! Reusable ownership around the canonical callable descriptor selection.
//!
//! This owner deliberately does not reimplement `feval` classification. It
//! selects once through the same descriptor selector, keeps the selected
//! target and metadata intact, and passes independently rooted arguments
//! through the semantic, dynamic, external, method, and forwarding ports.

use runmat_value::Value;

use super::{CallableDescriptor, CallableTarget, FunctionNameResolver};
use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::execution::{RetainedValueReadLease, RootedValueSequence, RootedValues};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub(crate) struct RepeatableCallable {
    selected: CallableDescriptor,
    // Drops after the descriptor's shared callable aliases. Its roots retire
    // before the final generation and physical shallow backing.
    _target_owner: Option<RetainedValueReadLease>,
    // Canonical descriptors retain a root-free closure generation. Ordinary
    // repeated execution keeps one external root lease across all calls.
    _closure_roots: Option<crate::call::RetainedCallableTarget>,
}

impl RepeatableCallable {
    pub(crate) fn select(
        runtime: &RuntimeContext,
        target: Value,
        resolver: &impl FunctionNameResolver,
    ) -> Result<Self, RuntimeError> {
        let policy = HostAllocationReceipt::policy(Some(runtime)).map_err(|source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Callable:RepeatableSelection",
                "repeatable callable policy unavailable",
                source,
            )
        })?;
        let (selected, target_owner) = if policy == HostAllocationPolicy::Managed {
            let owner = RetainedValueReadLease::prepare_callable_owned(target, runtime)?;
            let selected = CallableDescriptor::from_retained_feval_value(&owner, 1, resolver)?;
            (selected, Some(owner))
        } else {
            (
                CallableDescriptor::from_feval_value(target, Vec::new(), 1, resolver),
                None,
            )
        };
        let closure_roots = match (&target_owner, &selected.target) {
            (Some(_), _) => None,
            (
                None,
                CallableTarget::BoundLexical { closure, .. }
                | CallableTarget::CapturedResolved { closure, .. },
            ) => Some(closure.retain_for_call(Some(runtime))?),
            (None, _) => None,
        };
        Ok(Self {
            selected,
            _target_owner: target_owner,
            _closure_roots: closure_roots,
        })
    }

    pub(crate) async fn invoke_rooted(
        &self,
        runtime: &RuntimeContext,
        arguments: RootedValues,
        requested_outputs: usize,
    ) -> Result<RootedValueSequence, RuntimeError> {
        if let CallableTarget::Bound(binding) = &self.selected.target {
            let function = binding.function;
            return crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                runtime,
                binding.clone(),
                crate::call::RootedFunctionArguments::ordinary(arguments),
                requested_outputs,
                None,
            )
            .await
            .unwrap_or_else(|| {
                Err(semantic_error(
                    "UndefinedSemanticFunction",
                    format!("selected repeatable bound function {function} is no longer installed"),
                ))
            });
        }
        if let CallableTarget::BoundLexical { binding, closure } = &self.selected.target {
            let callable = if let Some(owner) = &self._target_owner {
                crate::call::RetainedCallableTarget::Direct(owner.clone())
            } else if let Some(roots) = &self._closure_roots {
                roots.clone()
            } else {
                closure.retain_for_call(Some(runtime))?
            };
            let environment = closure.as_closure().lexical_environment.clone();
            let function = binding.function;
            return crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                runtime,
                binding.clone(),
                crate::call::RootedFunctionArguments::captured(callable, arguments)?,
                requested_outputs,
                environment,
            )
            .await
            .unwrap_or_else(|| {
                Err(semantic_error(
                    "UndefinedSemanticFunction",
                    format!("selected repeatable bound function {function} is no longer installed"),
                ))
            });
        }

        match &self.selected.target {
            CallableTarget::Resolved {
                identity,
                fallback_policy,
            } => {
                super::resolved_execution::invoke_descriptor_required(
                    Some(runtime),
                    identity.clone(),
                    crate::call::RootedFunctionArguments::ordinary(arguments),
                    requested_outputs,
                    &self.selected.metadata,
                    *fallback_policy,
                )
                .await
            }
            CallableTarget::FevalForward(value) => {
                crate::call_feval_sequence_async_with_outputs_in_context(
                    runtime,
                    value.clone(),
                    arguments.values(),
                    requested_outputs,
                )
                .await
            }
            CallableTarget::RetainedFevalForward(value) => {
                let name = retained_target_name(value.value()).ok_or_else(|| {
                    semantic_error(
                        "RunMat:Callable:UnsupportedRepeatableTarget",
                        "repeatable target has no callable name",
                    )
                })?;
                let rooted = if matches!(value.value(), Value::Closure(_)) {
                    let owner = self._target_owner.as_ref().ok_or_else(|| {
                        semantic_error(
                            "RunMat:Callable:RepeatableOwner",
                            "retained closure target lost its generation owner",
                        )
                    })?;
                    crate::call::RootedFunctionArguments::captured(owner.clone(), arguments)?
                } else {
                    crate::call::RootedFunctionArguments::ordinary(arguments)
                };
                match crate::user_functions::try_call_rooted_semantic_name_in_context(
                    Some(runtime),
                    name,
                    rooted,
                    requested_outputs,
                )
                .await
                {
                    std::ops::ControlFlow::Break(result) => result,
                    std::ops::ControlFlow::Continue(arguments) => {
                        let (identity, fallback) = match value.value() {
                            Value::ExternalFunctionHandle(name) => (
                                crate::external_callable_identity_for_name(name),
                                runmat_types::CallableFallbackPolicy::ExternalBoundary,
                            ),
                            Value::MethodFunctionHandle(name) => (
                                runmat_types::CallableIdentity::Method(runmat_types::MethodId(
                                    name.name().to_owned(),
                                )),
                                runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
                            ),
                            _ => (
                                crate::callable_identity_for_handle_name(name).0,
                                runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
                            ),
                        };
                        let request = crate::user_functions::CallableRequest::rooted_resolved(
                            identity,
                            fallback,
                            arguments,
                            requested_outputs,
                        )
                        .with_function_input_callsite(
                            self.selected.metadata.function_input_callsite.clone(),
                        )
                        .with_selected_runtime(Some(runtime));
                        match crate::user_functions::try_call_semantic_descriptor_owned(request)
                            .await
                        {
                            std::ops::ControlFlow::Break(result) => result,
                            std::ops::ControlFlow::Continue(_) => Err(semantic_error(
                                "UndefinedFunction",
                                format!("repeatable callable target '{name}' is unavailable"),
                            )),
                        }
                    }
                }
            }
            CallableTarget::CapturedResolved {
                identity,
                fallback_policy,
                closure,
            } => {
                let callable = if let Some(owner) = &self._target_owner {
                    crate::call::RetainedCallableTarget::Direct(owner.clone())
                } else if let Some(roots) = &self._closure_roots {
                    roots.clone()
                } else {
                    closure.retain_for_call(Some(runtime))?
                };
                let rooted = crate::call::RootedFunctionArguments::captured(callable, arguments)?;
                if let runmat_types::CallableIdentity::BoundFunction(function) = identity {
                    return crate::user_functions::try_call_rooted_semantic_function_in_context(
                        Some(runtime),
                        function.0,
                        rooted,
                        requested_outputs,
                    )
                    .await
                    .unwrap_or_else(|| {
                        Err(semantic_error(
                            "UndefinedSemanticFunction",
                            format!(
                                "selected repeatable semantic function {} is unavailable",
                                function.0
                            ),
                        ))
                    });
                }
                let request = crate::user_functions::CallableRequest::rooted_resolved(
                    identity.clone(),
                    *fallback_policy,
                    rooted,
                    requested_outputs,
                )
                .with_function_input_callsite(
                    self.selected.metadata.function_input_callsite.clone(),
                )
                .with_selected_runtime(Some(runtime));
                match crate::user_functions::try_call_semantic_descriptor_owned(request).await {
                    std::ops::ControlFlow::Break(result) => result,
                    std::ops::ControlFlow::Continue(_) => Err(semantic_error(
                        "UndefinedFunction",
                        format!("repeatable callable target {identity:?} is unavailable"),
                    )),
                }
            }
            CallableTarget::Bound(_) | CallableTarget::BoundLexical { .. } => unreachable!(),
        }
    }
}

fn retained_target_name(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => Some(text),
        Value::StringArray(array) if array.data.len() == 1 => Some(&array.data[0]),
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Some(name.name()),
        Value::Closure(closure) => Some(&closure.function_name),
        _ => None,
    }
}
