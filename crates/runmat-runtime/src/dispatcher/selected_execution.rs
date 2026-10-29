//! Execution after implementation selection, shared by normal and admitted calls.
use super::*;
use crate::call::BuiltinArguments;
use futures::{future::Either, FutureExt};

/// Legacy inventory remains an explicit ordinary-only adapter.
#[derive(Clone, Copy)]
pub(super) enum SelectedImplementation {
    Canonical(crate::builtin::RuntimeBuiltinBinding),
    Legacy(fn(&[Value]) -> runmat_builtins::BuiltinFuture),
}

impl SelectedImplementation {
    /// Only the selected canonical binding's declared role can bypass object
    /// method selection. Neither requested spelling nor receiver class grants
    /// query authority; legacy and undeclared binding variants remain ordinary.
    pub(super) fn is_handle_validity_query(self, receiver: Option<&Value>) -> bool {
        // Sealing belongs to the canonical handle/listener domain, not to
        // arbitrary value classes that may declare an ordinary isvalid method.
        if !matches!(receiver, Some(Value::HandleObject(_) | Value::Listener(_))) {
            return false;
        }
        let Self::Canonical(binding) = self else {
            return false;
        };
        runmat_builtins::builtin_catalog_entry_by_name(binding.identity.builtin.name).is_some_and(
            |entry| {
                entry.identity == binding.identity.builtin
                    && entry
                        .bindings
                        .iter()
                        .any(|declared| entry.binding_identity(declared) == binding.identity)
                    && entry.contract.semantic_kind
                        == runmat_builtins::BuiltinSemanticKind::HandleValidityQuery
            },
        )
    }

    fn invoke_retained(
        self,
        args: BuiltinArguments,
        context: Option<crate::context::RuntimeContext>,
        requested_outputs: Option<usize>,
        observation: Option<direct_call_observation::CallAdmissionSink>,
    ) -> crate::builtin::RuntimeBuiltinFuture {
        match self {
            Self::Canonical(binding) => (binding.implementation)(
                crate::builtin::RuntimeBuiltinInvocation::retained(
                    args,
                    context.map(|context| {
                        context.with_vm_fusion_selected_binding(Some(binding.identity))
                    }),
                    requested_outputs,
                )
                .with_call_observation(observation),
            ),
            Self::Legacy(implementation) => crate::builtin::admitted_builtin_future_with(
                context,
                "legacy inventory",
                move |context| async move {
                    let ordinary = args.materialize_legacy_contiguous(context.as_ref())?;
                    let sequence = crate::call::arguments::adapt_legacy_builtin_result(
                        implementation(&ordinary).await?,
                    )?;
                    let result = crate::execution::RootedValueSequence::new(sequence)
                        .map_err(|error| error.into_runtime_error());
                    drop(ordinary);
                    drop(args);
                    result
                },
            ),
        }
    }
}

/// Borrowed API retained solely for embedding and direct test fixtures.
#[cfg(test)]
pub(super) async fn execute(
    implementation: SelectedImplementation,
    execution: &[Value],
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    observation: Option<&direct_call_observation::CallObservation>,
    authority: Option<direct_call_observation::RegistrationAuthority>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let args = BuiltinArguments::from_borrowed(execution, context.as_ref())?;
    execute_retained(
        implementation,
        args,
        context,
        requested_outputs,
        observation,
        authority,
    )
    .await
}

pub(super) fn execute_retained<'a>(
    implementation: SelectedImplementation,
    execution: BuiltinArguments,
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    observation: Option<&'a direct_call_observation::CallObservation>,
    authority: Option<direct_call_observation::RegistrationAuthority>,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + 'a
{
    // Selection/admission remains lazy. Retire ordinary invocation before
    // constructing a gather retry, so its cold preparation frame does not
    // surround every nested CPU call. Both phases retain the original roots.
    let first_execution = execution.fork();
    let first_context = context.clone();
    let first = futures::future::lazy(move |_| {
        let sink = match (observation, implementation, authority) {
            (Some(observation), SelectedImplementation::Canonical(binding), Some(authority)) => {
                observation.sink(binding, authority, requested_outputs)
            }
            _ => None,
        };
        implementation.invoke_retained(first_execution, first_context, requested_outputs, sink)
    })
    .flatten();
    crate::execution::after_result(first.map(Ok), move |result| match result {
        Err(error) if should_retry_with_gpu_gather_values(&error, execution.iter()) => {
            if let Some(observation) = observation {
                observation.invalidate_retry();
            }
            if let Some(recorder) =
                runmat_accelerate_api::observation::current_provider_observation()
            {
                recorder.execution_disposition(runmat_accelerate_api::observation::ProviderExecutionDisposition::SharedRuntimeFallback);
            }
            Either::Right(retry(implementation, execution, context, requested_outputs))
        }
        result => Either::Left(std::future::ready(result)),
    })
}

async fn retry(
    implementation: SelectedImplementation,
    execution: BuiltinArguments,
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    // Gather changes numeric backings, not GC identities. Build each
    // successor directly in its final admitted slot and retain the
    // original registration forest through all suspended downloads.
    let preparation =
        execution.try_map_preserving_roots_async(execution.len(), context.as_ref(), |index| {
            let value = &execution[index];
            async move {
                if value_contains_gpu(value) {
                    gather_if_needed_async(value).await
                } else {
                    Ok(value.clone())
                }
            }
        });
    let gathered = match &context {
        Some(context) => context.scope(preparation).await?,
        None => preparation.await?,
    };
    implementation
        .invoke_retained(gathered, context, requested_outputs, None)
        .await
}

#[cfg(test)]
mod tests {
    use super::{execute, SelectedImplementation};
    use crate::builtin::{RuntimeBuiltinBinding, RuntimeBuiltinInvocation};
    use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};
    use runmat_value::{Value, ValueSequence};

    fn fixture(invocation: RuntimeBuiltinInvocation<'_>) -> crate::builtin::RuntimeBuiltinFuture {
        let requested_outputs = invocation.requested_outputs;
        crate::builtin::admitted_builtin_future(None, "fixture", async move {
            let sequence = ValueSequence::single(Value::Num(requested_outputs.unwrap_or(0) as f64))
                .map_err(crate::sequence::sequence_error_to_runtime)?;
            crate::execution::RootedValueSequence::new(sequence)
                .map_err(|error| error.into_runtime_error())
        })
    }

    fn selected_identity_fixture(
        invocation: RuntimeBuiltinInvocation<'_>,
    ) -> crate::builtin::RuntimeBuiltinFuture {
        let selected = invocation
            .context
            .as_ref()
            .and_then(|context| context.vm_fusion_selected_binding());
        let exact = selected
            == Some(BuiltinBindingIdentity {
                builtin: BuiltinCatalogIdentity {
                    name: "__selected_binding_identity_fixture",
                },
                variant: "alternate",
            });
        crate::builtin::admitted_builtin_future(None, "fixture", async move {
            let sequence = ValueSequence::single(Value::Num(if exact { 1.0 } else { 0.0 }))
                .map_err(crate::sequence::sequence_error_to_runtime)?;
            crate::execution::RootedValueSequence::new(sequence)
                .map_err(|error| error.into_runtime_error())
        })
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn selected_canonical_binding_retains_nondefault_variant_through_invocation() {
        let identity = BuiltinBindingIdentity {
            builtin: BuiltinCatalogIdentity {
                name: "__selected_binding_identity_fixture",
            },
            variant: "alternate",
        };
        let selected =
            SelectedImplementation::Canonical(RuntimeBuiltinBinding::new(identity, fixture));
        let SelectedImplementation::Canonical(binding) = selected else {
            panic!("selected canonical binding became legacy");
        };
        assert_eq!(binding.identity, identity);
        let result = futures::executor::block_on(execute(selected, &[], None, Some(2), None, None))
            .expect("selected implementation executes without a global inventory");
        assert_eq!(result.as_slice(), &[Value::Num(2.0)]);
        let SelectedImplementation::Canonical(binding) = selected else {
            panic!("selected canonical binding became legacy after invocation");
        };
        assert_eq!(binding.identity, identity);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn only_selected_dispatch_projects_binding_identity_into_explicit_context() {
        let identity = BuiltinBindingIdentity {
            builtin: BuiltinCatalogIdentity {
                name: "__selected_binding_identity_fixture",
            },
            variant: "alternate",
        };
        let binding = RuntimeBuiltinBinding::new(identity, selected_identity_fixture);
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        assert!(runtime.vm_fusion_selected_binding().is_none());
        let selected = futures::executor::block_on(execute(
            SelectedImplementation::Canonical(binding),
            &[],
            Some(runtime.clone()),
            Some(1),
            None,
            None,
        ))
        .expect("selected canonical implementation");
        assert_eq!(selected.as_slice(), &[Value::Num(1.0)]);
        assert!(runtime.vm_fusion_selected_binding().is_none());
        let unauthenticated = futures::executor::block_on((binding.implementation)(
            RuntimeBuiltinInvocation::new(&[], Some(runtime), Some(1)),
        ))
        .expect("direct invocation remains unauthenticated");
        assert_eq!(unauthenticated.as_slice(), &[Value::Num(0.0)]);
    }
}
