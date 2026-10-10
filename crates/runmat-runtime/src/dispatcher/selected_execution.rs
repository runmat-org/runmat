//! Execution after implementation selection, shared by normal and admitted calls.
use super::*;

/// The old inventory remains separate until its L23 removal. Canonical
/// implementations have exactly one context-bearing ABI.
#[derive(Clone, Copy)]
pub(super) enum SelectedImplementation {
    Canonical(crate::builtin::RuntimeBuiltinBinding),
    Legacy(fn(&[Value]) -> runmat_builtins::BuiltinFuture),
}

impl SelectedImplementation {
    fn invoke(
        self,
        args: &[Value],
        context: Option<crate::context::RuntimeContext>,
        requested_outputs: Option<usize>,
        observation: Option<direct_call_observation::CallAdmissionSink>,
    ) -> crate::builtin::RuntimeBuiltinFuture {
        match self {
            Self::Canonical(binding) => (binding.implementation)(
                crate::builtin::RuntimeBuiltinInvocation::new(
                    args,
                    context.map(|context| {
                        context.with_vm_fusion_selected_binding(Some(binding.identity))
                    }),
                    requested_outputs,
                )
                .with_call_observation(observation),
            ),
            Self::Legacy(implementation) => {
                let future = implementation(args);
                Box::pin(async move {
                    crate::call::arguments::adapt_legacy_builtin_result(future.await?)
                })
            }
        }
    }
}

pub(super) async fn execute(
    implementation: SelectedImplementation,
    execution: &[Value],
    context: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    observation: Option<&direct_call_observation::CallObservation>,
    authority: Option<direct_call_observation::RegistrationAuthority>,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    let sink = match (observation, implementation, authority) {
        (Some(observation), SelectedImplementation::Canonical(binding), Some(authority)) => {
            observation.sink(binding, authority, requested_outputs)
        }
        _ => None,
    };
    let result = match implementation
        .invoke(execution, context.clone(), requested_outputs, sink)
        .await
    {
        Ok(result) => result,
        Err(error) if should_retry_with_gpu_gather(&error, execution) => {
            if let Some(observation) = observation {
                observation.invalidate_retry();
            }
            if let Some(recorder) =
                runmat_accelerate_api::observation::current_provider_observation()
            {
                recorder.execution_disposition(runmat_accelerate_api::observation::ProviderExecutionDisposition::SharedRuntimeFallback);
            }
            match gather_args_for_retry_async(execution).await? {
                Some(gathered) => {
                    implementation
                        .invoke(&gathered, context, requested_outputs, None)
                        .await?
                }
                None => return Err(error),
            }
        }
        Err(error) => return Err(error),
    };
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{execute, SelectedImplementation};
    use crate::builtin::{RuntimeBuiltinBinding, RuntimeBuiltinInvocation};
    use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};
    use runmat_value::{Value, ValueSequence};

    fn fixture(invocation: RuntimeBuiltinInvocation<'_>) -> crate::builtin::RuntimeBuiltinFuture {
        let requested_outputs = invocation.requested_outputs;
        Box::pin(async move {
            ValueSequence::single(Value::Num(requested_outputs.unwrap_or(0) as f64))
                .map_err(crate::sequence::sequence_error_to_runtime)
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
        Box::pin(async move {
            ValueSequence::single(Value::Num(if exact { 1.0 } else { 0.0 }))
                .map_err(crate::sequence::sequence_error_to_runtime)
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
