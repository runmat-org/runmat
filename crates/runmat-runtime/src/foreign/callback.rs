use super::{foreign_error, ForeignErrorKind};
use std::future::Future;
use std::pin::Pin;

use crate::call::descriptor::{
    CallableCallKind, CallableDescriptor, CallableMetadata, CallableTarget, RetainedClosureValue,
};
use crate::context::{RuntimeCallRequest, RuntimeContext};
use crate::RuntimeError;
use runmat_types::{CallableFallbackPolicy, CallableIdentity};
use runmat_value::Value;

/// Invocation-scoped callables are local to the pumped original evaluator.
/// Refuse before foreign callback registration or transmission, not after a
/// foreign effect has already occurred.
pub(crate) fn reject_scoped_callback(value: &Value) -> Result<(), RuntimeError> {
    if matches!(value, Value::ScopedFunctionHandle(_)) {
        let mut error = foreign_error(
            ForeignErrorKind::CallbackFailed,
            "invocation-scoped callables cannot cross a foreign boundary",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(error);
    }
    Ok(())
}

/// Normalize one retained RunMat callable into the executor-neutral callback
/// request. Captures remain in the callable generation; `arguments` contains
/// only the explicitly supplied suffix.
pub fn foreign_callback_request(
    context: &RuntimeContext,
    callback: &crate::call::RetainedCallableTarget,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<RuntimeCallRequest, RuntimeError> {
    let arguments = crate::execution::RootedValues::in_context(arguments, context)?;
    foreign_callback_request_rooted(context, callback, arguments, requested_outputs)
}

/// Build a callback request by moving the original rooted argument collection
/// into the selected descriptor. This is the canonical adapter boundary for
/// decoded foreign callback arguments: nested payloads are never cloned merely
/// to preserve their roots across the call service future.
pub(crate) fn foreign_callback_request_rooted(
    context: &RuntimeContext,
    callback: &crate::call::RetainedCallableTarget,
    arguments: crate::execution::RootedValues,
    requested_outputs: usize,
) -> Result<RuntimeCallRequest, RuntimeError> {
    use runmat_types::MethodId;
    reject_scoped_callback(callback.value())?;

    let target = match callback.value() {
        Value::FunctionHandle(name) => {
            let identity = crate::callable_identity_for_handle_name(name).0;
            CallableTarget::Resolved {
                fallback_policy: fallback_policy(&identity),
                identity: identity.into(),
            }
        }
        Value::ExternalFunctionHandle(name) => {
            let identity = crate::external_callable_identity_for_name(name);
            CallableTarget::Resolved {
                fallback_policy: fallback_policy(&identity),
                identity: identity.into(),
            }
        }
        Value::MethodFunctionHandle(name) => {
            let identity = CallableIdentity::Method(MethodId(name.name().to_owned()));
            CallableTarget::Resolved {
                fallback_policy: fallback_policy(&identity),
                identity: identity.into(),
            }
        }
        Value::BoundFunctionHandle(handle) => {
            CallableTarget::Bound(runmat_value::BoundFunctionBinding {
                function: handle.function,
                authority: handle.authority.clone(),
            })
        }
        Value::Closure(closure) => {
            let retained = RetainedClosureValue::from_callable_owner(callback)?;
            match &closure.binding {
                Some(binding) => CallableTarget::BoundLexical {
                    binding: binding.clone(),
                    closure: retained,
                },
                None => {
                    let identity =
                        crate::callable_identity_for_handle_name(&closure.function_name).0;
                    CallableTarget::CapturedResolved {
                        fallback_policy: fallback_policy(&identity),
                        identity: identity.into(),
                        closure: retained,
                    }
                }
            }
        }
        _ => {
            return Err(foreign_error(
                ForeignErrorKind::CallbackFailed,
                "foreign callback target is not a callable RunMat value",
            ));
        }
    };
    let display_name = match &target {
        CallableTarget::Resolved { identity, .. }
        | CallableTarget::CapturedResolved { identity, .. } => {
            crate::call::identity::strict_callable_display_name(identity.as_identity())
        }
        _ => None,
    };
    let supplied_roots =
        crate::execution::RootedValueLease::for_values_in_context(arguments.values(), context)?;
    let arguments = arguments.try_into_values()?.collect();
    RuntimeCallRequest::new(
        CallableDescriptor {
            target,
            args: arguments,
            requested_outputs,
            metadata: CallableMetadata {
                call_kind: CallableCallKind::Direct,
                display_name: display_name.map(Into::into),
                ..CallableMetadata::default()
            },
        },
        Some(callback.clone()),
        supplied_roots,
    )
}

fn fallback_policy(identity: &CallableIdentity) -> CallableFallbackPolicy {
    match identity {
        CallableIdentity::DynamicName(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_) => CallableFallbackPolicy::RuntimeNameResolution,
        CallableIdentity::ExternalName(_) => CallableFallbackPolicy::ExternalBoundary,
        _ => CallableFallbackPolicy::None,
    }
}

pub fn invoke_foreign_callback(
    context: RuntimeContext,
    request: RuntimeCallRequest,
) -> Pin<
    Box<dyn Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + 'static>,
> {
    Box::pin(async move {
        if context.is_cancelled() {
            return Err(foreign_error(
                ForeignErrorKind::CallbackFailed,
                "foreign callback was cancelled before invocation",
            ));
        }
        let service = context
            .service_ports()
            .require_call("foreign callback")
            .map_err(|error| error.into_runtime_error())?
            .clone();
        context
            .scope(service.invoke(context.clone(), request))
            .await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeCallRequest, RuntimeCallService, RuntimeServicePorts};
    use crate::execution::{RetainedValueReadLease, RuntimeExecutionService};
    use runmat_types::SymbolName;
    use runmat_value::ValueSequence;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct RecordingCallService {
        requests: RefCell<Vec<RuntimeCallRequest>>,
    }

    impl RuntimeCallService for RecordingCallService {
        fn resolve(&self, _name: &str) -> Option<usize> {
            None
        }

        fn invoke(
            &self,
            _runtime: RuntimeContext,
            request: RuntimeCallRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>>
                    + 'static,
            >,
        > {
            self.requests.borrow_mut().push(request);
            Box::pin(async {
                let value = ValueSequence::single(Value::Num(42.0)).map_err(|error| {
                    crate::runtime_error::semantic_error("ForeignCallbackOutput", error.to_string())
                })?;
                crate::execution::RootedValueSequence::new(value)
                    .map_err(|error| error.into_runtime_error())
            })
        }
    }

    #[test]
    fn callback_reenters_through_the_originating_runtime_context() {
        let service = Rc::new(RecordingCallService::default());
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_call(service.clone()));
        let args = vec![Value::Num(7.0)];
        let roots =
            crate::execution::RootedValueLease::for_values_in_context(&args, &context).unwrap();
        let result = futures::executor::block_on(invoke_foreign_callback(
            context,
            RuntimeCallRequest::new(
                CallableDescriptor::resolved(
                    CallableIdentity::DynamicName(SymbolName("callback".into())),
                    args,
                    1,
                    CallableFallbackPolicy::RuntimeNameResolution,
                    CallableCallKind::Direct,
                ),
                None,
                roots,
            )
            .unwrap(),
        ))
        .unwrap();

        result.with_owned_sequence(|values| {
            assert_eq!(values.as_slice(), &[Value::Num(42.0)]);
        });
        let requests = service.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].descriptor().requested_outputs, 1);
        assert_eq!(requests[0].descriptor().args, vec![Value::Num(7.0)]);
    }

    #[test]
    fn callback_observes_originating_session_cancellation() {
        let service = Rc::new(RecordingCallService::default());
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_cancellation_token(cancellation.token())
            .with_service_ports(RuntimeServicePorts::default().with_call(service.clone()));
        let roots =
            crate::execution::RootedValueLease::for_values_in_context(&[], &context).unwrap();
        let request = RuntimeCallRequest::new(
            CallableDescriptor::resolved(
                CallableIdentity::DynamicName(SymbolName("callback".into())),
                Vec::new(),
                0,
                CallableFallbackPolicy::RuntimeNameResolution,
                CallableCallKind::Direct,
            ),
            None,
            roots,
        )
        .unwrap();
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let error =
            futures::executor::block_on(invoke_foreign_callback(context, request)).unwrap_err();

        assert_eq!(error.identifier(), Some("RunMat:Foreign:CallbackFailed"));
        assert!(service.requests.borrow().is_empty());
    }

    #[test]
    fn bound_callback_retains_exact_authority_without_flattening_captures() {
        let binding = runmat_value::BoundFunctionBinding::unavailable(77);
        let callback = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "display_is_not_authority".into(),
                binding: Some(binding.clone()),
                source_role: runmat_types::FunctionSourceRole::Anonymous,
                source_file: None,
                captures: vec![Value::Num(3.0)].into(),
            },
        ));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let callback = RetainedValueReadLease::prepare_owned(callback, &runtime).unwrap();
        let owner = crate::call::RetainedCallableTarget::Direct(callback.clone());
        let request = foreign_callback_request(&runtime, &owner, vec![Value::Num(4.0)], 1)
            .expect("callback normalization");
        assert_eq!(request.descriptor().args, vec![Value::Num(4.0)]);
        let CallableTarget::BoundLexical {
            binding: selected,
            closure,
        } = &request.descriptor().target
        else {
            panic!("bound callback must retain its closure");
        };
        assert_eq!(selected, &binding);
        assert_eq!(closure.as_closure().captures, vec![Value::Num(3.0)]);
        assert!(closure
            .generation()
            .expect("direct callback generation")
            .ptr_eq(&callback.shared_generation()));
    }

    #[test]
    fn queued_callback_request_roots_supplied_gc_values_without_copying_them() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("supplied".into()))
                .expect("allocate supplied value");
            let handle = initial.handle();
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let callback = RetainedValueReadLease::prepare_owned(
                Value::FunctionHandle("callback".into()),
                &runtime,
            )
            .expect("retain callable");
            let owner = crate::call::RetainedCallableTarget::Direct(callback);
            let request = foreign_callback_request(
                &runtime,
                &owner,
                vec![Value::HandleObject(runmat_value::HandleRef {
                    class_name: "CallbackInput".into(),
                    target: handle,
                    valid: true,
                })],
                1,
            )
            .expect("prepare request");
            let queued = invoke_foreign_callback(runtime, request);
            drop(owner);
            initial.unroot().expect("transfer supplied root to request");
            runmat_gc::gc_collect_major().expect("collect with unpolled callback");
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            drop(queued);
            runmat_gc::gc_collect_major().expect("collect after callback retirement");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
