use super::*;
use crate::call::descriptor::{
    CallableCallKind, CallableDescriptor, CallableMetadata, CallableTarget, RetainedClosureValue,
};
use runmat_gc::{GcHandle, GcRoot, Trace, Tracer, ValueRoot, ValueRootRegistration};
use runmat_value::{BoundFunctionBinding, Closure, HandleRef, LexicalEnvironment};
use std::rc::Rc;

struct DeferredServiceRoot(Rc<RuntimeExecutionService>);

struct Handles(Vec<GcHandle>);

impl Tracer for Handles {
    fn mark(&mut self, handle: GcHandle) {
        self.0.push(handle);
    }
}

impl GcRoot for DeferredServiceRoot {
    fn scan(&self) -> Vec<GcHandle> {
        let mut handles = Handles(Vec::new());
        self.0
            .visit_retained_values(&mut |value| {
                value.trace(&mut handles);
                Some(())
            })
            .expect("deferred service state must be available to its root scanner");
        handles.0
    }

    fn description(&self) -> String {
        "deferred execution service".into()
    }
}

impl ValueRoot for DeferredServiceRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        self.0.visit_retained_values(visitor)
    }
}

fn handle_value(handle: GcHandle) -> Value {
    Value::HandleObject(HandleRef {
        class_name: "RetainedHandle".into(),
        target: handle,
        valid: true,
    })
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn deferred_callable_targets_keep_activation_ancestors_anonymous_captures_and_args_alive() {
    runmat_gc::gc_test_context(|| {
        let parent = runmat_gc::gc_allocate_rooted(Value::String("parent".into())).unwrap();
        let activation = runmat_gc::gc_allocate_rooted(Value::String("activation".into())).unwrap();
        let captured = runmat_gc::gc_allocate_rooted(Value::String("capture".into())).unwrap();
        let anonymous_capture =
            runmat_gc::gc_allocate_rooted(Value::String("anonymous capture".into())).unwrap();
        let argument = runmat_gc::gc_allocate_rooted(Value::String("argument".into())).unwrap();
        let handles = [
            parent.handle(),
            activation.handle(),
            captured.handle(),
            anonymous_capture.handle(),
            argument.handle(),
        ];
        let binding = BoundFunctionBinding::unavailable(7);
        let closure = Closure::from_data(runmat_value::ClosureData {
            function_name: "nested".into(),
            binding: Some(binding.clone()),
            source_role: runmat_types::FunctionSourceRole::Nested,
            source_file: None,
            captures: vec![handle_value(captured.handle())].into(),
            lexical_environment: Some(LexicalEnvironment::new(
                activation.handle(),
                Some(LexicalEnvironment::new(parent.handle(), None)),
            )),
        });
        let service = Rc::new(RuntimeExecutionService::new());
        let registration =
            ValueRootRegistration::new(Box::new(DeferredServiceRoot(Rc::clone(&service)))).unwrap();
        let call = DeferredCall {
            spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
            invocation: DeferredInvocation::callable(CallableDescriptor {
                target: CallableTarget::BoundLexical {
                    binding,
                    closure: RetainedClosureValue::new(closure),
                },
                args: vec![handle_value(argument.handle())],
                requested_outputs: 1,
                metadata: CallableMetadata {
                    call_kind: CallableCallKind::Direct,
                    ..CallableMetadata::default()
                },
            }),
            retry: runmat_execution::RetryPolicy::Never,
            program_revision: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            console: None,
            capabilities: Default::default(),
            program: DeferredProgram::Local,
        };
        service.create_future(call).unwrap();
        service
            .create_future(DeferredCall {
                spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
                invocation: DeferredInvocation::callable(CallableDescriptor {
                    target: CallableTarget::FevalForward(Value::Closure(Closure::from_data(
                        runmat_value::ClosureData {
                            function_name: "anonymous".into(),
                            binding: None,
                            source_role: runmat_types::FunctionSourceRole::Anonymous,
                            source_file: None,
                            captures: vec![handle_value(anonymous_capture.handle())].into(),
                            lexical_environment: None,
                        },
                    ))),
                    args: Vec::new(),
                    requested_outputs: 1,
                    metadata: CallableMetadata::default(),
                }),
                retry: runmat_execution::RetryPolicy::Never,
                program_revision: None,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                console: None,
                capabilities: Default::default(),
                program: DeferredProgram::Local,
            })
            .unwrap();
        drop((parent, activation, captured, anonymous_capture, argument));

        runmat_gc::gc_collect_major().unwrap();
        for handle in handles {
            assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_ok());
        }
        let stored_target_address = {
            let state = service.state.lock().unwrap();
            state
                .futures
                .values()
                .find_map(|future| {
                    let FutureState::Deferred(call) = future else {
                        return None;
                    };
                    let DeferredInvocation::Callable(descriptor) = &call.invocation else {
                        return None;
                    };
                    let CallableTarget::BoundLexical { closure, .. } = &descriptor.target else {
                        return None;
                    };
                    Some(closure.as_value() as *const Value)
                })
                .expect("stored lexical target")
        };
        let lexical_target_address = || {
            let mut address = None;
            service
                .visit_retained_values(&mut |value| {
                    if matches!(value, Value::Closure(closure) if closure.lexical_environment.is_some()) {
                        address = Some(value as *const Value);
                    }
                    Some(())
                })
                .unwrap();
            address.expect("deferred lexical closure must be visited")
        };
        assert_eq!(
            lexical_target_address(),
            stored_target_address,
            "first scan must borrow the stored closure"
        );
        assert_eq!(
            lexical_target_address(),
            stored_target_address,
            "repeated scans must not clone captured values"
        );
        let mut visited = Vec::new();
        service
            .visit_retained_values(&mut |value| {
                visited.push(value.clone());
                Some(())
            })
            .unwrap();
        assert_eq!(
            visited.len(),
            3,
            "argument and both callable targets are visible"
        );
        assert_eq!(
            visited
                .iter()
                .filter(|value| matches!(value, Value::Closure(_)))
                .count(),
            2
        );
        drop(visited);

        registration.unregister().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        for handle in handles {
            assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
        }
    });
}
