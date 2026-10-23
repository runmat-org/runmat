use super::*;

struct ConstructorControl(crate::context::host_effect::ManagedHostEffectLedger);

impl crate::context::RuntimeParallelService for ConstructorControl {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        crate::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }

    fn managed_host_effect_ledger(
        &self,
    ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[test]
fn prepared_callback_and_child_have_independent_admission_and_retirement() {
    use std::cell::Cell;
    use std::rc::Rc;

    for capacity in [0, 1 << 20] {
        let ledger = crate::context::host_effect::ManagedHostEffectLedger::new(capacity);
        let context = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            crate::context::RuntimeServicePorts::default()
                .with_parallel(Rc::new(ConstructorControl(ledger.clone()))),
        );
        let factories = Rc::new(Cell::new(0));
        let polls = Rc::new(Cell::new(0));
        for (invoke_child, replace_owner) in [(false, false), (true, false), (true, true)] {
            let factories = Rc::clone(&factories);
            let polls = Rc::clone(&polls);
            let callback = PreparedConstructorInvoke::new(&context, move |_| {
                factories.set(factories.get() + 1);
                async move {
                    polls.set(polls.get() + 1);
                    crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                        .map_err(|error| error.into_runtime_error())
                }
            });
            if capacity == 0 {
                assert!(callback.is_err(), "callback storage must be admitted");
            } else {
                let callback = callback.expect("funded callback");
                assert!(ledger.snapshot().live_reserved_bytes > 0);
                if invoke_child {
                    let (guard, body) = callback.into_parts();
                    if replace_owner {
                        let other_ledger =
                            crate::context::host_effect::ManagedHostEffectLedger::new(capacity);
                        let other = context.clone().with_service_ports(
                            crate::context::RuntimeServicePorts::default()
                                .with_parallel(Rc::new(ConstructorControl(other_ledger.clone()))),
                        );
                        assert!(
                            body.invoke(other, &guard).is_err(),
                            "a different owner cannot consume the callback"
                        );
                        assert_eq!(other_ledger.snapshot().live_reserved_bytes, 0);
                    } else {
                        let future = body.invoke(context.clone(), &guard).expect("funded child");
                        assert!(ledger.snapshot().live_reserved_bytes > 0);
                        drop(future);
                        assert!(
                            ledger.snapshot().live_reserved_bytes > 0,
                            "the outer guard retains callback funding through child retirement"
                        );
                    }
                    drop(guard);
                } else {
                    drop(callback);
                }
            }
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
        assert_eq!(factories.get(), usize::from(capacity != 0));
        assert_eq!(polls.get(), 0, "an unpolled child has no source effects");
    }
}

#[test]
fn exact_binding_rejects_an_ordinary_owner_before_default_creation() {
    let class =
        runmat_types::ClassIdentity::new("MissingBoundConstructor").expect("valid class identity");
    let owner = runmat_types::ClassMethodOwner {
        declaring_class: class.clone(),
        method: runmat_types::MethodName::from("MissingBoundConstructor"),
        is_static: false,
        role: runmat_types::ClassMethodRole::Ordinary,
    };
    let error = futures::executor::block_on(super::call_class_constructor(
        class,
        ConstructorArguments::Borrowed(&[]),
        Some(1),
        None,
        ConstructorInvocationTarget::ExactBound {
            owner,
            binding: runmat_value::BoundFunctionBinding::unavailable(1),
        },
    ))
    .expect_err("ordinary method cannot enter constructor orchestration");
    assert_eq!(error.identifier(), Some("RunMat:ConstructorOwnerInvalid"));
}
