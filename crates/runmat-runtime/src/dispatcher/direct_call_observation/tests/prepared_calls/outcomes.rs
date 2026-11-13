//! Prepared sum/nansum route, fence, and compatibility outcomes.
use super::*;
use runmat_value::{IntegerStorage, SparseTensor, ValueSequence};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sum_and_nansum_deposit_once_after_each_outer_fence() {
    for name in ["sum", "nansum"] {
        for route in 0..3 {
            let PreparedRuntime {
                runtime,
                binding,
                published,
                prepared,
                checkpoints,
                scoped: _scoped,
            } = prepared_runtime(name, false, false, false, None);
            let args = arguments();
            let result = match route {
                0 => futures::executor::block_on(
                    crate::call_builtin_sequence_async_with_outputs_in_context(
                        &runtime, name, &args, 1,
                    ),
                ),
                1 => futures::executor::block_on(
                    crate::dispatcher::try_call_builtin_sequence_async_with_outputs(
                        name,
                        &args,
                        1,
                        Some(&runtime),
                    ),
                )
                .map(|found| found.expect("selected optional binding")),
                _ => futures::executor::block_on(
                    crate::dispatcher::call_selected_builtin_async_with_outputs(
                        binding,
                        &args,
                        1,
                        Some(runtime.clone()),
                    ),
                ),
            }
            .expect("real reduction body completes");
            assert_eq!(result.len(), 1);
            assert_eq!(prepared.get(), 1);
            assert_eq!(checkpoints.get(), 1);
            assert_eq!(*published.borrow(), [PreparedCallStatus::Succeeded]);
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fence_failure_and_binding_mutation_suppress_completed_success() {
    let PreparedRuntime {
        runtime,
        published,
        prepared,
        checkpoints,
        ..
    } = prepared_runtime("sum", false, false, true, None);
    let result =
        futures::executor::block_on(crate::call_builtin_sequence_async_with_outputs_in_context(
            &runtime,
            "sum",
            &arguments(),
            1,
        ));
    assert!(result.is_err());
    assert_eq!(prepared.get(), 1);
    assert_eq!(checkpoints.get(), 1);
    assert!(published.borrow().is_empty());

    let binding = reduction_binding("sum");
    let (base, scoped, _placement) = harness(binding);
    let changed = scoped.clone();
    let published = Rc::new(RefCell::new(Vec::new()));
    let prepared = Rc::new(Cell::new(0));
    let checkpoints = Rc::new(Cell::new(0));
    let ports = base
        .service_ports()
        .clone()
        .with_acceleration(Rc::new(AdmittedService {
            published: published.clone(),
            prepared,
            fail_body: false,
            fail_cleanup: false,
        }))
        .with_workspace(Rc::new(Fence {
            published: published.clone(),
            checkpoints,
            fail: false,
            on_checkpoint: Some(Box::new(move || changed.generation.set(2))),
        }));
    let runtime = base.with_service_ports(ports);
    assert!(futures::executor::block_on(
        crate::call_builtin_sequence_async_with_outputs_in_context(
            &runtime,
            "sum",
            &arguments(),
            1
        )
    )
    .is_ok());
    assert!(
        published.borrow().is_empty(),
        "stale selected generation cannot publish"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn body_and_cleanup_errors_preserve_primary_cause_but_failed_fence_cannot_train() {
    for (fail_body, fail_cleanup) in [(true, false), (false, true)] {
        for route in 0..3 {
            let PreparedRuntime {
                runtime,
                binding,
                published,
                prepared,
                checkpoints,
                ..
            } = prepared_runtime("sum", fail_body, fail_cleanup, false, None);
            let primary = call_outer_route(&runtime, binding, route).unwrap_err();
            assert_eq!(prepared.get(), 1);
            assert_eq!(checkpoints.get(), 1);
            assert_eq!(*published.borrow(), [PreparedCallStatus::CompletedFailure]);

            let PreparedRuntime {
                runtime,
                binding,
                published,
                prepared,
                checkpoints,
                ..
            } = prepared_runtime("sum", fail_body, fail_cleanup, true, None);
            let after_revocation = call_outer_route(&runtime, binding, route).unwrap_err();
            assert_eq!(after_revocation.identifier(), primary.identifier());
            assert_eq!(after_revocation.to_string(), primary.to_string());
            assert_eq!(prepared.get(), 1);
            assert_eq!(checkpoints.get(), 1);
            assert!(
                published.borrow().is_empty(),
                "failed fence cannot train a negative sample"
            );
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn compatibility_failure_does_not_publish_a_successful_prepared_body() {
    thread_local! {
        static PUBLISHED: RefCell<Option<Rc<RefCell<Vec<PreparedCallStatus>>>>> = const { RefCell::new(None) };
    }
    fn sparse_return(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
        let sink = invocation.call_observation;
        crate::builtin::admitted_builtin_future(None, "fixture", async move {
            if let Some(right) = sink.and_then(|sink| {
                sink.prepared_right(
                    runmat_builtins::BuiltinBindingIdentity {
                        builtin: runmat_builtins::sum_invocation_entry(
                            runmat_types::SumInvocation::Sum,
                        )
                        .identity,
                        variant: "default",
                    },
                    Some(1),
                )
            }) {
                let binding = right.binding();
                let published = PUBLISHED.with(|slot| slot.borrow().as_ref().unwrap().clone());
                right.deposit(Box::new(Ticket {
                    binding,
                    status: PreparedCallStatus::Succeeded,
                    published,
                }));
            }
            let sparse = SparseTensor::new_integer(
                1,
                1,
                vec![0, 1],
                vec![0],
                IntegerStorage::U64(vec![u64::MAX]),
            )
            .unwrap();
            let sequence = ValueSequence::single(Value::SparseTensor(sparse))
                .map_err(crate::sequence::sequence_error_to_runtime)?;
            crate::execution::RootedValueSequence::new(sequence)
                .map_err(|error| error.into_runtime_error())
        })
    }
    let actual = reduction_binding("sum");
    let selected = RuntimeBuiltinBinding::new(actual.identity, sparse_return);
    let (runtime, _scoped, _placement) = harness(selected);
    let published = Rc::new(RefCell::new(Vec::new()));
    PUBLISHED.with(|slot| *slot.borrow_mut() = Some(published.clone()));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let result =
        futures::executor::block_on(crate::call_builtin_sequence_async_with_outputs_in_context(
            &runtime,
            "sum",
            &arguments(),
            1,
        ));
    assert_eq!(
        result.unwrap_err().identifier(),
        Some("RunMat:compatibility:SparseIntegerExtension")
    );
    assert!(published.borrow().is_empty());
    PUBLISHED.with(|slot| *slot.borrow_mut() = None);
}
