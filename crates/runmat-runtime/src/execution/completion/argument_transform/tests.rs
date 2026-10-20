use super::*;
use crate::call::BuiltinArguments;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_value::{HandleRef, Value};
use std::rc::Rc;

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn runtime(capacity: u64) -> (RuntimeContext, ManagedHostEffectLedger, CancellationSource) {
    let ledger = ManagedHostEffectLedger::new_with_epoch(capacity, 1);
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    (runtime, ledger, cancellation)
}

#[test]
fn final_slots_retain_original_payload_and_roots_until_last_fork_retires() {
    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::Num(29.)).unwrap();
        let handle = initial.handle();
        let text: runmat_value::Utf8Text = "original bytes".into();
        let pointer = text.as_str().as_ptr();
        let arguments: BuiltinArguments = RootedValues::new(vec![
            Value::String(text),
            Value::HandleObject(HandleRef {
                class_name: "TransformFixture".into(),
                target: handle,
                valid: true,
            }),
        ])
        .unwrap()
        .into();
        initial.unroot().unwrap();
        let (runtime, ledger, _) = runtime(1 << 20);
        let inserted = RootedValue::in_context(Value::Num(7.), &runtime).unwrap();
        let transformed = arguments
            .try_insert_in_context(1, inserted, &runtime)
            .unwrap();
        assert_eq!(transformed.len(), 3);
        let Value::String(text) = &transformed[0] else {
            panic!("retained text")
        };
        assert_eq!(text.as_str().as_ptr(), pointer);
        assert_eq!(transformed[1], Value::Num(7.));
        let retained_bytes = ledger.snapshot().live_reserved_bytes;
        assert!(retained_bytes > 0);
        let original_slots = transformed
            .fork()
            .into_function_arguments()
            .supplied()
            .as_ptr();
        let alias = futures::executor::block_on(crate::dispatcher::gather_builtin_arguments_async(
            transformed.fork(),
            Some(&runtime),
        ))
        .unwrap();
        assert_eq!(
            alias.fork().into_function_arguments().supplied().as_ptr(),
            original_slots,
            "host-only gather must retain the original argument slots",
        );
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained_bytes);
        drop(transformed);
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained_bytes);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[test]
fn denied_or_cancelled_transform_preserves_independent_source_owner() {
    let source: BuiltinArguments = RootedValues::new(vec![Value::Num(1.)]).unwrap().into();
    for capacity in [0, 1 << 20] {
        let (runtime, ledger, cancellation) = runtime(capacity);
        if capacity != 0 {
            cancellation.cancel(CancellationReason::User);
        }
        let inserted = RootedValue::new(Value::Num(2.)).unwrap();
        let error = source
            .fork()
            .try_insert_in_context(0, inserted, &runtime)
            .unwrap_err();
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        let mut matched = false;
        while let Some(source) = cause {
            if let Some(host) = source.downcast_ref::<HostAllocationError>() {
                matched = capacity == 0 && host.is_capacity_denial();
                if matched {
                    break;
                }
            }
            if capacity != 0
                && matches!(
                    source.downcast_ref::<crate::execution::ExecutionServiceError>(),
                    Some(crate::execution::ExecutionServiceError::Cancelled)
                )
            {
                matched = true;
                break;
            }
            cause = source.source();
        }
        assert!(
            matched,
            "preserved typed capacity/cancellation cause: {error}"
        );
        assert_eq!(source[0], Value::Num(1.));
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn async_projection_lends_one_mutable_cooperation_state_across_each_await() {
    let source: BuiltinArguments = RootedValues::new(vec![Value::Num(1.), Value::Num(2.)])
        .unwrap()
        .into();
    let (runtime, ledger, _) = runtime(1 << 20);
    let mut completed = 0usize;
    let output = futures::executor::block_on(source.try_map_preserving_roots_async(
        source.len(),
        Some(&runtime),
        async |index| {
            let mut first_poll = true;
            futures::future::poll_fn(|context| {
                if first_poll {
                    first_poll = false;
                    context.waker().wake_by_ref();
                    std::task::Poll::Pending
                } else {
                    std::task::Poll::Ready(())
                }
            })
            .await;
            completed += 1;
            Ok(source[index].clone())
        },
    ))
    .unwrap();
    assert_eq!(completed, 2);
    assert!(output.iter().eq(source.iter()));
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn cancellation_during_projection_wins_and_releases_partial_destination() {
    let source: BuiltinArguments = RootedValues::new(vec![Value::Num(1.)]).unwrap().into();
    let (runtime, ledger, cancellation) = runtime(1 << 20);
    let error = futures::executor::block_on(source.try_map_preserving_roots_async(
        1,
        Some(&runtime),
        |_| async {
            cancellation.cancel(CancellationReason::User);
            Err(crate::runtime_error::semantic_error(
                "FixtureDomainFailure",
                "cancel wins",
            ))
        },
    ))
    .unwrap_err();
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(source[0], Value::Num(1.));
}

#[test]
fn mixed_materialization_roots_new_edges_before_next_projection() {
    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::Num(31.)).unwrap();
        let handle = initial.handle();
        let setup = std::cell::RefCell::new(Some(initial));
        let class_name = runmat_value::HandleRef {
            class_name: "MixedFixture".into(),
            target: handle,
            valid: true,
        };
        let source: BuiltinArguments = RootedValues::new(vec![Value::Num(1.), Value::Num(2.)])
            .unwrap()
            .into();
        let (runtime, ledger, _) = runtime(1 << 20);
        let output =
            futures::executor::block_on(source.try_map_mixed_async(2, Some(&runtime), |slot| {
                let source = &source;
                let runtime = &runtime;
                let setup = &setup;
                let class_name = &class_name;
                async move {
                    if slot == 0 {
                        let rooted = RootedValue::in_context(
                            Value::HandleObject(class_name.clone()),
                            runtime,
                        )?;
                        setup.borrow_mut().take().unwrap().unroot().unwrap();
                        Ok(ArgumentProjection::Rooted(rooted))
                    } else {
                        runmat_gc::gc_collect_major().unwrap();
                        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                        Ok(ArgumentProjection::Source(source[slot].clone()))
                    }
                }
            }))
            .unwrap();
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        assert_eq!(output[1], Value::Num(2.));
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
