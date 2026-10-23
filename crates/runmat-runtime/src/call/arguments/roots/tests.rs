use super::*;
use crate::call::arguments::{expand_arguments, MaterializedArgument, MaterializedExpansionSource};
use crate::class_registry::{RuntimeClass, RuntimeMethod};
use futures::{channel::oneshot, executor::block_on, task::noop_waker};
use runmat_value::{HandleRef, ObjectInstance, ValueSequence};
use std::{
    cell::Cell,
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

fn managed(label: &str) -> (Value, GcHandle) {
    let initial = runmat_gc::gc_allocate_rooted(Value::String(label.into())).unwrap();
    let handle = initial.handle();
    let value = Value::HandleObject(HandleRef {
        class_name: "ExpansionPayload".into(),
        target: handle,
        valid: true,
    });
    initial.unroot().unwrap();
    (value, handle)
}

#[test]
fn compact_roots_trace_new_children_without_cloning_host_payloads() {
    runmat_gc::gc_test_context(|| {
        let (value, parent) = managed("parent");
        let roots = ExpansionRoots::new().unwrap();
        roots.retain_iter(std::iter::once(&value));
        let (child, child_handle) = managed("new child");
        runmat_gc::gc_with_value_mut(&parent, |value| *value = child).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&child_handle, |_| ()).is_ok());
        let tensor =
            Value::Tensor(runmat_value::Tensor::new(vec![3.0; 65_536], vec![256, 256]).unwrap());
        roots.retain_iter(std::iter::once(&tensor));
        assert_eq!(
            roots.handles.borrow().values.len(),
            1,
            "only the managed parent is retained"
        );
        drop(roots);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&parent, |_| ()).is_err());
        assert!(runmat_gc::gc_with_value(&child_handle, |_| ()).is_err());
    });
}

#[derive(Clone, Copy)]
enum Completion {
    Success,
    Error,
    Cancel,
}

#[test]
fn all_materialized_argument_carriers_keep_unconsumed_managed_inputs() {
    runmat_gc::gc_test_context(|| {
        let roots = ExpansionRoots::new().unwrap();
        let mut identities = Vec::new();
        let mut next = || {
            let (value, handle) = managed("argument carrier");
            identities.push(handle);
            value
        };
        let arguments = vec![
            MaterializedArgument::Single(next()),
            MaterializedArgument::Sequence(
                crate::execution::RootedValueSequence::new(ValueSequence::single(next()).unwrap())
                    .unwrap(),
            ),
            MaterializedArgument::Expansion(MaterializedExpansionSource::ReturnedOutputs(
                crate::execution::RootedValueSequence::new(ValueSequence::single(next()).unwrap())
                    .unwrap(),
            )),
            MaterializedArgument::Expansion(MaterializedExpansionSource::CellContents {
                base: next(),
                indices: vec![next()].into(),
                expand_all: false,
            }),
            MaterializedArgument::Expansion(MaterializedExpansionSource::Member {
                base: next(),
                member: "payload".into(),
            }),
            MaterializedArgument::Expansion(MaterializedExpansionSource::DynamicMember {
                base: next(),
                member: next(),
            }),
        ];
        for argument in &arguments {
            argument.trace(&mut *roots.handles.borrow_mut());
        }
        drop(arguments);
        runmat_gc::gc_collect_major().unwrap();
        for handle in &identities {
            assert!(runmat_gc::gc_with_value(handle, |_| ()).is_ok());
        }
        drop(roots);
        runmat_gc::gc_collect_major().unwrap();
        for handle in &identities {
            assert!(runmat_gc::gc_with_value(handle, |_| ()).is_err());
        }
    });
}

fn expansion_lifecycle(completion: Completion) {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let calls = Rc::new(Cell::new(0));
        let fresh = Rc::new(RefCell::new(None));
        let (sender, receiver) = oneshot::channel::<()>();
        let receiver = Rc::new(RefCell::new(Some(receiver)));
        let (unconsumed, original) = managed("unconsumed input");
        let observed = fresh.clone();
        let count = calls.clone();
        let mut future = Box::pin(runtime.scope(async {
            let _resolver =
                crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                    (name == "expansion_root_subsref").then_some(91)
                }))).expect("call-port scope");
            let _invoker = crate::user_functions::install_local_semantic_function_invoker(Rc::new(
                move |_runtime, function, _, _| {
                    assert_eq!(function, 91);
                    let call = count.get();
                    count.set(call + 1);
                    let fresh = observed.clone();
                    let receiver = receiver.clone();
                    crate::user_functions::admit_user_function_with(_runtime.as_ref(), move || async move {
                        if call == 0 {
                            let (value, handle) = managed("fresh method result");
                            *fresh.borrow_mut() = Some(handle);
                            return crate::user_functions::test_support::rooted_single_value_sequence(value);
                        }
                        assert_eq!(call, 1);
                        // The first result has left its producer; only the runtime
                        // expansion owner can keep it alive across this safepoint.
                        runmat_gc::gc_collect_major().unwrap();
                        assert!(runmat_gc::gc_with_value(&fresh.borrow().unwrap(), |_| ()).is_ok());
                        assert!(runmat_gc::gc_with_value(&original, |_| ()).is_ok());
                        let pending = receiver.borrow_mut().take().unwrap();
                        pending.await.expect("continuation gate");
                        match completion {
                            Completion::Error => {
                                Err(crate::build_runtime_error("second expansion failed").build())
                            }
                            _ => crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(7.0)),
                        }
                    })
                },
            )).expect("call-port scope");
            let method = RuntimeMethod {
                name: "subsref".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "expansion_root_subsref".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            };
            crate::class_registry::register_class(RuntimeClass {
                events: Default::default(),
                name: "ExpansionRootFixture".into(),
                parent: None,
                properties: Default::default(),
                methods: [(method.name.clone(), method)].into(),
            });
            let expansion = || {
                MaterializedArgument::Expansion(MaterializedExpansionSource::Member {
                    base: Value::Object(ObjectInstance::new("ExpansionRootFixture")),
                    member: "value".into(),
                })
            };
            expand_arguments(
                &runtime,
                vec![
                    expansion(),
                    expansion(),
                    MaterializedArgument::Single(unconsumed),
                ],
            )
            .await
        }));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(&noop_waker())),
            Poll::Pending
        ));
        assert_eq!(calls.get(), 2);
        let fresh = fresh.borrow().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&fresh, |_| ()).is_ok());
        assert!(runmat_gc::gc_with_value(&original, |_| ()).is_ok());
        if matches!(completion, Completion::Cancel) {
            drop(future);
        } else {
            sender.send(()).unwrap();
            let result = block_on(future);
            match completion {
                Completion::Success => {
                    let values = result.unwrap();
                    assert_eq!(values.len(), 3);
                    assert_eq!(values[1], Value::Num(7.0));
                    assert!(runmat_gc::gc_with_value(&fresh, |_| ()).is_ok());
                    drop(values);
                }
                Completion::Error => assert!(result
                    .unwrap_err()
                    .message()
                    .contains("second expansion failed")),
                Completion::Cancel => unreachable!(),
            }
        }
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&fresh, |_| ()).is_err());
        assert!(runmat_gc::gc_with_value(&original, |_| ()).is_err());
    });
}

#[test]
fn expanded_fresh_and_unconsumed_values_survive_pending_then_success() {
    expansion_lifecycle(Completion::Success);
}
#[test]
fn expanded_values_release_roots_on_later_error() {
    expansion_lifecycle(Completion::Error);
}
#[test]
fn expanded_values_release_roots_on_pending_drop() {
    expansion_lifecycle(Completion::Cancel);
}

#[test]
fn expansion_result_survives_collecting_source_registration_retirement() {
    thread_local! {
        static OBSERVED_HANDLE: Cell<Option<GcHandle>> = const { Cell::new(None) };
    }
    runmat_gc::gc_test_context(|| {
        let (value, handle) = managed("returned argument");
        OBSERVED_HANDLE.with(|selected| selected.set(Some(handle)));
        let observed = Arc::new(AtomicBool::new(false));
        let callback_observed = Arc::clone(&observed);
        let output_root_retired = Arc::new(AtomicBool::new(false));
        let callback_root_retired = Arc::clone(&output_root_retired);
        let output_returned = Arc::new(AtomicBool::new(false));
        let callback_output_returned = Arc::clone(&output_returned);
        let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
            let Some(handle) = OBSERVED_HANDLE.with(Cell::get) else {
                return;
            };
            if event != runmat_gc::RootRetirement::Explicit(handle) {
                return;
            }
            if callback_output_returned.load(Ordering::SeqCst) {
                callback_root_retired.store(true, Ordering::SeqCst);
                return;
            }
            // The canonical raw-input lease now retains exact handle roots,
            // not ExpansionRoots' old scanner registration. Collect at the
            // actual source-lease retirement before the result is returned.
            if callback_observed.swap(true, Ordering::SeqCst) {
                return;
            }
            runmat_gc::gc_collect_major().expect("collect during temporary-root retirement");
            assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_ok());
        }))
        .expect("register collector observer");
        let arguments = block_on(expand_arguments(
            &crate::context::RuntimeContext::new(Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            )),
            vec![MaterializedArgument::Single(value)],
        ))
        .expect("return rooted argument batch");
        output_returned.store(true, Ordering::SeqCst);
        assert!(observed.load(Ordering::SeqCst));
        assert!(!output_root_retired.load(Ordering::SeqCst));
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_ok());
        drop(arguments);
        assert!(output_root_retired.load(Ordering::SeqCst));
        assert_ne!(
            runmat_gc::gc_is_reachable_excluding(handle, &[]),
            Some(true),
            "no registered value root may retain the completed argument"
        );
        let major_collections = runmat_gc::gc_stats()
            .major_collections
            .load(Ordering::Relaxed);
        runmat_gc::gc_collect_major().expect("collect after result retirement");
        // Collection can defer while another test thread has registered VM
        // roots; root retirement and available reachability are the invariant.
        if runmat_gc::gc_stats()
            .major_collections
            .load(Ordering::Relaxed)
            > major_collections
        {
            assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
        }
        OBSERVED_HANDLE.with(|selected| selected.set(None));
    });
}
