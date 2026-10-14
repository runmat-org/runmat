use super::*;
use futures::channel::oneshot;
use futures::task::noop_waker_ref;
use runmat_value::HandleRef;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

const METHOD_ID: usize = 947_801;

fn install_pending_loadobj(
    gate: oneshot::Receiver<()>,
    calls: Arc<AtomicUsize>,
) -> (
    crate::user_functions::FunctionResolverGuard,
    crate::user_functions::FunctionInvokerGuard,
) {
    let resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "pending_mat_loadobj").then_some(METHOD_ID)
        })));
    let gate = Arc::new(Mutex::new(Some(gate)));
    let invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, arguments, outputs| {
            assert_eq!(function, METHOD_ID);
            assert_eq!(outputs, 1);
            assert_eq!(arguments.len(), 1);
            calls.fetch_add(1, Ordering::SeqCst);
            let payload = arguments[0].clone();
            let gate = gate
                .lock()
                .expect("loadobj gate")
                .take()
                .expect("no replay");
            Box::pin(async move {
                gate.await.map_err(|_| {
                    crate::runtime_error::semantic_error(
                        "PendingLoadAbandoned",
                        "loadobj gate abandoned",
                    )
                })?;
                crate::user_functions::test_support::rooted_single_value_sequence(payload)
            })
        },
    )));
    (resolver, invoker)
}

fn register_pending_class() {
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "PendingMatLoad".into(),
        parent: None,
        properties: Default::default(),
        methods: [(
            "loadobj".into(),
            crate::class_registry::RuntimeMethod {
                name: "loadobj".into(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "pending_mat_loadobj".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]
        .into(),
    });
}

fn managed_envelope(target: runmat_gc::GcHandle) -> Value {
    let mut payload = StructValue::new();
    payload.insert(
        "child",
        Value::HandleObject(HandleRef {
            class_name: "PendingMatChild".into(),
            target,
            valid: true,
        }),
    );
    serialized_object_envelope(
        "PendingMatLoad".into(),
        SERIALIZED_KIND_VALUE,
        true,
        Value::Struct(payload),
    )
}

fn poll_to_method_gate(
    future: &mut std::pin::Pin<Box<impl Future<Output = crate::BuiltinResult<RootedValue>>>>,
    calls: &AtomicUsize,
) {
    for _ in 0..8 {
        let mut context = Context::from_waker(noop_waker_ref());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        if calls.load(Ordering::SeqCst) == 1 {
            for _ in 0..2 {
                assert!(future.as_mut().poll(&mut context).is_pending());
                assert_eq!(calls.load(Ordering::SeqCst), 1, "loadobj replayed");
            }
            return;
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1, "loadobj not reached");
}

#[test]
fn pending_loadobj_keeps_managed_payload_live_and_retires_on_completion_or_drop() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _activation = runtime.enter();
        register_pending_class();

        for complete in [true, false] {
            let setup = runmat_gc::gc_allocate_rooted(Value::String("pending child".into()))
                .expect("allocate managed child");
            let target = setup.handle();
            let (release, gate) = oneshot::channel();
            let calls = Arc::new(AtomicUsize::new(0));
            let (_resolver, _invoker) = install_pending_loadobj(gate, Arc::clone(&calls));
            let mut future = Box::pin(restore_value_after_load(managed_envelope(target)));
            poll_to_method_gate(&mut future, &calls);
            setup
                .unroot()
                .expect("transfer setup root to pending restoration");
            runmat_gc::gc_collect_major().expect("collect while loadobj is pending");
            assert_eq!(
                runmat_gc::gc_clone_value(&target).expect("pending payload remains rooted"),
                Value::String("pending child".into())
            );
            if complete {
                release.send(()).expect("resume loadobj");
                let restored = futures::executor::block_on(future).expect("loadobj result");
                let Value::Struct(payload) = restored.value() else {
                    panic!("loadobj must return its exact structure payload");
                };
                assert!(matches!(
                    payload.fields.get("child"),
                    Some(Value::HandleObject(handle)) if handle.target == target
                ));
                runmat_gc::gc_collect_major().expect("collect with completed output");
                assert!(runmat_gc::gc_clone_value(&target).is_ok());
                drop(restored);
            } else {
                drop(future);
                assert!(
                    release.send(()).is_err(),
                    "dropped loadobj retained its gate"
                );
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1, "loadobj replayed");
            runmat_gc::gc_collect_major().expect("collect after final owner retires");
            assert!(runmat_gc::gc_clone_value(&target).is_err());
        }
    });
}

#[test]
fn pending_loadobj_effect_precedes_a_later_depth_error_without_replay() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _activation = runtime.enter();
        register_pending_class();

        let setup = runmat_gc::gc_allocate_rooted(Value::String("ordered child".into()))
            .expect("allocate ordered child");
        let target = setup.handle();
        let mut too_deep = Value::Num(1.0);
        for _ in 0..=MAX_SERIALIZATION_DEPTH {
            let mut nested = StructValue::new();
            nested.insert("next", too_deep);
            too_deep = Value::Struct(nested);
        }
        let mut ordered = StructValue::new();
        ordered.insert("first", managed_envelope(target));
        ordered.insert("later", too_deep);

        let (release, gate) = oneshot::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let (_resolver, _invoker) = install_pending_loadobj(gate, Arc::clone(&calls));
        let mut future = Box::pin(restore_value_after_load(Value::Struct(ordered)));
        poll_to_method_gate(&mut future, &calls);
        setup
            .unroot()
            .expect("transfer setup root to ordered restoration");
        runmat_gc::gc_collect_major().expect("collect while first field is pending");
        assert!(runmat_gc::gc_clone_value(&target).is_ok());

        release.send(()).expect("complete first field loadobj");
        let error = futures::executor::block_on(future).expect_err("later depth must fail");
        assert_eq!(error.identifier(), SERIALIZATION_ERROR_RECURSION.identifier);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "first field replayed");
        runmat_gc::gc_collect_major().expect("collect after ordered error");
        assert!(runmat_gc::gc_clone_value(&target).is_err());
    });
}
