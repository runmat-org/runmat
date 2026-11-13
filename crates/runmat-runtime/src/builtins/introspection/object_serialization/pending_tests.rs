use super::*;
use futures::channel::oneshot;
use futures::task::noop_waker_ref;
use runmat_value::{
    BoundFunctionBinding, FunctionDefinitionOwner, LocalExport, SelectedObjectClassFacts,
    SelectedReceiverClassDeclaration, SelectedReceiverDeclarationSeal,
    SelectedReceiverDeclarations, SelectedReceiverMethodDeclaration,
    SelectedReceiverPropertyDeclaration,
};
use std::cell::Cell;
use std::collections::HashMap;
use std::future::Future;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

const METHOD_ID: usize = 947_802;
const LOAD_METHOD_ID: usize = 947_803;

fn install_pending_saveobj(
    gate: oneshot::Receiver<()>,
    calls: Arc<AtomicUsize>,
) -> (
    crate::user_functions::FunctionResolverGuard,
    crate::user_functions::FunctionInvokerGuard,
) {
    let resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "pending_mat_saveobj").then_some(METHOD_ID)
        })))
        .expect("call-port scope");
    let gate = Arc::new(Mutex::new(Some(gate)));
    let invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, arguments, outputs| {
            assert_eq!(function, METHOD_ID);
            assert_eq!(outputs, 1);
            assert_eq!(arguments.len(), 1);
            calls.fetch_add(1, Ordering::SeqCst);
            let receiver = arguments[0].clone();
            let gate = gate
                .lock()
                .expect("saveobj gate")
                .take()
                .expect("no replay");
            Box::pin(async move {
                gate.await.map_err(|_| {
                    crate::runtime_error::semantic_error(
                        "PendingSaveAbandoned",
                        "saveobj gate abandoned",
                    )
                })?;
                let Value::Object(object) = receiver else {
                    panic!("saveobj receiver must be the typed MAT object");
                };
                let object = object
                    .into_unbounded_builder()
                    .expect("fixture receiver is ordinary");
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Struct(
                    StructValue::from_fields_unbounded(object.into_properties_unbounded()),
                ))
            })
        },
    )))
    .expect("call-port scope");
    (resolver, invoker)
}

fn register_pending_class() {
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "PendingMatSave".into(),
        parent: None,
        properties: Default::default(),
        methods: [(
            "saveobj".into(),
            crate::class_registry::RuntimeMethod {
                name: "saveobj".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "pending_mat_saveobj".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]
        .into(),
    });
}

#[test]
fn typed_mat_object_preparation_retains_pending_callback_receiver_and_retires_on_drop() {
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
            let mut object = runmat_value::UnboundedObjectBuilder::new("PendingMatSave");
            object.insert(
                "child".into(),
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "PendingMatChild".into(),
                    target,
                    valid: true,
                }),
            );
            let (release, gate) = oneshot::channel();
            let calls = Arc::new(AtomicUsize::new(0));
            let (_resolver, _invoker) = install_pending_saveobj(gate, Arc::clone(&calls));
            let mut future = Box::pin(prepare_mat_object_element(
                Value::Object(object.finish()),
                Some(&runtime),
            ));
            for _ in 0..8 {
                let mut context = Context::from_waker(noop_waker_ref());
                assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
                if calls.load(Ordering::SeqCst) == 1 {
                    break;
                }
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1, "saveobj not reached");
            setup
                .unroot()
                .expect("transfer setup root to pending saveobj");
            runmat_gc::gc_collect_major().expect("collect while callback is pending");
            assert!(runmat_gc::gc_clone_value(&target).is_ok());
            if complete {
                release.send(()).expect("resume saveobj");
                let saved = futures::executor::block_on(future).expect("saveobj result");
                assert!(matches!(saved.value(), Value::Struct(payload)
                    if matches!(payload.fields.get("child"), Some(Value::HandleObject(handle)) if handle.target == target)));
                drop(saved);
            } else {
                drop(future);
                assert!(
                    release.send(()).is_err(),
                    "dropped callback retained its gate"
                );
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1, "saveobj replayed");
            runmat_gc::gc_collect_major().expect("collect after final owner retires");
            assert!(runmat_gc::gc_clone_value(&target).is_err());
        }
    });
}

fn selected_load_product(runtime: &crate::context::RuntimeContext) -> Rc<SelectedObjectClassFacts> {
    let class = runmat_types::ClassIdentity::new("PendingMatLoad").expect("class identity");
    let method = LOADOBJ_METHOD.owned();
    let owner = runmat_types::ClassMethodOwner {
        declaring_class: class.clone(),
        method: method.clone(),
        is_static: true,
        role: runmat_types::ClassMethodRole::Ordinary,
    };
    let declarations = Arc::new(
        SelectedReceiverDeclarations::new(
            class.clone(),
            false,
            vec![class.clone()],
            HashMap::from([(
                class.clone(),
                SelectedReceiverClassDeclaration {
                    parent: None,
                    properties: HashMap::from([(
                        runmat_types::MemberName::from("child"),
                        SelectedReceiverPropertyDeclaration {
                            is_static: false,
                            is_constant: false,
                            is_dependent: false,
                            get_access: runmat_types::MemberAccess::Public,
                            set_access: runmat_types::MemberAccess::Public,
                        },
                    )]),
                    events: HashMap::new(),
                },
            )]),
            HashMap::from([(
                owner.clone(),
                Arc::new(SelectedReceiverMethodDeclaration {
                    owner: owner.clone(),
                    is_abstract: false,
                    access: runmat_types::MemberAccess::Public,
                    standard_substruct: false,
                }),
            )]),
        )
        .expect("selected class declarations"),
    );
    let definition = Arc::new(FunctionDefinitionOwner::default());
    let (seal, key) = SelectedReceiverDeclarationSeal::issue();
    assert!(definition.admit_selected_receiver_seal(Arc::clone(&seal)));
    assert!(seal.fill(&key, Arc::clone(&declarations)));
    let binding = |function| {
        BoundFunctionBinding::local(
            function,
            runtime.session_lease(),
            FunctionDefinitionOwner::issue(&definition, function),
            Vec::new(),
            LocalExport::Ephemeral,
        )
    };
    Rc::new(
        SelectedObjectClassFacts::new(
            declarations,
            binding(LOAD_METHOD_ID + 1),
            HashMap::from([(owner, Some(binding(LOAD_METHOD_ID)))]),
        )
        .expect("selected class product"),
    )
}

fn typed_object_record(
    class_name: &str,
    handle: bool,
    payload: crate::builtins::io::mat::format::MatArray,
) -> crate::builtins::io::mat::format::MatArray {
    use crate::builtins::io::mat::format::{MatArray, MatClass, MatData};
    MatArray {
        class: MatClass::Object,
        dims: vec![1, 1],
        data: MatData::Object {
            class_name: class_name.to_owned(),
            field_names: vec![
                "__runmat_saved_kind".into(),
                "__runmat_saved_payload".into(),
            ],
            field_values: vec![
                MatArray {
                    class: MatClass::Logical,
                    dims: vec![1, 1],
                    data: MatData::Logical {
                        data: vec![u8::from(handle)],
                    },
                },
                payload,
            ],
        },
    }
}

fn pending_load_record() -> crate::builtins::io::mat::format::MatArray {
    use crate::builtins::io::mat::format::{MatArray, MatClass, MatData};
    let child = typed_object_record(
        "PendingMatChild",
        true,
        MatArray {
            class: MatClass::Struct,
            dims: vec![1, 1],
            data: MatData::Struct {
                field_names: Vec::new(),
                field_values: Vec::new(),
            },
        },
    );
    typed_object_record(
        "PendingMatLoad",
        false,
        MatArray {
            class: MatClass::Struct,
            dims: vec![1, 1],
            data: MatData::Struct {
                field_names: vec!["child".into()],
                field_values: vec![child],
            },
        },
    )
}

fn pending_load_record_with_later_depth() -> crate::builtins::io::mat::format::MatArray {
    use crate::builtins::io::mat::format::{MatArray, MatClass, MatData};
    let mut later = MatArray {
        class: MatClass::Double,
        dims: vec![1, 1],
        data: MatData::Numeric {
            real: runmat_value::NumericStorage::F64(vec![1.0]),
            imag: None,
        },
    };
    for _ in 0..=MAX_SERIALIZATION_DEPTH {
        later = MatArray {
            class: MatClass::Struct,
            dims: vec![1, 1],
            data: MatData::Struct {
                field_names: vec!["next".into()],
                field_values: vec![later],
            },
        };
    }
    MatArray {
        class: MatClass::Struct,
        dims: vec![1, 1],
        data: MatData::Struct {
            field_names: vec!["first".into(), "later".into()],
            field_values: vec![pending_load_record(), later],
        },
    }
}

#[test]
fn typed_mat_load_keeps_decoded_child_rooted_during_pending_selected_loadobj() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _activation = runtime.enter();
        crate::class_registry::register_builtin_class(
            runmat_types::StaticClassIdentity::new("PendingMatChild"),
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: "PendingMatChild".into(),
                parent: None,
                properties: Default::default(),
                methods: Default::default(),
            },
        );
        let selected = selected_load_product(&runtime);
        runtime.set_current_class_product_resolver(Some(Rc::new(move |runtime, class| {
            let selected = Rc::clone(&selected);
            crate::context::admit_current_class_product_with(&runtime.clone(), move || async move {
                if class.display_name() == "PendingMatLoad" {
                    return Ok(crate::context::CurrentClassProductResolution::Selected(
                        selected,
                    ));
                }
                crate::class_registry::issue_builtin_class_in_context(&runtime, &class)
                    .map(crate::context::CurrentClassProductResolution::RegisteredBuiltin)
                    .ok_or_else(|| {
                        crate::build_runtime_error("test class product is absent").build()
                    })
            })
        })));

        for (complete, later_depth) in [(true, false), (false, false), (true, true)] {
            let (release, gate) = oneshot::channel();
            let calls = Rc::new(Cell::new(0));
            let observed = Rc::clone(&calls);
            let target = Rc::new(Cell::new(None));
            let observed_target = Rc::clone(&target);
            let gate = Rc::new(std::cell::RefCell::new(Some(gate)));
            let callback_gate = Rc::clone(&gate);
            let _invoker = crate::user_functions::install_local_bound_function_invoker(Rc::new(
                move |bound_runtime, binding, arguments, outputs, _| {
                    assert_eq!(binding.function, LOAD_METHOD_ID);
                    assert_eq!(outputs, 1);
                    let Value::Struct(payload) = &arguments.supplied()[0] else {
                        panic!("selected loadobj requires decoded property struct");
                    };
                    let Some(Value::HandleObject(handle)) = payload.fields.get("child") else {
                        panic!("typed child object must be a handle");
                    };
                    observed_target.set(Some(handle.target));
                    observed.set(observed.get() + 1);
                    let gate = callback_gate.borrow_mut().take().expect("one callback");
                    crate::execution::admit_executable_sequence_with(
                        &bound_runtime.clone(),
                        move || async move {
                            gate.await.map_err(|_| {
                                crate::runtime_error::semantic_error(
                                    "PendingLoadAbandoned",
                                    "loadobj gate abandoned",
                                )
                            })?;
                            let Value::Struct(payload) = arguments.supplied()[0].clone() else {
                                panic!("selected loadobj payload changed");
                            };
                            let mut object =
                                runmat_value::UnboundedObjectBuilder::new("PendingMatLoad");
                            for (name, value) in payload.clone().into_fields_unbounded() {
                                object.insert(name, value);
                            }
                            crate::user_functions::test_support::rooted_single_value_sequence(
                                Value::Object(object.finish()),
                            )
                        },
                    )
                },
            ))
            .expect("bound call port");
            let mut future = if later_depth {
                crate::builtins::io::mat::load::restore_typed_mat_entry_for_test(
                    pending_load_record_with_later_depth(),
                    &runtime,
                )
            } else {
                crate::builtins::io::mat::load::restore_typed_mat_array_for_test(
                    pending_load_record(),
                    &runtime,
                )
            };
            for _ in 0..8 {
                let mut context = Context::from_waker(noop_waker_ref());
                assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
                if calls.get() == 1 {
                    break;
                }
            }
            assert_eq!(calls.get(), 1, "selected loadobj not reached");
            let target = target.get().expect("decoded child target");
            runmat_gc::gc_collect_major().expect("collect during typed loadobj");
            assert!(runmat_gc::gc_clone_value(&target).is_ok());
            if complete {
                release.send(()).expect("resume loadobj");
                if later_depth {
                    let error = futures::executor::block_on(future)
                        .expect_err("later structure depth must fail after callback");
                    assert_eq!(error.identifier(), SERIALIZATION_ERROR_RECURSION.identifier);
                } else {
                    let restored =
                        futures::executor::block_on(future).expect("typed object result");
                    assert!(matches!(restored.value(), Value::Object(object)
                        if matches!(object.property("child"), Some(Value::HandleObject(handle)) if handle.target == target)));
                    drop(restored);
                }
            } else {
                drop(future);
                assert!(
                    release.send(()).is_err(),
                    "dropped loadobj retained its gate"
                );
            }
            assert_eq!(calls.get(), 1, "loadobj replayed");
            runmat_gc::gc_collect_major().expect("collect after typed result retires");
            assert!(runmat_gc::gc_clone_value(&target).is_err());
        }
    });
}
