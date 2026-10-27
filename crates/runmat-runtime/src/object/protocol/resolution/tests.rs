use super::*;
use crate::class_registry::{RuntimeClass, RuntimeMethod};
use runmat_value::ObjectInstance;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static CLASS_ID: AtomicU64 = AtomicU64::new(0);

fn class_name(prefix: &str) -> ClassIdentity {
    ClassIdentity::from(format!(
        "{prefix}_{}",
        CLASS_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn method(name: &str, access: MemberAccess, is_static: bool) -> RuntimeMethod {
    RuntimeMethod {
        name: name.into(),
        is_static,
        is_abstract: false,
        is_sealed: false,
        access,
        function_name: format!("protocol_fixture.{name}"),
        implicit_class_argument: None,
        role: runmat_types::ClassMethodRole::Ordinary,
    }
}

fn register(
    name: ClassIdentity,
    parent: Option<ClassIdentity>,
    methods: impl IntoIterator<Item = RuntimeMethod>,
) {
    crate::class_registry::register_class(RuntimeClass {
        events: Default::default(),
        name,
        parent,
        properties: HashMap::new(),
        methods: methods
            .into_iter()
            .map(|method| (method.name.clone(), method))
            .collect(),
    });
}

#[test]
fn resolves_inherited_protocol_to_a_typed_callable_once() {
    let parent = class_name("protocol_parent");
    let child = class_name("protocol_child");
    register(
        parent.clone(),
        None,
        [method("subsref", MemberAccess::Public, false)],
    );
    register(child.clone(), Some(parent.clone()), []);
    let base = Value::Object(ObjectInstance::new(child.clone()));
    let Some(resolved) = resolve_object_protocol(
        &base,
        ObjectProtocol::Subsref,
        &ObjectAccessContext::default(),
    )
    .unwrap() else {
        panic!("expected inherited subsref protocol");
    };
    assert_eq!(resolved.class, child);
    assert_eq!(resolved.declaring_class, parent);
    assert_eq!(resolved.method, MethodName::from("subsref"));
    assert_eq!(
        resolved.convention,
        ObjectProtocolCallingConvention::StandardSubstruct
    );
}

#[test]
fn missing_protocol_selects_default_indexing() {
    let class = class_name("protocol_default");
    register(class.clone(), None, []);
    let base = Value::Object(ObjectInstance::new(class));
    assert_eq!(
        resolve_object_protocol(
            &base,
            ObjectProtocol::Subsref,
            &ObjectAccessContext::default(),
        )
        .unwrap(),
        None
    );
}

#[test]
fn inaccessible_or_static_protocol_is_rejected_during_resolution() {
    for (access, is_static, identifier) in [
        (MemberAccess::Private, false, "RunMat:MethodPrivate"),
        (MemberAccess::Public, true, "RunMat:MethodStaticOnInstance"),
    ] {
        let class = class_name("protocol_invalid");
        register(class.clone(), None, [method("subsasgn", access, is_static)]);
        let base = Value::Object(ObjectInstance::new(class));
        let error = resolve_object_protocol(
            &base,
            ObjectProtocol::Subsasgn,
            &ObjectAccessContext::default(),
        )
        .expect_err("invalid instance protocol must be rejected");
        assert_eq!(error.identifier(), Some(identifier));
    }
}

#[test]
fn method_frame_context_requires_the_exact_current_binding() {
    let class = class_name("method_frame");
    register(
        class.clone(),
        None,
        [method("subsref", MemberAccess::Public, false)],
    );
    let method_name = MethodName::from("subsref");
    let bound =
        crate::class_registry::lookup_bound_method(&class, &method_name).expect("bound method");
    let owner = runmat_types::ClassMethodOwner {
        declaring_class: class.clone(),
        method: method_name,
        is_static: false,
        role: runmat_types::ClassMethodRole::Ordinary,
    };
    let context = ObjectAccessContext::from_method_frame(Some(&owner), &bound.callable)
        .expect("matching frame");
    assert_eq!(context.active_protocol, Some(ObjectProtocol::Subsref));
    assert_eq!(
        context
            .active_method
            .as_ref()
            .map(|active| active.registry_generation),
        Some(bound.registry_generation)
    );
    let mismatch =
        CallableIdentity::DynamicName(runmat_types::SymbolName("different_executable".into()));
    let error = ObjectAccessContext::from_method_frame(Some(&owner), &mismatch)
        .expect_err("mismatched executable must be rejected");
    assert_eq!(
        error.identifier(),
        Some("RunMat:ObjectMethodFrameIdentityMismatch")
    );

    let mut constructor = method("construct", MemberAccess::Public, false);
    constructor.role = runmat_types::ClassMethodRole::Constructor;
    constructor.function_name = format!("{class}.construct");
    register(
        class.clone(),
        None,
        [
            constructor,
            method("inspect", MemberAccess::Public, true),
            method("subsref", MemberAccess::Public, false),
            method("subsasgn", MemberAccess::Public, false),
        ],
    );
    let receiver = Value::Object(ObjectInstance::new(class.clone()));
    let bare_class = ObjectAccessContext {
        caller_class: Some(class.clone()),
        ..Default::default()
    };
    for protocol in [ObjectProtocol::Subsref, ObjectProtocol::Subsasgn] {
        assert!(resolve_object_protocol(&receiver, protocol, &bare_class)
            .unwrap()
            .is_some());
    }
    for (name, is_static, role) in [
        (
            "construct",
            false,
            runmat_types::ClassMethodRole::Constructor,
        ),
        ("inspect", true, runmat_types::ClassMethodRole::Ordinary),
    ] {
        let bound = crate::class_registry::lookup_bound_method(&class, &MethodName::from(name))
            .expect("own class method");
        let owner = runmat_types::ClassMethodOwner {
            declaring_class: class.clone(),
            method: MethodName::from(name),
            is_static,
            role,
        };
        let access = ObjectAccessContext::from_method_frame(Some(&owner), &bound.callable)
            .expect("actual own-class method frame");
        for protocol in [ObjectProtocol::Subsref, ObjectProtocol::Subsasgn] {
            assert!(resolve_object_protocol(&receiver, protocol, &access)
                .unwrap()
                .is_none());
        }
    }
    let legacy =
        ObjectAccessContext::from_legacy_function_name(Some(&format!("{class}.construct")));
    assert!(
        resolve_object_protocol(&receiver, ObjectProtocol::Subsasgn, &legacy)
            .unwrap()
            .is_none()
    );
    let mut replacement = method("construct", MemberAccess::Public, false);
    replacement.role = runmat_types::ClassMethodRole::Constructor;
    replacement.function_name = format!("{class}.replacement");
    register(
        class,
        None,
        [
            replacement,
            method("subsref", MemberAccess::Public, false),
            method("subsasgn", MemberAccess::Public, false),
        ],
    );
    let error = resolve_object_protocol(&receiver, ObjectProtocol::Subsasgn, &legacy)
        .expect_err("ordinary constructor frame cannot bypass after redefinition");
    assert_eq!(error.identifier(), Some("RunMat:StaleObjectMethodFrame"));

    let class = class_name("selected_constructor");
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _runtime = runtime.enter();
    let selected_facts = || {
        let definition = std::sync::Arc::new(runmat_value::FunctionDefinitionOwner::default());
        let binding = runmat_value::BoundFunctionBinding::local(
            1,
            runtime.session_lease(),
            runmat_value::FunctionDefinitionOwner::issue(&definition, 1),
            Vec::new(),
            runmat_value::LocalExport::Ephemeral,
        );
        let projection = crate::class_registry::prepare_selected_class_projection(vec![
            crate::class_registry::CheckedClassRegistration {
                class: RuntimeClass {
                    name: class.clone(),
                    parent: None,
                    properties: HashMap::from([(
                        "x".into(),
                        crate::class_registry::RuntimeProperty {
                            name: "x".into(),
                            is_static: false,
                            is_constant: false,
                            is_dependent: false,
                            get_access: MemberAccess::Private,
                            set_access: MemberAccess::Private,
                            default_value: None,
                        },
                    )]),
                    methods: HashMap::new(),
                    events: HashMap::new(),
                },
                method_bindings: HashMap::new(),
                is_sealed: false,
                is_abstract: false,
                enumerations: Vec::new(),
            },
        ])
        .expect("selected constructor projection");
        let declarations = projection.receiver_declarations(&class).unwrap();
        let (seal, key) = runmat_value::SelectedReceiverDeclarationSeal::issue();
        assert!(definition.admit_selected_receiver_seal(std::sync::Arc::clone(&seal)));
        assert!(seal.fill(&key, std::sync::Arc::clone(&declarations)));
        let facts = std::rc::Rc::new(
            runmat_value::SelectedObjectClassFacts::new(declarations, binding, HashMap::new())
                .expect("selected constructor facts"),
        );
        (projection, facts)
    };
    let (projection, original) = selected_facts();
    let (_, replacement) = selected_facts();
    // Private property access needs the exact active declaration product as
    // well as the issuing session and selected method binding.
    let runtime = runtime.with_selected_class_projection(projection);
    let _runtime = runtime.enter();
    let selected_owner = runmat_types::ClassMethodOwner {
        declaring_class: class.clone(),
        method: MethodName::from("construct"),
        is_static: false,
        role: runmat_types::ClassMethodRole::Constructor,
    };
    let selected_access =
        ObjectAccessContext::from_selected_method_frame(&selected_owner, &original)
            .expect("selected original constructor frame");
    for (facts, matches) in [(original, true), (replacement, false)] {
        let mut object = ObjectInstance::new(class.clone())
            .with_selected_class(facts)
            .expect("selected receiver");
        object.replace_property_unbounded("x".into(), Value::Num(1.0));
        let receiver = Value::Object(object);
        let result = resolve_object_protocol(&receiver, ObjectProtocol::Subsasgn, &selected_access);
        if matches {
            assert!(result.expect("original binding").is_none());
            let denied = futures::executor::block_on(
                crate::object::resolve::store_member_traced_in_context(
                    &runtime,
                    receiver.clone(),
                    "x".into(),
                    Value::Num(3.0),
                    None,
                ),
            )
            .unwrap_err();
            assert_eq!(denied.identifier(), Some("RunMat:PropertyPrivateAccess"));
            let updated = futures::executor::block_on(
                crate::object::resolve::store_member_traced_with_access(
                    &runtime,
                    receiver.clone(),
                    "x".into(),
                    Value::Num(7.0),
                    None,
                    &selected_access,
                ),
            )
            .unwrap();
            let value =
                futures::executor::block_on(crate::object::resolve::load_member_with_access(
                    Some(&runtime),
                    updated.value().clone(),
                    "x".into(),
                    false,
                    None,
                    &selected_access,
                ))
                .unwrap();
            assert_eq!(value.value(), &Value::Num(7.0));
            // The typed proof must survive the default member-to-cell path,
            // not just the first correct protocol resolution.
            use crate::object::indexing::{
                ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
            };
            let cell =
                Value::Cell(runmat_value::CellArray::new(vec![Value::Num(9.0)], 1, 1).unwrap());
            let cell_object = futures::executor::block_on(
                crate::object::resolve::store_member_traced_with_access(
                    &runtime,
                    updated.value().clone(),
                    "x".into(),
                    cell,
                    None,
                    &selected_access,
                ),
            )
            .unwrap();
            let path = ObjectSubscriptPath::new(vec![
                ObjectSubscript::member("x"),
                ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                    components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                        ObjectIndexComponent::Value(Value::Num(1.0)),
                    ]),
                }),
            ])
            .unwrap();
            let selected = futures::executor::block_on(
                crate::object::protocol::read_subscript_path_with_access_in_context(
                    &runtime,
                    cell_object.value().clone(),
                    path,
                    selected_access.clone(),
                ),
            )
            .unwrap();
            assert_eq!(selected.value(), &Value::Num(9.0));
            // A prepared comma-separated destination retains one exact frame
            // through End, assignment cardinality, default prefix and writeback.
            use crate::sequence::{
                AssignmentStepSpec, SequenceDestinationBuilder, SequenceEndpointSpec,
            };
            let captures = std::cell::Cell::new(0);
            let mut builder =
                SequenceDestinationBuilder::new_in_context(&runtime, cell_object.value())
                    .unwrap()
                    .with_method_frame_factory_in_context(&runtime, || {
                        captures.set(captures.get() + 1);
                        Ok(selected_access.clone())
                    })
                    .unwrap();
            futures::executor::block_on(builder.push_step_in_context(
                &runtime,
                AssignmentStepSpec::Member("x".into()),
                None,
            ))
            .unwrap();
            let end_receiver = futures::executor::block_on(
                builder.prepare_current_receiver_for_end_in_context(&runtime, None),
            )
            .unwrap();
            let end = futures::executor::block_on(
                crate::object::protocol::resolve_subscript_end_in_context(
                    &runtime,
                    &end_receiver,
                    0,
                    1,
                ),
            )
            .unwrap();
            assert_eq!(end.value(), &Value::Num(1.0));
            let endpoint = SequenceEndpointSpec::CellContents(
                runmat_value::ExactArrayStorage::shared_unbounded(vec![
                    ObjectIndexComponent::Value(end.value().clone()),
                ]),
            );
            let prepared =
                futures::executor::block_on(builder.finish_in_context(&runtime, endpoint)).unwrap();
            assert_eq!(prepared.cardinality(), 1);
            let updated_cell = futures::executor::block_on(prepared.assign_in_context(
                &runtime,
                cell_object.value().clone(),
                vec![Value::Num(11.0)],
                None,
            ))
            .unwrap();
            let selected =
                futures::executor::block_on(crate::object::resolve::load_member_with_access(
                    Some(&runtime),
                    updated_cell.value().clone(),
                    "x".into(),
                    false,
                    None,
                    &selected_access,
                ))
                .unwrap();
            let Value::Cell(cell) = selected.value() else {
                panic!("retained cell destination");
            };
            assert_eq!(cell.data()[0], Value::Num(11.0));
            assert_eq!(captures.get(), 1);

            // Stored-callable dotted fallback uses that same proof too.
            let callable_object = futures::executor::block_on(
                crate::object::resolve::store_member_traced_with_access(
                    &runtime,
                    updated.value().clone(),
                    "x".into(),
                    Value::FunctionHandle("abs".into()),
                    None,
                    &selected_access,
                ),
            )
            .unwrap();
            let path = ObjectSubscriptPath::new(
                ObjectSubscript::dotted_invoke("x", vec![Value::Num(-3.0)]).into(),
            )
            .unwrap();
            let selected = futures::executor::block_on(
                crate::object::protocol::read_subscript_path_with_access_in_context(
                    &runtime,
                    callable_object.value().clone(),
                    path,
                    selected_access.clone(),
                ),
            )
            .unwrap();
            assert_eq!(selected.value(), &Value::Num(3.0));
        } else {
            assert_eq!(
                result
                    .expect_err("same class and function number are not original binding")
                    .identifier(),
                Some("RunMat:ObjectMethodFrameIdentityMismatch")
            );
        }
    }
}

#[test]
fn resolution_stamp_ignores_unrelated_registration_but_rejects_redefinition() {
    let class = class_name("stamp_owner");
    register(
        class.clone(),
        None,
        [method("subsref", MemberAccess::Public, false)],
    );
    let base = Value::Object(ObjectInstance::new(class.clone()));
    let Some(resolved) = resolve_object_protocol(
        &base,
        ObjectProtocol::Subsref,
        &ObjectAccessContext::default(),
    )
    .expect("resolve") else {
        panic!("expected method");
    };
    register(class_name("unrelated_stamp"), None, []);
    assert!(resolved_method_is_current(&resolved));
    let mut replacement = method("subsref", MemberAccess::Public, false);
    replacement.function_name = "replacement_subsref_fixture".into();
    register(class, None, [replacement]);
    assert!(!resolved_method_is_current(&resolved));
}

#[test]
fn active_inherited_binding_bypasses_only_that_exact_protocol_method() {
    let parent = class_name("active_parent");
    let child = class_name("active_child");
    register(
        parent.clone(),
        None,
        [method("subsref", MemberAccess::Public, false)],
    );
    register(child.clone(), Some(parent.clone()), []);
    let parent_bound =
        crate::class_registry::lookup_bound_method(&parent, &MethodName::from("subsref"))
            .expect("parent binding");
    let owner = runmat_types::ClassMethodOwner {
        declaring_class: parent,
        method: MethodName::from("subsref"),
        is_static: false,
        role: runmat_types::ClassMethodRole::Ordinary,
    };
    let access = ObjectAccessContext::from_method_frame(Some(&owner), &parent_bound.callable)
        .expect("active parent frame");
    let child_value = Value::Object(ObjectInstance::new(child.clone()));
    assert_eq!(
        resolve_object_protocol(&child_value, ObjectProtocol::Subsref, &access).unwrap(),
        None
    );

    register(
        child.clone(),
        Some(owner.declaring_class.clone()),
        [method("subsref", MemberAccess::Public, false)],
    );
    let Some(override_method) =
        resolve_object_protocol(&child_value, ObjectProtocol::Subsref, &access).unwrap()
    else {
        panic!("a distinct subclass override must not inherit the active-method exemption");
    };
    assert_eq!(override_method.declaring_class, child);
}

#[test]
fn protected_access_distinguishes_subclass_and_external_callers() {
    let parent = class_name("protected_parent");
    let child = class_name("protected_child");
    register(
        parent.clone(),
        None,
        [method("subsref", MemberAccess::Protected, false)],
    );
    register(child.clone(), Some(parent.clone()), []);
    let base = Value::Object(ObjectInstance::new(parent));
    let subclass_access = ObjectAccessContext {
        caller_class: Some(child),
        ..ObjectAccessContext::default()
    };
    assert!(
        resolve_object_protocol(&base, ObjectProtocol::Subsref, &subclass_access)
            .unwrap()
            .is_some()
    );
    let error = resolve_object_protocol(
        &base,
        ObjectProtocol::Subsref,
        &ObjectAccessContext::default(),
    )
    .expect_err("external protected access must be rejected");
    assert_eq!(error.identifier(), Some("RunMat:MethodProtected"));
}

#[test]
fn active_protocol_frame_rejects_redefinition_before_dispatch() {
    let class = class_name("active_stale");
    register(
        class.clone(),
        None,
        [method("subsref", MemberAccess::Public, false)],
    );
    let bound = crate::class_registry::lookup_bound_method(&class, &MethodName::from("subsref"))
        .expect("initial binding");
    let owner = runmat_types::ClassMethodOwner {
        declaring_class: class.clone(),
        method: MethodName::from("subsref"),
        is_static: false,
        role: runmat_types::ClassMethodRole::Ordinary,
    };
    let access = ObjectAccessContext::from_method_frame(Some(&owner), &bound.callable)
        .expect("active method frame");
    let mut replacement = method("subsref", MemberAccess::Public, false);
    replacement.function_name = "replacement_active_subsref_fixture".into();
    register(class.clone(), None, [replacement]);
    let error = resolve_object_protocol(
        &Value::Object(ObjectInstance::new(class)),
        ObjectProtocol::Subsref,
        &access,
    )
    .expect_err("stale active binding must fail before dispatch");
    assert_eq!(error.identifier(), Some("RunMat:StaleObjectMethodFrame"));
}
