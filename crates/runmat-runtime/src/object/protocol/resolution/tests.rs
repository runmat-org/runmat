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
    }
}

fn register(
    name: ClassIdentity,
    parent: Option<ClassIdentity>,
    methods: impl IntoIterator<Item = RuntimeMethod>,
) {
    crate::class_registry::register_class(RuntimeClass {
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
    let ProtocolResolution::Method(resolved) = resolve_object_protocol(
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
        ProtocolResolution::DefaultIndexing
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
    let ProtocolResolution::Method(resolved) = resolve_object_protocol(
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
    };
    let access = ObjectAccessContext::from_method_frame(Some(&owner), &parent_bound.callable)
        .expect("active parent frame");
    let child_value = Value::Object(ObjectInstance::new(child.clone()));
    assert_eq!(
        resolve_object_protocol(&child_value, ObjectProtocol::Subsref, &access).unwrap(),
        ProtocolResolution::DefaultIndexing
    );

    register(
        child.clone(),
        Some(owner.declaring_class.clone()),
        [method("subsref", MemberAccess::Public, false)],
    );
    let ProtocolResolution::Method(override_method) =
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
    assert!(matches!(
        resolve_object_protocol(&base, ObjectProtocol::Subsref, &subclass_access).unwrap(),
        ProtocolResolution::Method(_)
    ));
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
