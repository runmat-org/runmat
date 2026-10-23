use super::{
    is_possible_graphics_handle_value, load_member, load_static_member,
    read_member_sequence_with_context, store_member,
};
use runmat_types::{ClassIdentity, MemberAccess};
use runmat_value::{IntValue, ObjectArray, ObjectInstance, UnboundedObjectBuilder, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn static_callable_fallback_observes_canonical_and_selected_runtime_inventory() {
    use crate::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
    struct Empty;
    impl RuntimeBuiltinService for Empty {
        fn bindings_by_name(&self, _: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
            Vec::new()
        }
    }
    // This actual namespace callable has a canonical binding, not legacy
    // descriptor metadata. Looking it up must not execute its filesystem body.
    assert!(!runmat_builtins::builtin_functions()
        .iter()
        .any(|entry| entry.name == "data.open"));
    let class = ClassIdentity::from("data");
    let Value::Closure(target) = load_static_member(&class, "open", None).unwrap() else {
        panic!("canonical callable closure");
    };
    assert_eq!(target.function_name.as_str(), "data.open");
    let selected = RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(std::rc::Rc::new(Empty)));
    let _scope = selected.enter();
    assert!(
        load_static_member(&class, "open", None).is_err(),
        "a selected inventory must not fall back to global registrations"
    );
}

#[test]
fn graphics_handle_predicate_preserves_wide_integer_sign() {
    assert!(is_possible_graphics_handle_value(&Value::Int(
        IntValue::U64(u64::MAX)
    )));
    assert!(!is_possible_graphics_handle_value(&Value::Int(
        IntValue::I64(-1)
    )));
}

fn unique_class_name(prefix: &str) -> ClassIdentity {
    let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
    ClassIdentity::from(format!("{}_{}", prefix, id))
}

#[test]
fn object_array_member_access_returns_comma_separated_values_in_order() {
    let values = ["first", "second"]
        .into_iter()
        .map(|name| {
            let mut object = UnboundedObjectBuilder::new("matlab.unittest.TestResult");
            object.insert("Name".into(), Value::String(name.into()));
            Value::Object(object.finish())
        })
        .collect();
    let array = Value::ObjectArray(ObjectArray::row("matlab.unittest.TestResult", values).unwrap());
    let loaded = futures::executor::block_on(read_member_sequence_with_context(
        None,
        array,
        "Name".into(),
        false,
        None,
    ))
    .unwrap();
    assert_eq!(
        loaded.as_slice(),
        &[
            Value::String("first".into()),
            Value::String("second".into())
        ]
    );
}

#[test]
fn load_static_member_resolves_inherited_static_property_value() {
    let parent_name = unique_class_name("vm_static_parent");
    let child_name = unique_class_name("vm_static_child");

    let mut parent_properties = HashMap::new();
    parent_properties.insert(
        "version".into(),
        crate::class_registry::RuntimeProperty {
            name: "version".into(),
            is_static: true,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Num(1.0)),
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: parent_name.clone(),
        parent: None,
        properties: parent_properties,
        methods: HashMap::new(),
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: child_name.clone(),
        parent: Some(parent_name.clone()),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });

    crate::class_registry::set_static_property_value(&parent_name, "version", Value::Num(3.0));
    let value = load_static_member(&child_name, "version", None)
        .expect("inherited static property should resolve through parent metadata owner");
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn store_member_updates_inherited_static_property_owner_slot() {
    let parent_name = unique_class_name("vm_store_static_parent");
    let child_name = unique_class_name("vm_store_static_child");

    let mut parent_properties = HashMap::new();
    parent_properties.insert(
        "version".into(),
        crate::class_registry::RuntimeProperty {
            name: "version".into(),
            is_static: true,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Num(1.0)),
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: parent_name.clone(),
        parent: None,
        properties: parent_properties,
        methods: HashMap::new(),
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: child_name.clone(),
        parent: Some(parent_name.clone()),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });

    let out = futures::executor::block_on(store_member(
        Value::ClassRef(child_name.clone().into()),
        "version".to_string(),
        Value::Num(9.0),
        None,
        |_old, _new| {},
    ))
    .expect("storing inherited static property via child class ref should succeed");
    assert_eq!(out, Value::ClassRef(child_name.into()));
    assert_eq!(
        crate::class_registry::get_static_property_value(&parent_name, "version"),
        Some(Value::Num(9.0))
    );
}

#[test]
fn load_static_member_resolves_inherited_static_method() {
    let parent_name = unique_class_name("vm_method_parent");
    let child_name = unique_class_name("vm_method_child");

    let mut parent_methods = HashMap::new();
    parent_methods.insert(
        "build".into(),
        crate::class_registry::RuntimeMethod {
            name: "build".into(),
            is_static: true,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "build_impl".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: parent_name.clone(),
        parent: None,
        properties: HashMap::new(),
        methods: parent_methods,
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: child_name.clone(),
        parent: Some(parent_name.clone()),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });

    let value = load_static_member(&child_name, "build", None)
        .expect("inherited static method should resolve through parent metadata");
    let Value::Closure(closure) = value else {
        panic!("expected static method lookup to return closure");
    };
    assert_eq!(closure.function_name, "build_impl");
}

#[test]
fn load_member_uses_inherited_subsref_for_missing_property() {
    let parent_name = unique_class_name("vm_subsref_parent");
    let child_name = unique_class_name("vm_subsref_child");

    let mut parent_methods = HashMap::new();
    parent_methods.insert(
        "subsref".into(),
        crate::class_registry::RuntimeMethod {
            name: "subsref".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "OverIdx.subsref".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: parent_name.clone(),
        parent: None,
        properties: HashMap::new(),
        methods: parent_methods,
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: child_name.clone(),
        parent: Some(parent_name),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });

    let obj = Value::Object(ObjectInstance::new(child_name));
    let value = futures::executor::block_on(load_member(obj, "missing".to_string(), false, None))
        .expect("missing member should dispatch to inherited subsref");
    assert_eq!(value, Value::Num(77.0));
}

#[test]
fn store_member_uses_inherited_subsasgn_for_missing_property() {
    let parent_name = unique_class_name("vm_subsasgn_parent");
    let child_name = unique_class_name("vm_subsasgn_child");

    let mut parent_methods = HashMap::new();
    parent_methods.insert(
        "subsasgn".into(),
        crate::class_registry::RuntimeMethod {
            name: "subsasgn".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "OverIdx.subsasgn".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: parent_name.clone(),
        parent: None,
        properties: HashMap::new(),
        methods: parent_methods,
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: child_name.clone(),
        parent: Some(parent_name),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });

    let out = futures::executor::block_on(store_member(
        Value::Object(ObjectInstance::new(child_name)),
        "missing".to_string(),
        Value::Num(13.0),
        None,
        |_old, _new| {},
    ))
    .expect("missing member store should dispatch to inherited subsasgn");
    let Value::Object(obj) = &*out else {
        panic!("expected object result from inherited subsasgn");
    };
    assert_eq!(obj.property("missing"), Some(&Value::Num(13.0)));
}
