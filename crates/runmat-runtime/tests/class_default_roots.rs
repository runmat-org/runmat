//! Run separately to prove class defaults install their own GC root provider.
use runmat_runtime::class_registry::{register_class, RuntimeClass, RuntimeProperty};
use runmat_types::MemberAccess;
use runmat_value::{HandleRef, Value};

#[test]
fn class_default_is_the_sole_owner_before_any_static_setter() {
    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::String("class-default".into())).unwrap();
        let handle = initial.handle();
        let property = RuntimeProperty {
            name: "payload".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::HandleObject(HandleRef {
                class_name: "DefaultPayload".into(),
                target: handle,
                valid: true,
            })),
        };
        register_class(RuntimeClass {
            events: Default::default(),
            name: "SoleDefaultOwner".into(),
            parent: None,
            properties: [(property.name.clone(), property)].into(),
            methods: Default::default(),
        });
        initial.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).unwrap(),
            Value::String("class-default".into())
        );
    });
}
