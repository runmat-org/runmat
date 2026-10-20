use super::lifecycle::delete_metadata_property;
use super::metadata::*;
use super::operations::*;
use super::*;
use futures::executor::block_on;
use runmat_builtins::{
    BuiltinIntegerAuditKind, ADDPROP_INTEGER_AUDIT, DYNAMIC_ERROR_INVALID_TARGET,
    DYNAMIC_PROPERTY_DELETE_INTEGER_AUDIT, FINDPROP_INTEGER_AUDIT,
};
use runmat_value::{HandleRef, IntValue, ObjectInstance, Value};
use std::collections::HashMap;

#[test]
fn addprop_descriptor_is_integer_inapplicable() {
    assert_eq!(
        ADDPROP_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    assert!(ADDPROP_INTEGER_AUDIT.canonical_builtin.is_none());
}

#[test]
fn dynamic_property_delete_is_integer_inapplicable() {
    assert_eq!(
        DYNAMIC_PROPERTY_DELETE_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    assert!(DYNAMIC_PROPERTY_DELETE_INTEGER_AUDIT
        .canonical_builtin
        .is_none());
}

#[test]
fn metadata_bool_accepts_only_exact_typed_logical_values() {
    assert!(metadata_bool(&Value::Int(IntValue::U64(1)), "Hidden").expect("logical one"));
    assert!(!metadata_bool(&Value::Int(IntValue::U64(0)), "Hidden").expect("logical zero"));
    assert!(metadata_bool(&Value::Int(IntValue::U64(u64::MAX)), "Hidden").is_err());
    assert!(metadata_bool(&Value::Int(IntValue::I64(-1)), "Hidden").is_err());
}

fn target_handle(class_name: &str) -> (Value, runmat_gc::ExplicitRoot) {
    let obj = ObjectInstance::new(class_name.to_string());
    let root = runmat_gc::gc_allocate_rooted(Value::Object(obj)).expect("gc");
    let value = Value::HandleObject(HandleRef {
        class_name: class_name.into(),
        target: root.handle(),
        valid: true,
    });
    (value, root)
}

fn dynamic_target_handle(class_name: &str) -> (Value, runmat_gc::ExplicitRoot) {
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: class_name.into(),
        parent: Some(DYNAMICPROPS_CLASS.into()),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });
    target_handle(class_name)
}

#[test]
fn addprop_duplicate_and_excess_outputs_leave_existing_metadata_unchanged() {
    let (target, _root) = dynamic_target_handle("DynPublicationGuard");
    let first_root =
        block_on(addprop_builtin(None, target.clone(), "gain".into())).expect("first property");
    let first = first_root.value().clone();
    let Value::HandleObject(first_handle) = first else {
        panic!("metadata handle")
    };
    let duplicate = block_on(addprop_builtin(None, target.clone(), "gain".into())).unwrap_err();
    assert_eq!(
        duplicate.identifier(),
        runmat_builtins::DYNAMIC_ERROR_DUPLICATE.identifier
    );
    let found_root =
        findprop_builtin(None, target.clone(), "gain".into()).expect("existing metadata");
    let found = found_root.value().clone();
    let Value::HandleObject(found_handle) = found else {
        panic!("metadata alias")
    };
    assert_eq!(found_handle.target, first_handle.target);
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = block_on(addprop_builtin(None, target.clone(), "other".into())).unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::ADDPROP_ERROR_OUTPUTS.identifier
        );
    }
    assert!(matches!(
        findprop_builtin(None, target, "other".into()).expect("no publication").value(),
        Value::Tensor(tensor) if tensor.is_empty()
    ));
}

#[test]
fn qualified_delete_rejects_requested_output_before_removing_property() {
    let (target, _root) = dynamic_target_handle("DynDeleteOutputGuard");
    let prop_root =
        block_on(addprop_builtin(None, target.clone(), "gain".into())).expect("property");
    let prop = prop_root.value().clone();
    {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let error = block_on(dynamic_property_delete_builtin(None, prop.clone())).unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::DYNAMIC_PROPERTY_DELETE_ERROR_OUTPUTS.identifier
        );
    }
    let Value::HandleObject(handle) = prop else {
        panic!("metadata")
    };
    assert!(crate::is_handle_valid(&handle));
    assert!(matches!(
        findprop_builtin(None, target, "gain".into())
            .expect("property retained")
            .value(),
        Value::HandleObject(_)
    ));
}

#[test]
fn delete_keeps_both_objects_live_during_collection_between_invalidity_and_removal() {
    let (target, target_root) = dynamic_target_handle("DynDeleteCollectionGuard");
    let prop_root =
        block_on(addprop_builtin(None, target.clone(), "gain".into())).expect("property");
    let prop = prop_root.value().clone();
    let Value::HandleObject(metadata) = prop else {
        panic!("metadata handle")
    };
    let Value::HandleObject(owner) = target else {
        panic!("owner handle")
    };
    // Only the deletion operation may retain the owner at the collection
    // midpoint: invalidation removes the metadata's edge to it.
    drop(target_root);
    delete_metadata_property(None, &metadata, || {
        assert!(!crate::is_handle_valid(&metadata));
        runmat_gc::gc_collect_major().expect("collect between deletion phases");
        assert!(runmat_gc::gc_clone_value(&metadata.target).is_ok());
        assert!(runmat_gc::gc_clone_value(&owner.target).is_ok());
        Ok(())
    })
    .expect("deletion commits");
    assert!(!crate::is_handle_valid(&metadata));
    let Value::Object(object) = runmat_gc::gc_clone_value(&owner.target).expect("owner survives")
    else {
        panic!("owner object")
    };
    assert!(!object.has_dynamic_property("gain"));
    assert!(!object.properties().contains_key("gain"));
}

#[test]
fn delete_fault_after_alias_invalidation_restores_property_and_alias() {
    let (target, _target_root) = dynamic_target_handle("DynDeleteFaultGuard");
    let prop_root =
        block_on(addprop_builtin(None, target.clone(), "gain".into())).expect("property");
    let prop = prop_root.value().clone();
    let Value::HandleObject(metadata) = prop else {
        panic!("metadata handle")
    };
    let fault = delete_metadata_property(None, &metadata, || {
        assert!(!crate::is_handle_valid(&metadata));
        Err(
            crate::build_runtime_error("injected failure before property removal")
                .with_identifier("RunMat:DynamicDeleteInjectedFault")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        )
    })
    .unwrap_err();
    assert_eq!(
        fault.identifier(),
        Some("RunMat:DynamicDeleteInjectedFault")
    );
    assert!(crate::is_handle_valid(&metadata));
    let Value::HandleObject(owner) = target else {
        panic!("owner handle")
    };
    let Value::Object(object) = runmat_gc::gc_clone_value(&owner.target).expect("owner survives")
    else {
        panic!("owner object")
    };
    assert!(object.has_dynamic_property("gain"));
    assert!(object.properties().contains_key("gain"));
}

#[test]
fn addprop_adds_dynamic_property_and_returns_metadata_handle() {
    {
        let (target, _target_root) = dynamic_target_handle("DynTarget");
        let prop_root =
            block_on(addprop_builtin(None, target.clone(), "gain".into())).expect("addprop");
        let prop = prop_root.value().clone();
        assert!(matches!(prop, Value::HandleObject(_)));
        let found_root = findprop_builtin(None, target.clone(), "gain".into()).expect("findprop");
        let found = found_root.value().clone();
        assert!(matches!(found, Value::HandleObject(_)));
        let Value::HandleObject(handle) = target else {
            panic!("target handle");
        };
        let target_obj = runmat_gc::gc_clone_value(&handle.target).expect("target");
        let Value::Object(obj) = target_obj else {
            panic!("target object");
        };
        assert!(obj.has_dynamic_property("gain"));
        assert!(obj.properties().contains_key("gain"));
    }
}

#[test]
fn dynamic_property_delete_removes_target_property() {
    {
        let (target, _target_root) = dynamic_target_handle("DynTarget");
        let prop_root =
            block_on(addprop_builtin(None, target.clone(), "gain".into())).expect("addprop");
        let prop = prop_root.value().clone();
        block_on(dynamic_property_delete_builtin(None, prop)).expect("delete");
        let Value::HandleObject(handle) = target else {
            panic!("target handle");
        };
        let target_obj = runmat_gc::gc_clone_value(&handle.target).expect("target");
        let Value::Object(obj) = target_obj else {
            panic!("target object");
        };
        assert!(!obj.has_dynamic_property("gain"));
        assert!(!obj.properties().contains_key("gain"));
    }
}

#[test]
fn dynamic_property_supports_value_access_and_metadata_mutation() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    {
        let (target, _target_root) = dynamic_target_handle("DynTarget");
        let prop_root =
            block_on(addprop_builtin(None, target.clone(), "gain".into())).expect("addprop");
        let prop = prop_root.value().clone();

        block_on(crate::call_builtin_async(
            "setfield",
            &[
                target.clone(),
                Value::String("gain".into()),
                Value::Num(7.0),
            ],
        ))
        .expect("set dynamic property");
        let value = block_on(crate::call_builtin_async(
            "getfield",
            &[target.clone(), Value::String("gain".into())],
        ))
        .expect("get dynamic property");
        assert_eq!(value, Value::Num(7.0));

        block_on(crate::call_builtin_async(
            "setfield",
            &[
                prop.clone(),
                Value::String("Hidden".to_string().into()),
                Value::Bool(true),
            ],
        ))
        .expect("set metadata");

        let Value::HandleObject(target_handle) = target else {
            panic!("target handle");
        };
        let target_obj = runmat_gc::gc_clone_value(&target_handle.target).expect("target");
        let Value::Object(obj) = target_obj else {
            panic!("target object");
        };
        assert!(
            obj.dynamic_property("gain")
                .expect("dynamic property")
                .hidden
        );
    }
}

#[test]
fn addprop_rejects_handle_classes_without_dynamicprops_parent() {
    {
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: "PlainHandleForDynamicProps".into(),
            parent: Some(runmat_types::standard::HANDLE.into()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        let (target, _target_root) = target_handle("PlainHandleForDynamicProps");
        let err = block_on(addprop_builtin(None, target, "gain".into()))
            .expect_err("plain handles cannot add dynamic properties");
        assert!(
            err.to_string().contains("not a dynamicprops subclass"),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn addprop_rejects_integer_receiver_instead_of_treating_it_as_property_data() {
    let err = block_on(addprop_builtin(
        None,
        Value::Int(IntValue::U64(u64::MAX)),
        "gain".into(),
    ))
    .expect_err("integer is not a dynamicprops receiver");
    assert_eq!(err.identifier(), Some("RunMat:dynamicprops:InvalidTarget"));
}

#[test]
fn findprop_aliases_share_delete_invalidation() {
    {
        let (target, _target_root) = dynamic_target_handle("DynAliasTarget");
        let first_root =
            block_on(addprop_builtin(None, target.clone(), "gain".into())).expect("addprop");
        let first = first_root.value().clone();
        let second_root = findprop_builtin(None, target.clone(), "gain".into()).expect("findprop");
        let second = second_root.value().clone();
        let (Value::HandleObject(first_handle), Value::HandleObject(second_handle)) =
            (&first, &second)
        else {
            panic!("metadata handles");
        };
        assert_eq!(first_handle.target, second_handle.target);

        block_on(dynamic_property_delete_builtin(None, second)).expect("delete alias");
        assert!(!crate::is_handle_valid(first_handle));
        let Value::HandleObject(metadata) = first else {
            panic!("metadata handle");
        };
        let metadata_value = runmat_gc::gc_clone_value(&metadata.target).expect("metadata");
        let Value::Object(metadata_obj) = metadata_value else {
            panic!("metadata object");
        };
        assert!(!metadata_obj.properties().contains_key(TARGET_FIELD));
    }
}

#[test]
fn findprop_integer_audit_rejects_every_integer_target_class() {
    assert_eq!(
        FINDPROP_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    for target in [
        Value::Int(runmat_value::IntValue::I8(1)),
        Value::Int(runmat_value::IntValue::I16(1)),
        Value::Int(runmat_value::IntValue::I32(1)),
        Value::Int(runmat_value::IntValue::I64(1)),
        Value::Int(runmat_value::IntValue::U8(1)),
        Value::Int(runmat_value::IntValue::U16(1)),
        Value::Int(runmat_value::IntValue::U32(1)),
        Value::Int(runmat_value::IntValue::U64(1)),
    ] {
        let error = findprop_builtin(None, target, "Name".into())
            .expect_err("integer is not an object target");
        assert_eq!(error.identifier(), DYNAMIC_ERROR_INVALID_TARGET.identifier);
    }
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("valid unregistered resident fixture")
            .handle,
    );
    let error = findprop_builtin(None, resident, "Name".into())
        .expect_err("resident numeric value is not an object target");
    assert_eq!(error.identifier(), DYNAMIC_ERROR_INVALID_TARGET.identifier);
}
