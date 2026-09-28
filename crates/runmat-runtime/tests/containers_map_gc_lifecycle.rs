//! GC lifecycle checks run in their own test binary. Resetting the process-wide
//! collector here cannot invalidate handles owned by parallel runtime unit tests.

use runmat_runtime::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use runmat_value::{
    ContainersMapEntry, ContainersMapKey, ContainersMapKeyType, ContainersMapStorage,
    ContainersMapValueType, ObjectInstance, Value,
};

fn map_handle(value: &Value) -> runmat_gc::GcHandle {
    let Value::HandleObject(handle) = value else {
        panic!("expected containers.Map handle")
    };
    handle.target
}

fn subscript(key: &str) -> Value {
    ObjectSubscriptPath::single(ObjectSubscript::parentheses(
        ObjectIndexSelector::IndexValues {
            components: vec![ObjectIndexComponent::Value(Value::from(key))],
        },
    ))
    .to_standard_substruct_value()
    .unwrap()
}

fn assign(map: Value, key: &str, value: Value) -> Value {
    runmat_runtime::call_builtin("containers.Map.subsasgn", &[map, subscript(key), value]).unwrap()
}

fn lookup(map: Value, key: &str) -> Value {
    runmat_runtime::call_builtin("containers.Map.subsref", &[map, subscript(key)]).unwrap()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn aliases_share_map_storage_across_collection() {
    runmat_gc::gc_test_context(|| {
        let map = runmat_runtime::call_builtin("containers.Map", &[]).unwrap();
        let alias = map.clone();
        let root = runmat_gc::gc_root(map_handle(&alias)).unwrap();
        let map = assign(map, "answer", Value::Num(42.0));
        assert_eq!(lookup(alias.clone(), "answer"), Value::Num(42.0));
        drop(map);
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(lookup(alias, "answer"), Value::Num(42.0));
        drop(root);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn self_referential_map_is_collected_after_last_root() {
    runmat_gc::gc_test_context(|| {
        let map = runmat_runtime::call_builtin("containers.Map", &[]).unwrap();
        let target = map_handle(&map);
        let root = runmat_gc::gc_root(target).unwrap();
        let map = assign(map.clone(), "self", map);
        let stored = lookup(map.clone(), "self");
        assert_eq!(map_handle(&stored), target);
        drop(stored);
        drop(map);
        drop(root);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
    });
}

#[test]
fn collecting_map_releases_nested_provider_value() {
    runmat_gc::gc_test_context(|| {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let resident = allocation.handle.clone();
        let weak = resident.downgrade();
        drop(allocation);
        let mut storage = ContainersMapStorage::new(
            ContainersMapKeyType::Char,
            ContainersMapValueType::Any,
            false,
        );
        storage.insert_unique(ContainersMapEntry {
            normalized: ContainersMapKey::String("resident".into()),
            key_value: Value::from("resident"),
            value: Value::GpuTensor(resident),
        });
        let owner = runmat_gc::gc_allocate(Value::Object(
            ObjectInstance::new("containers.Map").with_containers_map_payload(storage),
        ))
        .unwrap();
        let root = runmat_gc::gc_root(owner).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(weak.acquire().is_ok());
        drop(root);
        runmat_gc::gc_collect_major().unwrap();
        assert!(weak.acquire().is_err());
    });
}
