use super::*;

fn key(name: &str) -> Value {
    crate::make_cell(vec![Value::from(name)], 1, 1).unwrap()
}

fn handle(value: &Value) -> runmat_gc::GcHandle {
    let Value::HandleObject(handle) = value else {
        panic!("expected map handle")
    };
    handle.target
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn map_aliases_share_gc_owned_payload() {
    runmat_gc::gc_test_context(|| {
        let map = containers_map_builtin(Vec::new()).unwrap();
        let alias = map.clone();
        let root = runmat_gc::gc_root(handle(&alias)).unwrap();
        let map =
            containers_map_subsasgn(map, "()".into(), key("answer"), Value::Num(42.0)).unwrap();
        assert_eq!(
            containers_map_subsref(alias.clone(), "()".into(), key("answer")).unwrap(),
            Value::Num(42.0)
        );
        drop(map);
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(
            containers_map_subsref(alias, "()".into(), key("answer")).unwrap(),
            Value::Num(42.0)
        );
        drop(root);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unrooted_self_referential_map_is_collected() {
    runmat_gc::gc_test_context(|| {
        let map = containers_map_builtin(Vec::new()).unwrap();
        let target = handle(&map);
        let root = runmat_gc::gc_root(target).unwrap();
        let map = containers_map_subsasgn(map.clone(), "()".into(), key("self"), map).unwrap();
        let stored = containers_map_subsref(map.clone(), "()".into(), key("self")).unwrap();
        assert_eq!(handle(&stored), target);
        drop(stored);
        drop(map);
        drop(root);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn failed_multi_key_remove_preserves_all_entries() {
    let keys = crate::make_cell(vec![Value::from("present"), Value::from("other")], 1, 2).unwrap();
    let values = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
    let map = containers_map_builtin(vec![keys, values]).unwrap();
    let remove =
        crate::make_cell(vec![Value::from("present"), Value::from("missing")], 1, 2).unwrap();
    assert!(containers_map_remove(map.clone(), remove).is_err());
    assert_eq!(
        containers_map_subsref(map, "()".into(), key("present")).unwrap(),
        Value::Num(1.0)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn duplicate_multi_key_remove_is_atomic() {
    let map = containers_map_builtin(vec![
        key("present"),
        crate::make_cell(vec![Value::Num(1.0)], 1, 1).unwrap(),
    ])
    .unwrap();
    let remove =
        crate::make_cell(vec![Value::from("present"), Value::from("present")], 1, 2).unwrap();
    assert!(containers_map_remove(map.clone(), remove).is_err());
    assert_eq!(
        containers_map_subsref(map, "()".into(), key("present")).unwrap(),
        Value::Num(1.0)
    );
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
        let mut storage = MapStore::new(KeyType::Char, ValueType::Any, false);
        storage.insert_unique(MapEntry {
            normalized: NormalizedKey::String("resident".into()),
            key_value: Value::from("resident"),
            value: Value::GpuTensor(resident),
        });
        let owner = runmat_gc::gc_allocate(Value::Object(
            ObjectInstance::new(CLASS_IDENTITY).with_containers_map_payload(storage),
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
