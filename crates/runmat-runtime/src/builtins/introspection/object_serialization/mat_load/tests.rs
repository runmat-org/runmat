use super::*;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, SparseTensor,
    StringArray, StructArray, Tensor,
};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn flat_mat_classes_root_without_recursive_restoration_and_preserve_exact_values() {
    let values = vec![
        Value::Int(IntValue::U64(u64::MAX)),
        Value::Num(42.25),
        Value::Complex(1.5, -2.25),
        Value::Bool(true),
        Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
        Value::String("plain".into()),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        Value::CharArray(CharArray::new_row("plain")),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1]).unwrap(),
        ),
        Value::SparseTensor(SparseTensor::new(2, 2, vec![0, 1, 1], vec![1], vec![3.5]).unwrap()),
        Value::ComplexTensor(ComplexTensor::from_f32(vec![(1.0, -2.0)], vec![1, 1]).unwrap()),
    ];
    for value in values {
        let expected = value.clone();
        let MatLoadRestoration::Ready(leaf) = classify_mat_load_restoration(value) else {
            panic!("flat MAT class entered recursive restoration");
        };
        let rooted = leaf.into_rooted().expect("root flat MAT value");
        assert_eq!(rooted.value(), &expected);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn containers_keep_recursive_restoration() {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("value", Value::Num(1.0));
    let structure = structure.finish();
    let array = StructArray::with_fields(
        vec!["value".into()],
        vec![structure.clone(), structure.clone()],
        vec![1, 2],
    )
    .unwrap();
    for value in [
        Value::Struct(structure),
        Value::StructArray(array),
        Value::Cell(CellArray::new_with_shape(vec![Value::Num(1.0)], vec![1, 1]).unwrap()),
    ] {
        assert!(matches!(
            classify_mat_load_restoration(value),
            MatLoadRestoration::Recursive(_)
        ));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn recursive_restore_still_enforces_the_declared_depth_limit() {
    let mut nested = Value::Num(1.0);
    for _ in 0..=super::super::MAX_SERIALIZATION_DEPTH {
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("next", nested);
        nested = Value::Struct(structure.finish());
    }
    let error = futures::executor::block_on(super::super::restore_value_from_mat_load(nested))
        .expect_err("recursive MAT values must retain the depth bound");
    assert_eq!(
        error.identifier(),
        super::super::SERIALIZATION_ERROR_RECURSION.identifier
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn flat_handle_leaf_owns_its_gc_target_until_result_retirement() {
    runmat_gc::gc_test_context(|| {
        let target =
            runmat_gc::gc_allocate_rooted(Value::String("target".into())).expect("root target");
        let handle = target.handle();
        let value = Value::HandleObject(runmat_value::HandleRef {
            class_name: "Leaf".into(),
            target: handle,
            valid: true,
        });
        let MatLoadRestoration::Ready(leaf) = classify_mat_load_restoration(value) else {
            panic!("flat handle entered recursive restoration");
        };
        let rooted = leaf.into_rooted().expect("root handle result");
        target.unroot().expect("release setup root");
        assert_eq!(
            runmat_gc::gc_is_reachable_excluding(handle, &[]),
            Some(true)
        );
        drop(rooted);
        assert_eq!(
            runmat_gc::gc_is_reachable_excluding(handle, &[]),
            Some(false)
        );
    });
}
