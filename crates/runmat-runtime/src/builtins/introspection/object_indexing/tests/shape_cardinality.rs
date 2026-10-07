use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_target_and_selector_low_rank_shapes_keep_exact_cardinality() {
    for (shape, values) in [(Vec::new(), vec![1.0]), (vec![2], vec![1.0, 2.0])] {
        let value = Value::Tensor(Tensor::new(values.clone(), shape.clone()).unwrap());
        assert_eq!(target_shape(&value), Some(shape));
        assert_eq!(
            single_subscript_count(&value, &value, 0, 1).unwrap(),
            values.len()
        );
    }
    let complex = Value::ComplexTensor(
        runmat_value::ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, 4.0)], vec![2]).unwrap(),
    );
    assert_eq!(target_shape(&complex), Some(vec![2]));
}

#[test]
fn final_colon_folds_trailing_dimensions_including_empty_extents() {
    for (shape, count) in [(vec![2, 3, 4], 24), (vec![2, 3, 0], 0)] {
        let target =
            Value::Cell(CellArray::new_with_shape(vec![Value::Num(0.0); count], shape).unwrap());
        let path = substruct("{}", cell(vec![Value::String(":".into()); 2]));
        assert_eq!(
            call(target, path, Value::String("Expression".into())),
            Value::Num(count as f64)
        );
    }
}

#[test]
fn character_colon_uses_full_nd_shape() {
    let target = Value::CharArray(
        runmat_value::CharArray::new_with_shape(vec!['a'; 6], vec![1, 2, 3]).unwrap(),
    );
    assert_eq!(
        single_subscript_count(&target, &Value::String(":".into()), 0, 1).unwrap(),
        6
    );
    assert_eq!(
        single_subscript_count(&target, &Value::String(":".into()), 1, 2).unwrap(),
        6
    );
}
