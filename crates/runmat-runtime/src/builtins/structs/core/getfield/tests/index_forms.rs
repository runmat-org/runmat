use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_supports_end_index() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::Tensor(tensor));
    let idx_cell = CellArray::new(vec![Value::CharArray(CharArray::new_row("end"))], 1, 1).unwrap();
    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(idx_cell)],
    )
    .expect("end index");
    assert_eq!(result, Value::Num(3.0));
}

#[test]
fn getfield_nd_end_collapses_dimensions_into_final_selector() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let elements = (1..=24)
        .map(|value| {
            let mut structure = StructValue::new();
            structure.insert("payload", Value::Num(value as f64));
            structure
        })
        .collect();
    let array = StructArray::with_fields(vec!["payload".into()], elements, vec![2, 3, 4])
        .expect("N-D structure array");
    let index = CellArray::new(
        vec![Value::Num(1.0), Value::CharArray(CharArray::new_row("end"))],
        1,
        2,
    )
    .expect("index selector");
    let result = run_getfield(
        Value::StructArray(array),
        vec![Value::Cell(index), Value::from("payload")],
    )
    .expect("collapsed end index");
    assert_eq!(result, Value::Num(23.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_char_array_single_element() {
    let chars = CharArray::new_row("entry-a");
    let mut st = StructValue::new();
    st.fields
        .insert("name".to_string(), Value::CharArray(chars));
    let index = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("name"), Value::Cell(index)],
    )
    .expect("char indexing");
    match result {
        Value::CharArray(ca) => {
            assert_eq!(ca.rows, 1);
            assert_eq!(ca.cols, 1);
            assert_eq!(ca.data, vec!['n']);
        }
        other => panic!("expected 1x1 CharArray, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_complex_tensor_index() {
    let tensor =
        ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2, 1]).expect("complex tensor");
    let mut st = StructValue::new();
    st.fields
        .insert("vals".to_string(), Value::ComplexTensor(tensor));
    let index = CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("vals"), Value::Cell(index)],
    )
    .expect("complex index");
    assert_eq!(result, Value::Complex(3.0, 4.0));
}
