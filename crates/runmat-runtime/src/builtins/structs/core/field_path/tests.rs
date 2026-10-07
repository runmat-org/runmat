use super::parse::parse_float;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_field_path_indices_preserve_scalar_and_vector_rank() {
    use runmat_value::{CellArray, Tensor, Value};

    for (shape, values) in [(Vec::new(), vec![1.0]), (vec![2], vec![2.0, 1.0])] {
        let selector = CellArray::new(
            vec![Value::Tensor(Tensor::new(values, shape.clone()).unwrap())],
            1,
            1,
        )
        .unwrap();
        let path = super::parse::parse(
            vec![Value::Cell(selector), Value::from("values")],
            "getfield",
        )
        .unwrap();
        let selector = path.leading_index.expect("leading index");
        assert_eq!(selector.components.len(), 1);
        if shape.is_empty() {
            assert!(matches!(
                selector.components[0],
                super::IndexComponent::Scalar(1)
            ));
        } else {
            let super::IndexComponent::Vector(indices, actual_shape) = &selector.components[0]
            else {
                panic!("vector index");
            };
            assert_eq!(indices, &[2, 1]);
            assert_eq!(actual_shape, &shape);
        }
    }
}

#[test]
fn rejects_float_indices_at_or_beyond_the_platform_limit() {
    assert!(parse_float(usize::MAX as f64).is_err());
    assert!(parse_float(usize::MAX as f64 + 1.0).is_err());
}
