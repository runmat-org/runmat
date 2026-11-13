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
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Cell(selector),
                Value::from("values"),
            ]),
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
fn logical_field_path_indices_keep_mask_shape_for_linear_selection() {
    use runmat_value::{CellArray, LogicalArray, Tensor, Value};

    for (source_shape, mask_shape, expected_shape) in [
        (vec![1, 6], vec![1, 6], vec![1, 3]),
        (vec![6, 1], vec![1, 6], vec![3, 1]),
        (vec![1, 6], vec![2, 3], vec![3, 1]),
    ] {
        let source =
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], source_shape).unwrap());
        let mask = LogicalArray::new(vec![1, 0, 1, 0, 1, 0], mask_shape.clone()).unwrap();
        let cell = CellArray::new(vec![Value::LogicalArray(mask)], 1, 1).unwrap();
        let path = super::parse::parse(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Cell(cell),
                Value::from("values"),
            ]),
            "getfield",
        )
        .unwrap();
        let selector = path.leading_index.expect("leading logical selector");
        let plan = super::plan::build_plan(&source, &selector, false).unwrap();
        assert_eq!(plan.output_shape, expected_shape);
        let super::IndexComponent::Logical(parsed) = &selector.components[0] else {
            unreachable!()
        };
        assert_eq!(parsed.shape, mask_shape);
    }
}

#[test]
fn rejects_float_indices_at_or_beyond_the_platform_limit() {
    assert!(parse_float(usize::MAX as f64).is_err());
    assert!(parse_float(usize::MAX as f64 + 1.0).is_err());
}
