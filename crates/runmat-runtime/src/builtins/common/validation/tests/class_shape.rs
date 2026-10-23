use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dense_rank_boundaries_preserve_validation_shape_contracts() {
    let scalar = Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap());
    assert_eq!(value_shape_2d(&scalar), (0, 0));
    assert!(value_is_scalar_or_empty(&scalar));
    assert!(value_is_column(&scalar));
    assert!(value_is_vector(&scalar).unwrap());

    let row = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
    assert_eq!(value_shape_2d(&row), (1, 3));
    assert!(!value_is_scalar_or_empty(&row));
    assert!(value_is_column(&row));
    assert!(value_is_vector(&row).unwrap());
}

#[test]
fn underlying_type_uses_allocation_class_not_owner_preference() {
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};
    test_support::with_test_provider(|provider| {
        let source = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let weak = source.downgrade();
        let value = Value::GpuTensor(source);
        assert!(value_underlying_type_matches(&value, vec!["single".into()]).unwrap());
        assert!(!value_underlying_type_matches(&value, vec!["double".into()]).unwrap());
        must_be_underlying_type::evaluate(
            crate::builtins::common::validation::tests::retained_arguments(vec![
                value.clone(),
                Value::from("single"),
            ]),
        )
        .unwrap();
        let error = must_be_underlying_type::evaluate(
            crate::builtins::common::validation::tests::retained_arguments(vec![
                value,
                Value::from("double"),
            ]),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:mustBeUnderlyingType:ValidationFailed")
        );
        assert!(weak.acquire().is_err());
    });
}

#[test]
fn symbolic_and_object_arrays_use_stored_shape_without_overloads() {
    for shape in [vec![1, 1, 2], vec![2, 3, 0], vec![1, 2, 1]] {
        let count = shape.iter().product();
        let symbolic = Value::SymbolicArray(
            runmat_value::SymbolicArray::new(
                vec![runmat_value::SymbolicExpr::Constant(1.0); count],
                shape.clone(),
            )
            .unwrap(),
        );
        let objects = Value::ObjectArray(
            runmat_value::ObjectArray::from_objects(
                "ShapeValue",
                vec![runmat_value::ObjectInstance::new("ShapeValue"); count],
                shape.clone(),
            )
            .unwrap(),
        );
        for value in [symbolic, objects] {
            assert_eq!(value_is_empty(&value), count == 0);
            assert_eq!(value_is_scalar_or_empty(&value), count <= 1);
            assert_eq!(value_is_vector(&value).unwrap(), shape == vec![1, 2, 1]);
            assert!(!value_is_column(&value));
        }
    }
}

#[test]
fn scalar_or_empty_uses_every_dimension() {
    for (shape, expected) in [
        (vec![1, 1, 2], false),
        (vec![1, 1, 1], true),
        (vec![2, 3, 0], true),
        (vec![1, 0, 7], true),
    ] {
        let count = shape.iter().product();
        let value = Value::Tensor(Tensor::new(vec![1.0; count], shape.clone()).unwrap());
        assert_eq!(value_is_scalar_or_empty(&value), expected, "{shape:?}");
        let chars = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['x'; count], shape.clone()).unwrap(),
        );
        assert_eq!(value_is_scalar_or_empty(&chars), expected, "char {shape:?}");
        assert_eq!(value_is_empty(&chars), count == 0, "char {shape:?}");
    }
}

#[test]
fn sparse_logical_is_not_a_numeric_or_floating_class() {
    let value = Value::SparseTensor(SparseTensor::new_logical(2, 1, vec![0, 1], vec![0]).unwrap());
    assert!(!value_is_numeric(&value));
    assert!(!value_is_float(&value));
    assert!(value_is_numeric_or_logical(&value));
    for name in ["logical", "LoGiCaL", "sparse"] {
        assert!(value_matches_class(&value, name), "{name}");
    }
    for name in ["numeric", "float", "double", "single"] {
        assert!(!value_matches_class(&value, name), "{name}");
    }
}

#[test]
fn class_queries_preserve_custom_name_and_underlying_case_rules() {
    let value = Value::Object(runmat_value::ObjectInstance::new("Example.Widget"));
    assert!(value_matches_class(&value, " example.WIDGET "));
    assert!(!value_matches_class(&value, "Widget"));
    assert!(!value_matches_class(&value, "bad name"));
    assert!(value_underlying_type_matches(&value, vec!["Example.Widget".into()]).unwrap());
    assert!(!value_underlying_type_matches(&value, vec!["example.widget".into()]).unwrap());
}

#[test]
fn resident_nd_and_class_queries_preserve_owning_facts() {
    test_support::with_test_provider(|provider| {
        for (shape, expected) in [(vec![1, 1, 2], false), (vec![1, 1, 1], true)] {
            let tensor = Tensor::new(vec![1.0; shape.iter().product()], shape).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let value = Value::GpuTensor(handle.clone());
            assert_eq!(value_is_scalar_or_empty(&value), expected);
            assert!(value_matches_class(&value, "double"));
            assert!(value_underlying_type_matches(&value, vec!["double".into()]).unwrap());
            let logical = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &vec![1; tensor.shape().iter().product()],
                    shape: tensor.shape(),
                })
                .unwrap();
            let logical_weak = logical.downgrade();
            let value = Value::GpuTensor(logical);
            assert!(!value_is_numeric(&value));
            assert!(!value_is_float(&value));
            assert!(value_matches_class(&value, "logical"));
            assert!(value_underlying_type_matches(&value, vec!["logical".into()]).unwrap());
            drop(value);
            assert!(logical_weak.acquire().is_err());
            handle.allocation_lease().validate_live().unwrap();
        }
    });
}
