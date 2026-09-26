use super::*;

#[test]
fn underlying_type_keeps_exact_owner_precision_admission() {
    use runmat_accelerate_api::{GpuTensorStorage, NumericElementType, ProviderPrecision};
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let source =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let (wrong_type, wrong_class) = match provider.precision() {
            ProviderPrecision::F64 => (NumericElementType::F32, runmat_types::standard::SINGLE),
            ProviderPrecision::F32 => (NumericElementType::F64, runmat_types::standard::DOUBLE),
        };
        let malformed = source
            .clone()
            .with_numeric_descriptor(wrong_type, GpuTensorStorage::Real);
        runmat_accelerate_api::set_handle_class_identity(&malformed, wrong_class);
        let value = Value::GpuTensor(malformed);
        assert!(
            !value_underlying_type_matches(&value, vec![wrong_class.display_name().into()])
                .unwrap()
        );
        let error = crate::builtins::introspection::underlying_type::underlying_class_identity_for_value_checked(&value).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:isUnderlyingType:ProviderPayloadMismatch")
        );
        let error = must_be_underlying_type::evaluate(vec![
            value,
            Value::String(wrong_class.display_name().into()),
        ])
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:mustBeUnderlyingType:ValidationFailed")
        );
        runmat_accelerate_api::set_handle_class_identity(
            &source,
            source.descriptor.element_type.unwrap().class_identity(),
        );
        provider.free(&source).unwrap();
        runmat_accelerate_api::clear_handle_metadata(&source);
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
        let chars =
            Value::CharArray(CharArray::new_with_shape(vec!['x'; count], shape.clone()).unwrap());
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
fn resident_nd_and_class_queries_never_download_payloads() {
    test_support::with_test_provider(|provider| {
        for (shape, expected) in [(vec![1, 1, 2], false), (vec![1, 1, 1], true)] {
            let tensor = Tensor::new(vec![1.0; shape.iter().product()], shape).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            provider.free(&handle).unwrap();
            let value = Value::GpuTensor(handle.clone());
            assert_eq!(value_is_scalar_or_empty(&value), expected);
            assert!(value_matches_class(&value, "double"));
            assert!(value_underlying_type_matches(&value, vec!["double".into()]).unwrap());
            runmat_accelerate_api::set_handle_logical(&handle, true);
            assert!(!value_is_numeric(&value));
            assert!(!value_is_float(&value));
            assert!(value_matches_class(&value, "logical"));
            assert!(value_underlying_type_matches(&value, vec!["logical".into()]).unwrap());
            runmat_accelerate_api::clear_handle_metadata(&handle);
        }
    });
}
