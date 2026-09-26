use super::*;

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
