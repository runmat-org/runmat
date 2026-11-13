use super::*;
use runmat_builtins::ABS_LOGICAL_INPUT_EXTENSION;

#[test]
fn abs_logical_extension_returns_double_and_gates_resident_input_before_dispatch() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let Value::Tensor(output) =
            abs_builtin(Value::LogicalArray(logical)).expect("logical extension")
        else {
            panic!("expected numeric tensor");
        };
        assert_eq!(output.as_f64_slice(), Some(&[1.0, 0.0, 1.0][..]));
    }

    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1, 0],
                shape: &[1, 2],
            })
            .expect("logical producer upload");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let weak = handle.downgrade();
        provider.reset_telemetry();
        let error = abs_builtin(Value::GpuTensor(handle.clone()))
            .expect_err("resident logical input must gate before provider dispatch");
        assert_eq!(
            error.identifier(),
            ABS_LOGICAL_INPUT_EXTENSION.error_identifier
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        handle
            .allocation_lease()
            .validate_live()
            .expect("caller remains live after rejection");
        drop(error);
        handle.release().expect("release caller");
        assert!(weak.acquire().is_err());
    });
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1, 0],
                shape: &[1, 2],
            })
            .expect("logical producer upload");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = abs_builtin(Value::GpuTensor(handle)).expect("resident logical extension");
        let Value::GpuTensor(ref output) = result else {
            panic!("expected resident numeric output");
        };
        assert!(!runmat_accelerate_api::handle_is_logical(output));
        let gathered = test_support::gather(result).expect("gather output");
        assert_eq!(gathered.as_f64_slice(), Some(&[1.0, 0.0][..]));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_tensor_elements() {
    let tensor = Tensor::new(vec![-1.0, -2.0, 3.0, -4.0], vec![2, 2]).unwrap();
    let result = abs_builtin(Value::Tensor(tensor)).expect("abs");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_complex_scalar() {
    let result = abs_builtin(Value::Complex(3.0, 4.0)).expect("abs");
    match result {
        Value::Num(n) => assert!((n - 5.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_complex_tensor_to_real_tensor() {
    let complex = ComplexTensor::new(vec![(3.0, 4.0), (1.0, -1.0)], vec![2, 1]).unwrap();
    let result = abs_builtin(Value::ComplexTensor(complex)).expect("abs");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 1]);
            assert!((t.materialize_f64()[0] - 5.0).abs() < 1e-12);
            assert!((t.materialize_f64()[1] - (2f64).sqrt()).abs() < 1e-12);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_char_array_codes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let char_array = CharArray::from_unicode_scalars("Az".chars().collect(), 1, 2).unwrap();
    let result = abs_builtin(Value::CharArray(char_array)).expect("abs");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[1, 2]);
            assert_eq!(t.materialize_f64(), vec![65.0, 122.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_string_rejected() {
    let err = abs_builtin(Value::from("hello")).expect_err("should error");
    let identifier = err.identifier().map(str::to_string);
    assert!(err.message().contains("expected numeric"));
    assert_eq!(identifier.as_deref(), ABS_ERROR_INVALID_INPUT.identifier);
}
