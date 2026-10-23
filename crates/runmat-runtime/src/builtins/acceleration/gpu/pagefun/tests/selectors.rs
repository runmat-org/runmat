use super::*;
#[test]
fn pagefun_extensions_follow_compatibility_mode() {
    let matrix = || Tensor::new(vec![1.0], vec![1, 1]).expect("matrix");
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::Tensor(matrix()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(matrix())]),
        ))
        .expect_err("MATLAB mode rejects all-host pagefun");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:PagefunHostInputsExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::Tensor(matrix()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(matrix())]),
        ))
        .expect("RunMat mode accepts all-host pagefun");
    }

    test_support::with_test_provider(|provider| {
        let tensor = matrix();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let invoke = || {
            pagefun_builtin(
                Value::from("@mtimes"),
                Value::GpuTensor(handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    handle.clone(),
                )]),
            )
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(invoke()).expect_err("MATLAB mode rejects text pagefun callable");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:PagefunTextCallableExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(invoke()).expect("RunMat mode accepts text pagefun callable");
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_mtimes_char_array_handle() {
    let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
    let func = CharArray::from_unicode_scalars("@mtimes".chars().collect(), 1, 7).unwrap();
    let result = pagefun_impl(
        Value::CharArray(func),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("pagefun char array");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![19.0, 43.0, 22.0, 50.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn pagefun_mtimes_external_function_handle() {
    let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
    let result = pagefun_impl(
        Value::ExternalFunctionHandle("mtimes".into()),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("pagefun external function handle");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![19.0, 43.0, 22.0, 50.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn pagefun_mtimes_semantic_function_handle() {
    let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
    let result = pagefun_impl(
        Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
            runmat_value::BoundFunctionHandleData {
                name: "mtimes".into(),
                function: 17,
                authority: runmat_value::BoundFunctionAuthority::unavailable(17),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
            },
        )),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("pagefun semantic function handle");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![19.0, 43.0, 22.0, 50.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_mtimes_string_array_handle() {
    let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
    let strings = StringArray::new(vec!["@mtimes".to_string()], vec![1]).unwrap();
    let result = pagefun_impl(
        Value::StringArray(strings),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("pagefun string array");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![19.0, 43.0, 22.0, 50.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_char_array_multirow_error() {
    let chars = CharArray::from_unicode_scalars("@mtimes@".chars().collect(), 2, 4).unwrap();
    let lhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let rhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let err = pagefun_impl(
        Value::CharArray(chars),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let err = block_on(err).expect_err("expected multi-row char array error");
    assert!(
        err.contains("char array"),
        "unexpected error for multi-row char array: {err}"
    );
    assert_eq!(err.identifier(), PAGEFUN_ERROR_INVALID_CALLABLE.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_string_array_multi_value_error() {
    let strings =
        StringArray::new(vec!["@mtimes".to_string(), "@mtimes".to_string()], vec![2]).unwrap();
    let lhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let rhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let err = pagefun_impl(
        Value::StringArray(strings),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let err = block_on(err).expect_err("expected multi-element string array error");
    assert!(
        err.contains("string array"),
        "unexpected error for string array: {err}"
    );
    assert_eq!(err.identifier(), PAGEFUN_ERROR_INVALID_CALLABLE.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_unsupported_operation_identifier() {
    let lhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let rhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let err = pagefun_impl(
        Value::FunctionHandle("plus".into()),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let err = block_on(err).expect_err("expected unsupported operation");
    assert_eq!(
        err.identifier(),
        PAGEFUN_ERROR_UNSUPPORTED_OPERATION.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_invalid_arity_identifier() {
    let lhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let err = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    );
    let err = block_on(err).expect_err("expected invalid arity");
    assert_eq!(err.identifier(), PAGEFUN_ERROR_INVALID_ARITY.identifier);
}
