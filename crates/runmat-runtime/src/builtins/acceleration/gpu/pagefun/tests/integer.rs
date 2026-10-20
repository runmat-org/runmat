use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_preserves_wide_integer_pages_without_float_materialization() {
    let lhs = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![1, 2],
    )
    .unwrap();
    let rhs = Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).unwrap();

    let result = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("typed integer page inputs");
    let Value::Tensor(result) = result else {
        panic!("expected tensor result");
    };
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
}

#[test]
fn pagefun_integer_capability_records_typed_scalar_mtimes() {
    assert_eq!(PAGEFUN_INTEGER_CAPABILITIES.len(), 1);
    let capability = &PAGEFUN_INTEGER_CAPABILITIES[0];
    assert_eq!(
        capability.inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(capability.overload, BuiltinIntegerOverloadKind::ScalarOnly);
    assert!(!capability.notes.contains("[integer-audit-open]"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_matrix_result_reader_preserves_wide_typed_integer_storage() {
    let result = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, 7]),
        vec![1, 2],
    )
    .unwrap();

    let (storage, rows, cols) =
        tensor_matrix_storage(Value::Tensor(result)).expect("typed callback result");
    assert_eq!(rows, 1);
    assert_eq!(cols, 2);
    assert_eq!(storage, NumericStorage::U64(vec![9_007_199_254_740_993, 7]));
}

#[test]
fn pagefun_preserves_wide_integer_scalar_without_float_materialization() {
    let result = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    );

    let result = block_on(result).expect("wide integer scalar");
    let Value::Tensor(result) = result else {
        panic!("expected tensor result");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX]))
    );
}

#[test]
fn pagefun_resident_integer_fallback_restores_exact_owner_and_class() {
    test_support::with_test_provider(|provider| {
        let source = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[9_007_199_254_740_993, u64::MAX]),
                shape: &[1, 2],
            })
            .expect("resident integer input");
        let result = block_on(pagefun_impl(
            Value::FunctionHandle("mtimes".into()),
            Value::GpuTensor(source.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                runmat_value::IntValue::U64(1),
            )]),
        ))
        .expect("owner-aware integer fallback");
        let Value::GpuTensor(result) = result else {
            panic!("expected resident result");
        };
        assert_eq!(result.device_id(), source.device_id());
        assert_eq!(
            block_on(provider.download_integer(&result))
                .expect("download exact result")
                .data,
            HostIntegerDataOwned::U64(vec![9_007_199_254_740_993, u64::MAX])
        );
        let input_lease = source.allocation_lease().downgrade();
        let result_lease = result.allocation_lease().downgrade();
        drop(result);
        assert!(input_lease.acquire().is_some());
        drop(source);
        assert!(input_lease.acquire().is_none());
        assert!(result_lease.acquire().is_none());
    });
}
