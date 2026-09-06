use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_gpu_pair_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let ha = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let hb = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = minus_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu minus");
        let gathered = test_support::gather(result).expect("gather");
        let expected = vec![0.0; tensor.len()];
        assert_eq!(gathered.as_f64_slice().expect("double result"), expected);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_gpu_scalar_right() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = minus_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
            .expect("gpu scalar minus");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(
            gathered.as_f64_slice().expect("double result"),
            &[-1.0, 0.0, 1.0]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_gpu_scalar_left() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = minus_builtin(Value::Num(3.0), Value::GpuTensor(handle), Vec::new())
            .expect("gpu scalar minus");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(
            gathered.as_f64_slice().expect("double result"),
            &[1.0, -1.0]
        );
    });
}

#[test]
fn minus_gpu_host_integer_scalar_reenters_exact_dispatch_in_both_directions() {
    test_support::with_test_provider(|provider| {
        let wide = 9_007_199_254_740_993_u64;
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2]).unwrap();

        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = minus_builtin(
            Value::GpuTensor(handle),
            Value::Int(IntValue::U64(1)),
            Vec::new(),
        )
        .expect("gpu integer scalar right");
        let Value::Tensor(result) = result else {
            panic!("expected exact integer tensor");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide - 1, u64::MAX - 1]))
        );

        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = minus_builtin(
            Value::Int(IntValue::U64(1)),
            Value::GpuTensor(handle),
            Vec::new(),
        )
        .expect("gpu integer scalar left");
        let Value::Tensor(result) = result else {
            panic!("expected exact integer tensor");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![0, 0]))
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_like_gpu_prototype_keeps_residency() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto = provider.upload(&proto_view).expect("upload");
        let result = minus_builtin(
            Value::Tensor(lhs.clone()),
            Value::Tensor(rhs.clone()),
            vec![Value::from("like"), Value::GpuTensor(proto.clone())],
        )
        .expect("minus like gpu");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(gathered.shape, vec![2, 1]);
                assert_eq!(
                    gathered.as_f64_slice().expect("double result"),
                    &[7.0, 16.0]
                );
            }
            other => panic!("expected GPU tensor result, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_like_gpu_prototype_uploads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new_integer(IntegerStorage::I16(vec![10, 20]), vec![2, 1]).unwrap();
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto = provider.upload(&proto_view).expect("upload");

        let result = minus_builtin(
            Value::Tensor(lhs),
            Value::Num(3.0),
            vec![Value::from("like"), Value::GpuTensor(proto)],
        )
        .expect("minus like gpu");

        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::I16(vec![7, 17]))
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_like_host_gathers_gpu_value() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
        let ha = gpu_helpers::upload_tensor(provider, &lhs).expect("upload lhs");
        let hb = gpu_helpers::upload_tensor(provider, &rhs).expect("upload rhs");
        let result = minus_builtin(
            Value::GpuTensor(ha),
            Value::GpuTensor(hb),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("minus like host");
        let Value::Tensor(t) = result else {
            panic!("expected tensor result after host gather");
        };
        assert_eq!(t.shape, vec![2, 1]);
        assert_eq!(t.as_f64_slice().expect("double result"), &[5.0, 14.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_like_complex_prototype_yields_complex() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let lhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
    let result = minus_builtin(
        Value::Tensor(lhs),
        Value::Tensor(rhs),
        vec![Value::from("like"), Value::Complex(0.0, 1.0)],
    )
    .expect("minus like complex");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![2, 1]);
            let expected = [(-2.0, 0.0), (-2.0, 0.0)];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
            }
        }
        Value::Complex(re, im) => {
            assert!((re + 2.0).abs() < EPS && im.abs() < EPS);
        }
        other => panic!("unexpected result {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_like_missing_prototype_errors() {
    let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = minus_builtin(
        Value::Tensor(lhs),
        Value::Num(1.0),
        vec![Value::from("like")],
    )
    .expect_err("expected error");
    assert!(err.message().contains("prototype"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_like_keyword_case_insensitive() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
    let result = minus_builtin(
        Value::Tensor(tensor.clone()),
        Value::Num(1.0),
        vec![Value::from("LIKE"), Value::Num(0.0)],
    )
    .expect("minus like upper");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![2, 1]);
            assert_eq!(out.as_f64_slice().expect("double result"), &[-1.0, 0.0]);
        }
        other => panic!("unexpected result {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_like_char_array_keyword() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let keyword = CharArray::new_row("like");
    let result = minus_builtin(
        Value::Num(0.0),
        Value::Num(1.0),
        vec![Value::CharArray(keyword), Value::Num(0.0)],
    )
    .expect("minus like char");
    match result {
        Value::Num(v) => assert!((v + 1.0).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}
