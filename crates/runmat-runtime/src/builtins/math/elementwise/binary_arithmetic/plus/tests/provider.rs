use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_gpu_pair_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let ha = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let hb = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = plus_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu plus");
        let gathered = test_support::gather(result).expect("gather");
        let expected = tensor
            .as_f64_slice()
            .expect("double input")
            .iter()
            .zip(tensor.as_f64_slice().expect("double input").iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<_>>();
        assert_eq!(gathered.as_f64_slice().expect("double output"), expected);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_gpu_scalar_right() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = plus_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
            .expect("gpu scalar plus");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(
            gathered.as_f64_slice().expect("double output"),
            &[3.0, 4.0, 5.0]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_gpu_scalar_left() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = plus_builtin(Value::Num(3.0), Value::GpuTensor(handle), Vec::new())
            .expect("gpu scalar plus");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.as_f64_slice().expect("double output"), &[5.0, 7.0]);
    });
}

#[test]
fn plus_gpu_host_integer_scalar_reenters_exact_dispatch() {
    test_support::with_test_provider(|provider| {
        let wide = 9_007_199_254_740_993_u64;
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let result = plus_builtin(
            Value::GpuTensor(handle),
            Value::Int(IntValue::U64(1)),
            Vec::new(),
        )
        .expect("exact gpu-host integer plus");
        let Value::GpuTensor(result) = result else {
            panic!("expected resident exact integer tensor");
        };
        let result = test_support::gather(Value::GpuTensor(result)).expect("gather result");
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide + 1, u64::MAX]))
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_like_gpu_prototype_keeps_residency() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto = provider.upload(&proto_view).expect("upload");
        let result = plus_builtin(
            Value::Tensor(lhs.clone()),
            Value::Tensor(rhs.clone()),
            vec![Value::from("like"), Value::GpuTensor(proto.clone())],
        )
        .expect("plus like gpu");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(gathered.shape, vec![2, 1]);
                assert_eq!(gathered.as_f64_slice().expect("double output"), &[4.0, 6.0]);
            }
            other => panic!("expected GPU tensor result, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_like_host_gathers_gpu_value() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
        let ha = gpu_helpers::upload_tensor(provider, &lhs).expect("upload lhs");
        let hb = gpu_helpers::upload_tensor(provider, &rhs).expect("upload rhs");
        let result = plus_builtin(
            Value::GpuTensor(ha),
            Value::GpuTensor(hb),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("plus like host");
        let Value::Tensor(t) = result else {
            panic!("expected tensor result after host gather");
        };
        assert_eq!(t.shape, vec![2, 1]);
        assert_eq!(t.as_f64_slice().expect("double output"), &[6.0, 8.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_like_complex_prototype_yields_complex() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let lhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
    let result = plus_builtin(
        Value::Tensor(lhs),
        Value::Tensor(rhs),
        vec![Value::from("like"), Value::Complex(0.0, 1.0)],
    )
    .expect("plus like complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![2, 1]);
            let expected = [(6.0, 0.0), (8.0, 0.0)];
            for (got, exp) in ct.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < EPS);
                assert!((got.1 - exp.1).abs() < EPS);
            }
        }
        Value::Complex(re, im) => {
            assert!((re - 6.0).abs() < EPS && im.abs() < EPS);
        }
        other => panic!("expected complex output, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_like_missing_prototype_errors() {
    let lhs = Value::Num(2.0);
    let rhs = Value::Num(4.0);
    let err = plus_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
    assert!(
        err.message().contains("prototype"),
        "unexpected error: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plus_like_keyword_char_array() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let keyword = CharArray::new_row("LIKE");
        let lhs = Value::Num(2.0);
        let rhs = Value::Num(5.0);
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto = provider.upload(&proto_view).expect("upload");
        let result = plus_builtin(
            lhs,
            rhs,
            vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
        )
        .expect("plus like char");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(gathered.as_f64_slice().expect("double output"), &[7.0]);
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        }
    });
}
