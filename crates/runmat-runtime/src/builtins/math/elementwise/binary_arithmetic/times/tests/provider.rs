use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_gpu_pair_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let ha = provider
            .upload(&view)
            .expect("upload")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let hb = provider.upload(&view).expect("upload");
        let result = times_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu times");
        assert!(matches!(
            &result,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle)
        ));
        let gathered = test_support::gather(result).expect("gather");
        let expected = tensor
            .materialize_f64()
            .iter()
            .zip(tensor.materialize_f64().iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<_>>();
        assert_eq!(gathered.materialize_f64(), expected);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_gpu_scalar_right() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = times_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
            .expect("gpu scalar times");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.materialize_f64(), vec![2.0, 4.0, 6.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_gpu_scalar_left() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = times_builtin(Value::Num(3.0), Value::GpuTensor(handle), Vec::new())
            .expect("gpu scalar times");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.materialize_f64(), vec![6.0, 12.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_like_gpu_prototype_keeps_residency() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto = provider.upload(&proto_view).expect("upload");
        let result = times_builtin(
            Value::Tensor(lhs.clone()),
            Value::Tensor(rhs.clone()),
            vec![Value::from("like"), Value::GpuTensor(proto.clone())],
        )
        .expect("times like gpu");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(gathered.shape, vec![2, 1]);
                assert_eq!(gathered.materialize_f64(), vec![3.0, 8.0]);
            }
            other => panic!("expected GPU tensor result, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_like_gpu_prototype_uploads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new_integer(IntegerStorage::U16(vec![10, 20]), vec![2, 1]).unwrap();
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto = provider.upload(&proto_view).expect("upload");

        let result = times_builtin(
            Value::Tensor(lhs),
            Value::Num(3.0),
            vec![Value::from("like"), Value::GpuTensor(proto)],
        )
        .expect("times like gpu");

        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![30.0, 60.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_like_host_gathers_gpu_value() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
        let view_l = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        };
        let ha = provider.upload(&view_l).expect("upload lhs");
        let hb = provider.upload(&view_r).expect("upload rhs");
        let result = times_builtin(
            Value::GpuTensor(ha),
            Value::GpuTensor(hb),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("times like host");
        let Value::Tensor(t) = result else {
            panic!("expected tensor result after host gather");
        };
        assert_eq!(t.shape, vec![2, 1]);
        assert_eq!(t.materialize_f64(), vec![5.0, 12.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_like_complex_prototype_yields_complex() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let lhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
    let result = times_builtin(
        Value::Tensor(lhs),
        Value::Tensor(rhs),
        vec![Value::from("like"), Value::Complex(0.0, 1.0)],
    )
    .expect("times like complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![2, 1]);
            let expected = [(8.0, 0.0), (15.0, 0.0)];
            for (got, exp) in ct.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < EPS);
                assert!((got.1 - exp.1).abs() < EPS);
            }
        }
        Value::Complex(re, im) => {
            assert!((re - 8.0).abs() < EPS && im.abs() < EPS);
        }
        other => panic!("expected complex output, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_like_missing_prototype_errors() {
    let lhs = Value::Num(2.0);
    let rhs = Value::Num(4.0);
    let err = times_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
    assert!(
        err.message().contains("prototype"),
        "unexpected error: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_like_keyword_char_array() {
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
        let result = times_builtin(
            lhs,
            rhs,
            vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
        )
        .expect("times like char");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(gathered.materialize_f64(), vec![10.0]);
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        }
    });
}
