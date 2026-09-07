use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_gpu_pair_roundtrip() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![2.0, 5.0, 10.0], vec![3, 1]).unwrap();
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
        let result = ldivide_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb), Vec::new())
            .expect("gpu ldivide");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![3, 1]);
        let expected = [0.2, 0.25, 0.3333333333333333];
        for (got, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < GPU_EPS);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_like_gpu_prototype_keeps_residency() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto = provider.upload(&proto_view).expect("upload proto");
        let result = ldivide_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::from("like"), Value::GpuTensor(proto)],
        )
        .expect("ldivide like gpu");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert!(gathered
                    .materialize_f64()
                    .iter()
                    .all(|v| (v - 0.5).abs() < GPU_EPS));
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_like_gpu_prototype_uploads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let divisor = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).unwrap();
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto = provider.upload(&proto_view).expect("upload");

        let result = ldivide_builtin(
            Value::Tensor(divisor),
            Value::Num(20.0),
            vec![Value::from("like"), Value::GpuTensor(proto)],
        )
        .expect("ldivide like gpu");

        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![10.0, 5.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_like_host_gathers_gpu_value() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![8.0, 18.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
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
        let result = ldivide_builtin(
            Value::GpuTensor(ha),
            Value::GpuTensor(hb),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("ldivide like host");
        let Value::Tensor(t) = result else {
            panic!("expected tensor result after host gather");
        };
        assert_eq!(t.shape, vec![2, 1]);
        let expected = [0.25, 1.0 / 6.0];
        for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < GPU_EPS);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_like_complex_prototype_yields_complex() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let result = ldivide_builtin(
        Value::Tensor(lhs),
        Value::Tensor(rhs),
        vec![Value::from("like"), Value::Complex(0.0, 1.0)],
    )
    .expect("ldivide like complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![2, 1]);
            let expected = [(0.5, 0.0), (0.5, 0.0)];
            for (got, exp) in ct.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < EPS);
                assert!((got.1 - exp.1).abs() < EPS);
            }
        }
        Value::Complex(re, im) => {
            assert!((re - 0.5).abs() < EPS && im.abs() < EPS);
        }
        other => panic!("expected complex output, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_like_missing_prototype_errors() {
    let lhs = Value::Num(2.0);
    let rhs = Value::Num(4.0);
    let err = ldivide_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
    assert!(
        err.message().contains("prototype"),
        "unexpected error: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_like_keyword_char_array() {
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
        let result = ldivide_builtin(
            lhs,
            rhs,
            vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
        )
        .expect("ldivide like char");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert!((gathered.materialize_f64()[0] - 2.5).abs() < GPU_EPS);
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        }
    });
}
