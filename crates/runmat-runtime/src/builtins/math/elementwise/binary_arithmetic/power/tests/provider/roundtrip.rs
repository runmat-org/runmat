use super::super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_like_gpu_residency() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let exp = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let proto = gpu_helpers::upload_tensor(provider, &proto_tensor).expect("upload");
        let result = power_builtin(
            Value::Tensor(base.clone()),
            Value::Tensor(exp.clone()),
            vec![Value::from("like"), Value::GpuTensor(proto.clone())],
        )
        .expect("power");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                let expected = [1.0, 8.0, 81.0];
                for (got, exp) in gathered
                    .as_f64_slice()
                    .expect("double result")
                    .iter()
                    .zip(expected.iter())
                {
                    assert!((got - exp).abs() < 1e-9);
                }
            }
            other => panic!("expected gpu result, got {other:?}"),
        }
        proto.release().expect("release prototype");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_gpu_pair_roundtrip() {
    test_support::with_test_provider(|provider| {
        let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let exp = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let hb = gpu_helpers::upload_tensor(provider, &base).expect("upload");
        let he = gpu_helpers::upload_tensor(provider, &exp).expect("upload");
        let result =
            power_builtin(Value::GpuTensor(hb), Value::GpuTensor(he), Vec::new()).expect("power");
        let gathered = test_support::gather(result).expect("gather");
        let expected = [1.0, 8.0, 81.0];
        for (got, exp) in gathered
            .as_f64_slice()
            .expect("double result")
            .iter()
            .zip(expected.iter())
        {
            assert!((got - exp).abs() < 1e-9);
        }
    });
}
