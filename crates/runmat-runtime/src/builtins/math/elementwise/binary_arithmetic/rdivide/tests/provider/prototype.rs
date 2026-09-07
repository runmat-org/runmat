use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_like_gpu_prototype_keeps_residency() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let proto = gpu_helpers::upload_tensor(provider, &proto_tensor).expect("upload proto");
        let result = rdivide_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::from("like"), Value::GpuTensor(proto)],
        )
        .expect("rdivide like gpu");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(double_values(&gathered), &[2.0, 2.0]);
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_like_gpu_prototype_uploads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new_integer(IntegerStorage::U32(vec![10, 20]), vec![2, 1]).unwrap();
        let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let proto = gpu_helpers::upload_tensor(provider, &proto_tensor).expect("upload");

        let result = rdivide_builtin(
            Value::Tensor(lhs),
            Value::Num(2.0),
            vec![Value::from("like"), Value::GpuTensor(proto)],
        )
        .expect("rdivide like gpu");

        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U32(vec![5, 10]))
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_like_host_gathers_gpu_value() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![8.0, 18.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let ha = gpu_helpers::upload_tensor(provider, &lhs).expect("upload lhs");
        let hb = gpu_helpers::upload_tensor(provider, &rhs).expect("upload rhs");
        let result = rdivide_builtin(
            Value::GpuTensor(ha),
            Value::GpuTensor(hb),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("rdivide like host");
        let Value::Tensor(t) = result else {
            panic!("expected tensor result after host gather");
        };
        assert_eq!(t.shape, vec![2, 1]);
        assert_eq!(double_values(&t), &[4.0, 6.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_like_complex_prototype_yields_complex() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let result = rdivide_builtin(
        Value::Tensor(lhs),
        Value::Tensor(rhs),
        vec![Value::from("like"), Value::Complex(0.0, 1.0)],
    )
    .expect("rdivide like complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![2, 1]);
            let expected = [(2.0, 0.0), (2.0, 0.0)];
            for (got, exp) in ct.materialize_f64().iter().zip(expected.iter()) {
                assert!((got.0 - exp.0).abs() < EPS);
                assert!((got.1 - exp.1).abs() < EPS);
            }
        }
        Value::Complex(re, im) => {
            assert!((re - 2.0).abs() < EPS && im.abs() < EPS);
        }
        other => panic!("expected complex output, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_like_missing_prototype_errors() {
    let lhs = Value::Num(2.0);
    let rhs = Value::Num(4.0);
    let err = rdivide_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
    assert!(
        err.message().contains("prototype"),
        "unexpected error: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_like_keyword_char_array() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let keyword = CharArray::new_row("LIKE");
        let lhs = Value::Num(2.0);
        let rhs = Value::Num(5.0);
        let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let proto = gpu_helpers::upload_tensor(provider, &proto_tensor).expect("upload");
        let result = rdivide_builtin(
            lhs,
            rhs,
            vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
        )
        .expect("rdivide like char");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(double_values(&gathered), &[0.4]);
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        }
    });
}

#[test]
fn rdivide_like_gpu_uses_the_prototype_owner_not_the_ambient_provider() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = test_support::accel_test_lock();
    let owner = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    let ambient = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    unsafe {
        runmat_accelerate_api::register_provider(owner);
        runmat_accelerate_api::register_provider(ambient);
    }
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(ambient));
    let prototype_tensor = Tensor::new(vec![0.0], vec![1, 1]).expect("prototype tensor");
    let prototype = gpu_helpers::upload_tensor(owner, &prototype_tensor).expect("prototype upload");

    let result = rdivide_builtin(
        Value::Num(12.0),
        Value::Num(3.0),
        vec![Value::from("like"), Value::GpuTensor(prototype)],
    )
    .expect("rdivide like exact owner");
    let Value::GpuTensor(result) = result else {
        panic!("expected resident result")
    };
    let result_owner = runmat_accelerate_api::provider_for_handle(&result).expect("result owner");
    assert!(std::ptr::eq(result_owner, owner));
    assert!(!std::ptr::eq(result_owner, ambient));
}
