use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn host_like_conversion_retains_rank_zero_and_rank_one_shapes() {
    for shape in [Vec::new(), vec![1]] {
        let source = || Tensor::new(vec![7.0], shape.clone()).unwrap();
        let logical = finalize_like(source(), &Value::Bool(false)).unwrap();
        let Value::LogicalArray(logical) = logical else {
            panic!("expected logical output");
        };
        assert_eq!(logical.shape, shape);
        assert_eq!(logical.data, vec![1]);

        let integer = finalize_like(source(), &Value::Int(IntValue::U16(0))).unwrap();
        let Value::Tensor(integer) = integer else {
            panic!("expected integer output");
        };
        assert_eq!(integer.shape(), shape);
        assert_eq!(
            integer.integer_storage(),
            Some(&IntegerStorage::U16(vec![7]))
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn resident_like_conversion_retains_exact_shape_and_separate_ownership() {
    test_support::with_test_provider(|provider| {
        let prototype =
            gpu_helpers::upload_tensor(provider, &Tensor::new(vec![0.0], vec![1, 1]).unwrap())
                .unwrap();
        for shape in [Vec::new(), vec![1]] {
            let Value::GpuTensor(output) =
                tensor_to_gpu(Tensor::new(vec![7.0], shape.clone()).unwrap(), &prototype).unwrap()
            else {
                panic!("expected resident output");
            };
            assert_eq!(output.shape(), shape);
            assert!(!output.shares_allocation_with(&prototype));
            assert_eq!(
                runmat_accelerate_api::handle_provenance(&output),
                Some(runmat_accelerate_api::GpuHandleProvenance::Explicit),
            );
            let gathered = test_support::gather(Value::GpuTensor(output)).unwrap();
            assert_eq!(gathered.shape(), shape);
            assert_eq!(gathered.materialize_f64(), vec![7.0]);
        }
        assert!(block_on(provider.download(&prototype)).is_ok());
    });
}
