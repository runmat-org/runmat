use super::*;
#[test]
fn power_like_complex_conversion_preserves_single_storage() {
    let tensor = Tensor::from_f32(vec![-4.0, 5.0], vec![1, 2]).unwrap();

    let result = block_on(super::real_to_complex(
        OUTPUT_PROTOTYPE_CONTEXT,
        Value::Tensor(tensor),
    ))
    .expect("complex conversion");

    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(-4.0, 0.0), (5.0, 0.0)])
    );
}

#[test]
fn power_like_gpu_upload_reads_typed_integer_storage_exactly() {
    test_support::with_test_provider(|provider| {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 2]), vec![1, 2]).unwrap();

        let uploaded = gpu_helpers::upload_tensor(provider, &tensor).expect("gpu upload");
        let gathered = test_support::gather(Value::GpuTensor(uploaded)).expect("gather");

        assert_eq!(gathered.shape, vec![1, 2]);
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 2]))
        );
    });
}

#[test]
fn power_like_complex_promotes_output() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
    let result = power_builtin(
        Value::Tensor(base),
        Value::Num(0.5),
        vec![Value::from("like"), Value::Complex(0.0, 1.0)],
    )
    .expect("power");
    match result {
        Value::Complex(re, im) => {
            assert!(re.abs() < 1e-8);
            assert!((im - std::f64::consts::SQRT_2).abs() < 1e-8);
        }
        other => panic!("expected complex result, got {other:?}"),
    }
}
