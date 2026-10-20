use super::*;

#[test]
fn pdf_broadcasts_parameters_preserves_single_and_restores_residency() {
    use crate::builtins::common::test_support;

    let density = block_on(pdf_builtin(
        Value::String("Normal".into()),
        Value::Tensor(Tensor::from_f32(vec![0.0, 2.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Num(1.0),
        ]),
    ))
    .expect("broadcast single pdf");
    let Value::Tensor(density) = density else {
        panic!("expected single tensor");
    };
    assert_eq!(density.numeric_dtype(), NumericDType::F32);
    assert_eq!(density.shape(), &[2, 1]);

    test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let integer = Tensor::new_integer(IntegerStorage::I16(vec![0, 1]), vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let density = block_on(pdf_builtin(
            Value::String("Normal".into()),
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(1.0),
            ]),
        ))
        .expect("resident integer pdf");
        assert!(matches!(density, Value::GpuTensor(_)));
        let gathered = test_support::gather(density).expect("gather pdf");
        assert_eq!(gathered.shape(), &[2, 1]);
    });
}

#[test]
fn pdf_logical_points_and_named_parameters_follow_numeric_evaluator() {
    let density = block_on(pdf_builtin(
        Value::from("Normal"),
        Value::Bool(false),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Bool(false),
            Value::Bool(true),
        ]),
    ))
    .expect("logical values are evaluated as zero and one");
    let Value::Num(density) = density else {
        panic!("expected scalar density");
    };
    assert!((density - distribution_math::standard_normal_pdf(0.0)).abs() < 1.0e-12);
}
