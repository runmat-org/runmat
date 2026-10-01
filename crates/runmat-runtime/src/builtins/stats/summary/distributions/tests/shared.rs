use super::*;

#[test]
fn distribution_context_preserves_provider_failure_identity_and_retry_policy() {
    let original = build_runtime_error("gpu download failed")
        .with_identifier("RunMat:gpu:DownloadFailed")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    let annotated = with_distribution_context("normcdf", original);
    assert_eq!(annotated.identifier(), Some("RunMat:gpu:DownloadFailed"));
    assert_eq!(annotated.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert_eq!(annotated.context.builtin.as_deref(), Some("normcdf"));
}

#[test]
fn distribution_broadcast_indexing_appends_trailing_singletons() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    assert_eq!(
        broadcast_tensor_to("normcdf", &tensor, &[2, 1, 3]).unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn binomial_chi_square_and_weibull_distribution_values() {
    let binomial = block_on(binocdf::binocdf_builtin(
        Value::Num(55.0),
        vec![Value::Num(100.0), Value::Num(0.5)],
    ))
    .unwrap();
    match binomial {
        Value::Num(value) => assert_close(value, 0.864_373_487_963_083, 1.0e-12),
        other => panic!("expected scalar binocdf, got {other:?}"),
    }

    let upper = block_on(binocdf::binocdf_builtin(
        Value::Num(55.0),
        vec![Value::Num(100.0), Value::Num(0.5), Value::from("upper")],
    ))
    .unwrap();
    match upper {
        Value::Num(value) => assert_close(value, 0.135_626_512_036_917, 1.0e-12),
        other => panic!("expected scalar upper binocdf, got {other:?}"),
    }

    let chi2 = block_on(chi2cdf::chi2cdf_builtin(
        Value::Num(3.0),
        vec![Value::Num(5.0)],
    ))
    .unwrap();
    match chi2 {
        Value::Num(value) => assert_close(value, 0.300_014_164_121_372, 1.0e-12),
        other => panic!("expected scalar chi2cdf, got {other:?}"),
    }

    let weibull = block_on(wblinv::wblinv_builtin(
        Value::Num(0.5),
        vec![Value::Num(3.0), Value::Num(4.0)],
    ))
    .unwrap();
    match weibull {
        Value::Num(value) => assert_close(value, 3.0 * std::f64::consts::LN_2.powf(0.25), 1.0e-12),
        other => panic!("expected scalar wblinv, got {other:?}"),
    }
}

#[test]
fn distribution_helpers_accept_typed_integer_tensors_at_f64_boundary() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(normcdf::normcdf_builtin(
        mirrorless_int_tensor(IntegerStorage::I16(vec![0, 1]), vec![1, 2]),
        vec![
            mirrorless_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
            mirrorless_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        ],
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_close(tensor.materialize_f64()[0], 0.5, 1.0e-12);
            assert_close(tensor.materialize_f64()[1], 0.841_344_746_068_543, 1.0e-12);
        }
        other => panic!("expected tensor normcdf, got {other:?}"),
    }

    let chi2 = block_on(chi2cdf::chi2cdf_builtin(
        mirrorless_int_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]),
        vec![mirrorless_int_tensor(
            IntegerStorage::U16(vec![5]),
            vec![1, 1],
        )],
    ))
    .unwrap();
    match chi2 {
        Value::Num(value) => assert_close(value, 0.300_014_164_121_372, 1.0e-12),
        other => panic!("expected scalar chi2cdf, got {other:?}"),
    }
}

#[test]
fn distribution_helpers_broadcast_and_return_nan_for_bad_parameters() {
    let out = block_on(chi2cdf::chi2cdf_builtin(
        Value::Num(3.0),
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 5.0], vec![1, 3]).unwrap(),
        )],
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 3]);
            assert_close(tensor.materialize_f64()[2], 0.300_014_164_121_372, 1.0e-12);
        }
        other => panic!("expected tensor chi2cdf, got {other:?}"),
    }

    let invalid = block_on(wblinv::wblinv_builtin(
        Value::Num(0.5),
        vec![Value::Num(-1.0), Value::Num(2.0)],
    ))
    .unwrap();
    assert!(matches!(invalid, Value::Num(value) if value.is_nan()));
}
