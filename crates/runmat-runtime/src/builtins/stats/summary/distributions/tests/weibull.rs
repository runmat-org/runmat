use super::*;

#[test]
fn wblinv_preserves_tiny_positive_probabilities() {
    let out = block_on(wblinv::wblinv_builtin(
        Value::Num(1.0e-20),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(3.0),
            Value::Num(4.0),
        ]),
    ))
    .unwrap();
    match out {
        Value::Num(value) => {
            assert!(value > 0.0);
            assert_close(value, 3.0e-5, 1.0e-16);
        }
        other => panic!("expected scalar wblinv, got {other:?}"),
    }
}

#[test]
fn wblinv_gates_integer_roles_and_rejects_lossy_values() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(wblinv::wblinv_builtin(
            Value::Int(runmat_value::IntValue::U8(1)),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("integer probability gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:WblinvIntegerProbabilityExtension")
        );
        let error = block_on(wblinv::wblinv_builtin(
            Value::Num(0.5),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(runmat_value::IntValue::U8(1)),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("integer scale gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:WblinvIntegerScaleExtension")
        );
    }
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(wblinv::wblinv_builtin(
        Value::Num(0.5),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
        ]),
    ))
    .expect_err("lossy shape boundary");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn wblinv_preserves_documented_single_output() {
    let output = block_on(wblinv::wblinv_builtin(
        Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(3.0),
            Value::Num(4.0),
        ]),
    ))
    .expect("single wblinv");
    let Value::Tensor(output) = output else {
        panic!("expected single tensor output")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
}

#[test]
fn wblinv_distinguishes_automatic_and_explicit_residency() {
    test_support::with_test_provider(|provider| {
        let source = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![0.5], vec![1, 1]).expect("probability"),
        )
        .expect("resident probability");
        runmat_accelerate::fusion_residency::mark(&source);
        let automatic = block_on(wblinv::wblinv_builtin(
            Value::GpuTensor(source.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("automatic fallback");
        let Value::GpuTensor(output) = automatic else {
            panic!("expected resident automatic output")
        };
        assert!(
            gpu_helpers::exact_provider_for_handle(&output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );

        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(wblinv::wblinv_builtin(
            Value::GpuTensor(source),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("explicit gpu gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:WblinvExplicitGpuInputExtension")
        );
    });
}
