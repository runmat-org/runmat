use super::*;
#[test]
fn kstest_default_normal_accepts_centered_sample() {
    let sample = Value::Tensor(Tensor::new(vec![-1.0, -0.25, 0.0, 0.25, 1.0], vec![5, 1]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    assert_eq!(outputs[0], Value::Bool(false));
    match outputs[1] {
        Value::Num(p) => assert_close(p, 0.973_180_532_073_205, 1.0e-12),
        ref other => panic!("expected p value, got {other:?}"),
    }
    match outputs[2] {
        Value::Num(stat) => assert_close(stat, 0.201_293_674_317_076, 1.0e-12),
        ref other => panic!("expected statistic, got {other:?}"),
    }
    match outputs[3] {
        Value::Num(cv) => assert_close(cv, 0.607_361_461_908_305, 1.0e-12),
        ref other => panic!("expected critical value, got {other:?}"),
    }
}

#[test]
fn kstest_rejects_shifted_sample_and_supports_tail() {
    let sample = Value::Tensor(Tensor::new(vec![2.0, 2.5, 3.0, 3.5, 4.0], vec![5, 1]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Alpha"),
                Value::Num(0.05),
                Value::from("Tail"),
                Value::from("smaller"),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(outputs[0], Value::Bool(true));
    match outputs[1] {
        Value::Num(p) => assert!(p < 0.05),
        ref other => panic!("expected p value, got {other:?}"),
    }

    let sample = Value::Tensor(Tensor::new(vec![-4.0, -3.5, -3.0, -2.5], vec![4, 1]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Tail"),
                Value::from("larger"),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(outputs[0], Value::Bool(true));
    match outputs[1] {
        Value::Num(p) => assert!(p < 0.05),
        ref other => panic!("expected p value, got {other:?}"),
    }
}

#[test]
fn kstest_accepts_two_column_cdf_table() {
    let sample = Value::Tensor(Tensor::new(vec![0.2, 0.4, 0.6, 0.8], vec![4, 1]).unwrap());
    let cdf = Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0], vec![3, 2]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("CDF"), cdf]),
        ))
        .unwrap(),
    );
    assert_eq!(outputs[0], Value::Bool(false));
    match outputs[2] {
        Value::Num(stat) => assert_close(stat, 0.2, 1.0e-12),
        ref other => panic!("expected statistic, got {other:?}"),
    }

    let sample = Value::Tensor(Tensor::new(vec![0.25, 0.75], vec![2, 1]).unwrap());
    let unsorted_cdf = Value::Tensor(
        Tensor::new(vec![1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.4, 0.5], vec![4, 2]).unwrap(),
    );
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("CDF"),
                unsorted_cdf,
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(outputs[0], Value::Bool(false));
    match outputs[2] {
        Value::Num(stat) => assert_close(stat, 0.25, 1.0e-12),
        ref other => panic!("expected statistic, got {other:?}"),
    }

    let sample = Value::Tensor(Tensor::new(vec![-0.1, 0.2, 0.4], vec![3, 1]).unwrap());
    let cdf = Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0], vec![3, 2]).unwrap());
    let err = block_on(kstest_builtin(
        sample,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("CDF"), cdf]),
    ))
    .unwrap_err();
    assert!(err.message.contains("CDF table range"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn kstest_rejects_rank_zero_and_rank_one_cdf_tables() {
    let sample = || Value::Tensor(Tensor::new(vec![0.25, 0.75], vec![2, 1]).unwrap());
    for cdf in [
        Tensor::new(vec![0.0], Vec::new()).unwrap(),
        Tensor::new(vec![0.0, 1.0], vec![2]).unwrap(),
    ] {
        let error = block_on(kstest_builtin(
            sample(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("CDF"),
                Value::Tensor(cdf),
            ]),
        ))
        .expect_err("CDF tables must be rank-two");
        assert!(error.message.contains("n-by-2"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn kstest_accepts_rank_zero_and_rank_one_sample_vectors() {
    for sample in [
        Tensor::new(vec![0.0], Vec::new()).unwrap(),
        Tensor::new(vec![-0.5, 0.5], vec![2]).unwrap(),
    ] {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let outputs = output_list(
            block_on(kstest_builtin(
                Value::Tensor(sample),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("low-rank sample remains a vector"),
        );
        assert!(matches!(outputs.as_slice(), [Value::Bool(_)]));
    }
}

#[test]
fn kstest_reads_typed_integer_sample_and_cdf_table_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let sample = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1]), vec![2, 1]);
    let cdf = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1, 0, 1]), vec![2, 2]);
    let cdf =
        Value::Cell(runmat_value::CellArray::new(vec![cdf], 1, 1).expect("singleton CDF cell"));
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("CDF"), cdf]),
        ))
        .unwrap(),
    );
    assert_eq!(outputs[0], Value::Bool(false));
    match outputs[2] {
        Value::Num(stat) => assert!(stat.is_finite()),
        ref other => panic!("expected statistic, got {other:?}"),
    }
}

#[test]
fn kstest_gates_integer_roles_before_gather_and_rejects_lossy_values() {
    let sample = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1]), vec![2, 1]);
    let cdf = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1, 0, 1]), vec![2, 2]);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = ensure_kstest_extensions(&sample, &[])
        .expect_err("strict mode must reject integer samples before gather");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:KstestIntegerSampleExtension")
    );
    let err = ensure_kstest_extensions(&Value::Num(0.0), &[Value::from("CDF"), cdf])
        .expect_err("strict mode must reject integer CDF before gather");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:KstestIntegerCdfExtension")
    );
    drop(_strict);

    let wide =
        Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();
    let err = ensure_exact_kstest_integer_tensor(&wide, "sample")
        .expect_err("lossy integer sample must reject");
    assert!(err.message.contains("exactly representable"));
}

#[test]
fn kstest_residency_gate_distinguishes_automatic_and_explicit_handles() {
    let automatic = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let automatic = automatic
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .expect("validated provenance");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    ensure_kstest_extensions(&Value::GpuTensor(automatic.clone()), &[])
        .expect("automatic residency must retain transparent gather fallback");
    drop(automatic);

    let explicit = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let explicit = explicit
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let err = ensure_kstest_extensions(&Value::GpuTensor(explicit.clone()), &[])
        .expect_err("explicit gpuArray input must retain its compatibility gate");
    drop(explicit);
    assert_eq!(
        err.identifier(),
        KSTEST_RESIDENT_INPUT_EXTENSION.error_identifier
    );
}

#[test]
fn kstest_recursively_classifies_singleton_cell_cdf_inputs() {
    let integer_cdf = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1, 0, 1]), vec![2, 2]);
    let integer_cdf =
        Value::Cell(runmat_value::CellArray::new(vec![integer_cdf], 1, 1).expect("singleton cell"));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = ensure_kstest_extensions(&Value::Num(0.0), &[Value::from("CDF"), integer_cdf])
        .expect_err("nested integer CDF must gate");
    assert_eq!(
        err.identifier(),
        KSTEST_INTEGER_CDF_EXTENSION.error_identifier
    );

    let explicit = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let explicit = explicit
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let resident_cdf = Value::Cell(
        runmat_value::CellArray::new(vec![Value::GpuTensor(explicit.clone())], 1, 1)
            .expect("singleton cell"),
    );
    let err = ensure_kstest_extensions(&Value::Num(0.0), &[Value::from("CDF"), resident_cdf])
        .expect_err("nested explicit CDF must gate");
    drop(explicit);
    assert_eq!(
        err.identifier(),
        KSTEST_RESIDENT_INPUT_EXTENSION.error_identifier
    );
}

#[test]
fn kstest_rejects_malformed_or_text_only_roles_before_resident_access() {
    let poison = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let poison = poison
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let err = block_on(kstest_builtin(
        Value::GpuTensor(poison.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("CDF")]),
    ))
    .expect_err("arity must reject before resident sample access");
    assert_eq!(err.identifier(), Some("RunMat:kstest:InvalidArgument"));
    assert!(err.message.contains("supplied in pairs"));

    let sample = Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    let err = block_on(kstest_builtin(
        sample,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Tail"),
            Value::GpuTensor(poison.clone()),
        ]),
    ))
    .expect_err("text-only Tail must reject before resident access");
    drop(poison);
    assert_eq!(err.identifier(), Some("RunMat:kstest:InvalidArgument"));
    assert!(err.message.contains("Tail must be a string scalar"));
}

#[test]
fn kstest_rejects_bad_options_cdf_and_outputs() {
    let sample = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap());
    let err = block_on(kstest_builtin(
        sample.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Tail"),
            Value::from("sideways"),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("Tail"));

    let cdf = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.8, 0.2], vec![2, 2]).unwrap());
    let err = block_on(kstest_builtin(
        sample.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("CDF"), cdf]),
    ))
    .unwrap_err();
    assert!(err.message.contains("nondecreasing"));

    let matrix = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap());
    let err = block_on(kstest_builtin(
        matrix,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("vector"));

    let empty = Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap());
    let err = block_on(kstest_builtin(
        empty,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("non-NaN"));

    let all_nan = Value::Tensor(Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap());
    let err = block_on(kstest_builtin(
        all_nan,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("non-NaN"));

    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Alpha"),
                Value::Num(0.5),
            ]),
        ))
        .unwrap(),
    );
    match outputs[3] {
        Value::Num(cv) => assert!(cv.is_nan()),
        ref other => panic!("expected cv, got {other:?}"),
    }
    drop(_guard);

    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Alpha"),
                Value::Num(0.005),
                Value::from("Tail"),
                Value::from("smaller"),
            ]),
        ))
        .unwrap(),
    );
    match outputs[3] {
        Value::Num(cv) => assert!(cv.is_finite()),
        ref other => panic!("expected cv, got {other:?}"),
    }
    drop(_guard);

    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(kstest_builtin(
            sample.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Alpha"),
                Value::Num(0.15),
                Value::from("Tail"),
                Value::from("smaller"),
            ]),
        ))
        .unwrap(),
    );
    match outputs[3] {
        Value::Num(cv) => assert!(cv.is_nan()),
        ref other => panic!("expected cv, got {other:?}"),
    }
    drop(_guard);

    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let err = block_on(kstest_builtin(
        sample,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("too many output arguments"));
}

#[test]
fn kstest_integer_alpha_follows_the_ordinary_open_interval_rule() {
    let sample = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap());
    let err = block_on(kstest_builtin(
        sample,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Alpha"),
            poisoned_integer_scalar(IntegerStorage::U64(vec![1])),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("open interval"));
}
