use super::*;
use runmat_value::{IntegerStorage, LogicalArray, NumericStorage};

#[test]
fn classify_accepts_typed_integer_numeric_inputs_and_groups() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    with_outputs(2, || {
        let values = classify(
            int_tensor(IntegerStorage::I16(vec![0, 2]), vec![2, 1]),
            int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]),
            int_tensor(IntegerStorage::U8(vec![1, 1, 2, 2]), vec![4, 1]),
            Vec::new(),
        );
        let Value::Tensor(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(
            labels.integer_storage(),
            Some(&IntegerStorage::U8(vec![1, 2]))
        );
        assert!(matches!(values[1], Value::Num(err) if err <= EPS));
    });
}

#[test]
fn classify_preserves_native_single_and_wide_integer_group_labels() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    with_outputs(1, || {
        let values = classify(
            tensor(vec![0.0, 3.0], vec![2, 1]),
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
            Value::Tensor(
                Tensor::from_numeric_storage(
                    NumericStorage::F32(vec![1.0, 1.0, 2.0, 2.0]),
                    vec![4, 1],
                )
                .expect("single labels"),
            ),
            Vec::new(),
        );
        let Value::Tensor(labels) = values.into_iter().next().expect("single output") else {
            panic!("labels");
        };
        assert_eq!(
            labels.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![1.0, 2.0])
        );
    });

    with_outputs(1, || {
        let lower = u64::MAX - 1;
        let upper = u64::MAX;
        let values = classify(
            tensor(vec![0.0, 3.0], vec![2, 1]),
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
            int_tensor(
                IntegerStorage::U64(vec![lower, lower, upper, upper]),
                vec![4, 1],
            ),
            Vec::new(),
        );
        let Value::Tensor(labels) = values.into_iter().next().expect("integer output") else {
            panic!("labels");
        };
        assert_eq!(
            labels.into_numeric_storage().expect("integer storage"),
            NumericStorage::U64(vec![lower, upper])
        );
    });
}

#[test]
fn classify_checked_integer_solver_boundary_accepts_exact_wide_values() {
    for exact in [1_i64 << 54, i64::MIN] {
        ensure_exact_integer_scalar_boundary(NumericScalar::I64(exact), "sample")
            .expect("exact wide integer is representable in binary64");
    }
    let inexact = (1_i64 << 53) + 1;
    let err = ensure_exact_integer_scalar_boundary(NumericScalar::I64(inexact), "sample")
        .expect_err("adjacent wide integer loses binary64 precision");
    assert!(err.message().contains("exactly representable"));
}

#[test]
fn classify_preserves_all_integer_group_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_integer_storages(&[1, 1, 2, 2]) {
        let expected_dtype = storage.numeric_dtype();
        with_outputs(1, || {
            let (sample, training, _) = valid_floating_inputs();
            let values = classify(
                sample,
                training,
                int_tensor(storage, vec![4, 1]),
                Vec::new(),
            );
            let Value::Tensor(labels) = &values[0] else {
                panic!("integer labels");
            };
            assert_eq!(labels.numeric_dtype(), expected_dtype);
            assert_eq!(tensor::tensor_value_f64(labels, 0), 1.0);
            assert_eq!(tensor::tensor_value_f64(labels, 1), 2.0);
        });
    }
}

#[test]
fn classify_executes_all_integer_predictor_and_prior_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for ((sample, training), prior) in all_integer_storages(&[0, 2])
        .into_iter()
        .zip(all_integer_storages(&[0, 1, 2, 3]))
        .zip(all_integer_storages(&[1, 1]))
    {
        with_outputs(1, || {
            let values = classify(
                int_tensor(sample, vec![2, 1]),
                int_tensor(training, vec![4, 1]),
                tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
                vec![Value::from("linear"), int_tensor(prior, vec![2, 1])],
            );
            assert_eq!(values.len(), 1);
        });
    }
}

#[test]
fn classify_integer_roles_are_independently_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let cases = [
        (
            int_tensor(IntegerStorage::U8(vec![0, 2]), vec![2, 1]),
            tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]),
            tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
            Vec::new(),
            CLASSIFY_INTEGER_SAMPLE_EXTENSION.error_identifier,
        ),
        (
            tensor(vec![0.2, 2.2], vec![2, 1]),
            int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]),
            tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
            Vec::new(),
            CLASSIFY_INTEGER_TRAINING_EXTENSION.error_identifier,
        ),
        (
            tensor(vec![0.2, 2.2], vec![2, 1]),
            tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]),
            int_tensor(IntegerStorage::U32(vec![1, 1, 2, 2]), vec![4, 1]),
            Vec::new(),
            CLASSIFY_INTEGER_GROUP_EXTENSION.error_identifier,
        ),
        (
            tensor(vec![0.2, 2.2], vec![2, 1]),
            tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]),
            tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
            vec![
                Value::from("linear"),
                int_tensor(IntegerStorage::U16(vec![1, 1]), vec![2, 1]),
            ],
            CLASSIFY_INTEGER_PRIOR_EXTENSION.error_identifier,
        ),
    ];
    for (sample, training, group, rest, identifier) in cases {
        let err = block_on(classify_builtin(
            sample,
            training,
            group,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect_err("integer role must be gated");
        assert_eq!(err.identifier(), identifier);
    }
}

#[test]
fn classify_logical_numeric_roles_are_independently_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let (sample, training, group) = valid_floating_inputs();
    let logical_sample =
        Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![2, 1]).expect("logical sample"));
    let err = block_on(classify_builtin(
        logical_sample,
        training.clone(),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("logical sample must be gated");
    assert_eq!(
        err.identifier(),
        CLASSIFY_LOGICAL_PREDICTOR_EXTENSION.error_identifier
    );

    let logical_training = Value::LogicalArray(
        LogicalArray::new(vec![0, 0, 1, 1], vec![4, 1]).expect("logical training"),
    );
    let err = block_on(classify_builtin(
        sample.clone(),
        logical_training,
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("logical training must be gated");
    assert_eq!(
        err.identifier(),
        CLASSIFY_LOGICAL_PREDICTOR_EXTENSION.error_identifier
    );

    let err = block_on(classify_builtin(
        sample,
        training,
        group,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("linear"),
            Value::Bool(true),
        ]),
    ))
    .expect_err("logical prior must reject");
    assert_eq!(err.identifier(), Some("RunMat:classify:InvalidArgument"));
}

#[test]
fn classify_guards_wide_integer_floating_boundaries_but_not_labels() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = (1_u64 << 53) + 1;
    let (sample, training, group) = valid_floating_inputs();
    let err = block_on(classify_builtin(
        int_tensor(IntegerStorage::U64(vec![0, wide]), vec![2, 1]),
        training.clone(),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("wide sample must reject");
    assert!(err.message().contains("sample integer values"));

    let err = block_on(classify_builtin(
        sample.clone(),
        int_tensor(IntegerStorage::U64(vec![0, 1, 2, wide]), vec![4, 1]),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("wide training must reject");
    assert!(err.message().contains("training integer values"));

    let err = block_on(classify_builtin(
        sample,
        training,
        group,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("linear"),
            int_tensor(IntegerStorage::U64(vec![1, wide]), vec![2, 1]),
        ]),
    ))
    .expect_err("wide prior must reject");
    assert!(err.message().contains("prior integer values"));
}

#[test]
fn classify_resident_integer_gate_precedes_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let sample = Tensor::new_integer(IntegerStorage::U8(vec![0, 2]), vec![2, 1])
            .expect("integer sample");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &sample)
            .expect("integer upload");
        let (_, training, group) = valid_floating_inputs();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(classify_builtin(
            Value::GpuTensor(handle.clone()),
            training,
            group,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("resident integer sample must gate before gather");
        assert_eq!(
            err.identifier(),
            CLASSIFY_INTEGER_SAMPLE_EXTENSION.error_identifier
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            Some(runmat_accelerate_api::IntegerElementType::U8)
        );
        let floating = Tensor::new(vec![0.2, 2.2], vec![2, 1]).expect("floating sample");
        let floating_handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &floating)
                .expect("floating upload");
        let (_, training, group) = valid_floating_inputs();
        let err = block_on(classify_builtin(
            Value::GpuTensor(floating_handle.clone()),
            training,
            group,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("resident floating input must be independently gated");
        assert_eq!(
            err.identifier(),
            CLASSIFY_RESIDENT_INPUT_EXTENSION.error_identifier
        );
        let logical_handle = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1],
                shape: &[2, 1],
            })
            .expect("logical upload");
        let (_, training, group) = valid_floating_inputs();
        let err = block_on(classify_builtin(
            Value::GpuTensor(logical_handle.clone()),
            training,
            group,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("resident logical sample must gate by type before residency");
        assert_eq!(
            err.identifier(),
            CLASSIFY_LOGICAL_PREDICTOR_EXTENSION.error_identifier
        );
        drop(handle);
        drop(floating_handle);
        drop(logical_handle);
    });
}

#[test]
fn classify_admits_known_geometry_and_type_before_resident_materialization() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let sample = Tensor::new(vec![0.2, 2.2], vec![2, 1]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &sample).unwrap();
        let (_, training, group) = valid_floating_inputs();
        let err = block_on(classify_builtin(
            Value::GpuTensor(handle.clone()),
            training.clone(),
            group.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "unsupported",
            )]),
        ))
        .expect_err("invalid type must be rejected before materialization");
        assert!(err.message().contains("unsupported discriminant type"));

        let err = block_on(classify_builtin(
            Value::GpuTensor(handle),
            tensor(vec![0.0, 0.5, 2.0, 2.5, 0.0, 0.5, 2.0, 2.5], vec![4, 2]),
            group,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("known mismatched columns must be rejected before materialization");
        assert!(err.message().contains("same number of columns"));

        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &sample).unwrap();
        let err = block_on(classify_builtin(
            Value::GpuTensor(handle),
            training,
            tensor(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("known matrix labels must be rejected before materialization");
        assert!(err.message().contains("group must be a vector"));
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn classify_wgpu_integer_sample_uses_checked_host_fallback() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("wgpu provider");
    let sample = Tensor::new_integer(IntegerStorage::U8(vec![0, 3]), vec![2, 1]).unwrap();
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &sample)
        .expect("integer upload");
    let (_, training, group) = valid_floating_inputs();
    let result = block_on(classify_builtin(
        Value::GpuTensor(handle),
        training,
        group,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("classify");
    let Value::Tensor(labels) = result else {
        panic!("expected host label tensor");
    };
    assert_eq!(labels.materialize_f64(), vec![1.0, 2.0]);
}
