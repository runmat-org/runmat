use super::*;
use futures::executor::block_on;
use runmat_builtins::builtin_function_by_name;
use runmat_value::{IntegerStorage, NumericStorage};

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
}

fn classify(sample: Value, training: Value, group: Value, rest: Vec<Value>) -> Vec<Value> {
    let Value::OutputList(values) =
        block_on(classify_builtin(sample, training, group, rest)).expect("classify")
    else {
        panic!("expected output list");
    };
    values
}

fn with_outputs<T>(count: usize, f: impl FnOnce() -> T) -> T {
    let _guard = crate::output_count::push_output_count(Some(count));
    f()
}

fn valid_floating_inputs() -> (Value, Value, Value) {
    (
        tensor(vec![0.2, 2.2], vec![2, 1]),
        tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]),
        tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
    )
}

fn all_integer_storages(values: &[i8]) -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(values.to_vec()),
        IntegerStorage::I16(values.iter().map(|value| i16::from(*value)).collect()),
        IntegerStorage::I32(values.iter().map(|value| i32::from(*value)).collect()),
        IntegerStorage::I64(values.iter().map(|value| i64::from(*value)).collect()),
        IntegerStorage::U8(values.iter().map(|value| *value as u8).collect()),
        IntegerStorage::U16(values.iter().map(|value| *value as u16).collect()),
        IntegerStorage::U32(values.iter().map(|value| *value as u32).collect()),
        IntegerStorage::U64(values.iter().map(|value| *value as u64).collect()),
    ]
}

#[test]
fn classify_registers_independent_integer_extensions_and_capabilities() {
    let builtin = builtin_function_by_name(CLASSIFY_NAME).expect("classify builtin");
    assert_eq!(builtin.extensions.len(), 6);
    assert_eq!(builtin.integer_capabilities.len(), 4);
    assert!(builtin
        .integer_capabilities
        .iter()
        .all(|capability| capability.inputs[0].classes.len() == 8));
    assert_eq!(
        builtin.integer_capabilities[2].output_class,
        BuiltinIntegerOutputClassRule::PreserveInput
    );
}

#[test]
fn classify_linear_predicts_numeric_groups() {
    with_outputs(5, || {
        let values = classify(
            tensor(vec![0.2, 2.2], vec![2, 1]),
            tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]),
            tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
            Vec::new(),
        );
        let Value::Tensor(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.as_f64_slice(), Some(&[1.0, 2.0][..]));
        assert!(matches!(values[1], Value::Num(err) if err <= EPS));
        let Value::Tensor(posterior) = &values[2] else {
            panic!("posterior");
        };
        assert_eq!(posterior.shape, vec![2, 2]);
        let posterior = posterior.as_f64_slice().expect("double posterior");
        assert!(posterior[0] > posterior[2]);
        assert!(posterior[3] > posterior[1]);
        let Value::Cell(coeff) = &values[4] else {
            panic!("coeff");
        };
        assert_eq!(coeff.rows, 2);
        assert_eq!(coeff.cols, 2);
        assert!(matches!(&coeff.data[1], Value::Struct(st) if st.fields.contains_key("linear")));
    });
}

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
        let err = block_on(classify_builtin(sample, training, group, rest))
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
        Vec::new(),
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
        Vec::new(),
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
        vec![Value::from("linear"), Value::Bool(true)],
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
        Vec::new(),
    ))
    .expect_err("wide sample must reject");
    assert!(err.message().contains("sample integer values"));

    let err = block_on(classify_builtin(
        sample.clone(),
        int_tensor(IntegerStorage::U64(vec![0, 1, 2, wide]), vec![4, 1]),
        group.clone(),
        Vec::new(),
    ))
    .expect_err("wide training must reject");
    assert!(err.message().contains("training integer values"));

    let err = block_on(classify_builtin(
        sample,
        training,
        group,
        vec![
            Value::from("linear"),
            int_tensor(IntegerStorage::U64(vec![1, wide]), vec![2, 1]),
        ],
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
            Vec::new(),
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
            Vec::new(),
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
            Vec::new(),
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
            vec![Value::from("unsupported")],
        ))
        .expect_err("invalid type must be rejected before materialization");
        assert!(err.message().contains("unsupported discriminant type"));

        let err = block_on(classify_builtin(
            Value::GpuTensor(handle),
            tensor(vec![0.0, 0.5, 2.0, 2.5, 0.0, 0.5, 2.0, 2.5], vec![4, 2]),
            group,
            Vec::new(),
        ))
        .expect_err("known mismatched columns must be rejected before materialization");
        assert!(err.message().contains("same number of columns"));

        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &sample).unwrap();
        let err = block_on(classify_builtin(
            Value::GpuTensor(handle),
            training,
            tensor(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]),
            Vec::new(),
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
        Vec::new(),
    ))
    .expect("classify");
    let Value::Tensor(labels) = result else {
        panic!("expected host label tensor");
    };
    assert_eq!(labels.materialize_f64(), vec![1.0, 2.0]);
}

#[test]
fn classify_quadratic_preserves_text_labels() {
    with_outputs(5, || {
        let groups = Value::StringArray(
            StringArray::new(
                vec!["low".into(), "low".into(), "high".into(), "high".into()],
                vec![4, 1],
            )
            .unwrap(),
        );
        let values = classify(
            tensor(vec![0.1, 3.2], vec![2, 1]),
            tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
            groups,
            vec![Value::String("quadratic".into())],
        );
        let Value::StringArray(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.data, vec!["low", "high"]);
        let Value::Cell(coeff) = &values[4] else {
            panic!("coeff");
        };
        assert!(matches!(&coeff.data[1], Value::Struct(st) if st.fields.contains_key("quadratic")));
    });
}

#[test]
fn classify_empirical_prior_and_missing_labels() {
    with_outputs(3, || {
        let values = classify(
            tensor(vec![0.1, 10.0], vec![2, 1]),
            tensor(vec![0.0, 0.2, 10.0, 10.2, 99.0], vec![5, 1]),
            tensor(vec![1.0, 1.0, 2.0, 2.0, f64::NAN], vec![5, 1]),
            vec![
                Value::String("diagLinear".into()),
                Value::String("empirical".into()),
            ],
        );
        let Value::Tensor(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.as_f64_slice(), Some(&[1.0, 2.0][..]));
        let Value::Tensor(posterior) = &values[2] else {
            panic!("posterior");
        };
        assert_eq!(posterior.shape, vec![2, 2]);
    });
}

#[test]
fn classify_preserves_char_matrix_labels() {
    with_outputs(1, || {
        let groups = Value::CharArray(
            CharArray::new(vec!['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], 4, 2).unwrap(),
        );
        let values = classify(
            tensor(vec![0.1, 3.2], vec![2, 1]),
            tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
            groups,
            Vec::new(),
        );
        let Value::CharArray(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.rows, 2);
        assert_eq!(labels.cols, 2);
        assert_eq!(labels.data, vec!['a', 'a', 'b', 'b']);
    });
}

#[test]
fn classify_accepts_cellstr_labels_and_struct_priors() {
    with_outputs(2, || {
        let groups = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(CharArray::new_row("left")),
                    Value::CharArray(CharArray::new_row("left")),
                    Value::CharArray(CharArray::new_row("right")),
                    Value::CharArray(CharArray::new_row("right")),
                ],
                4,
                1,
            )
            .unwrap(),
        );
        let mut prior = StructValue::new();
        prior.insert(
            "group",
            Value::StringArray(
                StringArray::new(vec!["left".into(), "right".into()], vec![2, 1]).unwrap(),
            ),
        );
        prior.insert(
            "prob",
            Value::Tensor(Tensor::new(vec![0.25, 0.75], vec![2, 1]).unwrap()),
        );
        let values = classify(
            tensor(vec![0.1, 3.2], vec![2, 1]),
            tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
            groups,
            vec![Value::String("linear".into()), Value::Struct(prior)],
        );
        let Value::Cell(labels) = &values[0] else {
            panic!("labels");
        };
        assert_eq!(labels.rows, 2);
        assert!(matches!(values[1], Value::Num(err) if err <= EPS));
    });
}

#[test]
fn classify_rejects_bad_dimensions() {
    let err = block_on(classify_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(err.message.contains("same number of columns"));
}
