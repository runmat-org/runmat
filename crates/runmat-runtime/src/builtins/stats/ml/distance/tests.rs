use super::*;
use futures::executor::block_on;
use runmat_builtins::{KNNSEARCH_EXTENSIONS, KNNSEARCH_INTEGER_CAPABILITIES};
use runmat_value::IntegerStorage;
use runmat_value::NumericDType;

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new_integer(storage, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn cleared_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn tensor_out(value: Value) -> Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn knnsearch_documented_floating_gpu_fallback_preserves_explicit_residency() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.0, 2.0, 5.0], vec![3, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x)
            .expect("upload observations");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let output = block_on(knnsearch_builtin(
            Value::GpuTensor(handle),
            tensor(vec![1.0, 4.0], 2, 1),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("NSMethod"),
                Value::from("exhaustive"),
            ]),
        ))
        .expect("documented resident knnsearch");
        assert!(matches!(output, Value::GpuTensor(_)));
    });
}

#[test]
fn knnsearch_requested_outputs_preserve_padding_and_distinct_resident_allocations() {
    {
        let _requested =
            crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let output = block_on(knnsearch_builtin(
            tensor(vec![0.0, 2.0], 2, 1),
            tensor(vec![1.0], 1, 1),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        assert!(matches!(output, Value::OutputList(values) if values.is_empty()));
    }
    {
        let _requested =
            crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let output = block_on(knnsearch_builtin(
            tensor(vec![0.0, 2.0], 2, 1),
            tensor(vec![1.0], 1, 1),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let Value::OutputList(values) = output else {
            panic!("expected padded outputs");
        };
        assert_eq!(values.len(), 3);
        assert!(matches!(values[2], Value::Num(0.0)));
    }
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = crate::builtins::common::gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![0.0, 2.0], vec![2, 1]).unwrap(),
        )
        .expect("upload observations")
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("explicit provenance");
        let _requested =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let output = block_on(knnsearch_builtin(
            Value::GpuTensor(source.clone()),
            tensor(vec![1.0], 1, 1),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("NSMethod"),
                Value::from("exhaustive"),
            ]),
        ))
        .expect("resident outputs");
        let Value::OutputList(values) = output else {
            panic!("expected two resident outputs");
        };
        let (Value::GpuTensor(indices), Value::GpuTensor(distances)) = (&values[0], &values[1])
        else {
            panic!("expected two resident tensors");
        };
        assert!(!indices.shares_allocation_with(&source));
        assert!(!distances.shares_allocation_with(&source));
        assert!(!indices.shares_allocation_with(distances));
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn knnsearch_wgpu_fallback_enforces_explicit_residency_contract() {
    let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU provider");
    let x = Tensor::new(vec![0.0, 2.0, 5.0], vec![3, 1]).unwrap();
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x)
        .expect("upload observations");
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let result = block_on(knnsearch_builtin(
        Value::GpuTensor(handle.clone()),
        tensor(vec![1.0, 4.0], 2, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("NSMethod"),
            Value::from("exhaustive"),
        ]),
    ));
    let Value::GpuTensor(output) = result.expect("documented WGPU knnsearch") else {
        panic!("expected resident output");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert_eq!(
        output.device_id(),
        runmat_accelerate_api::AccelProvider::device_id(provider)
    );
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(runmat_accelerate_api::ProviderPrecision::F64)
    );
}

#[test]
fn pdist_default_and_cityblock_use_condensed_order() {
    let x = tensor(vec![0.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0, 2.0], 4, 2);
    let out = block_on(pdist_builtin(x.clone(), Vec::new())).unwrap();
    let data = tensor_out(out).materialize_f64();
    assert!((data[0] - 5.0).abs() < 1.0e-10);
    assert!((data[1] - 4.0).abs() < 1.0e-10);
    assert!((data[2] - 2.0).abs() < 1.0e-10);
    assert!((data[3] - 17.0_f64.sqrt()).abs() < 1.0e-10);
    assert!((data[4] - 13.0_f64.sqrt()).abs() < 1.0e-10);
    assert!((data[5] - 20.0_f64.sqrt()).abs() < 1.0e-10);

    let out = block_on(pdist_builtin(x, vec![Value::from("cityblock")])).unwrap();
    assert_eq!(
        tensor_out(out).materialize_f64(),
        vec![7.0, 4.0, 2.0, 5.0, 5.0, 6.0]
    );
}

#[test]
fn pdist_supports_parameterized_metrics() {
    let x = tensor(vec![0.0, 3.0, 4.0, 0.0, 4.0, 0.0], 3, 2);
    let out = block_on(pdist_builtin(
        x.clone(),
        vec![Value::from("minkowski"), Value::Num(1.0)],
    ))
    .unwrap();
    assert_eq!(tensor_out(out).materialize_f64(), vec![7.0, 4.0, 5.0]);

    let out = block_on(pdist_builtin(
        x,
        vec![Value::from("seuclidean"), tensor(vec![1.0, 4.0], 1, 2)],
    ))
    .unwrap();
    assert!((tensor_out(out).materialize_f64()[0] - 10.0_f64.sqrt()).abs() < 1.0e-10);

    let out = block_on(pdist2_builtin(
        tensor(vec![0.0, 3.0, 0.0, 4.0], 2, 2),
        tensor(vec![0.0, 6.0], 1, 2),
        vec![Value::from("seuclidean")],
    ))
    .unwrap();
    assert!((tensor_out(out).materialize_f64()[0] - 4.5_f64.sqrt()).abs() < 1.0e-10);
}

#[test]
fn euclidean_and_minkowski_keep_representable_large_distances_finite() {
    let observations = tensor(vec![0.0, 1.0e200, 0.0, 1.0e200], 2, 2);
    let expected = 2.0_f64.sqrt() * 1.0e200;
    for options in [
        vec![Value::from("euclidean")],
        vec![Value::from("minkowski"), Value::Num(2.0)],
    ] {
        let distance =
            tensor_out(block_on(pdist_builtin(observations.clone(), options.clone())).unwrap())
                .materialize_f64()[0];
        assert!(distance.is_finite());
        assert!((distance / expected - 1.0).abs() < 1.0e-12);

        let distance = tensor_out(
            block_on(pdist2_builtin(
                tensor(vec![0.0, 0.0], 1, 2),
                tensor(vec![1.0e200, 1.0e200], 1, 2),
                options,
            ))
            .unwrap(),
        )
        .materialize_f64()[0];
        assert!(distance.is_finite());
        assert!((distance / expected - 1.0).abs() < 1.0e-12);
    }

    let neighbor = tensor_out(
        block_on(knnsearch_builtin(
            tensor(vec![1.0e200, 2.0e200], 2, 1),
            tensor(vec![0.0], 1, 1),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    assert_eq!(neighbor.materialize_f64(), vec![1.0]);
}

#[test]
fn distance_helpers_read_typed_integer_inputs_and_parameters_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = poisoned_int_tensor(IntegerStorage::I16(vec![0, 3, 4, 0, 4, 0]), 3, 2);
    let out = block_on(pdist_builtin(
        x,
        vec![
            Value::from("minkowski"),
            cleared_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
        ],
    ))
    .unwrap();
    assert_eq!(tensor_out(out).materialize_f64(), vec![7.0, 4.0, 5.0]);

    let out = block_on(pdist_builtin(
        int_tensor(IntegerStorage::I16(vec![0, 3, 4, 0, 4, 0]), 3, 2),
        vec![
            Value::from("seuclidean"),
            poisoned_int_tensor(IntegerStorage::U16(vec![1, 4]), 1, 2),
        ],
    ))
    .unwrap();
    assert!((tensor_out(out).materialize_f64()[0] - 10.0_f64.sqrt()).abs() < 1.0e-10);

    let out = block_on(pdist_builtin(
        int_tensor(IntegerStorage::I16(vec![0, 1, 0, 2]), 2, 2),
        vec![
            Value::from("mahalanobis"),
            poisoned_int_tensor(IntegerStorage::U16(vec![1, 0, 0, 1]), 2, 2),
        ],
    ))
    .unwrap();
    assert!((tensor_out(out).materialize_f64()[0] - 5.0_f64.sqrt()).abs() < 1.0e-10);
}

#[test]
fn nan_rows_propagate_and_invalid_covariance_is_rejected() {
    let x = tensor(vec![0.0, f64::NAN, 0.0, 1.0], 2, 2);
    let out = block_on(pdist_builtin(x, vec![Value::from("hamming")])).unwrap();
    assert!(tensor_out(out).materialize_f64()[0].is_nan());

    let err = block_on(pdist_builtin(
        tensor(vec![0.0, 1.0, 0.0, 1.0], 2, 2),
        vec![
            Value::from("mahalanobis"),
            tensor(vec![1.0, 2.0, 0.0, 1.0], 2, 2),
        ],
    ))
    .expect_err("asymmetric covariance should fail");
    assert!(err.to_string().contains("symmetric positive definite"));
}

#[test]
fn pdist2_matrix_and_selection_modes_work() {
    let x = tensor(vec![0.0, 2.0, 0.0, 0.0], 2, 2);
    let y = tensor(vec![1.0, 3.0, 0.0, 0.0], 2, 2);
    let out = block_on(pdist2_builtin(
        x.clone(),
        y.clone(),
        vec![Value::from("squaredeuclidean")],
    ))
    .unwrap();
    assert_eq!(tensor_out(out).materialize_f64(), vec![1.0, 1.0, 9.0, 1.0]);

    let out = block_on(pdist2_builtin(
        x,
        y,
        vec![
            Value::from("euclidean"),
            Value::from("Smallest"),
            Value::Num(1.0),
        ],
    ))
    .unwrap();
    let tensor = tensor_out(out);
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 1.0]);
}

#[test]
fn pdist2_selection_reads_typed_integer_count_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(pdist2_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 2, 0, 0]), 2, 2),
        poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 0, 0]), 2, 2),
        vec![
            Value::from("euclidean"),
            Value::from("Smallest"),
            cleared_int_tensor(IntegerStorage::U64(vec![1]), 1, 1),
        ],
    ))
    .unwrap();
    let tensor = tensor_out(out);
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 1.0]);

    assert!(parse_positive_integer(
        PDIST2_NAME,
        &Value::Num(usize::MAX as f64),
        "pdist2 selection count"
    )
    .is_err());
    assert!(parse_positive_integer(
        PDIST2_NAME,
        &Value::Num(usize::MAX as f64 + 1.0),
        "pdist2 selection count"
    )
    .is_err());
}

#[test]
fn pdist2_selection_extension_uses_typed_case_insensitive_keyword() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for spelling in ["Smallest", "smallest", "LARGEST"] {
        let error = block_on(pdist2_builtin(
            tensor(vec![0.0, 2.0], 2, 1),
            tensor(vec![1.0], 1, 1),
            vec![
                Value::from(spelling),
                cleared_int_tensor(IntegerStorage::U16(vec![1]), 1, 1),
            ],
        ))
        .expect_err("typed selection count must be gated before the provider or CPU work");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Pdist2IntegerSelectionCountExtension")
        );
    }
}

#[test]
fn knnsearch_returns_indices_then_distances() {
    let x = tensor(vec![0.0, 2.0, 5.0, 0.0, 0.0, 0.0], 3, 2);
    let y = tensor(vec![1.0, 4.0, 0.0, 0.0], 2, 2);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(knnsearch_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let Value::OutputList(values) = out else {
        panic!("expected output list");
    };
    let idx = tensor_out(values[0].clone());
    let dist = tensor_out(values[1].clone());
    assert_eq!(idx.shape(), &[2, 2]);
    assert_eq!(dist.shape(), &[2, 2]);
    assert_eq!(idx.materialize_f64(), vec![1.0, 3.0, 2.0, 2.0]);
    assert_eq!(dist.materialize_f64(), vec![1.0, 1.0, 1.0, 2.0]);
}

#[test]
fn knnsearch_supports_metrics_and_ties() {
    let x = tensor(vec![0.0, 2.0, -2.0, 0.0, 0.0, 0.0], 3, 2);
    let y = tensor(vec![0.0, 0.0], 1, 2);
    let out = block_on(knnsearch_builtin(
        x.clone(),
        y.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Distance"),
            Value::from("cityblock"),
            Value::from("K"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let idx = tensor_out(out);
    assert_eq!(idx.shape(), &[1, 2]);
    assert_eq!(idx.materialize_f64(), vec![1.0, 2.0]);

    let out = block_on(knnsearch_builtin(
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(2.0),
            Value::from("IncludeTies"),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    let Value::Cell(cells) = out else {
        panic!("expected cell output");
    };
    assert_eq!(cells.shape, vec![1, 1]);
    let Value::Tensor(tied) = &cells.data()[0] else {
        panic!("expected tensor in cell");
    };
    assert_eq!(tied.shape(), &[1, 3]);
    assert_eq!(tied.materialize_f64(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn knnsearch_ties_require_equal_distances_including_infinity() {
    let inf = block_on(knnsearch_builtin(
        tensor(vec![f64::INFINITY; 3], 3, 1),
        tensor(vec![0.0], 1, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(1.0),
            Value::from("IncludeTies"),
            Value::Bool(true),
        ]),
    ))
    .expect("infinite distances remain exact ties");
    let Value::Cell(cells) = inf else {
        panic!("expected tied-neighbor cells");
    };
    let Value::Tensor(indices) = &cells.data()[0] else {
        panic!("expected index tensor");
    };
    assert_eq!(indices.materialize_f64(), vec![1.0, 2.0, 3.0]);

    let near = block_on(knnsearch_builtin(
        tensor(vec![1.0, 1.0 + 5.0e-13], 2, 1),
        tensor(vec![0.0], 1, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(1.0),
            Value::from("IncludeTies"),
            Value::Bool(true),
        ]),
    ))
    .expect("near but unequal distances are not ties");
    let Value::Cell(cells) = near else {
        panic!("expected tied-neighbor cells");
    };
    let Value::Tensor(indices) = &cells.data()[0] else {
        panic!("expected index tensor");
    };
    assert_eq!(indices.materialize_f64(), vec![1.0]);
}

#[test]
fn knnsearch_options_read_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(knnsearch_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 2, 5, 0, 0, 0]), 3, 2),
        poisoned_int_tensor(IntegerStorage::I16(vec![1, 4, 0, 0]), 2, 2),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            cleared_int_tensor(IntegerStorage::U16(vec![2]), 1, 1),
            Value::from("IncludeTies"),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
            Value::from("SortIndices"),
            cleared_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            Value::from("CacheSize"),
            cleared_int_tensor(IntegerStorage::U32(vec![16]), 1, 1),
            Value::from("BucketSize"),
            cleared_int_tensor(IntegerStorage::U32(vec![8]), 1, 1),
        ]),
    ))
    .unwrap();
    let idx = tensor_out(out);
    assert_eq!(idx.shape(), &[2, 2]);
    assert_eq!(idx.materialize_f64(), vec![1.0, 3.0, 2.0, 2.0]);
}

#[test]
fn knnsearch_integer_data_and_controls_are_mode_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(knnsearch_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1]), 2, 1),
        tensor(vec![0.0], 1, 1),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("strict mode rejects typed observation data");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:KnnsearchIntegerObservationDataExtension")
    );
    assert_eq!(KNNSEARCH_EXTENSIONS.len(), 8);
    assert_eq!(KNNSEARCH_INTEGER_CAPABILITIES.len(), 4);
}

#[test]
fn knnsearch_runmat_floating_integer_controls_require_exact_double_values() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(knnsearch_builtin(
        tensor(vec![0.0, 1.0], 2, 1),
        tensor(vec![0.0], 1, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Distance"),
            Value::from("minkowski"),
            Value::from("P"),
            poisoned_int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), 1, 1),
        ]),
    ))
    .expect_err("wide integer P cannot cross the floating boundary");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn knnsearch_handles_nan_rows_empty_queries_and_option_validation() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(knnsearch_builtin(
        tensor(vec![0.0, f64::NAN, 2.0], 3, 1),
        tensor(vec![1.0], 1, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let Value::OutputList(values) = out else {
        panic!("expected output list");
    };
    assert_eq!(
        tensor_out(values[0].clone()).materialize_f64(),
        vec![1.0, 3.0]
    );
    assert_eq!(
        tensor_out(values[1].clone()).materialize_f64(),
        vec![1.0, 1.0]
    );

    let out = block_on(knnsearch_builtin(
        tensor(vec![0.0, 2.0], 2, 1),
        tensor(Vec::new(), 0, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(1.0),
        ]),
    ))
    .unwrap();
    let Value::OutputList(values) = out else {
        panic!("expected output list");
    };
    assert_eq!(tensor_out(values[0].clone()).shape(), &[0, 1]);
    assert_eq!(tensor_out(values[1].clone()).shape(), &[0, 1]);

    let out = block_on(knnsearch_builtin(
        tensor(vec![0.0, 2.0, -2.0], 3, 1),
        tensor(vec![0.0], 1, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(2.0),
            Value::from("IncludeTies"),
            Value::Bool(true),
            Value::from("SortIndices"),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    assert!(matches!(out, Value::OutputList(values) if matches!(&values[0], Value::Cell(_))));

    let err = block_on(knnsearch_builtin(
        tensor(vec![1.0, 0.0], 2, 1),
        tensor(vec![1.0], 1, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Distance"),
            Value::from("cosine"),
            Value::from("NSMethod"),
            Value::from("kdtree"),
        ]),
    ))
    .expect_err("kdtree should reject cosine distance");
    assert!(err.to_string().contains("NSMethod='kdtree'"));
}

#[test]
fn squareform_converts_both_directions() {
    let vector = tensor(
        vec![
            5.0,
            4.0,
            2.0,
            17.0_f64.sqrt(),
            13.0_f64.sqrt(),
            20.0_f64.sqrt(),
        ],
        1,
        6,
    );
    let matrix = tensor_out(
        block_on(squareform_builtin(
            vector,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    assert_eq!(matrix.shape(), &[4, 4]);
    assert_eq!(
        matrix.materialize_f64(),
        vec![
            0.0,
            5.0,
            4.0,
            2.0,
            5.0,
            0.0,
            17.0_f64.sqrt(),
            13.0_f64.sqrt(),
            4.0,
            17.0_f64.sqrt(),
            0.0,
            20.0_f64.sqrt(),
            2.0,
            13.0_f64.sqrt(),
            20.0_f64.sqrt(),
            0.0,
        ]
    );

    let vector = tensor_out(
        block_on(squareform_builtin(
            Value::Tensor(matrix),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    assert_eq!(vector.shape(), &[1, 6]);
    assert_eq!(
        vector.materialize_f64(),
        vec![
            5.0,
            4.0,
            2.0,
            17.0_f64.sqrt(),
            13.0_f64.sqrt(),
            20.0_f64.sqrt(),
        ]
    );
}

#[test]
fn squareform_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let vector = cleared_int_tensor(IntegerStorage::U8(vec![5, 4, 2]), 1, 3);
    let matrix = tensor_out(
        block_on(squareform_builtin(
            vector,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    assert_eq!(matrix.shape(), &[3, 3]);
    assert_eq!(matrix.numeric_dtype(), NumericDType::U8);
    assert_eq!(
        matrix.materialize_f64(),
        vec![0.0, 5.0, 4.0, 5.0, 0.0, 2.0, 4.0, 2.0, 0.0]
    );

    let matrix = poisoned_int_tensor(IntegerStorage::U8(vec![0, 5, 4, 5, 0, 2, 4, 2, 0]), 3, 3);
    let vector = tensor_out(
        block_on(squareform_builtin(
            matrix,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    assert_eq!(vector.shape(), &[1, 3]);
    assert_eq!(vector.numeric_dtype(), NumericDType::U8);
    assert_eq!(vector.materialize_f64(), vec![5.0, 4.0, 2.0]);
}

#[test]
fn squareform_preserves_logical_class_in_both_directions() {
    let input =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap());
    let matrix = block_on(squareform_builtin(
        input,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::LogicalArray(matrix) = matrix else {
        panic!("logical vector must produce a logical matrix");
    };
    assert_eq!(matrix.shape, vec![3, 3]);
    assert_eq!(matrix.data.as_slice(), &[0, 1, 0, 1, 0, 1, 0, 1, 0]);

    let vector = block_on(squareform_builtin(
        Value::LogicalArray(matrix),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("tovector")]),
    ))
    .unwrap();
    let Value::LogicalArray(vector) = vector else {
        panic!("logical matrix must produce a logical vector");
    };
    assert_eq!(vector.shape, vec![1, 3]);
    assert_eq!(vector.data.as_slice(), &[1, 0, 1]);

    let bad_diagonal =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 0, 0], vec![2, 2]).unwrap());
    assert!(block_on(squareform_builtin(
        bad_diagonal,
        crate::builtins::common::test_support::builtin_arguments(Vec::new())
    ))
    .unwrap_err()
    .message()
    .contains("diagonal"));
}

#[test]
fn squareform_gates_typed_integer_extension_in_strict_mode() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let vector = cleared_int_tensor(IntegerStorage::U8(vec![5, 4, 2]), 1, 3);
    let error = block_on(squareform_builtin(
        vector,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("typed-integer squareform must be gated in strict mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:SquareformIntegerInputExtension")
    );
}

#[test]
fn seuclidean_scale_reads_typed_integer_storage_length_without_mirror() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(pdist_builtin(
        int_tensor(IntegerStorage::I16(vec![0, 3, 4, 0, 4, 0]), 3, 2),
        vec![
            Value::from("seuclidean"),
            cleared_int_tensor(IntegerStorage::U16(vec![1, 4]), 1, 2),
        ],
    ))
    .unwrap();
    assert!((tensor_out(out).materialize_f64()[0] - 10.0_f64.sqrt()).abs() < 1.0e-10);
}

#[test]
fn count_and_logical_parsers_read_all_integer_storage_variants() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let value = poisoned_int_tensor(storage, 1, 1);
        assert_eq!(
            parse_positive_integer(KNNSEARCH_NAME, &value, "K").unwrap(),
            1
        );
        assert!(parse_logical_scalar(&value, "IncludeTies").unwrap());
    }
}

#[test]
fn knn_preflights_options_and_geometry_before_provider_access() {
    let x = tensor(vec![0.0, 1.0, 2.0], 3, 1);
    let y = tensor(vec![0.5], 1, 1);
    let bad_option = block_on(knnsearch_builtin(
        x.clone(),
        y.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("not-an-option"),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert!(bad_option.message().contains("unsupported option"));

    let bad_metric_control = block_on(knnsearch_builtin(
        x.clone(),
        y.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Distance"),
            Value::from("euclidean"),
            Value::from("P"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap_err();
    assert!(bad_metric_control.message().contains("must match"));

    let bad_k = block_on(knnsearch_builtin(
        x.clone(),
        y.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(4.0),
        ]),
    ))
    .unwrap_err();
    assert!(bad_k.message().contains("K must be <="));

    let bad_scale = block_on(knnsearch_builtin(
        x.clone(),
        y.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Distance"),
            Value::from("seuclidean"),
            Value::from("Scale"),
            tensor(vec![1.0, 2.0], 1, 2),
        ]),
    ))
    .unwrap_err();
    assert!(bad_scale.message().contains("parameter shape"));

    let bad_columns = block_on(knnsearch_builtin(
        x,
        tensor(vec![1.0, 2.0], 1, 2),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(bad_columns.message().contains("same number of columns"));
}

#[test]
fn knn_rejects_complex_resident_observations_from_physical_type() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let complex = runmat_value::ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1])
            .expect("complex observations");
        let resident =
            crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &complex)
                .expect("resident complex observations");
        let error = block_on(knnsearch_builtin(
            Value::GpuTensor(resident),
            tensor(vec![0.0], 1, 1),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("complex storage must be rejected before host gather");
        assert!(error.message().contains("real numeric or logical storage"));
    });
}

#[test]
fn knn_preserves_legacy_near_integer_floating_count() {
    let output = block_on(knnsearch_builtin(
        tensor(vec![0.0, 2.0], 2, 1),
        tensor(vec![0.5], 1, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("K"),
            Value::Num(1.0 + 5.0e-13),
        ]),
    ))
    .expect("legacy floating-count tolerance");
    assert_eq!(tensor_out(output).shape(), &[1, 1]);
}
