use futures::executor::block_on;
use runmat_value::{CellArray, CharArray, IntegerStorage, Tensor};

use super::*;

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
    Tensor::new(data, vec![rows, cols]).unwrap()
}

fn tensor_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(tensor(data, rows, cols))
}

fn typed_integer_tensor_value(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-8,
        "expected {expected}, got {actual}"
    );
}

fn assert_matrix_close(tensor: &Tensor, rows: &[[f64; 3]]) {
    assert_eq!(tensor.shape(), [rows.len(), 3]);
    for (row_idx, expected) in rows.iter().enumerate() {
        for col in 0..3 {
            assert_close(
                tensor.materialize_f64()[col * rows.len() + row_idx],
                expected[col],
            );
        }
    }
}

#[test]
fn linkage_complete_accepts_condensed_distance_vector() {
    let y = tensor_value(vec![1.0, 4.0, 6.0, 5.0, 7.0, 2.0], 1, 6);
    let Value::Tensor(z) = block_on(linkage_builtin(
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "complete".into(),
        )]),
    ))
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_matrix_close(&z, &[[1.0, 2.0, 1.0], [3.0, 4.0, 2.0], [5.0, 6.0, 7.0]]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn linkage_preserves_rank_one_condensed_input_and_rank_zero_rejection() {
    let input = Value::Tensor(Tensor::new(vec![1.0, 4.0, 6.0, 5.0, 7.0, 2.0], vec![6]).unwrap());
    let Value::Tensor(tree) = block_on(linkage_builtin(
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "complete".into(),
        )]),
    ))
    .unwrap() else {
        panic!("expected clustering tree tensor");
    };
    assert_matrix_close(&tree, &[[1.0, 2.0, 1.0], [3.0, 4.0, 2.0], [5.0, 6.0, 7.0]]);

    let scalar = Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap());
    let error = block_on(linkage_builtin(
        scalar,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(error.contains("input must be a numeric vector or 2-D matrix"));
}

#[test]
fn linkage_condensed_vector_reads_typed_integer_storage_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = typed_integer_tensor_value(IntegerStorage::I16(vec![1, 4, 6, 5, 7, 2]), vec![1, 6]);
    let Value::Tensor(z) = block_on(linkage_builtin(
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "complete".into(),
        )]),
    ))
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_matrix_close(&z, &[[1.0, 2.0, 1.0], [3.0, 4.0, 2.0], [5.0, 6.0, 7.0]]);
}

#[test]
fn linkage_single_computes_from_observation_matrix() {
    let x = tensor_value(
        vec![
            0.0, 3.0, 4.0, 0.0, // x column
            0.0, 4.0, 0.0, 2.0, // y column
        ],
        4,
        2,
    );
    let Value::Tensor(z) = block_on(linkage_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "single".into(),
        )]),
    ))
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_matrix_close(
        &z,
        &[
            [1.0, 4.0, 2.0],
            [2.0, 5.0, 13.0_f64.sqrt()],
            [3.0, 6.0, 4.0],
        ],
    );
}

#[test]
fn linkage_column_vector_is_observation_matrix_not_condensed_vector() {
    let x = tensor_value(vec![0.0, 1.0, 2.0], 3, 1);
    let Value::Tensor(z) = block_on(linkage_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "single".into(),
        )]),
    ))
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_matrix_close(&z, &[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]);
}

#[test]
fn linkage_average_uses_pdist_metric_cell() {
    let x = tensor_value(
        vec![
            0.0, 3.0, 4.0, 0.0, // x column
            0.0, 4.0, 0.0, 2.0, // y column
        ],
        4,
        2,
    );
    let metric = Value::Cell(
        CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("minkowski")),
                Value::Num(3.0),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let Value::Tensor(z) = block_on(linkage_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("average".into()),
            metric,
        ]),
    ))
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_eq!(z.shape(), [3, 3]);
    assert_close(z.materialize_f64()[6], 2.0);
}

#[test]
fn linkage_ward_matches_singleton_euclidean_first_merge() {
    let x = tensor_value(
        vec![
            0.0, 3.0, 4.0, 0.0, // x column
            0.0, 4.0, 0.0, 2.0, // y column
        ],
        4,
        2,
    );
    let Value::Tensor(z) = block_on(linkage_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "ward".into(),
        )]),
    ))
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_eq!(z.shape(), [3, 3]);
    assert_close(z.materialize_f64()[6], 2.0);
}

#[test]
fn linkage_rejects_bad_distance_vector_length() {
    let err = block_on(linkage_builtin(
        tensor_value(vec![1.0, 2.0], 1, 2),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("condensed distance vector length"));
}

#[test]
fn linkage_preflights_resource_geometry_before_resident_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let oversized = tensor(vec![0.0; 701], 701, 1);
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &oversized)
            .expect("upload observations");
        let error = block_on(linkage_builtin(
            Value::GpuTensor(resident),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("historical observation cap must be checked before download");
        assert!(error.message().contains("too many observations"));

        let complex =
            runmat_value::ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).expect("complex input");
        let resident =
            crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &complex)
                .expect("upload complex input");
        let error = block_on(linkage_builtin(
            Value::GpuTensor(resident),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("physical complex storage must be rejected before download");
        assert!(error.message().contains("real numeric"));
    });
}

#[test]
fn linkage_rejects_metric_for_condensed_vector() {
    let err = block_on(linkage_builtin(
        tensor_value(vec![1.0, 4.0, 6.0], 1, 3),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("single".into()),
            Value::String("cityblock".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("does not accept a distance metric"));
}

#[test]
fn shared_metric_admission_uses_linkage_error_identity() {
    let error = block_on(linkage_builtin(
        tensor_value(vec![0.0, 1.0, 2.0], 3, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("single"),
            Value::from("unsupported-metric"),
        ]),
    ))
    .expect_err("unknown metric must fail at the linkage boundary");
    assert_eq!(error.identifier(), Some("RunMat:linkage:InvalidArgument"));

    let error = block_on(linkage_builtin(
        tensor_value(vec![0.0, 1.0, 2.0], 3, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("single"),
            Value::from("minkowski"),
            Value::Num(-2.0),
        ]),
    ))
    .expect_err("shared Minkowski parameter admission must use linkage identity");
    assert_eq!(error.identifier(), Some("RunMat:linkage:InvalidArgument"));
}

#[test]
fn linkage_rejects_nonfinite_distances() {
    let err = block_on(linkage_builtin(
        tensor_value(vec![1.0, f64::INFINITY, 2.0], 1, 3),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("distances must be finite"));
}

#[test]
fn linkage_rejects_noneuclidean_condensed_distances_for_ward_family() {
    let err = block_on(linkage_builtin(
        tensor_value(vec![1.0, 1.0, 3.0], 1, 3),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "ward".into(),
        )]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("require Euclidean"));
}

#[test]
fn linkage_savememory_requires_text_and_euclidean_ward_family() {
    let x = tensor_value(vec![0.0, 1.0, 2.0], 3, 1);
    let err = block_on(linkage_builtin(
        x.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("single".into()),
            Value::String("euclidean".into()),
            Value::String("savememory".into()),
            Value::String("on".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("centroid, median, or ward"));

    let err = block_on(linkage_builtin(
        x.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("ward".into()),
            Value::String("cityblock".into()),
            Value::String("savememory".into()),
            Value::String("on".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("Euclidean distance metric"));

    let err = block_on(linkage_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("ward".into()),
            Value::String("euclidean".into()),
            Value::String("savememory".into()),
            Value::Bool(true),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("must be 'on' or 'off'"));
}

#[test]
fn linkage_integer_roles_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let data_error = block_on(linkage_builtin(
        typed_integer_tensor_value(IntegerStorage::U16(vec![1, 2, 3]), vec![1, 3]),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(
        data_error.identifier(),
        Some("RunMat:compatibility:LinkageIntegerDataExtension")
    );

    let metric = Value::Cell(
        CellArray::new(
            vec![
                Value::from("minkowski"),
                Value::Int(runmat_value::IntValue::U16(3)),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let parameter_error = block_on(linkage_builtin(
        tensor_value(vec![0.0, 1.0, 2.0], 3, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("average"),
            metric,
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        parameter_error.identifier(),
        Some("RunMat:compatibility:LinkageIntegerDistanceParameterExtension")
    );
}

#[test]
fn linkage_rejects_lossy_integer_data_and_distance_parameters() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let data_error = block_on(linkage_builtin(
        typed_integer_tensor_value(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(data_error
        .message()
        .contains("exactly representable as double"));

    let metric = Value::Cell(
        CellArray::new(
            vec![
                Value::from("minkowski"),
                Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let parameter_error = block_on(linkage_builtin(
        tensor_value(vec![0.0, 1.0, 2.0], 3, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("average"),
            metric,
        ]),
    ))
    .unwrap_err();
    assert!(parameter_error
        .message()
        .contains("exactly representable as double"));
}

#[test]
fn linkage_explicit_gpu_fallback_is_gated_before_download() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 3],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let error = block_on(linkage_builtin(
        Value::GpuTensor(handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:LinkageExplicitGpuInputExtension")
    );
    drop(handle);
}

#[test]
fn linkage_automatic_residency_gathers_transparently() {
    use crate::builtins::common::{gpu_helpers, test_support};

    test_support::with_test_provider(|provider| {
        let input = tensor(vec![0.0, 1.0, 2.0], 3, 1);
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("validated provenance");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let Value::Tensor(output) = block_on(linkage_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("automatic residency may gather transparently") else {
            panic!("linkage remains a host implementation");
        };
        assert_matrix_close(&output, &[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]);
    });
}
