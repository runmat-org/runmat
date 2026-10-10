use super::*;

#[test]
fn scalar_upsample_runtime_shape_matches_catalog_axis() {
    let result = call_upsample(vec![Value::Num(7.0), Value::Num(3.0)]);
    let Value::Tensor(output) = result else {
        panic!("expanded scalar must become a tensor")
    };
    assert_eq!(output.shape(), [3, 1]);
    assert_eq!(output.materialize_f64(), vec![7.0, 0.0, 0.0]);
}

#[test]
fn first_non_singleton_axis_changes_with_concrete_leading_extent() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let singleton = call_downsample(vec![tensor(vec![1.0; 28], vec![1, 4, 7]), Value::Num(2.0)]);
    let Value::Tensor(singleton) = singleton else {
        panic!("expected tensor")
    };
    assert_eq!(singleton.shape(), [1, 2, 7]);

    let leading = call_downsample(vec![tensor(vec![1.0; 84], vec![3, 4, 7]), Value::Num(2.0)]);
    let Value::Tensor(leading) = leading else {
        panic!("expected tensor")
    };
    assert_eq!(leading.shape(), [2, 4, 7]);
}

#[test]
fn upsample_row_vector_inserts_zeros_after_each_sample() {
    let out = call_upsample(vec![
        tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
        Value::Num(2.0),
    ]);
    let Value::Tensor(upsampled) = out else {
        panic!("expected tensor");
    };
    assert_eq!(upsampled.shape(), [1, 6]);
    assert_eq!(
        upsampled.materialize_f64(),
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
    );
}

#[test]
fn upsample_column_vector_honors_phase() {
    let out = call_upsample(vec![
        tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
        Value::Num(3.0),
        Value::Num(1.0),
    ]);
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [9, 1]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0]
    );
}

#[test]
fn downsample_row_vector_keeps_every_nth_sample() {
    let out = call_downsample(vec![
        tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
        Value::Num(2.0),
    ]);
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [1, 3]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 3.0, 5.0]);
}

#[test]
fn downsample_phase_offsets_first_kept_sample() {
    let out = call_downsample(vec![
        tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
        Value::Num(2.0),
        Value::Num(1.0),
    ]);
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [1, 2]);
    assert_eq!(tensor.materialize_f64(), vec![2.0, 4.0]);
}

#[test]
fn sample_rate_operates_down_matrix_columns() {
    let out = call_upsample(vec![
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        Value::Num(2.0),
    ]);
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [4, 2]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
    );

    let down = call_downsample(vec![Value::Tensor(tensor), Value::Num(2.0)]);
    let Value::Tensor(tensor) = down else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn sample_rate_operates_along_first_non_singleton_nd_dimension() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = call_upsample(vec![
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]),
        Value::Num(2.0),
    ]);
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [1, 4, 2]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
    );
}

#[test]
fn sample_rate_preserves_empty_shape() {
    let out = call_upsample(vec![tensor(Vec::new(), vec![1, 0]), Value::Num(3.0)]);
    let Value::Tensor(array) = out else {
        panic!("expected tensor");
    };
    assert_eq!(array.shape(), [1, 0]);
    assert!(array.materialize_f64().is_empty());

    let out = call_downsample(vec![tensor(Vec::new(), vec![0, 1]), Value::Num(2.0)]);
    let Value::Tensor(array) = out else {
        panic!("expected tensor");
    };
    assert_eq!(array.shape(), [0, 1]);
    assert!(array.materialize_f64().is_empty());
}

#[test]
fn resample_custom_unit_filter_matches_zero_insert_and_drop() {
    let out = call_resample(vec![
        tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
        Value::Num(2.0),
        Value::Num(1.0),
        tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
    ]);
    let Value::Tensor(upsampled) = out else {
        panic!("expected tensor");
    };
    assert_eq!(upsampled.shape(), [1, 6]);
    assert_eq!(
        upsampled.materialize_f64(),
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
    );

    let down = call_resample(vec![
        tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
        Value::Num(1.0),
        Value::Num(2.0),
        tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
    ]);
    let Value::Tensor(downsampled) = down else {
        panic!("expected tensor");
    };
    assert_eq!(downsampled.shape(), [1, 3]);
    assert_eq!(downsampled.materialize_f64(), vec![1.0, 3.0, 5.0]);
}

#[test]
fn resample_rejects_integer_signal_storage() {
    let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]).unwrap();
    let error = block_on(resample_builtin(
        Value::Tensor(input),
        Value::Num(2.0),
        Value::Num(1.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_INPUT.identifier);
}
