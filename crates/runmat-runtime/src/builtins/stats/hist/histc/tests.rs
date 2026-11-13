use super::*;
use futures::executor::block_on;
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage};

mod sink;

#[test]
fn empty_literal_edges_accept_vector_and_matrix_data() {
    for (shape, expected) in [(vec![1, 3], vec![0, 0]), (vec![3, 2], vec![0, 2])] {
        let count = shape.iter().product();
        let result = block_on(evaluate(
            tensor(vec![1.; count], shape),
            tensor(vec![], vec![0, 0]),
            None,
        ))
        .unwrap();
        assert_eq!(result.counts.shape(), expected.as_slice());
        assert_eq!(result.indices.shape(), &[0, 0]);
    }
}

#[test]
fn floating_dimension_tolerance_remains_accepted() {
    let result = block_on(histc_builtin(
        tensor(vec![0., 1.], vec![1, 2]),
        tensor(vec![0., 1.], vec![1, 2]),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0000001)]),
    ))
    .unwrap();
    let Value::Tensor(counts) = result else {
        panic!("counts")
    };
    assert_eq!(counts.shape(), &[2, 2]);
}

#[test]
fn wide_complex_integer_data_and_edges_obey_compatibility_gate() {
    let value = Value::ComplexTensor(
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1, 2]),
                IntegerStorage::U64(vec![0, 0]),
            )
            .unwrap(),
            vec![1, 2],
        )
        .unwrap(),
    );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for (data, edges) in [
        (value.clone(), tensor(vec![0., 2.], vec![1, 2])),
        (tensor(vec![1., 2.], vec![1, 2]), value),
    ] {
        let error = block_on(evaluate(data, edges, None)).unwrap_err();
        assert_eq!(
            error.identifier(),
            HISTC_WIDE_INTEGER_EXTENSION.error_identifier
        );
    }
}

#[test]
fn excess_outputs_and_zero_dimension_fail_before_provider_access() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let resident = Value::GpuTensor(allocation.handle.clone());
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let error = block_on(histc_builtin(
        resident.clone(),
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(error.message().contains("too many output"));
    drop(_outputs);
    let error = block_on(evaluate(resident, Value::Num(0.0), Some(0))).unwrap_err();
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn resident_histogram_inputs_return_host_counts_without_consuming_sources() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[0.0, 1.0, 1.0, 2.0],
                shape: &[1, 4],
            })
            .unwrap();
        let result = block_on(evaluate(
            Value::GpuTensor(handle.clone()),
            tensor(vec![0.0, 1.0, 2.0], vec![1, 3]),
            None,
        ))
        .unwrap();
        assert_eq!(result.counts.materialize_f64(), vec![1.0, 2.0, 1.0]);
        assert_eq!(result.indices.materialize_f64(), vec![1.0, 2.0, 2.0, 3.0]);
        assert!(block_on(provider.download(&handle)).is_ok());
    });
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn values(value: Tensor) -> Vec<f64> {
    value.materialize_f64()
}

fn typed_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn histc_low_rank_vector_calls_preserve_input_and_edge_shapes() {
    for (x_values, edges, shape, expected_counts, expected_indices) in [
        (vec![0.5], vec![0.5], Vec::new(), vec![1.0], vec![1.0]),
        (
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![2],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
        ),
    ] {
        let result = block_on(evaluate(
            tensor(x_values, shape.clone()),
            tensor(edges, shape.clone()),
            None,
        ))
        .expect("low-rank vector histogram");
        assert_eq!(result.counts.shape(), shape.as_slice());
        assert_eq!(result.indices.shape(), shape.as_slice());
        assert_eq!(result.counts.materialize_f64(), expected_counts);
        assert_eq!(result.indices.materialize_f64(), expected_indices);
    }
}

#[test]
fn histc_counts_vector_and_returns_indices() {
    let ages = tensor(
        vec![
            3.0, 12.0, 24.0, 15.0, 5.0, 74.0, 23.0, 54.0, 31.0, 23.0, 64.0, 75.0,
        ],
        vec![1, 12],
    );
    let edges = tensor(vec![0.0, 10.0, 25.0, 50.0, 75.0], vec![1, 5]);
    let result = block_on(evaluate(ages, edges, None)).expect("histc");
    assert_eq!(result.counts.shape(), &[1, 5]);
    assert_eq!(values(result.counts), vec![2.0, 5.0, 1.0, 3.0, 1.0]);
    assert_eq!(
        values(result.indices),
        vec![1.0, 2.0, 2.0, 2.0, 1.0, 4.0, 2.0, 4.0, 3.0, 2.0, 4.0, 5.0]
    );
}

#[test]
fn histc_operates_down_matrix_columns_by_default() {
    let x = tensor(vec![0.0, 0.5, 1.5, 1.0, 2.0, 3.0], vec![3, 2]);
    let edges = tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]);
    let result = block_on(evaluate(x, edges, None)).expect("histc");
    assert_eq!(result.counts.shape(), &[4, 2]);
    assert_eq!(
        values(result.counts),
        vec![2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(values(result.indices), vec![1.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
}

#[test]
fn histc_respects_explicit_dimension() {
    let x = tensor(vec![0.0, 2.0, 1.0, 3.0, 2.0, 4.0], vec![2, 3]);
    let edges = tensor(vec![0.0, 2.0, 4.0], vec![1, 3]);
    let result = block_on(evaluate(x, edges, Some(2))).expect("histc");
    assert_eq!(result.counts.shape(), &[2, 3]);
    assert_eq!(values(result.counts), vec![2.0, 0.0, 1.0, 2.0, 0.0, 1.0]);
    assert_eq!(values(result.indices), vec![1.0, 2.0, 1.0, 2.0, 2.0, 3.0]);
}

#[test]
fn histc_supports_per_slice_edge_columns() {
    let x = tensor(vec![0.5, 1.5, 10.0, 20.0], vec![2, 2]);
    let edges = tensor(vec![0.0, 1.0, 2.0, 0.0, 15.0, 30.0], vec![3, 2]);
    let result = block_on(evaluate(x, edges, None)).expect("histc");
    assert_eq!(result.counts.shape(), &[3, 2]);
    assert_eq!(values(result.counts), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn histc_ignores_nan_out_of_range_and_imaginary_parts() {
    let x = Value::ComplexTensor(
        ComplexTensor::new(
            vec![
                (f64::NAN, 7.0),
                (-1.0, 2.0),
                (0.5, 9.0),
                (2.0, -4.0),
                (3.0, 1.0),
            ],
            vec![1, 5],
        )
        .unwrap(),
    );
    let edges = tensor(vec![0.0, 1.0, 2.0], vec![1, 3]);
    let result = block_on(evaluate(x, edges, None)).expect("histc");
    assert_eq!(values(result.counts), vec![1.0, 0.0, 1.0]);
    assert_eq!(values(result.indices), vec![0.0, 0.0, 1.0, 3.0, 0.0]);
}

#[test]
fn histc_accepts_duplicate_edges_with_rightmost_nonempty_bin() {
    let x = tensor(vec![1.0, 1.5], vec![1, 2]);
    let edges = tensor(vec![0.0, 1.0, 1.0, 2.0], vec![1, 4]);
    let result = block_on(evaluate(x, edges, None)).expect("histc");
    assert_eq!(values(result.counts), vec![0.0, 0.0, 2.0, 0.0]);
    assert_eq!(values(result.indices), vec![3.0, 3.0]);
}

#[test]
fn histc_reads_typed_integer_inputs_and_edges_exactly() {
    let x = typed_tensor(IntegerStorage::U16(vec![0, 1, 2, 3]), vec![1, 4]);
    let edges = typed_tensor(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]);

    let result = block_on(evaluate(x, edges, None)).expect("histc");

    assert_eq!(result.counts.shape(), &[1, 3]);
    assert_eq!(values(result.counts), vec![1.0, 2.0, 1.0]);
    assert_eq!(values(result.indices), vec![1.0, 2.0, 2.0, 3.0]);
}

#[test]
fn histc_wide_integer_inputs_are_strictly_extension_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let x = typed_tensor(IntegerStorage::U64(vec![0, 1]), vec![1, 2]);
    let edges = typed_tensor(IntegerStorage::U64(vec![0, 1]), vec![1, 2]);
    let error = block_on(evaluate(x, edges, None)).expect_err("wide integer gate");
    assert_eq!(
        error.identifier(),
        HISTC_WIDE_INTEGER_EXTENSION.error_identifier
    );
}

#[test]
fn histc_documented_integer_classes_do_not_require_extensions() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for storage in [
        IntegerStorage::I8(vec![0, 1]),
        IntegerStorage::I16(vec![0, 1]),
        IntegerStorage::I32(vec![0, 1]),
        IntegerStorage::U8(vec![0, 1]),
        IntegerStorage::U16(vec![0, 1]),
        IntegerStorage::U32(vec![0, 1]),
    ] {
        let x = typed_tensor(storage, vec![1, 2]);
        let edges = tensor(vec![0.0, 1.0], vec![1, 2]);
        block_on(evaluate(x, edges, None)).expect("documented integer class");
    }
}

#[test]
fn histc_reads_typed_integer_matrix_edge_columns_exactly() {
    let x = typed_tensor(IntegerStorage::I16(vec![0, 2, 10, 20]), vec![2, 2]);
    let edges = typed_tensor(IntegerStorage::I16(vec![0, 1, 3, 0, 15, 30]), vec![3, 2]);

    let result = block_on(evaluate(x, edges, None)).expect("histc");

    assert_eq!(result.counts.shape(), &[3, 2]);
    assert_eq!(values(result.counts), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn histc_reads_typed_complex_integer_real_storage_exactly() {
    let complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::I16(vec![0, 1, 2]),
            IntegerStorage::I16(vec![99, 99, 99]),
        )
        .unwrap(),
        vec![1, 3],
    )
    .unwrap();
    let edges = typed_tensor(IntegerStorage::I16(vec![0, 1, 2]), vec![1, 3]);

    let result = block_on(evaluate(Value::ComplexTensor(complex), edges, None)).expect("histc");

    assert_eq!(values(result.counts), vec![1.0, 1.0, 1.0]);
    assert_eq!(values(result.indices), vec![1.0, 2.0, 3.0]);
}

#[test]
fn histc_rejects_decreasing_or_nan_edges() {
    let x = tensor(vec![1.0], vec![1, 1]);
    let err = block_on(evaluate(
        x.clone(),
        tensor(vec![0.0, 2.0, 1.0], vec![1, 3]),
        None,
    ))
    .expect_err("decreasing edges should fail");
    assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);

    let err = block_on(evaluate(x, tensor(vec![0.0, f64::NAN], vec![1, 2]), None))
        .expect_err("NaN edge should fail");
    assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn histc_rejects_mismatched_edge_columns() {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let edges = tensor(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0], vec![2, 3]);
    let err = block_on(evaluate(x, edges, None)).expect_err("edge column mismatch");
    assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn histc_empty_edges_return_empty_counts_and_indices() {
    let x = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let edges = tensor(Vec::new(), vec![1, 0]);
    let result = block_on(evaluate(x, edges, None)).expect("histc empty edges");
    assert_eq!(result.counts.shape(), &[1, 0]);
    assert!(result.counts.materialize_f64().is_empty());
    assert_eq!(result.indices.shape(), &[0, 0]);
    assert!(result.indices.materialize_f64().is_empty());
}

#[test]
fn histc_rejects_excess_outputs_and_huge_dimensions() {
    let x = tensor(vec![0.0, 1.0], vec![1, 2]);
    let edges = tensor(vec![0.0, 1.0], vec![1, 2]);
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let err = block_on(histc_builtin(
        x,
        edges,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("too many outputs");
    assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    drop(_guard);

    let x = tensor(vec![0.0, 1.0], vec![1, 2]);
    let edges = tensor(vec![0.0, 1.0], vec![1, 2]);
    let err = block_on(histc_builtin(
        x,
        edges,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
            (MAX_DIMENSION_ARGUMENT + 1) as f64,
        )]),
    ))
    .expect_err("dim cap");
    assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn histc_runtime_builtin_handles_output_count_padding() {
    let x = tensor(vec![0.0, 1.0], vec![1, 2]);
    let edges = tensor(vec![0.0, 1.0], vec![1, 2]);
    let result = block_on(histc_builtin(
        x,
        edges,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("histc builtin");
    let Value::Tensor(counts) = result else {
        panic!("expected tensor");
    };
    assert_eq!(counts.materialize_f64(), vec![1.0, 1.0]);
}
