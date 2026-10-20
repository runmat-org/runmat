use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::{
    HISTCOUNTS_ERROR_BINMETHOD_CONFLICT, HISTCOUNTS_ERROR_BINWIDTH_INVALID,
    HISTCOUNTS_ERROR_EDGE_PRECISION,
};
use runmat_value::{IntValue, IntegerStorage, NumericStorage, Tensor, Value};

async fn evaluate(data: Value, rest: &[Value]) -> BuiltinResult<HistcountsEvaluation> {
    super::evaluate(data, crate::call::FunctionArgumentView::from_slice(rest)).await
}

async fn histcounts_builtin(data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    super::histcounts_builtin(data, rest.into()).await
}

fn values_from_tensor(value: Value) -> Vec<f64> {
    match value {
        Value::Tensor(t) => t.materialize_f64(),
        Value::Num(n) => vec![n],
        other => panic!("unexpected value {other:?}"),
    }
}

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn histcounts_three_outputs_preserve_low_rank_bin_indices() {
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    for (samples, shape, expected_counts, expected_bins) in [
        (vec![0.5], Vec::new(), vec![1.0, 0.0], vec![1.0]),
        (vec![0.5, 1.5], vec![2], vec![1.0, 1.0], vec![1.0, 2.0]),
    ] {
        let input = Value::Tensor(Tensor::new(samples, shape.clone()).unwrap());
        let edges = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap());
        let Value::OutputList(outputs) = block_on(histcounts_builtin(input, vec![edges]))
            .expect("three-output low-rank histogram")
        else {
            panic!("expected three histogram outputs")
        };
        let mut output_iter = outputs
            .try_into_owned()
            .expect("unique outputs")
            .into_iter();
        let outputs = [
            output_iter.next().expect("counts output"),
            output_iter.next().expect("edges output"),
            output_iter.next().expect("bins output"),
        ];
        assert!(output_iter.next().is_none(), "exactly three outputs");
        let [Value::Tensor(counts), Value::Tensor(bin_edges), Value::Tensor(bins)] = outputs else {
            panic!("expected tensor histogram outputs")
        };
        assert_eq!(counts.shape(), &[1, 2]);
        assert_eq!(counts.materialize_f64(), expected_counts);
        assert_eq!(bin_edges.shape(), &[1, 3]);
        assert_eq!(bin_edges.materialize_f64(), vec![0.0, 1.0, 2.0]);
        assert_eq!(bins.shape(), shape.as_slice());
        assert_eq!(bins.materialize_f64(), expected_bins);
    }
}

#[test]
fn wide_integer_edges_classify_without_rounding_and_keep_exact_widths() {
    let base = 9_007_199_254_740_992_u64;
    let data = integer_tensor(
        IntegerStorage::U64(vec![base, base + 1, base + 2]),
        vec![1, 3],
    );
    let edges = integer_tensor(
        IntegerStorage::U64(vec![base, base + 1, base + 2]),
        vec![1, 3],
    );
    let eval = block_on(evaluate(data.clone(), std::slice::from_ref(&edges))).expect("exact edges");
    let (counts, _) = eval.into_pair();
    assert_eq!(values_from_tensor(counts), vec![1.0, 2.0]);

    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let result = block_on(histcounts_builtin(
        data,
        vec![
            Value::String("BinEdges".into()),
            edges,
            Value::String("Normalization".into()),
            Value::String("countdensity".into()),
        ],
    ))
    .expect("density with exact integer widths");
    let Value::OutputList(outputs) = result else {
        panic!("expected three outputs");
    };
    assert_eq!(values_from_tensor(outputs[0].clone()), vec![1.0, 2.0]);
    assert_eq!(values_from_tensor(outputs[2].clone()), vec![1.0, 2.0, 2.0]);
}

#[test]
fn signed_minimum_unsigned_maximum_and_cross_class_edges_are_exact() {
    let signed = integer_tensor(
        IntegerStorage::I64(vec![i64::MIN, i64::MIN + 1, i64::MIN + 2]),
        vec![1, 3],
    );
    let signed_edges = integer_tensor(
        IntegerStorage::I64(vec![i64::MIN, i64::MIN + 1, i64::MIN + 2]),
        vec![1, 3],
    );
    let (counts, _) = block_on(evaluate(signed, &[signed_edges]))
        .expect("signed minimum")
        .into_pair();
    assert_eq!(values_from_tensor(counts), vec![1.0, 2.0]);

    let unsigned = integer_tensor(
        IntegerStorage::U64(vec![u64::MAX - 2, u64::MAX - 1, u64::MAX]),
        vec![1, 3],
    );
    let unsigned_edges = integer_tensor(
        IntegerStorage::U64(vec![u64::MAX - 2, u64::MAX - 1, u64::MAX]),
        vec![1, 3],
    );
    let (counts, _) = block_on(evaluate(unsigned, &[unsigned_edges]))
        .expect("unsigned maximum")
        .into_pair();
    assert_eq!(values_from_tensor(counts), vec![1.0, 2.0]);

    let signed = integer_tensor(IntegerStorage::I16(vec![-1, 0, 1]), vec![1, 3]);
    let unsigned_edges = integer_tensor(IntegerStorage::U64(vec![0, 1, 2]), vec![1, 3]);
    let (counts, _) = block_on(evaluate(signed, &[unsigned_edges]))
        .expect("mixed integer classes")
        .into_pair();
    assert_eq!(values_from_tensor(counts), vec![1.0, 1.0]);
}

#[test]
fn third_output_preserves_input_shape_and_marks_excluded_observations() {
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let input =
        Tensor::new(vec![-1.0, 0.0, 1.0, 2.0, 3.0, f64::NAN], vec![2, 3]).expect("nonsquare input");
    let edges = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).expect("edges");
    let result = block_on(histcounts_builtin(
        Value::Tensor(input),
        vec![Value::Tensor(edges)],
    ))
    .expect("three histogram outputs");
    let Value::OutputList(outputs) = result else {
        panic!("expected three outputs");
    };
    assert_eq!(outputs.len(), 3);
    let mut outputs = outputs
        .try_into_owned()
        .expect("unique outputs")
        .into_iter();
    assert_eq!(
        values_from_tensor(outputs.next().unwrap()),
        vec![1.0, 1.0, 2.0]
    );
    assert_eq!(
        values_from_tensor(outputs.next().unwrap()),
        vec![0.0, 1.0, 2.0, 3.0]
    );
    let Value::Tensor(bins) = outputs.next().unwrap() else {
        panic!("bin indices must be a tensor");
    };
    assert_eq!(bins.shape(), &[2, 3]);
    assert_eq!(bins.materialize_f64(), vec![0.0, 1.0, 2.0, 3.0, 3.0, 0.0]);
}

#[test]
fn excessive_outputs_are_rejected_before_invalid_bin_options() {
    let _outputs = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let result = block_on(histcounts_builtin(Value::Num(1.0), vec![Value::Num(-1.0)]));
    let error = match result {
        Ok(_) => panic!("four outputs must fail before invalid bins"),
        Err(error) => error,
    };
    assert!(error.message().contains("too many output arguments"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_basic_numbins() {
    let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0, 5.0, 7.0], vec![6, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(tensor),
        &[Value::Int(IntValue::I32(3))],
    ))
    .expect("histcounts");
    let (counts_val, edges_val) = eval.into_pair();
    assert_eq!(values_from_tensor(counts_val), vec![3.0, 1.0, 2.0]);
    assert_eq!(values_from_tensor(edges_val), vec![1.0, 3.0, 5.0, 7.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_data_reads_typed_integer_storage_exactly() {
    let eval = block_on(evaluate(
        integer_tensor(IntegerStorage::I16(vec![1, 2, 2, 4, 5, 7]), vec![6, 1]),
        &[Value::Int(IntValue::I32(3))],
    ))
    .expect("histcounts");
    let (counts_val, edges_val) = eval.into_pair();
    assert_eq!(values_from_tensor(counts_val), vec![3.0, 1.0, 2.0]);
    assert_eq!(values_from_tensor(edges_val), vec![1.0, 3.0, 5.0, 7.0]);
}

#[test]
fn histcounts_automatic_bins_compact_unrepresentable_wide_integer_edges() {
    let wide = 9_007_199_254_740_993_u64;
    let eval = block_on(evaluate(
        integer_tensor(
            IntegerStorage::U64(vec![wide, wide + 1, wide + 2, wide + 3]),
            vec![4, 1],
        ),
        &[],
    ))
    .expect("wide integer histcounts");
    let (counts, edges) = eval.into_pair();
    let counts = values_from_tensor(counts);
    let edges = values_from_tensor(edges);
    assert_eq!(counts.iter().sum::<f64>(), 4.0);
    assert!(edges.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn histcounts_accepts_every_host_integer_class_and_returns_double_outputs() {
    let storages = vec![
        IntegerStorage::I8(vec![1, 2, 3]),
        IntegerStorage::I16(vec![1, 2, 3]),
        IntegerStorage::I32(vec![1, 2, 3]),
        IntegerStorage::I64(vec![1, 2, 3]),
        IntegerStorage::U8(vec![1, 2, 3]),
        IntegerStorage::U16(vec![1, 2, 3]),
        IntegerStorage::U32(vec![1, 2, 3]),
        IntegerStorage::U64(vec![1, 2, 3]),
    ];
    for storage in storages {
        let eval = block_on(evaluate(
            integer_tensor(storage, vec![3, 1]),
            &[Value::Tensor(
                Tensor::new(vec![0.0, 2.0, 4.0], vec![1, 3]).expect("edges"),
            )],
        ))
        .expect("histcounts");
        let (counts, edges) = eval.into_pair();
        for (value, expected) in [(counts, vec![1.0, 2.0]), (edges, vec![0.0, 2.0, 4.0])] {
            let Value::Tensor(tensor) = value else {
                panic!("expected double tensor");
            };
            assert_eq!(tensor.materialize_f64(), expected);
            assert!(matches!(
                tensor.into_numeric_storage().unwrap(),
                NumericStorage::F64(_)
            ));
        }
    }
}

#[test]
fn histcounts_numbins_preserves_typed_integers_and_rejects_lossy_f64() {
    assert_eq!(
        positive_usize(&Value::Int(IntValue::U16(3)), "histcounts", "NumBins").unwrap(),
        3
    );
    assert_eq!(
        positive_usize(
            &integer_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
            "histcounts",
            "NumBins",
        )
        .unwrap(),
        4
    );
    assert_eq!(
        positive_usize(
            &integer_tensor(IntegerStorage::U16(vec![9]), vec![1, 1]),
            "histcounts",
            "NumBins",
        )
        .unwrap(),
        9
    );
    for value in [
        Value::Int(IntValue::I8(-1)),
        integer_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
        integer_tensor(IntegerStorage::U64(vec![usize::MAX as u64]), vec![1, 1]),
        Value::Num(1.5),
        Value::Num(usize::MAX as f64 + 1.0),
    ] {
        assert!(positive_usize(&value, "histcounts", "NumBins").is_err());
    }
}

#[test]
fn histcounts_numeric_vectors_read_typed_integer_storage_exactly() {
    assert_eq!(
        numeric_vector(
            &integer_tensor(IntegerStorage::I16(vec![1, 3, 5]), vec![1, 3]),
            "histcounts",
            "BinEdges",
        )
        .unwrap(),
        vec![1.0, 3.0, 5.0]
    );
    assert_eq!(
        scalar_value(
            &integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
            "histcounts",
            "BinWidth",
        )
        .unwrap(),
        2.0
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_binwidth_and_limits() {
    let tensor = Tensor::new(vec![5.0, 7.0, 8.0, 10.0, 12.0], vec![5, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(tensor),
        &[
            Value::from("BinWidth"),
            Value::Num(2.0),
            Value::from("BinLimits"),
            Value::Tensor(Tensor::new(vec![4.0, 12.0], vec![2, 1]).unwrap()),
        ],
    ))
    .expect("histcounts");
    let (counts_val, edges_val) = eval.into_pair();
    assert_eq!(values_from_tensor(counts_val), vec![1.0, 1.0, 1.0, 2.0]);
    assert_eq!(
        values_from_tensor(edges_val),
        vec![4.0, 6.0, 8.0, 10.0, 12.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_probability_normalization() {
    let data = Tensor::new(vec![0.2, 0.4, 1.1, 1.4, 1.8, 2.5], vec![6, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(data),
        &[
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            Value::from("Normalization"),
            Value::from("probability"),
        ],
    ))
    .expect("histcounts");
    let (counts_val, _) = eval.into_pair();
    let counts = values_from_tensor(counts_val);
    assert!((counts[0] - 0.3333).abs() < 5e-4);
    assert!((counts[1] - 0.5).abs() < 5e-4);
    assert!((counts[2] - 0.1667).abs() < 5e-4);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_cdf_normalization() {
    let data = Tensor::new(vec![1.0, 2.0, 2.0, 3.0], vec![4, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(data),
        &[
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            Value::from("Normalization"),
            Value::from("cdf"),
        ],
    ))
    .expect("histcounts");
    let (counts_val, _) = eval.into_pair();
    let counts = values_from_tensor(counts_val);
    assert_eq!(counts, vec![0.0, 0.25, 1.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_handles_nan() {
    let data = Tensor::new(vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0], vec![5, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(data),
        &[Value::Int(IntValue::I32(3))],
    ))
    .expect("histcounts");
    let (counts_val, _) = eval.into_pair();
    assert_eq!(values_from_tensor(counts_val), vec![1.0, 1.0, 1.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_constant_data_single_bin() {
    let data = Tensor::new(vec![4.0, 4.0, 4.0], vec![3, 1]).unwrap();
    let eval = block_on(evaluate(Value::Tensor(data), &[])).expect("histcounts");
    let (counts_val, edges_val) = eval.into_pair();
    assert_eq!(values_from_tensor(counts_val), vec![3.0]);
    assert_eq!(values_from_tensor(edges_val), vec![3.5, 4.5]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_binmethod_sqrt() {
    let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
    let tensor = Tensor::new(data, vec![16, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(tensor),
        &[Value::from("BinMethod"), Value::from("sqrt")],
    ))
    .expect("histcounts");
    let (counts_val, edges_val) = eval.into_pair();
    assert_eq!(values_from_tensor(counts_val).len(), 4);
    assert_eq!(values_from_tensor(edges_val).len(), 5);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_binmethod_integers_with_limits() {
    let tensor = Tensor::new(vec![2.2, 2.8, 3.4, 3.9], vec![4, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(tensor),
        &[
            Value::from("BinMethod"),
            Value::from("integers"),
            Value::from("BinLimits"),
            Value::Tensor(Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap()),
        ],
    ))
    .expect("histcounts");
    let (counts_val, edges_val) = eval.into_pair();
    let counts = values_from_tensor(counts_val);
    let edges = values_from_tensor(edges_val);
    assert_eq!(counts, vec![2.0, 2.0]);
    assert_eq!(edges, vec![2.0, 3.0, 4.0]);
}

#[test]
fn integer_method_rejects_unrepresentable_unit_edges_without_rejecting_wide_data() {
    let base = 9_007_199_254_740_992.0;
    let data = Value::Tensor(Tensor::new(vec![base, base + 2.0], vec![1, 2]).unwrap());
    let error = block_on(evaluate(
        data.clone(),
        &[Value::from("BinMethod"), Value::from("integers")],
    ))
    .err()
    .expect("unit edges above flintmax are not distinct doubles");
    assert_eq!(
        error.identifier(),
        HISTCOUNTS_ERROR_EDGE_PRECISION.identifier
    );
    let explicit_count_error = block_on(evaluate(
        data.clone(),
        &[Value::from("NumBins"), Value::Num(10.0)],
    ))
    .err()
    .expect("requested bins cannot be silently collapsed");
    assert_eq!(
        explicit_count_error.identifier(),
        HISTCOUNTS_ERROR_EDGE_PRECISION.identifier
    );
    let wide_default = block_on(evaluate(data, &[])).expect("wider double bins remain legal");
    let (counts, edges) = wide_default.into_pair();
    assert_eq!(values_from_tensor(counts).iter().sum::<f64>(), 2.0);
    assert!(values_from_tensor(edges)
        .windows(2)
        .all(|pair| pair[0] < pair[1]));
}

#[test]
fn empty_integer_method_input_uses_finite_default_edges() {
    let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap());
    let evaluation = block_on(evaluate(
        empty,
        &[Value::from("BinMethod"), Value::from("integers")],
    ))
    .expect("empty integer-method histogram");
    let (counts, edges) = evaluation.into_pair();
    assert_eq!(values_from_tensor(counts).iter().sum::<f64>(), 0.0);
    assert!(values_from_tensor(edges)
        .windows(2)
        .all(|pair| pair[0].is_finite() && pair[0] < pair[1]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_binmethod_conflict_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(evaluate(
        Value::Tensor(tensor),
        &[
            Value::from("BinMethod"),
            Value::from("auto"),
            Value::from("NumBins"),
            Value::Num(5.0),
        ],
    ))
    .err()
    .expect("BinMethod + NumBins should fail");
    assert_eq!(
        err.identifier(),
        HISTCOUNTS_ERROR_BINMETHOD_CONFLICT.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_invalid_binwidth_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = block_on(evaluate(
        Value::Tensor(tensor),
        &[Value::from("BinWidth"), Value::Num(0.0)],
    ))
    .err()
    .expect("non-positive BinWidth should fail");
    assert_eq!(
        err.identifier(),
        HISTCOUNTS_ERROR_BINWIDTH_INVALID.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let eval = block_on(evaluate(
            Value::GpuTensor(handle),
            &[Value::Tensor(
                Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap(),
            )],
        ))
        .expect("histcounts");
        let (counts_val, edges_val) = eval.into_pair();
        assert_eq!(values_from_tensor(counts_val), vec![1.0, 1.0, 1.0]);
        assert_eq!(values_from_tensor(edges_val), vec![0.0, 1.0, 2.0, 3.0]);
    });
}

#[test]
fn histcounts_rejects_64_bit_integer_gpu_inputs_before_gather() {
    test_support::with_test_provider(|provider| {
        for storage in [
            IntegerStorage::I64(vec![1, 2, 3]),
            IntegerStorage::U64(vec![1, 2, 3]),
        ] {
            let tensor = Tensor::new_integer(storage, vec![3, 1]).expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let error = block_on(evaluate(Value::GpuTensor(handle), &[]))
                .err()
                .expect("64-bit integer gpuArray should be rejected");
            assert!(error
                .message()
                .contains("64-bit integer gpuArray inputs are not supported"));
        }
    });
}

#[test]
fn public_dispatch_preserves_64_bit_integer_gpu_rejection() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3]), vec![3, 1])
            .expect("integer tensor");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let error = crate::dispatcher::call_builtin("histcounts", &[Value::GpuTensor(handle)])
            .expect_err("public dispatch must not gather an unsupported 64-bit GPU form");
        assert!(error
            .message()
            .contains("64-bit integer gpuArray inputs are not supported"));
    });
}

#[test]
fn public_dispatch_still_gathers_resident_bin_specifications() {
    test_support::with_test_provider(|provider| {
        let bins = Tensor::new(vec![0.0, 2.0, 4.0], vec![1, 3]).expect("bin edges");
        let bins = gpu_helpers::upload_tensor(provider, &bins).expect("resident bin edges");
        let data =
            Value::Tensor(Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).expect("histogram data"));
        let value = crate::dispatcher::call_builtin("histcounts", &[data, Value::GpuTensor(bins)])
            .expect("legacy dispatcher fallback gathers resident bin edges");
        let Value::Tensor(counts) = value.value() else {
            panic!("expected count tensor");
        };
        assert_eq!(counts.materialize_f64(), vec![2.0, 1.0]);
    });
}

#[test]
fn histcounts_accepts_supported_integer_gpu_classes() {
    test_support::with_test_provider(|provider| {
        for storage in [
            IntegerStorage::I8(vec![1, 2, 3]),
            IntegerStorage::I16(vec![1, 2, 3]),
            IntegerStorage::I32(vec![1, 2, 3]),
            IntegerStorage::U8(vec![1, 2, 3]),
            IntegerStorage::U16(vec![1, 2, 3]),
            IntegerStorage::U32(vec![1, 2, 3]),
        ] {
            let tensor = Tensor::new_integer(storage, vec![3, 1]).expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let eval = block_on(evaluate(
                Value::GpuTensor(handle),
                &[Value::Tensor(
                    Tensor::new(vec![0.0, 2.0, 4.0], vec![1, 3]).expect("edges"),
                )],
            ))
            .expect("supported integer gpuArray");
            let (counts, edges) = eval.into_pair();
            assert_eq!(values_from_tensor(counts), vec![1.0, 2.0]);
            assert_eq!(values_from_tensor(edges), vec![0.0, 2.0, 4.0]);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn histcounts_wgpu_roundtrip() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required WGPU provider");
    let tensor = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
    let eval = block_on(evaluate(
        Value::GpuTensor(handle),
        &[Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap(),
        )],
    ))
    .expect("histcounts");
    let (counts_val, edges_val) = eval.into_pair();
    assert_eq!(values_from_tensor(counts_val), vec![1.0, 1.0, 1.0]);
    assert_eq!(values_from_tensor(edges_val), vec![0.0, 1.0, 2.0, 3.0]);
}
