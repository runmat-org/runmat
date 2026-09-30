use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, NumericStorage, Tensor, Value};

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
    match classify_bin_argument(&integer_tensor(
        IntegerStorage::U16(vec![2, 4, 6]),
        vec![1, 3],
    ))
    .unwrap()
    {
        BinArgument::Edges(edges) => assert_eq!(edges, vec![2.0, 4.0, 6.0]),
        other => panic!("expected edge vector, got {other:?}"),
    }
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
        let Value::Tensor(counts) = value else {
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
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
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
