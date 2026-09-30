use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage};

fn tensor_from_value(value: Value) -> Tensor {
    match value {
        Value::Tensor(t) => t,
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
        other => panic!("expected tensor value, got {:?}", other),
    }
}

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

fn double_values(tensor: &Tensor) -> &[f64] {
    tensor.as_f64_slice().expect("expected double tensor")
}

#[test]
fn histcounts2_numbins_preserves_typed_integers_and_rejects_lossy_f64() {
    assert_eq!(
        positive_usize(&Value::Int(IntValue::U16(3)), NAME, "NumBins").unwrap(),
        3
    );
    assert_eq!(
        positive_usize(
            &integer_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
            NAME,
            "NumBins",
        )
        .unwrap(),
        4
    );
    assert_eq!(
        positive_usize(
            &integer_tensor(IntegerStorage::U16(vec![9]), vec![1, 1]),
            NAME,
            "NumBins",
        )
        .unwrap(),
        9
    );
    assert_eq!(
        num_bins_vector(&integer_tensor(
            IntegerStorage::U16(vec![7, 11]),
            vec![1, 2],
        ))
        .unwrap(),
        vec![7, 11]
    );
    for value in [
        Value::Int(IntValue::I8(-1)),
        integer_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
        integer_tensor(IntegerStorage::I16(vec![2, -1]), vec![1, 2]),
        integer_tensor(IntegerStorage::U64(vec![usize::MAX as u64]), vec![1, 1]),
        integer_tensor(IntegerStorage::U64(vec![2, usize::MAX as u64]), vec![1, 2]),
        Value::Num(1.5),
        Value::Num(usize::MAX as f64 + 1.0),
    ] {
        assert!(positive_usize(&value, NAME, "NumBins").is_err());
    }
    assert!(num_bins_vector(&integer_tensor(
        IntegerStorage::I16(vec![2, -1]),
        vec![1, 2],
    ))
    .is_err());
}

#[test]
fn histcounts2_numeric_vectors_read_typed_integer_storage_exactly() {
    assert_eq!(
        numeric_vector(
            &integer_tensor(IntegerStorage::I16(vec![1, 3, 5]), vec![1, 3]),
            NAME,
            "XBinEdges",
        )
        .unwrap(),
        vec![1.0, 3.0, 5.0]
    );
    assert_eq!(
        scalar_value(
            &integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
            NAME,
            "BinWidth",
        )
        .unwrap(),
        2.0
    );
}

#[test]
fn histcounts2_accepts_every_integer_class_and_native_single_at_floating_boundary() {
    let integer_cases = [
        IntegerStorage::I8(vec![0, 1, 2]),
        IntegerStorage::I16(vec![0, 1, 2]),
        IntegerStorage::I32(vec![0, 1, 2]),
        IntegerStorage::I64(vec![0, 1, 2]),
        IntegerStorage::U8(vec![0, 1, 2]),
        IntegerStorage::U16(vec![0, 1, 2]),
        IntegerStorage::U32(vec![0, 1, 2]),
        IntegerStorage::U64(vec![0, 1, 2]),
    ];
    for storage in integer_cases {
        let x = Tensor::new_integer(storage, vec![3, 1]).expect("integer observations");
        let y = Tensor::from_f32(vec![0.25, 1.25, 2.25], vec![3, 1]).expect("single observations");
        let eval = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                Value::Tensor(Tensor::from_f32(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            ],
        ))
        .expect("typed histcounts2");
        let counts = tensor_from_value(eval.into_counts_value());
        assert_eq!(
            double_values(&counts),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_basic_counts() {
    let x = Tensor::new(vec![0.5, 1.5, 2.5, 3.5], vec![4, 1]).unwrap();
    let y = Tensor::new(vec![0.2, 0.9, 1.4, 2.8], vec![4, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        &[
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
        ],
    ))
    .expect("histcounts2");
    let (counts, xedges, yedges) = eval.into_triple();
    let counts = tensor_from_value(counts);
    assert_eq!(counts.shape, vec![4, 3]);
    let xedges = tensor_from_value(xedges);
    let yedges = tensor_from_value(yedges);
    assert_eq!(double_values(&xedges), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(double_values(&yedges), &[0.0, 1.0, 2.0, 3.0]);
    let rows = counts.shape[0];
    let count = |ix: usize, iy: usize| double_values(&counts)[ix + iy * rows];
    assert_eq!(count(0, 0), 1.0);
    assert_eq!(count(1, 0), 1.0);
    assert_eq!(count(2, 1), 1.0);
    assert_eq!(count(3, 2), 1.0);
    for iy in 0..counts.shape[1] {
        for ix in 0..rows {
            let keep = matches!((ix, iy), (0, 0) | (1, 0) | (2, 1) | (3, 2));
            if !keep {
                assert_eq!(count(ix, iy), 0.0);
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_probability_normalization() {
    let x = Tensor::new(vec![0.2, 0.4, 1.1, 1.5], vec![4, 1]).unwrap();
    let y = Tensor::new(vec![0.1, 0.8, 1.2, 1.9], vec![4, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        &[
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
            Value::from("Normalization"),
            Value::from("probability"),
        ],
    ))
    .expect("histcounts2");
    let counts = tensor_from_value(eval.into_counts_value());
    assert_eq!(counts.shape, vec![2, 2]);
    assert_eq!(double_values(&counts), &[0.5, 0.0, 0.0, 0.5]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_nan_pairs_excluded() {
    let x = Tensor::new(vec![1.0, 2.0, f64::NAN, 3.0], vec![4, 1]).unwrap();
    let y = Tensor::new(vec![2.0, 2.0, 2.0, f64::NAN], vec![4, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        &[
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
        ],
    ))
    .expect("histcounts2");
    let counts = tensor_from_value(eval.into_counts_value());
    assert_eq!(double_values(&counts).iter().sum::<f64>(), 2.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_num_bins_vector() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        &[
            Value::from("NumBins"),
            Value::Tensor(Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap()),
        ],
    ))
    .expect("histcounts2");
    let counts = tensor_from_value(eval.into_counts_value());
    assert_eq!(counts.shape, vec![2, 4]);
    assert_eq!(double_values(&counts).iter().sum::<f64>(), 4.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_bin_width_and_limits() {
    let x_tensor = Tensor::new(vec![0.2, 1.2, 2.4, 2.6], vec![4, 1]).unwrap();
    let y_tensor = Tensor::new(vec![0.1, 0.6, 1.4, 2.2], vec![4, 1]).unwrap();
    let bin_limits = Tensor::new(vec![0.0, 3.0, 0.0, 2.5], vec![4, 1]).unwrap();

    let eval = block_on(evaluate(
        Value::Tensor(x_tensor.clone()),
        Value::Tensor(y_tensor.clone()),
        &[
            Value::from("XBinWidth"),
            Value::Num(1.0),
            Value::from("YBinWidth"),
            Value::Num(0.5),
            Value::from("BinLimits"),
            Value::Tensor(bin_limits.clone()),
        ],
    ))
    .expect("histcounts2");
    let (counts_v, xedges_v, yedges_v) = eval.into_triple();
    let counts = tensor_from_value(counts_v);
    let xedges = tensor_from_value(xedges_v);
    let yedges = tensor_from_value(yedges_v);

    assert_eq!(double_values(&xedges), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(double_values(&yedges), &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    assert_eq!(counts.shape, vec![3, 5]);
    assert_eq!(double_values(&counts).iter().sum::<f64>(), 4.0);

    let density_eval = block_on(evaluate(
        Value::Tensor(x_tensor),
        Value::Tensor(y_tensor),
        &[
            Value::from("XBinWidth"),
            Value::Num(1.0),
            Value::from("YBinWidth"),
            Value::Num(0.5),
            Value::from("BinLimits"),
            Value::Tensor(bin_limits),
            Value::from("Normalization"),
            Value::from("countdensity"),
        ],
    ))
    .expect("histcounts2 countdensity");
    let density = tensor_from_value(density_eval.into_counts_value());
    let positives: Vec<f64> = double_values(&density)
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .collect();
    assert!(!positives.is_empty());
    for value in positives {
        assert!((value - 2.0).abs() < 1e-12);
    }
    assert_eq!(double_values(&density)[0], 2.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_cdf_normalization() {
    let x = Tensor::new(vec![0.1, 0.9, 1.2, 1.8], vec![4, 1]).unwrap();
    let y = Tensor::new(vec![0.2, 0.7, 1.4, 1.6], vec![4, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        &[
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
            Value::from("Normalization"),
            Value::from("cdf"),
        ],
    ))
    .expect("histcounts2");
    let cdf = tensor_from_value(eval.into_counts_value());
    assert_eq!(cdf.shape, vec![2, 2]);
    assert_eq!(double_values(&cdf).last().copied().unwrap(), 1.0);
    assert!(double_values(&cdf).windows(2).all(|w| w[0] <= w[1] + 1e-12));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_integer_bin_method() {
    let x = Tensor::new(vec![0.2, 1.7, 2.1], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.2, 1.8, 2.9], vec![3, 1]).unwrap();
    let eval = block_on(evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        &[
            Value::from("XBinMethod"),
            Value::from("integers"),
            Value::from("YBinMethod"),
            Value::from("integers"),
            Value::from("BinLimits"),
            Value::Tensor(Tensor::new(vec![0.0, 3.0, 1.0, 3.0], vec![4, 1]).unwrap()),
        ],
    ))
    .expect("histcounts2");
    let (counts_v, xedges_v, yedges_v) = eval.into_triple();
    let counts = tensor_from_value(counts_v);
    let xedges = tensor_from_value(xedges_v);
    let yedges = tensor_from_value(yedges_v);
    assert_eq!(double_values(&xedges), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(double_values(&yedges), &[1.0, 2.0, 3.0]);
    assert_eq!(double_values(&counts).iter().sum::<f64>(), 3.0);
    assert!(double_values(&xedges)
        .windows(2)
        .all(|w| (w[1] - w[0] - 1.0).abs() < 1e-12));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_num_bins_zero_errors() {
    let x = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = block_on(evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        &[Value::from("NumBins"), Value::Num(0.0)],
    ))
    .err()
    .expect("NumBins=0 should fail");
    assert_eq!(
        err.identifier(),
        HISTCOUNTS2_ERROR_NUMBINS_INVALID.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_binmethod_conflict_errors() {
    let x = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
    let err = block_on(evaluate(
        Value::Tensor(x),
        Value::Tensor(y),
        &[
            Value::from("XBinEdges"),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::from("XBinMethod"),
            Value::from("auto"),
        ],
    ))
    .err()
    .expect("XBinEdges + XBinMethod should fail");
    assert_eq!(
        err.identifier(),
        HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT.identifier
    );
}

#[test]
fn histcounts2_returns_bin_indices_with_source_shapes() {
    let x = integer_tensor(IntegerStorage::I16(vec![0, 2, 9]), vec![1, 3]);
    let y = integer_tensor(IntegerStorage::I16(vec![0, 2, 1]), vec![1, 3]);
    let edges = integer_tensor(IntegerStorage::I16(vec![0, 1, 3]), vec![1, 3]);
    let eval = block_on(evaluate(x, y, &[edges.clone(), edges])).expect("histcounts2");
    let (_, _, _, bin_x, bin_y) = eval.into_all();
    let bin_x = tensor_from_value(bin_x);
    let bin_y = tensor_from_value(bin_y);
    assert_eq!(bin_x.shape, vec![1, 3]);
    assert_eq!(bin_y.shape, vec![1, 3]);
    assert_eq!(double_values(&bin_x), &[1.0, 2.0, 0.0]);
    assert_eq!(double_values(&bin_y), &[1.0, 2.0, 2.0]);
}

#[test]
fn histcounts2_wide_explicit_edges_classify_exactly() {
    let base = 9_007_199_254_740_992_u64;
    let x = integer_tensor(IntegerStorage::U64(vec![base, base + 1]), vec![1, 2]);
    let y = integer_tensor(IntegerStorage::U64(vec![base, base + 1]), vec![1, 2]);
    let edges = integer_tensor(
        IntegerStorage::U64(vec![base, base + 1, base + 2]),
        vec![1, 3],
    );
    let eval = block_on(evaluate(x, y, &[edges.clone(), edges])).expect("histcounts2");
    let (counts, _, _, bin_x, bin_y) = eval.into_all();
    assert_eq!(
        double_values(&tensor_from_value(counts)),
        &[1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(double_values(&tensor_from_value(bin_x)), &[1.0, 2.0]);
    assert_eq!(double_values(&tensor_from_value(bin_y)), &[1.0, 2.0]);
}

#[test]
fn histcounts2_percentage_and_cumulative_are_two_dimensional() {
    let counts = vec![1.0, 2.0, 3.0, 4.0];
    let edges = vec![0.0, 1.0, 2.0];
    assert_eq!(
        apply_normalization_2d(
            &counts,
            &edges,
            &edges,
            HistogramNormalization::Percentage,
            10
        ),
        vec![10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(
        apply_normalization_2d(
            &counts,
            &edges,
            &edges,
            HistogramNormalization::CumCount,
            10
        ),
        vec![1.0, 3.0, 4.0, 10.0]
    );
    assert_eq!(
        apply_normalization_2d(&counts, &edges, &edges, HistogramNormalization::Cdf, 10),
        vec![0.1, 0.3, 0.4, 1.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn histcounts2_gpu_roundtrip() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 1.1, 2.9], vec![3, 1]).unwrap();
        let x_view = runmat_accelerate_api::HostTensorView {
            data: double_values(&x),
            shape: &x.shape,
        };
        let y_view = runmat_accelerate_api::HostTensorView {
            data: double_values(&y),
            shape: &y.shape,
        };
        let x_handle = provider.upload(&x_view).expect("upload X");
        let y_handle = provider.upload(&y_view).expect("upload Y");
        let eval = block_on(evaluate(
            Value::GpuTensor(x_handle),
            Value::GpuTensor(y_handle),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            ],
        ))
        .expect("histcounts2");
        let counts = tensor_from_value(eval.into_counts_value());
        assert_eq!(counts.shape, vec![3, 2]);
        let rows = counts.shape[0];
        let count = |ix: usize, iy: usize| double_values(&counts)[ix + iy * rows];
        assert_eq!(count(0, 0), 1.0);
        assert_eq!(count(1, 0), 1.0);
        assert_eq!(count(2, 1), 1.0);
        assert_eq!(count(2, 0), 0.0);
        assert_eq!(count(0, 1), 0.0);
        assert_eq!(count(1, 1), 0.0);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn histcounts2_wgpu_roundtrip() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");

    let x = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
    let y = Tensor::new(vec![1.0, 1.1, 2.9], vec![3, 1]).unwrap();

    let x_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: double_values(&x),
            shape: &x.shape,
        })
        .expect("upload x");
    let y_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: double_values(&y),
            shape: &y.shape,
        })
        .expect("upload y");

    let eval = block_on(evaluate(
        Value::GpuTensor(x_handle),
        Value::GpuTensor(y_handle),
        &[
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 2.0, 3.0], vec![3, 1]).unwrap()),
        ],
    ))
    .expect("histcounts2");

    let (counts_v, xedges_v, yedges_v) = eval.into_triple();
    let counts = tensor_from_value(counts_v);
    let xedges = tensor_from_value(xedges_v);
    let yedges = tensor_from_value(yedges_v);

    assert_eq!(double_values(&xedges), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(double_values(&yedges), &[0.0, 2.0, 3.0]);

    assert_eq!(counts.shape, vec![3, 2]);
    let rows = counts.shape[0];
    let count = |ix: usize, iy: usize| double_values(&counts)[ix + iy * rows];
    assert_eq!(count(0, 0), 1.0);
    assert_eq!(count(1, 0), 1.0);
    assert_eq!(count(2, 1), 1.0);
    assert_eq!(count(2, 0), 0.0);
    assert_eq!(count(0, 1), 0.0);
    assert_eq!(count(1, 1), 0.0);
}
