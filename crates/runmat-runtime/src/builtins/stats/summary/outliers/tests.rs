use super::options::{logical_mask, numeric_vector, scalar_number, scalar_usize};
use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn filloutliers_preserves_low_rank_numeric_mask_geometry() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    for (shape, data, mask, expected) in [
        (Vec::new(), vec![9.0], vec![1.0], Value::Num(0.0)),
        (
            vec![2],
            vec![1.0, 9.0],
            vec![0.0, 1.0],
            tensor(vec![1.0, 0.0], vec![2]),
        ),
    ] {
        let output = block_on(filloutliers_builtin(
            tensor(data, shape.clone()),
            vec![
                Value::Num(0.0),
                Value::from("OutlierLocations"),
                tensor(mask.clone(), shape.clone()),
            ],
        ))
        .expect("fill with exact-shape numeric mask");
        let Value::OutputList(output) = output else {
            panic!("filled value and mask");
        };
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], expected);
        let Value::LogicalArray(actual_mask) = &output[1] else {
            panic!("logical mask");
        };
        assert_eq!(actual_mask.shape, shape);
        assert_eq!(
            actual_mask.data.as_slice(),
            mask.iter()
                .map(|value| u8::from(*value != 0.0))
                .collect::<Vec<_>>()
        );
    }
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn first_unrepresentable_usize_double() -> f64 {
    if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    }
}

#[test]
fn isoutlier_documented_floating_gpu_fallback_preserves_explicit_residency() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
            .expect("upload observations");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let output = block_on(isoutlier_builtin(Value::GpuTensor(handle), Vec::new()))
            .expect("documented resident isoutlier");
        assert!(matches!(output, Value::GpuTensor(_)));
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn isoutlier_wgpu_fallback_preserves_explicit_residency() {
    let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU provider");
    let input = Tensor::new(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]).unwrap();
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
        .expect("upload observations");
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let output = block_on(isoutlier_builtin(Value::GpuTensor(handle), Vec::new()))
        .expect("documented WGPU isoutlier");
    let Value::GpuTensor(output) = output else {
        panic!("expected resident output");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert_eq!(
        output.device_id(),
        runmat_accelerate_api::AccelProvider::device_id(provider)
    );
    assert_eq!(output.shape(), &[5, 1]);
    assert!(runmat_accelerate_api::handle_is_logical(&output));
}

#[test]
fn outlier_numeric_parsers_read_typed_integer_storage_exactly() {
    let wide = u64::MAX - 1;
    assert_eq!(
        numeric_vector(
            ISOUTLIER_NAME,
            &int_tensor(IntegerStorage::U64(vec![wide, wide - 1]), vec![1, 2]),
        )
        .unwrap(),
        vec![
            IntValue::U64(wide).to_f64(),
            IntValue::U64(wide - 1).to_f64()
        ]
    );
    assert_eq!(
        scalar_number(&int_tensor(IntegerStorage::U64(vec![wide]), vec![1, 1])).unwrap(),
        IntValue::U64(wide).to_f64()
    );
    assert_eq!(
        scalar_usize(
            ISOUTLIER_NAME,
            &int_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
            "window",
        )
        .unwrap(),
        4
    );
    assert_eq!(
        logical_mask(
            FILLOUTLIERS_NAME,
            &int_tensor(IntegerStorage::I16(vec![0, -2, 3]), vec![3, 1]),
        )
        .unwrap(),
        (vec![0, 1, 1], vec![3, 1])
    );
}

#[test]
fn outlier_scalar_usize_rejects_negative_typed_integer_storage() {
    let err = scalar_usize(
        ISOUTLIER_NAME,
        &int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
        "window",
    )
    .unwrap_err();
    assert!(
        err.message().contains("nonnegative integer"),
        "{}",
        err.message()
    );
}

#[test]
fn outlier_window_parser_ignores_all_typed_mirrors() {
    let storages = [
        IntegerStorage::I8(vec![4]),
        IntegerStorage::I16(vec![4]),
        IntegerStorage::I32(vec![4]),
        IntegerStorage::I64(vec![4]),
        IntegerStorage::U8(vec![4]),
        IntegerStorage::U16(vec![4]),
        IntegerStorage::U32(vec![4]),
        IntegerStorage::U64(vec![4]),
    ];

    for storage in storages {
        assert_eq!(
            scalar_usize(ISOUTLIER_NAME, &int_tensor(storage, vec![1, 1]), "window",).unwrap(),
            4
        );
    }
}

#[test]
fn outlier_scalar_usize_rejects_unrepresentable_double_boundary() {
    let err = scalar_usize(
        ISOUTLIER_NAME,
        &Value::Num(first_unrepresentable_usize_double()),
        "window",
    )
    .unwrap_err();
    assert!(
        err.message().contains("platform integer limits"),
        "{}",
        err.message()
    );
}

#[test]
fn filloutliers_accepts_typed_integer_input_fill_and_mask() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let value = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1], 0.0);
    let fill = poisoned_int_tensor(IntegerStorage::I16(vec![-7]), vec![1, 1], 99.0);
    let locations = poisoned_int_tensor(IntegerStorage::U8(vec![0, 0, 1, 0, 0]), vec![5, 1], 0.0);
    let out = block_on(filloutliers_builtin(
        value,
        vec![fill, Value::from("OutlierLocations"), locations],
    ))
    .unwrap();
    assert!(
        matches!(out, Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.0, 2.0, -7.0, 4.0, 5.0])
    );
}

#[test]
fn filloutliers_integer_roles_are_independently_gated() {
    let integer_data = int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1]);
    let floating_data = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

    let error = block_on(filloutliers_builtin(
        integer_data,
        vec![Value::from("center")],
    ))
    .expect_err("integer data gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FilloutliersIntegerDataExtension")
    );

    let error = block_on(filloutliers_builtin(
        floating_data.clone(),
        vec![Value::Int(IntValue::I16(-1))],
    ))
    .expect_err("integer fill gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FilloutliersIntegerFillExtension")
    );

    let numeric_mask = tensor(vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![5, 1]);
    let error = block_on(filloutliers_builtin(
        floating_data,
        vec![
            Value::from("center"),
            Value::from("OutlierLocations"),
            numeric_mask,
        ],
    ))
    .expect_err("numeric mask gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FilloutliersNumericMaskExtension")
    );

    let poison = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let error = block_on(filloutliers_builtin(
        tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]),
        vec![
            Value::from("center"),
            Value::from("ThresholdFactor"),
            poison,
        ],
    ))
    .expect_err("resident argument gate before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FilloutliersResidentInputExtension")
    );
}

#[test]
fn filloutliers_wide_integer_detection_precedes_double_output_boundary() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let base = 9_007_199_254_740_992_u64;
    let value = int_tensor(
        IntegerStorage::U64(vec![base, base + 1, base + 2, base + 3, base + 100]),
        vec![5, 1],
    );
    let output = block_on(filloutliers_builtin(value, vec![Value::from("center")]))
        .expect("wide integer extension");
    let Value::OutputList(values) = output else {
        panic!("expected output list");
    };
    assert!(
        matches!(&values[0], Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 && tensor.integer_storage().is_none())
    );
    assert!(matches!(&values[1], Value::LogicalArray(mask) if mask.data == vec![0, 0, 0, 0, 1]));
}

#[test]
fn filloutliers_rejects_integer_span_that_would_collapse_adjacent_wide_deltas() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact_limit = 1_u64 << 53;
    let value = int_tensor(
        IntegerStorage::U64(vec![0, exact_limit, exact_limit + 1]),
        vec![3, 1],
    );
    let error = block_on(filloutliers_builtin(value, vec![Value::from("center")]))
        .expect_err("inexact integer span must not be collapsed");
    assert!(error.message().contains("span exceeds 2^53"));
}

#[test]
fn filloutliers_integer_data_extension_covers_all_eight_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = [
        IntegerStorage::I8(vec![1, 2, 100, 4, 5]),
        IntegerStorage::I16(vec![1, 2, 100, 4, 5]),
        IntegerStorage::I32(vec![1, 2, 100, 4, 5]),
        IntegerStorage::I64(vec![1, 2, 100, 4, 5]),
        IntegerStorage::U8(vec![1, 2, 100, 4, 5]),
        IntegerStorage::U16(vec![1, 2, 100, 4, 5]),
        IntegerStorage::U32(vec![1, 2, 100, 4, 5]),
        IntegerStorage::U64(vec![1, 2, 100, 4, 5]),
    ];
    for storage in storages {
        let value = int_tensor(storage, vec![5, 1]);
        let output = block_on(filloutliers_builtin(value, vec![Value::from("center")]))
            .expect("integer filloutliers class");
        assert!(
            matches!(output, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 && tensor.materialize_f64()[2] == 4.0)
        );
    }
}

#[test]
fn filloutliers_preserves_documented_single_output_class() {
    let value = Value::Tensor(
        Tensor::new_with_dtype(
            vec![1.0, 2.0, 100.0, 4.0, 5.0],
            vec![5, 1],
            NumericDType::F32,
        )
        .unwrap(),
    );
    let output = block_on(filloutliers_builtin(value, vec![Value::from("center")]))
        .expect("single filloutliers");
    assert!(matches!(output, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32));
}

#[test]
fn filloutliers_logical_locations_require_exact_shape() {
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0], vec![2, 2]);
    let locations = Value::LogicalArray(LogicalArray::new(vec![0, 0, 1, 0], vec![4, 1]).unwrap());
    let error = block_on(filloutliers_builtin(
        value,
        vec![
            Value::from("center"),
            Value::from("OutlierLocations"),
            locations,
        ],
    ))
    .expect_err("mask shape mismatch");
    assert!(error.message().contains("exactly the same shape"));
}

#[test]
fn filloutliers_rejects_logical_locations_with_explicit_find_method() {
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0], vec![4, 1]);
    let locations = Value::LogicalArray(LogicalArray::new(vec![0, 0, 1, 0], vec![4, 1]).unwrap());
    let error = block_on(filloutliers_builtin(
        value,
        vec![
            Value::from("center"),
            Value::from("mean"),
            Value::from("OutlierLocations"),
            locations,
        ],
    ))
    .expect_err("find method and known locations are incompatible");
    assert!(error.message().contains("cannot be combined"));
}

#[test]
fn isoutlier_typed_integer_input_reads_exact_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let value = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1], 0.0);
    let out = block_on(isoutlier_builtin(value, Vec::new())).unwrap();
    let Value::LogicalArray(mask) = out else {
        panic!("expected logical mask");
    };
    assert_eq!(mask.data, vec![0, 0, 1, 0, 0]);
}

#[test]
fn isoutlier_moving_method_reads_typed_integer_input_and_window() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let value = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1], 0.0);
    let window = poisoned_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1], 0.0);
    let out = block_on(isoutlier_builtin(
        value,
        vec![Value::from("movmedian"), window],
    ))
    .unwrap();
    let Value::LogicalArray(mask) = out else {
        panic!("expected logical mask");
    };
    assert_eq!(mask.data, vec![0, 0, 1, 0, 0]);
}

#[test]
fn isoutlier_threshold_factor_reads_typed_integer_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let value = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 9, 10]), vec![5, 1], 0.0);
    let factor = poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1], 0.0);
    let out = block_on(isoutlier_builtin(
        value,
        vec![Value::from("ThresholdFactor"), factor],
    ))
    .unwrap();
    let Value::LogicalArray(mask) = out else {
        panic!("expected logical mask");
    };
    assert_eq!(mask.data, vec![0, 0, 0, 1, 1]);
}

#[test]
fn isoutlier_percentile_bounds_read_typed_integer_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let value = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1], 0.0);
    let bounds = poisoned_int_tensor(IntegerStorage::U8(vec![10, 90]), vec![1, 2], 0.0);
    let out = block_on(isoutlier_builtin(
        value,
        vec![Value::from("percentiles"), bounds],
    ))
    .unwrap();
    let Value::LogicalArray(mask) = out else {
        panic!("expected logical mask");
    };
    assert_eq!(mask.data, vec![1, 0, 1, 0, 0]);
}

#[test]
fn isoutlier_integer_data_is_strictly_gated_and_capabilities_are_declared() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let value = int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]);
    let error =
        block_on(isoutlier_builtin(value, Vec::new())).expect_err("strict mode rejects typed data");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:IsoutlierIntegerDataExtension")
    );
    assert_eq!(ISOUTLIER_EXTENSIONS.len(), 7);
    assert_eq!(ISOUTLIER_INTEGER_CAPABILITIES.len(), 3);
}

#[test]
fn isoutlier_runmat_integer_data_requires_individually_exact_double_values() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let value = int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
    let error = block_on(isoutlier_builtin(value, Vec::new()))
        .expect_err("wide integer data cannot cross the floating boundary");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn isoutlier_detects_columnwise_median_outliers() {
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let out = block_on(isoutlier_builtin(value, Vec::new())).unwrap();
    let Value::LogicalArray(mask) = out else {
        panic!("expected logical mask");
    };
    assert_eq!(mask.data, vec![0, 0, 1, 0, 0]);
}

#[test]
fn isoutlier_returns_threshold_outputs() {
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let out = block_on(isoutlier_builtin(value, vec![Value::from("quartiles")])).unwrap();
    let Value::OutputList(values) = out else {
        panic!("expected output list");
    };
    assert_eq!(values.len(), 4);
    assert!(matches!(&values[0], Value::LogicalArray(mask) if mask.data[2] == 1));
    assert!(matches!(&values[1], Value::Num(_) | Value::Tensor(_)));
}

#[test]
fn filloutliers_supports_center_and_mask_output() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let out = block_on(filloutliers_builtin(value, vec![Value::from("center")])).unwrap();
    let Value::OutputList(values) = out else {
        panic!("expected output list");
    };
    assert!(matches!(&values[0], Value::Tensor(tensor) if tensor.materialize_f64()[2] == 4.0));
    assert!(matches!(&values[1], Value::LogicalArray(mask) if mask.data[2] == 1));
}

#[test]
fn filloutliers_supports_linear_fill() {
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let out = block_on(filloutliers_builtin(value, vec![Value::from("linear")])).unwrap();
    assert!(
        matches!(out, Value::Tensor(tensor) if (tensor.materialize_f64()[2] - 3.0).abs() < 1.0e-12)
    );
}

#[test]
fn filloutliers_supports_numeric_scalar_constant_fill() {
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let out = block_on(filloutliers_builtin(value, vec![Value::Num(-1.0)])).unwrap();
    assert!(matches!(out, Value::Tensor(tensor) if tensor.materialize_f64()[2] == -1.0));
}

#[test]
fn filloutliers_uses_per_column_centers() {
    let value = tensor(
        vec![1.0, 2.0, 100.0, 4.0, 10.0, 11.0, 300.0, 13.0],
        vec![4, 2],
    );
    let out = block_on(filloutliers_builtin(value, vec![Value::from("center")])).unwrap();
    assert!(
        matches!(out, Value::Tensor(tensor) if tensor.materialize_f64()[2] == 3.0 && tensor.materialize_f64()[6] == 12.0)
    );
}

#[test]
fn isoutlier_percentiles_work_and_reject_threshold_factor() {
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let out = block_on(isoutlier_builtin(
        value.clone(),
        vec![
            Value::from("percentiles"),
            tensor(vec![10.0, 90.0], vec![1, 2]),
        ],
    ))
    .unwrap();
    assert!(matches!(out, Value::LogicalArray(mask) if mask.data[0] == 1 && mask.data[2] == 1));
    let err = block_on(isoutlier_builtin(
        value,
        vec![
            Value::from("percentiles"),
            tensor(vec![10.0, 90.0], vec![1, 2]),
            Value::from("ThresholdFactor"),
            Value::Num(2.0),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:outliers:InvalidArgument"));
}

#[test]
fn isoutlier_marks_infinity_when_threshold_is_finite() {
    let value = tensor(vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0], vec![5, 1]);
    let out = block_on(isoutlier_builtin(value, Vec::new())).unwrap();
    assert!(matches!(out, Value::LogicalArray(mask) if mask.data[2] == 1));
}

#[test]
fn moving_even_window_is_backward_weighted_like_matlab() {
    assert_eq!(super::centered_window(3, 8, 4), (1, 5));
}

#[test]
fn previous_and_next_do_not_cross_fill_endpoints() {
    let mask = DetectionResult {
        mask: vec![1, 0, 0],
        mask_shape: vec![3, 1],
        lower: vec![f64::NAN],
        upper: vec![f64::NAN],
        center: vec![f64::NAN],
        lower_full: vec![f64::NAN; 3],
        upper_full: vec![f64::NAN; 3],
        center_full: vec![f64::NAN; 3],
        threshold_shape: vec![1, 1],
    };
    let mut prev = vec![100.0, 2.0, 3.0];
    let prev_original = prev.clone();
    let mut filled = vec![0; 3];
    fill_slice(
        &mut prev,
        &prev_original,
        &[0, 1, 2],
        &mask,
        &FillMethod::Previous,
        &mut filled,
    );
    assert!(prev[0].is_nan());
    assert_eq!(filled, vec![0, 0, 0]);
    let mut next = vec![1.0, 2.0, 100.0];
    let next_original = next.clone();
    let mut mask = mask;
    mask.mask = vec![0, 0, 1];
    fill_slice(
        &mut next,
        &next_original,
        &[0, 1, 2],
        &mask,
        &FillMethod::Next,
        &mut filled,
    );
    assert!(next[2].is_nan());
}

#[test]
fn filloutliers_mask_excludes_detected_endpoint_that_cannot_be_filled() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = tensor(vec![100.0, 2.0, 3.0], vec![3, 1]);
    let locations = Value::LogicalArray(LogicalArray::new(vec![1, 0, 0], vec![3, 1]).unwrap());
    let output = block_on(filloutliers_builtin(
        value,
        vec![
            Value::from("previous"),
            Value::from("OutlierLocations"),
            locations,
        ],
    ))
    .unwrap();
    let Value::OutputList(values) = output else {
        panic!("expected output list");
    };
    assert!(matches!(&values[1], Value::LogicalArray(mask) if mask.data == vec![0, 0, 0]));
}

#[test]
fn outlier_locations_drive_filloutliers_mask() {
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let locations =
        Value::LogicalArray(LogicalArray::new(vec![0, 0, 1, 0, 0], vec![5, 1]).unwrap());
    let out = block_on(filloutliers_builtin(
        value,
        vec![
            Value::from("center"),
            Value::from("OutlierLocations"),
            locations,
        ],
    ))
    .unwrap();
    assert!(matches!(out, Value::Tensor(tensor) if tensor.materialize_f64()[2] == 4.0));
}

#[test]
fn too_many_outputs_error() {
    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
    let err = block_on(isoutlier_builtin(value, Vec::new())).unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:outliers:TooManyOutputs"));
}

#[test]
fn host_visible_invalid_options_precede_unregistered_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(
            vec![3, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("unregistered resident fixture");
    let resident = || Value::GpuTensor(allocation.handle.clone());

    let error = block_on(isoutlier_builtin(
        resident(),
        vec![Value::from("unsupported")],
    ))
    .expect_err("method rejection before gather");
    assert_eq!(error.identifier(), Some("RunMat:outliers:InvalidArgument"));

    let error = block_on(isoutlier_builtin(
        resident(),
        vec![Value::from("SamplePoints"), Value::Num(1.0)],
    ))
    .expect_err("unsupported option before gather");
    assert_eq!(error.identifier(), Some("RunMat:outliers:Unsupported"));

    let error = block_on(isoutlier_builtin(
        resident(),
        vec![Value::from("percentiles"), tensor(vec![10.0], vec![1, 1])],
    ))
    .expect_err("malformed host percentile bounds before gather");
    assert_eq!(error.identifier(), Some("RunMat:outliers:InvalidArgument"));

    let error = block_on(filloutliers_builtin(
        resident(),
        vec![Value::from("spline")],
    ))
    .expect_err("unsupported fill method before gather");
    assert_eq!(error.identifier(), Some("RunMat:outliers:Unsupported"));

    let error = block_on(filloutliers_builtin(
        resident(),
        vec![Value::from("center"), Value::from("OutlierLocations")],
    ))
    .expect_err("missing mask before gather");
    assert_eq!(error.identifier(), Some("RunMat:outliers:InvalidArgument"));
}

#[test]
fn implicit_trailing_singleton_dimension_keeps_bounded_nd_geometry() {
    let input = Tensor::new((1..=8).map(f64::from).collect(), vec![2, 2, 2]).unwrap();
    let options = DetectionOptions {
        dim: Some(usize::MAX),
        forced_mask: Some((vec![0, 0, 0, 1, 0, 0, 0, 0], vec![2, 2, 2])),
        ..DetectionOptions::default()
    };
    let result = detect_outliers(&input, &options, FILLOUTLIERS_NAME).unwrap();
    assert_eq!(result.threshold_shape, vec![2, 2, 2]);
    assert_eq!(result.mask, vec![0, 0, 0, 1, 0, 0, 0, 0]);
    let (filled, filled_mask) =
        fill_outlier_tensor(&input, &result, &options, &FillMethod::Constant(0.0)).unwrap();
    assert_eq!(filled_mask, vec![0, 0, 0, 1, 0, 0, 0, 0]);
    assert!(
        matches!(filled, Value::Tensor(tensor) if tensor.shape() == [2, 2, 2]
        && tensor.materialize_f64() == vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 8.0])
    );
}
