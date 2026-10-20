use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::BLSIMPV_NONDOUBLE_EXTENSION;
use runmat_value::{CellArray, IntegerStorage, StringArray};

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new_2d(data, rows, cols).unwrap())
}

fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

fn empty_numeric() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_pricing_admission_and_numeric_input_use_the_same_geometry() {
    for (shape, values, expected) in [
        (Vec::new(), vec![12.0], vec![1, 1]),
        (vec![2], vec![12.0, 15.0], vec![2, 1]),
        (vec![1, 2, 1], vec![12.0, 15.0], vec![1, 2, 1]),
    ] {
        let value = Value::Tensor(Tensor::new(values.clone(), shape).unwrap());
        admission::preflight_numeric_shapes("blsprice", [&value]).unwrap();
        let input = numeric_input("blsprice", value).unwrap();
        assert_eq!(input.shape, expected);
        assert_eq!(input.values, values);
    }
}

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique test output list")
            .try_into_unbounded_vec()
            .expect("ordinary unbounded test output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

fn as_scalar(value: &Value) -> f64 {
    match value {
        Value::Num(n) => *n,
        other => panic!("expected scalar, got {other:?}"),
    }
}

fn as_tensor(value: &Value) -> &Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn single(data: Vec<f32>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::from_f32(data, shape).expect("single input"))
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn black_scholes_broadcast_indexing_appends_trailing_singletons() {
    let inputs = BroadcastInputs::from_values(
        BLSPRICE,
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(
                Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 1, 3]).unwrap(),
            ),
        ],
        precision::PricingClass::Double,
    )
    .unwrap();
    assert_eq!(inputs.shape, vec![2, 1, 3]);
    assert_eq!(
        (0..inputs.len)
            .map(|index| inputs.arg(0, index))
            .collect::<Vec<_>>(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn blsprice_matches_documented_stock_option_example() {
    let guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = output_list(
        block_on(blsprice_builtin(
            Value::Num(100.0),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    drop(guard);
    assert!((as_scalar(&out[0]) - 13.6953).abs() < 5e-4);
    assert!((as_scalar(&out[1]) - 6.3497).abs() < 5e-4);
}

#[test]
fn blsprice_vectorizes_scalar_inputs() {
    let guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = output_list(
        block_on(blsprice_builtin(
            tensor(vec![100.0, 90.0], 2, 1),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
        ))
        .unwrap(),
    );
    drop(guard);
    let call = as_tensor(&out[0]);
    let put = as_tensor(&out[1]);
    assert_eq!(call.shape(), vec![2, 1]);
    assert_eq!(put.shape(), vec![2, 1]);
    assert!(call.materialize_f64()[0] > call.materialize_f64()[1]);
    assert!(put.materialize_f64()[0] < put.materialize_f64()[1]);
}

#[test]
fn blsprice_broadcast_inputs_read_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = output_list(
        block_on(blsprice_builtin(
            poisoned_integer_tensor(IntegerStorage::U16(vec![100, 90]), vec![2, 1]),
            poisoned_integer_tensor(IntegerStorage::U16(vec![95]), vec![1, 1]),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            crate::builtins::common::test_support::builtin_arguments(vec![
                poisoned_integer_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
            ]),
        ))
        .unwrap(),
    );
    drop(guard);
    let call = as_tensor(&out[0]);
    let put = as_tensor(&out[1]);
    assert_eq!(call.shape(), vec![2, 1]);
    assert_eq!(put.shape(), vec![2, 1]);
    assert!(call.integer_storage().is_none());
    assert!(put.integer_storage().is_none());
    assert!(call.materialize_f64()[0] > call.materialize_f64()[1]);
    assert!(put.materialize_f64()[0] < put.materialize_f64()[1]);
}

#[test]
fn nondouble_pricing_inputs_are_gated_before_provider_work() {
    let integer = || poisoned_integer_tensor(IntegerStorage::U16(vec![100]), vec![1, 1]);
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let failure = block_on(blsprice_builtin(
        integer(),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(0.50),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("integer pricing is a RunMat extension");
    assert_eq!(
        failure.identifier(),
        BLSPRICE_NONDOUBLE_EXTENSION.error_identifier
    );
    let failure = block_on(blsimpv_builtin_source(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        integer(),
        Vec::new(),
    ))
    .expect_err("integer option value is a RunMat extension");
    assert_eq!(
        failure.identifier(),
        BLSIMPV_NONDOUBLE_EXTENSION.error_identifier
    );

    test_support::with_test_provider(|provider| {
        let single = Tensor::from_f32(vec![100.0], vec![1, 1]).expect("single");
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &single).expect("upload");
        provider.reset_telemetry();
        let failure = block_on(blsprice_builtin(
            Value::GpuTensor(handle),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("resident single is a RunMat extension");
        assert_eq!(
            failure.identifier(),
            BLSPRICE_NONDOUBLE_EXTENSION.error_identifier
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    });
}

#[test]
fn invalid_host_pricing_operand_precedes_any_resident_download_or_upload() {
    test_support::with_test_provider(|provider| {
        let price = Tensor::new(vec![100.0], vec![1, 1]).expect("price");
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &price).expect("upload");
        provider.reset_telemetry();
        let failure = block_on(blsprice_builtin(
            Value::GpuTensor(handle.clone()),
            Value::String("bad strike".into()),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("known invalid host operand");
        assert_eq!(failure.identifier(), ERROR_INVALID_INPUT.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);

        provider.reset_telemetry();
        let failure = block_on(blsimpv_builtin_source(
            Value::GpuTensor(handle),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(13.6953),
            vec![Value::String("unsupported".into()), Value::Num(1.0)],
        ))
        .expect_err("known invalid option name");
        assert_eq!(failure.identifier(), ERROR_INVALID_ARGUMENT.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn blsprice_implicitly_expands_nonscalar_inputs() {
    let guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = output_list(
        block_on(blsprice_builtin(
            tensor(vec![100.0, 110.0], 2, 1),
            tensor(vec![90.0, 100.0, 110.0], 1, 3),
            Value::Num(0.05),
            Value::Num(0.5),
            Value::Num(0.2),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    drop(guard);

    let call = as_tensor(&out[0]);
    let put = as_tensor(&out[1]);
    assert_eq!(call.shape(), vec![2, 3]);
    assert_eq!(put.shape(), vec![2, 3]);
    for idx in 0..call.materialize_f64().len() {
        let price = [100.0, 110.0][idx % 2];
        let strike = [90.0, 100.0, 110.0][idx / 2];
        let expected = price_pair(PriceParams {
            price,
            strike,
            rate: 0.05,
            time: 0.5,
            volatility: 0.2,
            yield_rate: 0.0,
        });
        assert_close(call.materialize_f64()[idx], expected.call, 1e-12);
        assert_close(put.materialize_f64()[idx], expected.put, 1e-12);
    }
}

#[test]
fn blsprice_gpu_resident_inputs_preserve_residency() {
    test_support::with_test_provider(|provider| {
        let price = Tensor::new(vec![100.0, 110.0], vec![2, 1]).unwrap();
        let strike = Tensor::new(vec![90.0, 100.0, 110.0], vec![1, 3]).unwrap();
        let rate = Tensor::new(vec![0.05], vec![1, 1]).unwrap();
        let time = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
        let volatility = Tensor::new(vec![0.2], vec![1, 1]).unwrap();
        let yield_rate = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let upload = |tensor: &Tensor| {
            provider
                .upload(&HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: tensor.shape(),
                })
                .expect("upload")
        };
        let price_gpu = upload(&price);
        let strike_gpu = upload(&strike);
        let rate_gpu = upload(&rate);
        let time_gpu = upload(&time);
        let volatility_gpu = upload(&volatility);
        let yield_gpu = upload(&yield_rate);

        provider.reset_telemetry();
        let guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let out = output_list(
            block_on(blsprice_builtin(
                Value::GpuTensor(price_gpu.clone()),
                Value::GpuTensor(strike_gpu.clone()),
                Value::GpuTensor(rate_gpu.clone()),
                Value::GpuTensor(time_gpu.clone()),
                Value::GpuTensor(volatility_gpu.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    yield_gpu.clone(),
                )]),
            ))
            .unwrap(),
        );
        drop(guard);

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.upload_bytes, 0);
        assert_eq!(telemetry.download_bytes, 0);
        assert!(matches!(out[0], Value::GpuTensor(_)));
        assert!(matches!(out[1], Value::GpuTensor(_)));

        let call = test_support::gather(out[0].clone()).expect("gather call");
        let put = test_support::gather(out[1].clone()).expect("gather put");
        assert_eq!(call.shape(), vec![2, 3]);
        assert_eq!(put.shape(), vec![2, 3]);
        for idx in 0..call.materialize_f64().len() {
            let expected = price_pair(PriceParams {
                price: price.materialize_f64()[idx % 2],
                strike: strike.materialize_f64()[idx / 2],
                rate: rate.materialize_f64()[0],
                time: time.materialize_f64()[0],
                volatility: volatility.materialize_f64()[0],
                yield_rate: yield_rate.materialize_f64()[0],
            });
            assert_close(call.materialize_f64()[idx], expected.call, 1e-12);
            assert_close(put.materialize_f64()[idx], expected.put, 1e-12);
        }
    });
}

#[test]
fn blsprice_host_single_class_ignores_double_scalar_representation() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let price = single(vec![100.0, 110.0], vec![2, 1]);
    for strike in [Value::Num(95.0), tensor(vec![95.0], 1, 1)] {
        let result = block_on(blsprice_builtin(
            price.clone(),
            strike,
            Value::Num(0.1),
            Value::Num(0.25),
            Value::Num(0.5),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("single price with double scalar controls");
        assert_eq!(as_tensor(&result).numeric_dtype(), NumericDType::F32);
        assert_eq!(as_tensor(&result).shape(), vec![2, 1]);
    }

    let all_scalar = block_on(blsprice_builtin(
        single(vec![100.0], vec![1, 1]),
        Value::Num(95.0),
        Value::Num(0.1),
        Value::Num(0.25),
        Value::Num(0.5),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("single scalar result");
    assert_eq!(as_tensor(&all_scalar).numeric_dtype(), NumericDType::F32);
    assert_eq!(as_tensor(&all_scalar).shape(), vec![1, 1]);
}

#[test]
fn blsprice_nonscalar_double_array_forces_double() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = block_on(blsprice_builtin(
        single(vec![100.0, 110.0], vec![2, 1]),
        tensor(vec![95.0, 95.0], 2, 1),
        Value::Num(0.1),
        Value::Num(0.25),
        Value::Num(0.5),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("double array determines result precision");
    assert_eq!(as_tensor(&result).numeric_dtype(), NumericDType::F64);
}

#[test]
fn single_boundary_rejects_finite_double_scalar_overflow_and_underflow() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let price = single(vec![100.0, 110.0], vec![2, 1]);
    for strike in [Value::Num(1.0e100), tensor(vec![1.0e-100], 1, 1)] {
        let failure = block_on(blsprice_builtin(
            price.clone(),
            strike,
            Value::Num(0.1),
            Value::Num(0.25),
            Value::Num(0.5),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("double scalar outside single range");
        assert_eq!(failure.identifier(), ERROR_INVALID_INPUT.identifier);
    }
}

#[test]
fn black_scholes_nan_inputs_propagate_elementwise() {
    let guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let prices = output_list(
        block_on(blsprice_builtin(
            tensor(vec![100.0, f64::NAN], 2, 1),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    drop(guard);
    let call = as_tensor(&prices[0]);
    assert!(call.materialize_f64()[0].is_finite());
    assert!(call.materialize_f64()[1].is_nan());

    let vol = block_on(blsimpv_builtin_source(
        tensor(vec![100.0, f64::NAN], 2, 1),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        tensor(vec![13.6953, 13.6953], 2, 1),
        Vec::new(),
    ))
    .unwrap();
    let vols = as_tensor(&vol);
    assert!(vols.materialize_f64()[0].is_finite());
    assert!(vols.materialize_f64()[1].is_nan());
}

#[test]
fn blsprice_empty_yield_uses_default_zero_yield() {
    let with_empty_yield = block_on(blsprice_builtin(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(0.50),
        crate::builtins::common::test_support::builtin_arguments(vec![empty_numeric()]),
    ))
    .unwrap();
    let without_yield = block_on(blsprice_builtin(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(0.50),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert!((as_scalar(&with_empty_yield) - as_scalar(&without_yield)).abs() < 1e-12);
}

#[test]
fn blsprice_single_with_empty_yield_keeps_single_for_numeric_cell_and_string_arrays() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let price = single(vec![100.0, 110.0], vec![2, 1]);
    let no_yield = block_on(blsprice_builtin(
        price.clone(),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(0.50),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("default yield");
    for yield_value in [
        empty_numeric(),
        Value::Cell(CellArray::new(Vec::new(), 0, 0).expect("empty cell")),
        Value::StringArray(StringArray::new(Vec::new(), vec![0, 0]).expect("empty string array")),
    ] {
        let result = block_on(blsprice_builtin(
            price.clone(),
            Value::Num(95.0),
            Value::Num(0.10),
            Value::Num(0.25),
            Value::Num(0.50),
            crate::builtins::common::test_support::builtin_arguments(vec![yield_value]),
        ))
        .expect("empty yield uses default");
        let result = as_tensor(&result);
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.shape(), vec![2, 1]);
        assert_eq!(
            result.materialize_f64(),
            as_tensor(&no_yield).materialize_f64()
        );
    }
}

#[test]
fn blsimpv_inverts_call_price() {
    let call = block_on(blsprice_builtin(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(0.50),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let vol = block_on(blsimpv_builtin_source(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        call,
        Vec::new(),
    ))
    .unwrap();
    assert!((as_scalar(&vol) - 0.5).abs() < 1e-5);
}

#[test]
fn blsimpv_supports_put_class_and_yield() {
    let guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let prices = output_list(
        block_on(blsprice_builtin(
            Value::Num(910.0),
            Value::Num(980.0),
            Value::Num(0.02),
            Value::Num(0.25),
            Value::Num(0.25),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.025)]),
        ))
        .unwrap(),
    );
    drop(guard);
    let vol = block_on(blsimpv_builtin_source(
        Value::Num(910.0),
        Value::Num(980.0),
        Value::Num(0.02),
        Value::Num(0.25),
        prices[1].clone(),
        vec![
            Value::from("Yield"),
            Value::Num(0.025),
            Value::from("Class"),
            Value::from("put"),
        ],
    ))
    .unwrap();
    assert!((as_scalar(&vol) - 0.25).abs() < 1e-5);
}

#[test]
fn blsimpv_vectorizes_class_strings() {
    let values = tensor(vec![13.6953, 6.3497], 2, 1);
    let classes = Value::StringArray(
        StringArray::new(vec!["call".to_string(), "put".to_string()], vec![2, 1]).unwrap(),
    );
    let out = block_on(blsimpv_builtin_source(
        tensor(vec![100.0, 100.0], 2, 1),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        values,
        vec![Value::from("Class"), classes],
    ))
    .unwrap();
    let vols = as_tensor(&out);
    assert_eq!(vols.shape(), vec![2, 1]);
    assert!((vols.materialize_f64()[0] - 0.5).abs() < 1e-3);
    assert!((vols.materialize_f64()[1] - 0.5).abs() < 1e-3);
}

#[test]
fn blsimpv_returns_nan_when_no_solution_in_limit() {
    let out = block_on(blsimpv_builtin_source(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(50.0),
        vec![
            Value::from("Limit"),
            Value::Num(0.1),
            Value::from("Method"),
            Value::from("search"),
        ],
    ))
    .unwrap();
    assert!(as_scalar(&out).is_nan());
}

#[test]
fn finance_scalar_numeric_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
        .expect("typed finance scalar");

    assert_eq!(
        scalar_numeric(BLSIMPV, &Value::Tensor(tensor), "Limit").expect("scalar numeric"),
        2026.0
    );
}

#[test]
fn blsimpv_jackel_method_uses_default_bound_and_empty_defaults() {
    let call = block_on(blsprice_builtin(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(0.50),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let vol = block_on(blsimpv_builtin_source(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        call,
        vec![
            Value::from("Limit"),
            Value::Num(0.1),
            Value::from("Tolerance"),
            empty_numeric(),
            Value::from("Class"),
            empty_numeric(),
            Value::from("Method"),
            Value::from("jackel2016"),
        ],
    ))
    .unwrap();
    assert!((as_scalar(&vol) - 0.5).abs() < 1e-5);
}

#[test]
fn black_scholes_rejects_shape_mismatch_and_bad_options() {
    assert!(block_on(blsprice_builtin(
        tensor(vec![100.0, 90.0], 2, 1),
        tensor(vec![95.0, 96.0, 97.0], 3, 1),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(0.50),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .is_err());
    assert!(block_on(blsimpv_builtin_source(
        Value::Num(100.0),
        Value::Num(95.0),
        Value::Num(0.10),
        Value::Num(0.25),
        Value::Num(10.0),
        vec![Value::from("Method"), Value::from("unknown")],
    ))
    .is_err());
}
