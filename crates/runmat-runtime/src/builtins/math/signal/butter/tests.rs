use super::*;
use futures::executor::block_on;
use num_complex::Complex64;
use runmat_builtins::{
    builtin_function_by_name, BuiltinIntegerOutputClassRule, REQUIRED_DEFAULT_BINDING,
};
use runmat_value::{IntValue, IntegerStorage};

fn output_values(order: usize, cutoff: Value, rest: Vec<Value>, outputs: usize) -> Vec<Value> {
    let _guard = crate::output_count::push_output_count(Some(outputs)).expect("output-count scope");
    match block_on(butter_builtin(
        None,
        Value::Num(order as f64),
        cutoff,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .expect("butter")
    {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn character_options_and_row_column_cutoffs_match_catalog_forms() {
    for token in ["high", "s"] {
        let chars = runmat_value::CharArray::from_unicode_scalars(
            token.chars().collect(),
            1,
            token.encode_utf16().count(),
        )
        .unwrap();
        let values = output_values(3, Value::Num(0.25), vec![Value::CharArray(chars)], 2);
        assert_eq!(tensor_data(&values[0]).len(), 4);
    }
    for shape in [vec![1, 2], vec![2, 1]] {
        let cutoff = Value::Tensor(Tensor::new(vec![0.2, 0.4], shape).unwrap());
        let values = output_values(2, cutoff, vec![Value::String("bandpass".into())], 2);
        assert_eq!(tensor_data(&values[0]).len(), 5);
    }
}

fn call_values(
    order: Value,
    cutoff: Value,
    rest: Vec<Value>,
    outputs: usize,
) -> BuiltinResult<Value> {
    let _guard = crate::output_count::push_output_count(Some(outputs)).expect("output-count scope");
    block_on(butter_builtin(
        None,
        order,
        cutoff,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

#[cfg(feature = "wgpu")]
fn register_wgpu_provider_available() -> bool {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_ok()
        && runmat_accelerate_api::provider().is_some()
}

fn tensor_data(value: &Value) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor.materialize_f64().clone(),
        other => panic!("expected real tensor, got {other:?}"),
    }
}

#[test]
fn butter_rejects_non_finite_transfer_coefficients() {
    let err = numerical::normalize_transfer_function(
        vec![Complex64::new(f64::INFINITY, 0.0)],
        vec![Complex64::new(1.0, 0.0)],
    )
    .expect_err("non-finite numerator should fail");
    assert_eq!(err.identifier(), BUTTER_ERROR_INTERNAL.identifier);
    assert!(err.message().contains("non-finite numerator coefficients"));
}

#[test]
fn numerical_design_observes_invocation_cancellation() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let source = Rc::new(CancellationSource::new());
    let cancel_on_turn = Rc::clone(&source);
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4096).unwrap()),
        source.token(),
        move || {
            cancel_on_turn.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(evaluate_with_cooperation(
        Value::Num(100.0),
        Value::Num(0.25),
        crate::call::FunctionArgumentView::from_slice(&[]),
        &mut cooperation,
    ))
    .expect_err("cooperative design must cancel before publishing outputs");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

fn complex_column(value: &Value) -> Vec<Complex64> {
    match value {
        Value::Tensor(tensor) => tensor
            .materialize_f64()
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect(),
        Value::ComplexTensor(tensor) => tensor
            .materialize_f64()
            .iter()
            .map(|(re, im)| Complex64::new(*re, *im))
            .collect(),
        other => panic!("expected complex-compatible vector, got {other:?}"),
    }
}

fn assert_close(lhs: f64, rhs: f64, tol: f64) {
    let diff = (lhs - rhs).abs();
    assert!(
        diff <= tol,
        "expected {lhs} to be within {tol} of {rhs}; diff={diff}"
    );
}

fn assert_slice_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (left, right)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let diff = (left - right).abs();
        assert!(
            diff <= tol,
            "mismatch at {idx}: {left} vs {right}; diff={diff}"
        );
    }
}

fn dc_gain(b: &[f64], a: &[f64]) -> f64 {
    b.iter().sum::<f64>() / a.iter().sum::<f64>()
}

fn nyquist_gain(b: &[f64], a: &[f64]) -> f64 {
    let numerator = b
        .iter()
        .enumerate()
        .map(|(idx, coeff)| if idx % 2 == 0 { *coeff } else { -*coeff })
        .sum::<f64>();
    let denominator = a
        .iter()
        .enumerate()
        .map(|(idx, coeff)| if idx % 2 == 0 { *coeff } else { -*coeff })
        .sum::<f64>();
    numerator / denominator
}

#[test]
fn butter_descriptor_signatures_and_errors() {
    assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
    let builtin = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME)
        .expect("canonical butter builtin");
    assert_eq!(builtin.bindings, &REQUIRED_DEFAULT_BINDING);
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        builtin.binding_identity(&builtin.bindings[0])
    );
    let descriptor = builtin.descriptor;
    let labels = descriptor
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"b = butter(n, Wn)"));
    assert!(labels.contains(&"b = butter(n, Wn, 's')"));
    assert!(labels.contains(&"[b, a] = butter(n, Wn, ftype)"));
    assert!(labels.contains(&"[b, a] = butter(n, Wn, 's')"));
    assert!(labels.contains(&"[z, p, k] = butter(n, Wn, ftype, 's')"));
    assert!(labels.contains(&"[z, p, k] = butter(n, Wn, 's')"));
    assert!(labels.contains(&"[A, B, C, D] = butter(n, Wn)"));
    assert!(labels.contains(&"[A, B, C, D] = butter(n, Wn, 's')"));
    assert!(descriptor
        .errors
        .iter()
        .any(|error| error.code == "RM.BUTTER.INVALID_FREQUENCY"));
    assert_eq!(builtin.extensions.len(), 9);
    assert_eq!(builtin.integer_capabilities.len(), 2);
    assert!(builtin
        .integer_capabilities
        .iter()
        .all(|capability| capability.output_class == BuiltinIntegerOutputClassRule::Double));
}

#[test]
fn butter_second_order_coefficients_match_reference_values() {
    let low = output_values(2, Value::Num(0.25), Vec::new(), 2);
    assert_slice_close(
        &tensor_data(&low[0]),
        &[0.0976310729378175, 0.195262145875635, 0.0976310729378175],
        1e-12,
    );
    assert_slice_close(
        &tensor_data(&low[1]),
        &[1.0, -0.9428090415820632, 0.33333333333333326],
        1e-12,
    );

    let high = output_values(
        2,
        Value::Num(0.25),
        vec![Value::String("high".to_string().into())],
        2,
    );
    assert_slice_close(
        &tensor_data(&high[0]),
        &[0.5690355937288492, -1.1380711874576983, 0.5690355937288492],
        1e-12,
    );
    assert_slice_close(
        &tensor_data(&high[1]),
        &[1.0, -0.9428090415820632, 0.33333333333333326],
        1e-12,
    );

    let wn = Value::Tensor(Tensor::new(vec![0.2, 0.4], vec![1, 2]).unwrap());
    let bandpass = output_values(
        2,
        wn.clone(),
        vec![Value::String("bandpass".to_string().into())],
        2,
    );
    assert_slice_close(
        &tensor_data(&bandpass[0]),
        &[
            0.06745527388907191,
            0.0,
            -0.13491054777814382,
            0.0,
            0.06745527388907191,
        ],
        1e-12,
    );
    assert_slice_close(
        &tensor_data(&bandpass[1]),
        &[
            1.0,
            -1.942_468_776_547_884,
            2.119_202_397_144_283,
            -1.216651635515531,
            0.4128015980961886,
        ],
        1e-12,
    );

    let bandstop = output_values(2, wn, vec![Value::String("stop".to_string().into())], 2);
    assert_slice_close(
        &tensor_data(&bandstop[0]),
        &[
            0.6389455251590221,
            -1.579_560_206_031_707,
            2.254_112_944_922_426,
            -1.579_560_206_031_707,
            0.6389455251590221,
        ],
        1e-12,
    );
    assert_slice_close(
        &tensor_data(&bandstop[1]),
        &[
            1.0,
            -1.942_468_776_547_884,
            2.119_202_397_144_283,
            -1.216651635515531,
            0.4128015980961886,
        ],
        1e-12,
    );
}

#[test]
fn butter_first_order_lowpass_matches_closed_form() {
    let values = output_values(1, Value::Num(0.5), Vec::new(), 2);
    assert_slice_close(&tensor_data(&values[0]), &[0.5, 0.5], 1e-12);
    assert_slice_close(&tensor_data(&values[1]), &[1.0, 0.0], 1e-12);
}

#[test]
fn butter_order_reads_typed_integer_storage_exactly() {
    let order = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("order");
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let values = match block_on(butter_builtin(
        None,
        Value::Tensor(order),
        Value::Num(0.5),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("butter")
    {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => panic!("expected output list, got {other:?}"),
    };

    assert_slice_close(&tensor_data(&values[0]), &[0.5, 0.5], 1e-12);
    assert_slice_close(&tensor_data(&values[1]), &[1.0, 0.0], 1e-12);
}

#[test]
fn butter_cutoff_reads_typed_integer_storage_exactly() {
    let cutoff = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("cutoff");
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let values = output_values(
        1,
        Value::Tensor(cutoff),
        vec![Value::String("s".to_string().into())],
        2,
    );
    assert_slice_close(&tensor_data(&values[0]), &[0.0, 1.0], 1e-12);
    assert_slice_close(&tensor_data(&values[1]), &[1.0, 1.0], 1e-12);
}

#[test]
fn butter_accepts_all_integer_order_and_cutoff_classes_only_in_runmat_mode() {
    let integer_orders = [
        Value::Int(IntValue::I8(1)),
        Value::Int(IntValue::I16(1)),
        Value::Int(IntValue::I32(1)),
        Value::Int(IntValue::I64(1)),
        Value::Int(IntValue::U8(1)),
        Value::Int(IntValue::U16(1)),
        Value::Int(IntValue::U32(1)),
        Value::Int(IntValue::U64(1)),
    ];
    let integer_cutoffs = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];

    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = call_values(Value::Int(IntValue::U64(1)), Value::Num(0.5), Vec::new(), 2)
            .expect_err("typed order must be gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:ButterIntegerOrderExtension")
        );
        let cutoff = Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("cutoff");
        let err = call_values(
            Value::Num(1.0),
            Value::Tensor(cutoff),
            vec![Value::String("s".to_string().into())],
            2,
        )
        .expect_err("typed cutoff must be gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:ButterIntegerCutoffExtension")
        );
    }

    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    for order in integer_orders {
        let Value::OutputList(outputs) =
            call_values(order, Value::Num(0.5), Vec::new(), 2).expect("integer order")
        else {
            panic!("expected coefficient outputs");
        };
        assert!(outputs.iter().all(
            |value| matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64)
        ));
    }
    for storage in integer_cutoffs {
        let cutoff = Tensor::new_integer(storage, vec![1, 1]).expect("cutoff");
        let Value::OutputList(outputs) = call_values(
            Value::Num(1.0),
            Value::Tensor(cutoff),
            vec![Value::String("s".to_string().into())],
            2,
        )
        .expect("integer cutoff") else {
            panic!("expected coefficient outputs");
        };
        assert_slice_close(&tensor_data(&outputs[0]), &[0.0, 1.0], 1e-12);
        assert_slice_close(&tensor_data(&outputs[1]), &[1.0, 1.0], 1e-12);
    }
}

#[test]
fn butter_nondouble_alias_and_large_order_extensions_are_independent() {
    let single_one = || Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single");
    let cases = [
        (
            Value::Bool(true),
            Value::Num(0.5),
            Vec::new(),
            "RunMat:compatibility:ButterLogicalOrderExtension",
        ),
        (
            Value::Tensor(single_one()),
            Value::Num(0.5),
            Vec::new(),
            "RunMat:compatibility:ButterSingleOrderExtension",
        ),
        (
            Value::Num(1.0),
            Value::Bool(true),
            vec![Value::String("s".to_string().into())],
            "RunMat:compatibility:ButterLogicalCutoffExtension",
        ),
        (
            Value::Num(1.0),
            Value::Tensor(single_one()),
            vec![Value::String("s".to_string().into())],
            "RunMat:compatibility:ButterSingleCutoffExtension",
        ),
        (
            Value::Num(1.0),
            Value::Num(0.5),
            vec![Value::String("lowpass".to_string().into())],
            "RunMat:compatibility:ButterOptionAliasExtension",
        ),
        (
            Value::Num(501.0),
            Value::Num(0.5),
            Vec::new(),
            "RunMat:compatibility:ButterLargeOrderExtension",
        ),
    ];
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for (order, cutoff, rest, identifier) in cases {
        let err = call_values(order, cutoff, rest, 2).expect_err("extension must be gated");
        assert_eq!(err.identifier(), Some(identifier));
    }
}

#[test]
fn butter_rejects_wide_integer_order_without_float_rounding() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let err = call_values(
        Value::Int(IntValue::U64(u64::MAX)),
        Value::Num(0.5),
        Vec::new(),
        2,
    )
    .expect_err("wide order must reject exactly");
    assert_eq!(err.identifier(), BUTTER_ERROR_INVALID_ORDER.identifier);
    assert!(err.message().contains("resource limit"));
}

#[cfg(feature = "wgpu")]
#[test]
fn butter_gpu_input_is_gated_before_gather_and_runmat_fallback_is_host_double() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("provider");
    let cutoff = Tensor::new(vec![0.5], vec![1, 1]).expect("cutoff");
    let handle =
        crate::builtins::common::gpu_helpers::upload_tensor(provider, &cutoff).expect("upload");
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = call_values(
            Value::Num(1.0),
            Value::GpuTensor(handle.clone()),
            Vec::new(),
            2,
        )
        .expect_err("gpu input must be gated");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:ButterGpuInputExtension")
        );
    }
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let Value::OutputList(outputs) =
        call_values(Value::Num(1.0), Value::GpuTensor(handle), Vec::new(), 2)
            .expect("host fallback")
    else {
        panic!("expected coefficient outputs");
    };
    assert!(outputs.iter().all(
        |value| matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64)
    ));
}

#[test]
fn butter_first_order_highpass_matches_closed_form() {
    let values = output_values(
        1,
        Value::Num(0.5),
        vec![Value::String("high".to_string().into())],
        2,
    );
    assert_slice_close(&tensor_data(&values[0]), &[0.5, -0.5], 1e-12);
    assert_slice_close(&tensor_data(&values[1]), &[1.0, 0.0], 1e-12);
}

#[test]
fn butter_fourth_order_lowpass_has_unity_dc_gain() {
    let values = output_values(
        4,
        Value::Tensor(Tensor::new(vec![0.25], vec![1, 1]).unwrap()),
        vec![Value::String("low".to_string().into())],
        2,
    );
    let b = tensor_data(&values[0]);
    let a = tensor_data(&values[1]);
    assert_eq!(b.len(), 5);
    assert_eq!(a.len(), 5);
    assert_close(dc_gain(&b, &a), 1.0, 1e-8);
    assert!(nyquist_gain(&b, &a).abs() < 1e-10);
}

#[test]
fn butter_bandpass_and_bandstop_have_expected_edge_zeros() {
    let wn = Value::Tensor(Tensor::new(vec![0.2, 0.4], vec![1, 2]).unwrap());
    let bandpass = output_values(
        3,
        wn.clone(),
        vec![Value::String("bandpass".to_string().into())],
        2,
    );
    let b_pass = tensor_data(&bandpass[0]);
    let a_pass = tensor_data(&bandpass[1]);
    assert_eq!(b_pass.len(), 7);
    assert_eq!(a_pass.len(), 7);
    assert!(dc_gain(&b_pass, &a_pass).abs() < 1e-9);
    assert!(nyquist_gain(&b_pass, &a_pass).abs() < 1e-9);

    let bandstop = output_values(3, wn, vec![Value::String("stop".to_string().into())], 2);
    let b_stop = tensor_data(&bandstop[0]);
    let a_stop = tensor_data(&bandstop[1]);
    assert_eq!(b_stop.len(), 7);
    assert_eq!(a_stop.len(), 7);
    assert_close(dc_gain(&b_stop, &a_stop), 1.0, 1e-8);
    assert_close(nyquist_gain(&b_stop, &a_stop), 1.0, 1e-8);
}

#[test]
fn butter_zpk_outputs_stable_digital_poles() {
    let values = output_values(4, Value::Num(0.35), Vec::new(), 3);
    let zeros = complex_column(&values[0]);
    let poles = complex_column(&values[1]);
    assert_eq!(zeros.len(), 4);
    assert_eq!(poles.len(), 4);
    assert!(zeros.iter().all(|zero| (zero.re + 1.0).abs() < 1e-8));
    assert!(poles.iter().all(|pole| pole.norm() < 1.0));
    match &values[2] {
        Value::Num(value) => assert!(*value > 0.0),
        other => panic!("expected scalar gain, got {other:?}"),
    }
}

#[test]
fn butter_state_space_outputs_canonical_dimensions() {
    let values = output_values(2, Value::Num(0.25), Vec::new(), 4);
    match &values[0] {
        Value::Tensor(tensor) => assert_eq!(tensor.shape(), [2, 2]),
        other => panic!("expected state matrix, got {other:?}"),
    }
    match &values[1] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![1.0, 0.0]);
        }
        other => panic!("expected input matrix, got {other:?}"),
    }
    match &values[2] {
        Value::Tensor(tensor) => assert_eq!(tensor.shape(), [1, 2]),
        other => panic!("expected output matrix, got {other:?}"),
    }
    assert!(matches!(values[3], Value::Num(_)));
}

#[test]
fn butter_analog_lowpass_and_highpass_are_supported() {
    let low = output_values(
        1,
        Value::Num(2.0),
        vec![Value::String("s".to_string().into())],
        2,
    );
    assert_slice_close(&tensor_data(&low[0]), &[0.0, 2.0], 1e-12);
    assert_slice_close(&tensor_data(&low[1]), &[1.0, 2.0], 1e-12);

    let high = output_values(
        1,
        Value::Num(2.0),
        vec![
            Value::String("high".to_string().into()),
            Value::String("s".to_string().into()),
        ],
        2,
    );
    assert_slice_close(&tensor_data(&high[0]), &[1.0, 0.0], 1e-12);
    assert_slice_close(&tensor_data(&high[1]), &[1.0, 2.0], 1e-12);
}

#[test]
fn butter_rejects_invalid_frequency_and_outputs() {
    let err = block_on(butter_builtin(
        None,
        Value::Num(2.0),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("digital Wn at Nyquist should be rejected");
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:butter:InvalidFrequency")
    );

    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let err = block_on(butter_builtin(
        None,
        Value::Num(2.0),
        Value::Num(0.4),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("too many outputs should be rejected");
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:butter:TooManyOutputs")
    );
}
