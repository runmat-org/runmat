use super::*;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::set::set_builtin;
use crate::builtins::plotting::state::axis_display_bounds_snapshot_for_axes;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
use futures::executor::block_on;
use runmat_builtins::{BuiltinOutputMode, DASPECT_INTEGER_CAPABILITIES};
use runmat_value::{IntegerStorage, LogicalArray};

#[test]
fn daspect_query_output_count_rejects_before_default_state_creation() {
    use crate::builtins::plotting::state;
    let _guard = lock_plot_registry();
    state::reset_plot_state();
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    assert_eq!(
        daspect_builtin_source(vec![]).unwrap_err().identifier(),
        Some("RunMat:daspect:TooManyOutputs")
    );
    assert!(state::current_figure_handle_if_exists().is_none());
}

fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
    let guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    guard
}

fn set_daspect(args: Vec<Value>) -> BuiltinResult<Value> {
    let _output = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    daspect_builtin_source(args)
}

#[test]
fn daspect_descriptor_and_runtime_keep_setters_outputless() {
    for signature in runmat_builtins::DASPECT_DESCRIPTOR.signatures {
        if signature.label.starts_with("daspect(") {
            assert!(signature.outputs.is_empty(), "{}", signature.label);
        }
    }
    assert_eq!(
        runmat_builtins::DASPECT_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    let _guard = setup();
    let error = daspect_builtin_source(vec![ratio(&[1.0, 2.0, 1.0])])
        .expect_err("setter output must reject");
    assert!(error.message().contains("do not return"));
    assert_eq!(
        set_daspect(vec![ratio(&[1.0, 2.0, 1.0])]).unwrap(),
        Value::output_list(Vec::new())
    );
}

#[test]
fn daspect_queries_sets_ratio_and_mode() {
    let _guard = setup();
    let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
    assert_eq!(
        set_daspect(vec![ratio(&[1.0, 2.0, 1.0])]).unwrap(),
        Value::output_list(Vec::new())
    );
    assert_eq!(
        daspect_builtin_source(vec![Value::String("mode".into())]).unwrap(),
        Value::String("manual".into())
    );
    assert_eq!(
        get_builtin(vec![
            ax.clone(),
            Value::String("DataAspectRatioMode".into())
        ])
        .unwrap(),
        Value::String("manual".into())
    );

    let prop = get_builtin(vec![ax, Value::String("DataAspectRatio".into())]).unwrap();
    assert_eq!(tensor_data(prop), vec![1.0, 2.0, 1.0]);

    assert_eq!(
        set_daspect(vec![Value::String("auto".into())]).unwrap(),
        Value::output_list(Vec::new())
    );
}

#[test]
fn daspect_explicit_axes_does_not_change_current_axes() {
    let _guard = setup();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let ax1 = crate::builtins::plotting::subplot::subplot_builtin(
        Value::Num(1.0),
        Value::Num(2.0),
        Value::Num(1.0),
    )
    .unwrap();
    let ax2 = crate::builtins::plotting::subplot::subplot_builtin(
        Value::Num(1.0),
        Value::Num(2.0),
        Value::Num(2.0),
    )
    .unwrap();
    set_daspect(vec![Value::Num(ax1), ratio(&[3.0, 1.0, 1.0])]).unwrap();
    let current = daspect_builtin_source(Vec::new()).unwrap();
    assert_eq!(tensor_data(current), vec![1.0, 1.0, 1.0]);
    let left = daspect_builtin_source(vec![Value::Num(ax1)]).unwrap();
    assert_eq!(tensor_data(left), vec![3.0, 1.0, 1.0]);
    let right = daspect_builtin_source(vec![Value::Num(ax2)]).unwrap();
    assert_eq!(tensor_data(right), vec![1.0, 1.0, 1.0]);
}

#[test]
fn daspect_round_trips_through_axes_properties_and_bounds() {
    let _guard = setup();
    let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
    block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        ratio(&[0.0, 10.0]),
        ratio(&[0.0, 1.0]),
    ]))
    .unwrap();
    set_builtin(vec![
        ax.clone(),
        Value::String("DataAspectRatio".into()),
        ratio(&[2.0, 1.0, 1.0]),
    ])
    .unwrap();
    let bounds = axis_display_bounds_snapshot_for_axes(
        crate::builtins::plotting::current_figure_handle(),
        0,
    )
    .unwrap()
    .unwrap();
    assert_eq!(bounds, (0.0, 10.0, -2.0, 3.0));
}

#[test]
fn daspect_rejects_invalid_ratios() {
    let _guard = setup();
    let err = set_daspect(vec![ratio(&[1.0, 0.0, 1.0])]).unwrap_err();
    assert!(err.message.contains("positive"));
}

#[test]
fn daspect_accepts_all_documented_integer_ratio_classes() {
    let _guard = setup();
    for storage in [
        IntegerStorage::I8(vec![1, 2, 3]),
        IntegerStorage::I16(vec![1, 2, 3]),
        IntegerStorage::I32(vec![1, 2, 3]),
        IntegerStorage::I64(vec![1, 2, 3]),
        IntegerStorage::U8(vec![1, 2, 3]),
        IntegerStorage::U16(vec![1, 2, 3]),
        IntegerStorage::U32(vec![1, 2, 3]),
        IntegerStorage::U64(vec![1, 2, 3]),
    ] {
        let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 3]).unwrap());
        assert_eq!(
            set_daspect(vec![value]).unwrap(),
            Value::output_list(Vec::new())
        );
        assert_eq!(
            tensor_data(daspect_builtin_source(Vec::new()).unwrap()),
            vec![1.0, 2.0, 3.0]
        );
    }
    assert_eq!(DASPECT_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
}

#[test]
fn daspect_rejects_nonvector_logical_complex_and_resident_ratio_inputs() {
    let _guard = setup();
    let nonvector = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 1, 3]).unwrap());
    assert!(set_daspect(vec![nonvector]).is_err());
    let logical = Value::LogicalArray(LogicalArray::new(vec![1, 1, 1], vec![1, 3]).unwrap());
    assert!(set_daspect(vec![logical]).is_err());
    let complex = Value::ComplexTensor(
        runmat_value::ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], vec![1, 3])
            .unwrap(),
    );
    assert!(set_daspect(vec![complex]).is_err());
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 3],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    assert!(set_daspect(vec![resident]).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn daspect_rank_one_ratio_is_admitted_but_rank_zero_is_not() {
    let _guard = setup();
    let scalar = Value::Tensor(Tensor::new(vec![0.5], Vec::new()).expect("rank-zero ratio"));
    let error = ratio_from_value(&scalar).unwrap_err();
    assert!(error.message().contains("3-element row or column"));
    assert!(set_daspect(vec![scalar]).is_err());

    let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("rank-one ratio");
    assert_eq!(
        set_daspect(vec![Value::Tensor(vector)]).unwrap(),
        Value::output_list(Vec::new())
    );
    assert_eq!(
        tensor_data(daspect_builtin_source(Vec::new()).unwrap()),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn daspect_rejects_registered_resident_ratio_without_provider_access() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &[1, 3],
            })
            .unwrap();
        provider.reset_telemetry();
        let error = set_daspect(vec![Value::GpuTensor(handle.clone())])
            .expect_err("resident ratio must reject");
        assert!(error.message().contains("host row or column"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let _ = handle.release();
    });
}

#[test]
fn daspect_parses_all_eight_integer_handle_classes_exactly() {
    for value in [
        runmat_value::IntValue::I8(1),
        runmat_value::IntValue::I16(1),
        runmat_value::IntValue::I32(1),
        runmat_value::IntValue::I64(1),
        runmat_value::IntValue::U8(1),
        runmat_value::IntValue::U16(1),
        runmat_value::IntValue::U32(1),
        runmat_value::IntValue::U64(1),
    ] {
        assert_eq!(
            exact_numeric_axes_alias(&Value::Int(value)).unwrap(),
            Some(1.0)
        );
    }
    assert!(
        exact_numeric_axes_alias(&Value::Int(runmat_value::IntValue::U64(
            9_007_199_254_740_993,
        )))
        .is_err()
    );
}

#[test]
fn daspect_numeric_axes_alias_is_compatibility_gated() {
    let _guard = setup();
    let ax_value = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
    let Value::Num(ax) = ax_value else {
        panic!("expected numeric axes handle");
    };
    let alias = Value::Int(runmat_value::IntValue::U64(ax as u64));
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = daspect_builtin_source(vec![alias.clone()]).expect_err("numeric axes alias gate");
    assert_eq!(
        error.identifier(),
        DASPECT_NUMERIC_AXES_ALIAS_EXTENSION.error_identifier
    );
    drop(strict);
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    assert!(daspect_builtin_source(vec![alias]).is_ok());
    assert!(daspect_builtin_source(vec![Value::Num(ax + 0.5)]).is_err());
}

fn ratio(values: &[f64]) -> Value {
    Value::Tensor(
        Tensor::new(values.to_vec(), vec![1, values.len()]).expect("aspect-ratio test vector"),
    )
}

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor.materialize_f64(),
        other => panic!("expected tensor, got {other:?}"),
    }
}
