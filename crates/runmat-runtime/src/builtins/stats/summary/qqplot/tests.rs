use super::*;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::state::PlotTestLockGuard;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage};

fn setup() -> PlotTestLockGuard {
    let guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    guard
}

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new_integer(storage, vec![rows, cols]).unwrap())
}

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor.materialize_f64(),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn qqplot_numeric_helpers_read_typed_integer_storage_exactly() {
    let wide = u64::MAX - 1;
    let tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![wide, wide - 1]), vec![2, 1]).unwrap();
    assert_eq!(
        tensor_values_f64(&tensor),
        vec![
            IntValue::U64(wide).to_f64(),
            IntValue::U64(wide - 1).to_f64()
        ]
    );
    assert_eq!(
        columns_from_tensor(tensor, &[2, 1]).unwrap(),
        vec![vec![
            IntValue::U64(wide).to_f64(),
            IntValue::U64(wide - 1).to_f64()
        ]]
    );
    let pvec = Tensor::new_integer(IntegerStorage::U8(vec![75, 25]), vec![1, 2]).unwrap();
    assert_eq!(probabilities_from_tensor(pvec).unwrap(), vec![0.25, 0.75]);
}

#[test]
fn qqplot_scalar_typed_integer_sample_uses_storage_len_for_default_shape() {
    let input = Tensor::new_integer(IntegerStorage::I16(vec![42]), Vec::new()).unwrap();

    let series = normal_series_from_tensor(input, None).unwrap();

    assert_eq!(series.len(), 1);
    assert_eq!(series[0].observed, vec![42.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn qqplot_public_path_preserves_rank_zero_and_rank_one_samples() {
    let _guard = setup();
    for (sample, expected) in [
        (
            Value::Tensor(Tensor::new(vec![4.0], Vec::new()).unwrap()),
            vec![4.0],
        ),
        (
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2]).unwrap()),
            vec![1.0, 3.0],
        ),
    ] {
        let handles = tensor_data(
            block_on(qqplot_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![sample]),
            ))
            .unwrap(),
        );
        let observed = tensor_data(
            get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
        );
        assert_eq!(observed, expected);
    }
}

#[test]
fn qqplot_normal_vector_returns_data_reference_and_quartile_handles() {
    let _guard = setup();
    let out = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![1.0, 2.0, 4.0, 8.0],
            4,
            1,
        )]),
    ))
    .unwrap();
    let handles = tensor_data(out);
    assert_eq!(handles.len(), 3);

    let x = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("XData".into())]).unwrap(),
    );
    let y = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
    );
    assert_eq!(x.len(), 4);
    assert_eq!(y, vec![1.0, 2.0, 4.0, 8.0]);
    assert!(x[0] < x[1] && x[2] < x[3]);
    assert_eq!(
        get_builtin(vec![Value::Num(handles[0]), Value::String("Marker".into())]).unwrap(),
        Value::String("+".into())
    );
    assert_eq!(
        get_builtin(vec![
            Value::Num(handles[1]),
            Value::String("LineStyle".into())
        ])
        .unwrap(),
        Value::String("-".into())
    );
    assert_eq!(
        get_builtin(vec![
            Value::Num(handles[2]),
            Value::String("LineStyle".into())
        ])
        .unwrap(),
        Value::String("--".into())
    );
}

#[test]
fn qqplot_percentage_vector_interpolates_sample_quantiles() {
    let _guard = setup();
    let out = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![0.0, 10.0, 20.0], 3, 1),
            tensor(Vec::new(), 0, 0),
            tensor(vec![25.0, 50.0, 75.0], 1, 3),
        ]),
    ))
    .unwrap();
    let handles = tensor_data(out);
    let y = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
    );
    assert_eq!(y, vec![5.0, 10.0, 15.0]);
}

#[test]
fn qqplot_accepts_typed_integer_samples_and_pvec() {
    let _guard = setup();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            int_tensor(IntegerStorage::I16(vec![0, 10, 20]), 3, 1),
            int_tensor(IntegerStorage::I16(Vec::new()), 0, 0),
            int_tensor(IntegerStorage::U8(vec![25, 50, 75]), 1, 3),
        ]),
    ))
    .unwrap();
    let handles = tensor_data(out);
    let y = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
    );
    assert_eq!(y, vec![5.0, 10.0, 15.0]);
}

#[test]
fn qqplot_integer_extensions_are_gated_and_inexact_values_reject() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(
            IntegerStorage::I16(vec![1, 2]),
            2,
            1,
        )]),
    ))
    .expect_err("strict mode rejects integer samples");
    assert_eq!(
        error.identifier(),
        QQPLOT_INTEGER_SAMPLE_EXTENSION.error_identifier
    );
    drop(strict);

    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0], 2, 1),
            tensor(Vec::new(), 0, 0),
            int_tensor(IntegerStorage::U8(vec![25, 75]), 1, 2),
        ]),
    ))
    .expect_err("strict mode rejects integer pvec");
    assert_eq!(
        error.identifier(),
        QQPLOT_INTEGER_PVEC_EXTENSION.error_identifier
    );
    drop(strict);

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, 2]),
            2,
            1,
        )]),
    ))
    .expect_err("inexact integer sample rejects");
    assert!(error.message().contains("exactly representable as double"));
    drop(extensions);
}

#[test]
fn qqplot_strict_mode_gates_explicit_gpu_before_provider_access() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(handle)]),
    ))
    .expect_err("strict mode rejects explicit GPU input before gather");
    assert_eq!(
        error.identifier(),
        QQPLOT_EXPLICIT_GPU_EXTENSION.error_identifier
    );
}

#[test]
fn qqplot_logical_input_is_a_gated_extension() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)]),
    ))
    .expect_err("strict mode rejects logical sample extension");
    assert_eq!(
        error.identifier(),
        QQPLOT_LOGICAL_INPUT_EXTENSION.error_identifier
    );
}

#[test]
fn qqplot_two_sample_uses_common_probabilities() {
    let _guard = setup();
    let out = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 3.0, 5.0], 3, 1),
            tensor(vec![2.0, 4.0, 6.0], 3, 1),
            tensor(vec![50.0], 1, 1),
        ]),
    ))
    .unwrap();
    let handles = tensor_data(out);
    let x = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("XData".into())]).unwrap(),
    );
    let y = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
    );
    assert_eq!(x, vec![3.0]);
    assert_eq!(y, vec![4.0]);
}

#[test]
fn qqplot_two_sample_matrices_pair_columns_and_do_not_treat_y_as_pvec() {
    let _guard = setup();
    let out = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![0.1, 0.2, 1.0, 2.0], 2, 2),
            tensor(vec![0.3, 0.4, 3.0, 4.0], 2, 2),
            tensor(vec![50.0], 1, 1),
        ]),
    ))
    .unwrap();
    let handles = tensor_data(out);
    assert_eq!(handles.len(), 6);
    let x1 = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("XData".into())]).unwrap(),
    );
    let y1 = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
    );
    let x2 = tensor_data(
        get_builtin(vec![Value::Num(handles[3]), Value::String("XData".into())]).unwrap(),
    );
    let y2 = tensor_data(
        get_builtin(vec![Value::Num(handles[3]), Value::String("YData".into())]).unwrap(),
    );
    assert_eq!(x1, vec![0.15000000000000002]);
    assert_eq!(y1.len(), 1);
    // Convex binary64 interpolation may round by one ULP from the literal.
    assert!(y1[0].to_bits().abs_diff(0.35_f64.to_bits()) <= 1);
    assert_eq!(x2, vec![1.5]);
    assert_eq!(y2, vec![3.5]);
}

#[test]
fn qqplot_matrix_creates_one_series_per_column_and_omits_nan() {
    let _guard = setup();
    let out = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![1.0, 2.0, f64::NAN, 4.0, 10.0, 20.0],
            3,
            2,
        )]),
    ))
    .unwrap();
    let handles = tensor_data(out);
    assert_eq!(handles.len(), 6);
    let y = tensor_data(
        get_builtin(vec![Value::Num(handles[0]), Value::String("YData".into())]).unwrap(),
    );
    assert_eq!(y, vec![1.0, 2.0]);
}

#[test]
fn qqplot_rejects_bad_inputs() {
    let err = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0], 2, 1),
            tensor(vec![3.0, 4.0], 2, 1),
            tensor(vec![-1.0], 1, 1),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("pvec"));
}
