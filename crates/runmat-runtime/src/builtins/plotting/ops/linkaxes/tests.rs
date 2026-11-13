use super::*;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::set::set_builtin;
use crate::builtins::plotting::subplot::subplot_builtin;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
use runmat_value::Tensor;

#[test]
fn linkaxes_output_admission_preserves_existing_link_and_empty_arrays_are_noop() {
    let _guard = setup();
    let first = subplot(1., 2., 1.);
    let second = subplot(1., 2., 2.);
    let handles = tensor(vec![first, second], vec![1, 1, 2]);
    assert_eq!(
        linkaxes_builtin_source(vec![handles.clone()]).unwrap(),
        Value::output_list(vec![])
    );
    {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        assert_eq!(
            linkaxes_builtin_source(vec![handles, Value::from("off")])
                .unwrap_err()
                .identifier(),
            Some("RunMat:linkaxes:TooManyOutputs")
        );
    }
    set_limits(first, "XLim", [2., 7.]);
    assert_eq!(get_limits(second, "XLim"), vec![2., 7.]);
    assert_eq!(
        linkaxes_builtin_source(vec![tensor(vec![], vec![0, 0])]).unwrap(),
        Value::output_list(vec![])
    );
    assert!(
        linkaxes_builtin_source(vec![Value::Int(runmat_value::IntValue::U64(u64::MAX))]).is_err()
    );
    assert!(linkaxes_builtin_source(vec![
        tensor(vec![first, second], vec![1, 2]),
        Value::from("z")
    ])
    .is_err());
    set_limits(first, "XLim", [4., 8.]);
    assert_eq!(get_limits(second, "XLim"), vec![4., 8.]);
}

fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
    let guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    guard
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).expect("linkaxes test tensor"))
}

fn numeric_vec(value: Value) -> Vec<f64> {
    let tensor = Tensor::try_from(&value).expect("tensor value");
    tensor.materialize_f64()
}

fn subplot(rows: f64, cols: f64, index: f64) -> f64 {
    subplot_builtin(Value::Num(rows), Value::Num(cols), Value::Num(index)).expect("subplot")
}

fn set_limits(ax: f64, property: &str, limits: [f64; 2]) {
    set_builtin(vec![
        Value::Num(ax),
        Value::String(property.into()),
        tensor(limits.to_vec(), vec![1, 2]),
    ])
    .expect("set limits");
}

fn get_limits(ax: f64, property: &str) -> Vec<f64> {
    numeric_vec(
        get_builtin(vec![Value::Num(ax), Value::String(property.into())]).expect("get limits"),
    )
}

#[test]
fn linkaxes_handle_vector_reads_typed_integer_storage_exactly() {
    let _guard = setup();
    let first = subplot(1.0, 2.0, 1.0);
    let second = subplot(1.0, 2.0, 2.0);
    let handles = Tensor::new_integer(
        runmat_value::IntegerStorage::U32(vec![first as u32, second as u32]),
        vec![1, 2],
    )
    .expect("typed axes handles");
    let first_target = decode_axes_handle(first).expect("first axes handle");
    let second_target = decode_axes_handle(second).expect("second axes handle");

    assert_eq!(
        axes_handles_from_value(&Value::Tensor(handles)).expect("axes handles"),
        vec![first_target, second_target]
    );
}

#[test]
fn linkaxes_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = LINKAXES_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"linkaxes(ax)"));
    assert!(labels.contains(&"linkaxes(ax, option)"));
}

#[test]
fn linkaxes_propagates_selected_axis_limits_and_unlinks() {
    let _guard = setup();
    let ax1 = subplot(2.0, 1.0, 1.0);
    let ax2 = subplot(2.0, 1.0, 2.0);
    let handles = tensor(vec![ax1, ax2], vec![1, 2]);

    linkaxes_builtin_source(vec![handles.clone(), Value::String("x".into())]).expect("link x");
    set_limits(ax1, "XLim", [0.0, 10.0]);
    set_limits(ax2, "YLim", [5.0, 9.0]);

    assert_eq!(get_limits(ax2, "XLim"), vec![0.0, 10.0]);
    assert_ne!(get_limits(ax1, "YLim"), vec![5.0, 9.0]);

    linkaxes_builtin_source(vec![handles.clone(), Value::String("xy".into())]).expect("link xy");
    set_limits(ax2, "YLim", [2.0, 4.0]);
    assert_eq!(get_limits(ax1, "YLim"), vec![2.0, 4.0]);

    let x_before_off = get_limits(ax2, "XLim");
    linkaxes_builtin_source(vec![handles, Value::String("off".into())]).expect("unlink");
    set_limits(ax1, "XLim", [20.0, 30.0]);
    assert_eq!(get_limits(ax2, "XLim"), x_before_off);
}

#[test]
fn linkaxes_initial_sync_uses_union_of_current_limits() {
    let _guard = setup();
    let ax1 = subplot(2.0, 1.0, 1.0);
    let ax2 = subplot(2.0, 1.0, 2.0);
    set_limits(ax1, "XLim", [0.0, 2.0]);
    set_limits(ax2, "XLim", [5.0, 9.0]);

    linkaxes_builtin_source(vec![
        tensor(vec![ax1, ax2], vec![1, 2]),
        Value::String("x".into()),
    ])
    .expect("link x");

    assert_eq!(get_limits(ax1, "XLim"), vec![0.0, 9.0]);
    assert_eq!(get_limits(ax2, "XLim"), vec![0.0, 9.0]);
}

#[test]
fn linkaxes_preserves_overlapping_axis_specific_groups() {
    let _guard = setup();
    let ax1 = subplot(3.0, 1.0, 1.0);
    let ax2 = subplot(3.0, 1.0, 2.0);
    let ax3 = subplot(3.0, 1.0, 3.0);

    linkaxes_builtin_source(vec![
        tensor(vec![ax1, ax2], vec![1, 2]),
        Value::String("x".into()),
    ])
    .expect("link x group");
    linkaxes_builtin_source(vec![
        tensor(vec![ax1, ax3], vec![1, 2]),
        Value::String("y".into()),
    ])
    .expect("link y group");

    set_limits(ax2, "XLim", [10.0, 12.0]);
    assert_eq!(get_limits(ax1, "XLim"), vec![10.0, 12.0]);
    assert_ne!(get_limits(ax3, "XLim"), vec![10.0, 12.0]);

    set_limits(ax3, "YLim", [20.0, 24.0]);
    assert_eq!(get_limits(ax1, "YLim"), vec![20.0, 24.0]);
    assert_ne!(get_limits(ax2, "YLim"), vec![20.0, 24.0]);
}

#[test]
fn linkaxes_clear_figure_removes_stale_groups() {
    let _guard = setup();
    let ax1 = subplot(2.0, 1.0, 1.0);
    let ax2 = subplot(2.0, 1.0, 2.0);
    linkaxes_builtin_source(vec![
        tensor(vec![ax1, ax2], vec![1, 2]),
        Value::String("x".into()),
    ])
    .expect("link x");

    let _ = clear_figure(None).expect("clear figure");
    let ax1 = subplot(2.0, 1.0, 1.0);
    let ax2 = subplot(2.0, 1.0, 2.0);
    set_limits(ax1, "XLim", [30.0, 40.0]);

    assert_ne!(get_limits(ax2, "XLim"), vec![30.0, 40.0]);
}
