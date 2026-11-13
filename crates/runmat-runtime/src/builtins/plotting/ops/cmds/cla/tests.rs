use super::super::test_support::setup;
use super::*;
use crate::builtins::plotting::state::*;
use runmat_plot::plots::{Figure, LinePlot};

#[test]
fn cla_target_reset_preserves_selection_and_other_plot_identity() {
    let _guard = setup();
    reset_plot_state();
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    let first = figure.add_line_plot_on_axes(LinePlot::new(vec![1., 2.], vec![2., 3.]).unwrap(), 0);
    let second =
        figure.add_line_plot_on_axes(LinePlot::new(vec![1., 2.], vec![3., 4.]).unwrap(), 1);
    let handle = import_figure(figure);
    let removed = register_line_handle(handle, 0, first);
    let surviving = register_line_handle(handle, 1, second);
    select_axes_for_figure(handle, 1).unwrap();
    assert!(cla_builtin(vec![
        Value::Num(encode_axes_handle(handle, 0)),
        Value::String("reset".into())
    ])
    .unwrap());
    assert!(plot_child_handle_snapshot(removed).is_err());
    assert_eq!(
        plot_child_handle_snapshot(surviving).unwrap().plot_index(),
        Some(0)
    );
    assert_eq!(current_axes_state().active_index, 1);
}

#[test]
fn cla_invalid_syntax_and_strict_output_reject_before_creation() {
    let _guard = setup();
    reset_plot_state();
    assert!(cla_builtin(vec![Value::String("bad".into())]).is_err());
    assert!(current_figure_handle_if_exists().is_none());
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let _output = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    assert!(cla_builtin(vec![]).is_err());
    assert!(current_figure_handle_if_exists().is_none());
}
