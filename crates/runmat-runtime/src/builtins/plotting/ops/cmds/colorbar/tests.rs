use super::super::test_support::setup;
use super::*;

#[test]
fn strict_colorbar_reuses_live_handle_and_off_deletes_without_selection_drift() {
    let _guard = setup();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    crate::builtins::plotting::state::reset_plot_state();
    let state = current_axes_state();
    let first = colorbar_builtin_source(vec![]).unwrap();
    assert_eq!(first, colorbar_builtin_source(vec![]).unwrap());
    let Value::Num(bar) = first else {
        panic!("registered ColorBar handle");
    };
    assert_eq!(
        crate::builtins::plotting::state::plot_child_handle_snapshot(bar)
            .unwrap()
            .type_name(),
        "colorbar"
    );
    crate::builtins::plotting::state::configure_subplot(1, 2, 1).unwrap();
    let target = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
        state.handle,
        0,
    ));
    {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        assert!(
            colorbar_builtin_source(vec![target.clone(), Value::String("off".into())]).is_err()
        );
        assert!(crate::builtins::plotting::state::plot_child_handle_snapshot(bar).is_ok());
    }
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert_eq!(
        colorbar_builtin_source(vec![target, Value::String("off".into())]).unwrap(),
        Value::output_list(vec![])
    );
    assert!(crate::builtins::plotting::state::plot_child_handle_snapshot(bar).is_err());
    assert_eq!(current_axes_state().active_index, 1);
}

#[test]
fn runmat_colorbar_preserves_toggle_and_logical_results() {
    let _guard = setup();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::plotting::state::reset_plot_state();
    assert_eq!(colorbar_builtin_source(vec![]).unwrap(), Value::Bool(true));
    assert_eq!(colorbar_builtin_source(vec![]).unwrap(), Value::Bool(false));
}

#[test]
fn extra_arguments_do_not_create_figure() {
    let _guard = setup();
    crate::builtins::plotting::state::reset_plot_state();
    assert!(colorbar_builtin_source(vec![
        Value::String("on".into()),
        Value::String("extra".into())
    ])
    .is_err());
    assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_none());
}
