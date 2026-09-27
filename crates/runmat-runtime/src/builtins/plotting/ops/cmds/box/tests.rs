use super::super::test_support::setup;
use super::*;

#[test]
fn extra_arguments_do_not_create_figure() {
    let _guard = setup();
    crate::builtins::plotting::state::reset_plot_state();
    assert!(box_builtin(vec![
        Value::String("on".into()),
        Value::String("extra".into())
    ])
    .is_err());
    assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_none());
}
