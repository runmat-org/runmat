use super::super::test_support::setup;
use super::*;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::set::set_builtin;
use runmat_value::Value;
#[test]
fn invalid_mode_and_extra_arguments_do_not_create_figure() {
    let _guard = setup();
    crate::builtins::plotting::state::reset_plot_state();
    for args in [
        vec![Value::String("invalid".into())],
        vec![Value::String("on".into()), Value::String("off".into())],
    ] {
        assert!(
            hidden_builtin(crate::builtins::common::test_support::builtin_arguments(
                args
            ))
            .is_err()
        );
        assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_none());
    }
}
#[test]
fn hidden_toggles_and_sets_current_axes_hidden_line_removal() {
    let _guard = setup();
    let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("HiddenLineRemoval".into())]).unwrap(),
        Value::String("on".into())
    );
    assert!(
        !hidden_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![]
        ))
        .unwrap()
    );
    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("HiddenLineRemoval".into())]).unwrap(),
        Value::String("off".into())
    );
    assert!(
        hidden_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("on".into())]
        ))
        .unwrap()
    );
    assert_eq!(
        get_builtin(vec![ax, Value::String("HiddenLineRemoval".into())]).unwrap(),
        Value::String("on".into())
    );
}

#[test]
fn hidden_accepts_axes_target_without_selecting_current_axes() {
    let _guard = setup();
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

    hidden_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Num(ax1), Value::String("off".into())],
    ))
    .unwrap();

    assert_eq!(
        crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap(),
        Value::Num(ax2)
    );

    assert_eq!(
        get_builtin(vec![
            Value::Num(ax1),
            Value::String("HiddenLineRemoval".into())
        ])
        .unwrap(),
        Value::String("off".into())
    );
    assert_eq!(
        get_builtin(vec![
            Value::Num(ax2),
            Value::String("HiddenLineRemoval".into())
        ])
        .unwrap(),
        Value::String("on".into())
    );
}

#[test]
fn hidden_line_removal_property_set_get_and_rejects_invalid_values() {
    let _guard = setup();
    let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

    set_builtin(vec![
        ax.clone(),
        Value::String("HiddenLineRemoval".into()),
        Value::String("off".into()),
    ])
    .unwrap();
    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("HiddenLineRemoval".into())]).unwrap(),
        Value::String("off".into())
    );

    let err = set_builtin(vec![
        ax,
        Value::String("HiddenLineRemoval".into()),
        Value::String("maybe".into()),
    ])
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:set:InvalidArgument"));
}

#[test]
fn hidden_rejects_extra_arguments() {
    let _guard = setup();
    let err = hidden_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("on".into()), Value::String("off".into())],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:hidden:InvalidArgument"));
}
