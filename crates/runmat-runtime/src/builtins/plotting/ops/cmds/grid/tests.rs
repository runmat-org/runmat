use super::super::test_support::setup;
use super::*;
use crate::builtins::plotting::get::get_builtin;
use runmat_value::Value;
#[test]
fn grid_minor_toggles_minor_grid_without_changing_major_grid() {
    let _guard = setup();
    let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

    let enabled = grid_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("minor".into())],
    ))
    .unwrap();
    assert!(enabled);
    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("Grid".into())]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("MinorGrid".into())]).unwrap(),
        Value::Bool(true)
    );

    let enabled = grid_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("minor".into())],
    ))
    .unwrap();
    assert!(!enabled);
    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("Grid".into())]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("MinorGrid".into())]).unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn grid_off_disables_major_and_minor_grid() {
    let _guard = setup();
    let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

    grid_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("minor".into())],
    ))
    .unwrap();
    let enabled = grid_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("off".into())],
    ))
    .unwrap();
    assert!(!enabled);
    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("Grid".into())]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("MinorGrid".into())]).unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn grid_rejects_extra_arguments() {
    let _guard = setup();
    let err = grid_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("minor".into()), Value::String("on".into())],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:grid:InvalidArgument"));
}
