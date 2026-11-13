use crate::builtins::plotting::state::{axes_handle_exists, reset_plot_state, FigureHandle};
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clone_figure, current_figure_handle};
use runmat_value::Value;

fn invalid_style() -> Vec<Value> {
    vec![
        Value::String("heading".into()),
        Value::String("Color".into()),
    ]
}

#[test]
fn invalid_annotation_forms_do_not_create_an_implicit_scene() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    let default_figure = FigureHandle::default();
    let absent = || !axes_handle_exists(default_figure, 0);

    reset_plot_state();
    assert!(absent());
    assert!(
        super::sgtitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            invalid_style()
        ))
        .is_err()
    );
    assert!(absent());

    reset_plot_state();
    assert!(super::super::suptitle::suptitle_builtin(
        crate::builtins::common::test_support::builtin_arguments(invalid_style())
    )
    .is_err());
    assert!(absent());

    reset_plot_state();
    assert!(super::super::subtitle::subtitle_builtin(
        crate::builtins::common::test_support::builtin_arguments(invalid_style())
    )
    .is_err());
    assert!(absent());

    for invoke in [
        super::super::title::title_builtin
            as fn(crate::call::BuiltinArguments) -> crate::BuiltinResult<f64>,
        super::super::xlabel::xlabel_builtin,
        super::super::ylabel::ylabel_builtin,
        super::super::zlabel::zlabel_builtin,
    ] {
        reset_plot_state();
        assert!(absent());
        assert!(
            invoke(crate::builtins::common::test_support::builtin_arguments(
                invalid_style(),
            ))
            .is_err()
        );
        assert!(absent());
    }

    reset_plot_state();
    assert!(absent());
    assert!(super::super::xline::xline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(f64::NAN)])
    )
    .is_err());
    assert!(absent());

    reset_plot_state();
    assert!(absent());
    assert!(super::super::yline::yline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(f64::NAN)])
    )
    .is_err());
    assert!(absent());
}

#[test]
fn invalid_text_and_reference_line_leave_existing_scene_unchanged() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    super::super::title::title_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("existing title".into())],
    ))
    .expect("create the scene with a title");
    super::super::xline::xline_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Num(1.0)],
    ))
    .expect("seed the scene with a reference line");
    let figure = current_figure_handle();
    let before = clone_figure(figure).unwrap();
    let original_title = before.axes_metadata(0).and_then(|meta| meta.title.clone());
    let original_plots = before.plots().count();
    assert!(original_title.is_some());
    assert!(original_plots > 0);
    assert!(super::super::title::title_builtin(
        crate::builtins::common::test_support::builtin_arguments(invalid_style())
    )
    .is_err());
    assert!(super::super::xline::xline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(f64::NAN)])
    )
    .is_err());
    let after = clone_figure(figure).unwrap();
    assert_eq!(
        after.axes_metadata(0).and_then(|meta| meta.title.clone()),
        original_title
    );
    assert_eq!(after.plots().count(), original_plots);
}
