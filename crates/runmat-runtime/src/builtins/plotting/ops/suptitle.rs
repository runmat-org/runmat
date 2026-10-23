use runmat_builtins::{
    SUPTITLE_DESCRIPTOR, SUPTITLE_ERROR_INTERNAL, SUPTITLE_ERROR_INVALID_ARGUMENT,
    SUPTITLE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Value;

#[runtime_builtin(
    name = "suptitle",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::suptitle"
)]
pub fn suptitle_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &SUPTITLE_DESCRIPTOR,
        &SUPTITLE_ERROR_TOO_MANY_OUTPUTS,
        "suptitle",
    )?;
    super::sgtitle::sgtitle_impl(
        "suptitle",
        &SUPTITLE_ERROR_INVALID_ARGUMENT,
        &SUPTITLE_ERROR_INTERNAL,
        args,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, figure::figure_builtin,
        reset_hold_state_for_run,
    };

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn suptitle_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SUPTITLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = suptitle(txt)"));
        assert!(labels.contains(&"h = suptitle(fig, txt)"));
        assert!(labels.contains(&"h = suptitle(txt, Name, Value, ...)"));
    }

    #[test]
    fn suptitle_returns_handle_and_updates_current_figure() {
        let _guard = setup();
        let handle = suptitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Overview".into())],
        ))
        .unwrap();
        let (figure, axes, kind) = decode_plot_object_handle(handle).unwrap();
        assert_eq!(figure, current_figure_handle());
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::SuperTitle);

        let fig = clone_figure(figure).unwrap();
        assert_eq!(fig.sg_title.as_deref(), Some("Overview"));
    }

    #[test]
    fn suptitle_accepts_explicit_figure_target() {
        let _guard = setup();
        let fig = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(421.0)]),
        )
        .unwrap();
        suptitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Num(fig), Value::String("Target Figure".into())],
        ))
        .unwrap();

        let figure = clone_figure(crate::builtins::plotting::FigureHandle::from(421)).unwrap();
        assert_eq!(figure.sg_title.as_deref(), Some("Target Figure"));
    }

    #[test]
    fn suptitle_applies_text_style_pairs() {
        let _guard = setup();
        let fig = figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(422.0)]),
        )
        .unwrap();
        suptitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Num(fig),
                Value::String("Styled".into()),
                Value::String("FontSize".into()),
                Value::Num(20.0),
                Value::String("FontWeight".into()),
                Value::String("bold".into()),
                Value::String("Color".into()),
                Value::String("red".into()),
                Value::String("Interpreter".into()),
                Value::String("none".into()),
                Value::String("Visible".into()),
                Value::Bool(false),
            ],
        ))
        .unwrap();

        let figure = clone_figure(crate::builtins::plotting::FigureHandle::from(422)).unwrap();
        assert_eq!(figure.sg_title.as_deref(), Some("Styled"));
        assert_eq!(figure.sg_title_style.font_size, Some(20.0));
        assert_eq!(figure.sg_title_style.font_weight.as_deref(), Some("bold"));
        assert!(figure.sg_title_style.color.is_some());
        assert_eq!(figure.sg_title_style.interpreter.as_deref(), Some("none"));
        assert!(!figure.sg_title_style.visible);
    }

    #[test]
    fn suptitle_handle_supports_get_and_set() {
        let _guard = setup();
        let handle = suptitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Initial".into())],
        ))
        .unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("text".into()));

        set_builtin(vec![
            Value::Num(handle),
            Value::String("String".into()),
            Value::String("Updated".into()),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();

        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string, Value::String("Updated".into()));
        let weight =
            get_builtin(vec![Value::Num(handle), Value::String("FontWeight".into())]).unwrap();
        assert_eq!(weight, Value::String("bold".into()));
    }

    #[test]
    fn suptitle_reports_errors_with_suptitle_context() {
        let _guard = setup();
        let err = suptitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![],
        ))
        .unwrap_err();
        assert!(err.message.contains("suptitle"));
        assert!(!err.message.contains("sgtitle"));

        let err = suptitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Top".into()),
                Value::String("Bogus".into()),
                Value::Num(1.0),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("suptitle"));
        assert!(!err.message.contains("sgtitle"));
    }
}

#[cfg(test)]
mod output_contract_tests {
    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = super::suptitle_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:suptitle:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = super::suptitle_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:suptitle:TooManyOutputs"));
    }
}
