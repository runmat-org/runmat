use runmat_builtins::{
    TITLE_DESCRIPTOR, TITLE_ERROR_INTERNAL, TITLE_ERROR_INVALID_ARGUMENT,
    TITLE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;

use super::op_common::{annotation_error, map_figure_error, parse_numeric_text_command};
use super::state::set_figure_title_for_axes;

#[runtime_builtin(
    name = "title",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::title"
)]
pub fn title_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &TITLE_DESCRIPTOR,
        &TITLE_ERROR_TOO_MANY_OUTPUTS,
        "title",
    )?;
    let command = parse_numeric_text_command("title", args.view())
        .map_err(|error| annotation_error("title", &TITLE_ERROR_INVALID_ARGUMENT, error))?;
    let target = command.resolve_target();
    set_figure_title_for_axes(target.0, target.1, &command.text, command.style).map_err(|err| {
        annotation_error(
            "title",
            &TITLE_ERROR_INTERNAL,
            map_figure_error("title", err),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::Figure;
    use runmat_value::IntValue;
    use runmat_value::{CellArray, CharArray, StringArray, Value};

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn title_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TITLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = title(txt)"));
        assert!(labels.contains(&"h = title(ax, txt)"));
        assert!(labels.contains(&"h = title(txt, Name, Value, ...)"));
    }

    #[test]
    fn title_returns_plot_object_handle_and_updates_active_axes() {
        let _guard = setup_plot_tests();
        let handle = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Signal".into())],
        ))
        .unwrap();
        let (figure, axes, kind) = decode_plot_object_handle(handle).unwrap();
        assert_eq!(figure, current_figure_handle());
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::Title);

        let fig = clone_figure(figure).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.title.as_deref()),
            Some("Signal")
        );
    }

    #[test]
    fn numeric_text_with_property_pair_is_not_interpreted_as_axes_target() {
        let _guard = setup_plot_tests();
        let handle = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Num(42.0),
                Value::String("FontSize".into()),
                Value::Num(12.0),
            ],
        ))
        .expect("numeric text with a complete property pair");
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|meta| meta.title.as_deref()),
            Some("42")
        );
        let value = crate::builtins::plotting::get::get_builtin(vec![
            Value::Num(handle),
            Value::String("FontSize".into()),
        ])
        .unwrap();
        assert_eq!(value, Value::Num(12.0));
    }

    #[test]
    fn title_accepts_axes_target_and_char_array_with_properties() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.set_subplot_grid(1, 2);
        let handle = crate::builtins::plotting::state::import_figure(figure);
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            handle, 1,
        ));
        let title = Value::CharArray(CharArray::new_row("Right"));
        title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                ax,
                title,
                Value::String("FontSize".into()),
                Value::Num(18.0),
                Value::String("Visible".into()),
                Value::Bool(false),
            ],
        ))
        .unwrap();

        let fig = clone_figure(handle).unwrap();
        assert_eq!(
            fig.axes_metadata(1).and_then(|m| m.title.as_deref()),
            Some("Right")
        );
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.title_style.font_size, Some(18.0));
        assert!(!meta.title_style.visible);
    }

    #[test]
    fn title_rejects_invalid_axes_handle_and_bad_properties() {
        let _guard = setup_plot_tests();

        let invalid_handle = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            99,
        ));
        let err = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![invalid_handle, Value::String("Oops".into())],
        ))
        .unwrap_err();
        assert!(err.message.contains("invalid axes") || err.message.contains("out of range"));

        let err = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Oops".into()),
                Value::String("Bogus".into()),
                Value::Num(1.0),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("unsupported property"));

        let err = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Oops".into()),
                Value::String("FontSize".into()),
                Value::String("big".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("FontSize must be numeric"));
    }

    #[test]
    fn title_accepts_multiline_text_inputs() {
        let _guard = setup_plot_tests();

        title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::StringArray(
                StringArray::new(vec!["Line 1".into(), "Line 2".into()], vec![1, 2]).unwrap(),
            )],
        ))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.title.as_deref()),
            Some("Line 1\nLine 2")
        );

        let cell = CellArray::new(
            vec![Value::String("A".into()), Value::String("B".into())],
            1,
            2,
        )
        .unwrap();
        title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Cell(cell)],
        ))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.title.as_deref()),
            Some("A\nB")
        );
    }

    #[test]
    fn title_formats_wide_integer_text_without_binary64_conversion() {
        let _guard = setup_plot_tests();
        title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Int(IntValue::U64(u64::MAX))],
        ))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0)
                .and_then(|metadata| metadata.title.as_deref()),
            Some("18446744073709551615")
        );
    }
}

#[cfg(test)]
mod output_contract_tests {
    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = super::title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![],
        ))
        .expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:title:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = super::title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![],
        ))
        .expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:title:TooManyOutputs"));
    }
}
