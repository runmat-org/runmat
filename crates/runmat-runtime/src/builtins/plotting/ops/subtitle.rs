use runmat_builtins::SUBTITLE_INTEGER_AXES_EXTENSION;
use runmat_builtins::{
    SUBTITLE_DESCRIPTOR, SUBTITLE_ERROR_INTERNAL, SUBTITLE_ERROR_INVALID_ARGUMENT,
    SUBTITLE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;

use super::op_common::{annotation_error, map_figure_error, parse_text_command};
use super::state::set_figure_subtitle_for_axes;

#[runtime_builtin(
    name = "subtitle",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::subtitle"
)]
pub fn subtitle_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &SUBTITLE_DESCRIPTOR,
        &SUBTITLE_ERROR_TOO_MANY_OUTPUTS,
        "subtitle",
    )?;
    if args.len() >= 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SUBTITLE_INTEGER_AXES_EXTENSION,
            "subtitle",
        )?;
    }
    let command = parse_text_command("subtitle", args.view())
        .map_err(|error| annotation_error("subtitle", &SUBTITLE_ERROR_INVALID_ARGUMENT, error))?;
    let target = command.resolve_target();
    set_figure_subtitle_for_axes(target.0, target.1, &command.text, command.style).map_err(|err| {
        annotation_error(
            "subtitle",
            &SUBTITLE_ERROR_INTERNAL,
            map_figure_error("subtitle", err),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::PlotObjectKind;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_value::IntValue;
    use runmat_value::{CellArray, Value};

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn subtitle_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SUBTITLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = subtitle(txt)"));
        assert!(labels.contains(&"h = subtitle(ax, txt)"));
        assert!(labels.contains(&"h = subtitle(txt, Name, Value, ...)"));
        assert!(labels.contains(&"h = subtitle(ax, txt, Name, Value, ...)"));
    }

    #[test]
    fn subtitle_returns_text_handle_and_updates_active_axes() {
        let _guard = setup_plot_tests();
        let handle = subtitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Slope = 2".into())],
        ))
        .unwrap();
        let (figure, axes, kind) =
            crate::builtins::plotting::state::decode_plot_object_handle(handle).unwrap();
        assert_eq!(figure, current_figure_handle());
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::Subtitle);

        let fig = clone_figure(figure).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.subtitle.as_deref()),
            Some("Slope = 2")
        );
        assert!(fig.has_any_titles());
    }

    #[test]
    fn subtitle_accepts_axes_target_multiline_text_and_properties() {
        let _guard = setup_plot_tests();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let lines = Value::Cell(
            CellArray::new(
                vec![Value::String("left".into()), Value::String("right".into())],
                1,
                2,
            )
            .unwrap(),
        );
        subtitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Num(ax),
                lines,
                Value::String("FontSize".into()),
                Value::Num(13.0),
                Value::String("FontWeight".into()),
                Value::String("bold".into()),
                Value::String("Visible".into()),
                Value::Bool(false),
            ],
        ))
        .unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.subtitle.as_deref(), Some("left\nright"));
        assert_eq!(meta.subtitle_style.font_size, Some(13.0));
        assert_eq!(meta.subtitle_style.font_weight.as_deref(), Some("bold"));
        assert!(!meta.subtitle_style.visible);
    }

    #[test]
    fn get_axes_and_subtitle_handle_expose_subtitle() {
        let _guard = setup_plot_tests();
        let handle = subtitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Details".into())],
        ))
        .unwrap();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();

        let subtitle_prop =
            get_builtin(vec![ax.clone(), Value::String("Subtitle".into())]).unwrap();
        assert_eq!(subtitle_prop, Value::Num(handle));

        let string_prop =
            get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string_prop, Value::String("Details".into()));
    }

    #[test]
    fn subtitle_rejects_bad_properties() {
        let _guard = setup_plot_tests();
        let err = subtitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Oops".into()),
                Value::String("FontSize".into()),
                Value::String("large".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("FontSize must be numeric"));
    }

    #[test]
    fn subtitle_accepts_documented_integer_font_size() {
        let _guard = setup_plot_tests();
        let handle = subtitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Integer font size".into()),
                Value::String("FontSize".into()),
                Value::Int(IntValue::U16(14)),
            ],
        ))
        .expect("integer FontSize");
        let fig = clone_figure(current_figure_handle()).expect("figure");
        let (_, axes, _) =
            crate::builtins::plotting::state::decode_plot_object_handle(handle).expect("handle");
        assert_eq!(
            fig.axes_metadata(axes)
                .expect("axes")
                .subtitle_style
                .font_size,
            Some(14.0)
        );
    }

    #[test]
    fn subtitle_typed_integer_axes_alias_is_explicitly_gated() {
        let _guard = setup_plot_tests();
        let axes_handle = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .expect("axes handle") as u64;
        let subtitle_before = clone_figure(current_figure_handle()).and_then(|figure| {
            figure
                .axes_metadata(0)
                .and_then(|metadata| metadata.subtitle.clone())
        });
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = subtitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Int(IntValue::U64(axes_handle)),
                Value::String("Blocked".into()),
            ],
        ))
        .expect_err("strict mode rejects typed integer axes aliases");
        assert_eq!(
            error.identifier(),
            SUBTITLE_INTEGER_AXES_EXTENSION.error_identifier
        );
        let subtitle_after = clone_figure(current_figure_handle()).and_then(|figure| {
            figure
                .axes_metadata(0)
                .and_then(|metadata| metadata.subtitle.clone())
        });
        assert_eq!(subtitle_after, subtitle_before);
        drop(strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = subtitle_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Int(IntValue::U64(axes_handle)),
                Value::String("Allowed".into()),
            ],
        ))
        .expect("extension mode accepts a typed integer axes alias");
        assert!(handle.is_finite());
    }
}

#[cfg(test)]
mod output_contract_tests {
    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = super::subtitle_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:subtitle:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = super::subtitle_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:subtitle:TooManyOutputs"));
    }
}
