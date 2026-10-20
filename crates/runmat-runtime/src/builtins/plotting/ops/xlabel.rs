use runmat_builtins::XLABEL_INTEGER_AXES_EXTENSION;
use runmat_builtins::{
    XLABEL_DESCRIPTOR, XLABEL_ERROR_INTERNAL, XLABEL_ERROR_INVALID_ARGUMENT,
    XLABEL_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Value;

use super::op_common::{annotation_error, map_figure_error, parse_numeric_text_command};
use super::state::set_xlabel_for_axes;

#[runtime_builtin(
    name = "xlabel",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::xlabel"
)]
pub fn xlabel_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &XLABEL_DESCRIPTOR,
        &XLABEL_ERROR_TOO_MANY_OUTPUTS,
        "xlabel",
    )?;
    if args.len() >= 2
        && args.len().is_multiple_of(2)
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XLABEL_INTEGER_AXES_EXTENSION,
            "xlabel",
        )?;
    }
    let command = parse_numeric_text_command("xlabel", args.view())
        .map_err(|error| annotation_error("xlabel", &XLABEL_ERROR_INVALID_ARGUMENT, error))?;
    let target = command.resolve_target();
    set_xlabel_for_axes(target.0, target.1, &command.text, command.style).map_err(|err| {
        annotation_error(
            "xlabel",
            &XLABEL_ERROR_INTERNAL,
            map_figure_error("xlabel", err),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::ylabel::ylabel_builtin;
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::Figure;
    use runmat_value::{CellArray, StringArray};
    use runmat_value::{IntValue, IntegerStorage};

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn xlabel_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = XLABEL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = xlabel(txt)"));
        assert!(labels.contains(&"h = xlabel(ax, txt)"));
        assert!(labels.contains(&"h = xlabel(txt, Name, Value, ...)"));
    }

    #[test]
    fn x_and_y_labels_update_active_axes_metadata() {
        let _guard = setup_plot_tests();
        let xh = xlabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Time".into())],
        ))
        .unwrap();
        let yh = ylabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Amplitude".into()),
                Value::String("Color".into()),
                Value::String("g".into()),
            ],
        ))
        .unwrap();

        assert_eq!(
            decode_plot_object_handle(xh).unwrap().2,
            PlotObjectKind::XLabel
        );
        assert_eq!(
            decode_plot_object_handle(yh).unwrap().2,
            PlotObjectKind::YLabel
        );

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.x_label.as_deref(), Some("Time"));
        assert_eq!(meta.y_label.as_deref(), Some("Amplitude"));
        assert!(meta.y_label_style.color.is_some());
    }

    #[test]
    fn xlabel_and_ylabel_reject_invalid_property_values() {
        let _guard = setup_plot_tests();

        let err = xlabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Time".into()),
                Value::String("Bogus".into()),
                Value::Num(1.0),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("unsupported property"));

        let err = ylabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Amp".into()),
                Value::String("Interpreter".into()),
                Value::Num(5.0),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("Interpreter must be a string"));
    }

    #[test]
    fn xlabel_formats_native_integer_scalars_exactly() {
        let _guard = setup_plot_tests();
        xlabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Tensor(
                    runmat_value::Tensor::new_integer(
                        IntegerStorage::U64(vec![u64::MAX]),
                        vec![1, 1],
                    )
                    .expect("scalar integer label"),
                ),
                Value::String("FontSize".into()),
                Value::Int(IntValue::U8(14)),
            ],
        ))
        .expect("integer label");
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).unwrap().x_label.as_deref(),
            Some("18446744073709551615")
        );
        assert_eq!(
            fig.axes_metadata(0).unwrap().x_label_style.font_size,
            Some(14.0)
        );
    }

    #[test]
    fn xlabel_typed_integer_axes_alias_is_gated() {
        let _guard = setup_plot_tests();
        let axes_handle = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .expect("axes handle") as u64;
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = xlabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Int(IntValue::U64(axes_handle)),
                Value::String("Blocked".into()),
            ],
        ))
        .expect_err("strict mode rejects typed integer axes aliases");
        assert_eq!(
            error.identifier(),
            XLABEL_INTEGER_AXES_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[test]
    fn xlabel_and_ylabel_support_explicit_axes_targets_and_multiline_text() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.set_subplot_grid(1, 2);
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 1,
        ));

        let xhandle = xlabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                ax.clone(),
                Value::StringArray(
                    StringArray::new(vec!["Time".into(), "(s)".into()], vec![1, 2]).unwrap(),
                ),
                Value::String("FontWeight".into()),
                Value::String("bold".into()),
            ],
        ))
        .unwrap();
        let yhandle = ylabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                ax,
                Value::Cell(
                    CellArray::new(
                        vec![
                            Value::String("Amplitude".into()),
                            Value::String("(V)".into()),
                        ],
                        1,
                        2,
                    )
                    .unwrap(),
                ),
                Value::String("FontAngle".into()),
                Value::String("italic".into()),
            ],
        ))
        .unwrap();

        assert_eq!(decode_plot_object_handle(xhandle).unwrap().1, 1);
        assert_eq!(decode_plot_object_handle(yhandle).unwrap().1, 1);

        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.x_label.as_deref(), Some("Time\n(s)"));
        assert_eq!(meta.y_label.as_deref(), Some("Amplitude\n(V)"));
        assert_eq!(meta.x_label_style.font_weight.as_deref(), Some("bold"));
        assert_eq!(meta.y_label_style.font_angle.as_deref(), Some("italic"));
    }
}

#[cfg(test)]
mod output_contract_tests {
    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = super::xlabel_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:xlabel:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = super::xlabel_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:xlabel:TooManyOutputs"));
    }
}
