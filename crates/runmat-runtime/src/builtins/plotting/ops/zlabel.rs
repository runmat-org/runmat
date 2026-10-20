use runmat_builtins::ZLABEL_INTEGER_AXES_EXTENSION;
use runmat_builtins::{
    ZLABEL_DESCRIPTOR, ZLABEL_ERROR_INTERNAL, ZLABEL_ERROR_INVALID_ARGUMENT,
    ZLABEL_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Value;

use super::op_common::{annotation_error, map_figure_error, parse_numeric_text_command};
use super::state::set_zlabel_for_axes;

#[runtime_builtin(
    name = "zlabel",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::zlabel"
)]
pub fn zlabel_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &ZLABEL_DESCRIPTOR,
        &ZLABEL_ERROR_TOO_MANY_OUTPUTS,
        "zlabel",
    )?;
    if args.len() >= 2
        && args.len().is_multiple_of(2)
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ZLABEL_INTEGER_AXES_EXTENSION,
            "zlabel",
        )?;
    }
    let command = parse_numeric_text_command("zlabel", args.view())
        .map_err(|error| annotation_error("zlabel", &ZLABEL_ERROR_INVALID_ARGUMENT, error))?;
    let target = command.resolve_target();
    set_zlabel_for_axes(target.0, target.1, &command.text, command.style).map_err(|err| {
        annotation_error(
            "zlabel",
            &ZLABEL_ERROR_INTERNAL,
            map_figure_error("zlabel", err),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_value::{CellArray, StringArray};

    #[test]
    fn zlabel_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ZLABEL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = zlabel(txt)"));
        assert!(labels.contains(&"h = zlabel(ax, txt)"));
        assert!(labels.contains(&"h = zlabel(txt, Name, Value, ...)"));
    }

    #[test]
    fn zlabel_sets_axes_local_metadata_and_returns_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = zlabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Height".into())],
        ))
        .unwrap();
        assert_eq!(
            decode_plot_object_handle(handle).unwrap().2,
            PlotObjectKind::ZLabel
        );
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.z_label.as_deref()),
            Some("Height")
        );
    }

    #[test]
    fn zlabel_supports_axes_target_multiline_and_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            1,
        ));

        let handle = zlabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                ax,
                Value::StringArray(
                    StringArray::new(vec!["Height".into(), "(m)".into()], vec![1, 2]).unwrap(),
                ),
                Value::String("FontWeight".into()),
                Value::String("bold".into()),
            ],
        ))
        .unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.z_label.as_deref(), Some("Height\n(m)"));
        assert_eq!(meta.z_label_style.font_weight.as_deref(), Some("bold"));

        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert!(matches!(string, Value::StringArray(_)));
        set_builtin(vec![
            Value::Num(handle),
            Value::String("String".into()),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("Depth".into()), Value::String("(km)".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(1).unwrap().z_label.as_deref(),
            Some("Depth\n(km)")
        );
    }
}

#[cfg(test)]
mod output_contract_tests {
    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = super::zlabel_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:zlabel:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = super::zlabel_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:zlabel:TooManyOutputs"));
    }
}
