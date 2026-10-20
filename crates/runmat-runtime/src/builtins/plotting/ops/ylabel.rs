use runmat_builtins::YLABEL_INTEGER_AXES_EXTENSION;
use runmat_builtins::{
    YLABEL_DESCRIPTOR, YLABEL_ERROR_INTERNAL, YLABEL_ERROR_INVALID_ARGUMENT,
    YLABEL_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Value;

use super::op_common::{annotation_error, map_figure_error, parse_numeric_text_command};
use super::state::set_ylabel_for_axes;

#[runtime_builtin(
    name = "ylabel",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::ylabel"
)]
pub fn ylabel_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &YLABEL_DESCRIPTOR,
        &YLABEL_ERROR_TOO_MANY_OUTPUTS,
        "ylabel",
    )?;
    if args.len() >= 2
        && args.len().is_multiple_of(2)
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &YLABEL_INTEGER_AXES_EXTENSION,
            "ylabel",
        )?;
    }
    let command = parse_numeric_text_command("ylabel", args.view())
        .map_err(|error| annotation_error("ylabel", &YLABEL_ERROR_INVALID_ARGUMENT, error))?;
    let target = command.resolve_target();
    set_ylabel_for_axes(target.0, target.1, &command.text, command.style).map_err(|err| {
        annotation_error(
            "ylabel",
            &YLABEL_ERROR_INTERNAL,
            map_figure_error("ylabel", err),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn ylabel_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = YLABEL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = ylabel(txt)"));
        assert!(labels.contains(&"h = ylabel(ax, txt)"));
        assert!(labels.contains(&"h = ylabel(txt, Name, Value, ...)"));
    }

    #[test]
    fn ylabel_rejects_invalid_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let err = ylabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Num(crate::builtins::plotting::state::encode_axes_handle(
                    crate::builtins::plotting::current_figure_handle(),
                    42,
                )),
                Value::String("Amp".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("invalid axes") || err.message.contains("out of range"));
    }
}

#[cfg(test)]
mod output_contract_tests {
    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = super::ylabel_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:ylabel:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = super::ylabel_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )
        .expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:ylabel:TooManyOutputs"));
    }
}
