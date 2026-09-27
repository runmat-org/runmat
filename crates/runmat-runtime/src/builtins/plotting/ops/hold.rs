//! Per-axes hold policy through the shared graphics state.
use super::op_common::axes_target::{apply_axes_target, split_leading_axes_handle};
use super::op_common::cmd_parsing::parse_hold_mode;
use super::state::{set_hold, HoldMode};
use runmat_builtins::{HOLD_DESCRIPTOR, HOLD_ERROR_TOO_MANY_OUTPUTS};
use runmat_value::Value;
#[runmat_macros::runtime_builtin(
    name = "hold",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::hold"
)]
pub fn hold_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &HOLD_DESCRIPTOR,
        &HOLD_ERROR_TOO_MANY_OUTPUTS,
        "hold",
    )?;
    hold_state(args).map_err(|mut error| {
        if error.identifier.is_none() {
            error.identifier = Some("RunMat:hold:InvalidArgument".into());
        }
        error.context = error.context.with_builtin("hold");
        error
    })
}

fn hold_state(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (axes_target, rest) = split_leading_axes_handle(args, "hold")?;
    let mode = match rest.as_slice() {
        [] => HoldMode::Toggle,
        [value] => parse_hold_mode(value)?,
        _ => {
            return Err(crate::builtins::plotting::plotting_error(
                "hold",
                "hold: expected at most one input",
            ))
        }
    };
    apply_axes_target(axes_target, "hold")?;
    set_hold(mode);
    Ok(Value::OutputList(Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = HOLD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"hold()"));
        assert!(labels.contains(&"hold(mode)"));
        assert!(HOLD_DESCRIPTOR
            .signatures
            .iter()
            .all(|signature| signature.outputs.is_empty()));
    }

    #[test]
    fn hold_toggle_and_explicit_modes_work() {
        let _lock = crate::builtins::plotting::tests::lock_plot_registry();
        let _ = hold_builtin(Vec::new()).unwrap();
        assert_eq!(
            hold_builtin(vec![Value::String("on".into())]).unwrap(),
            Value::OutputList(Vec::new())
        );
        assert_eq!(
            hold_builtin(vec![Value::String("off".into())]).unwrap(),
            Value::OutputList(Vec::new())
        );
        assert!(hold_builtin(vec![Value::from("on"), Value::from("off")]).is_err());
    }

    #[test]
    fn hold_typed_integer_state_is_exact_in_compatibility_mode() {
        let _lock = crate::builtins::plotting::tests::lock_plot_registry();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(hold_builtin(vec![Value::Int(runmat_value::IntValue::U8(1))]).is_ok());
        assert!(hold_builtin(vec![Value::Int(runmat_value::IntValue::I64(0))]).is_ok());
    }

    #[test]
    fn hold_output_admission_precedes_state_creation_and_invalid_mode_preserves_state() {
        use crate::builtins::plotting::state;
        let _lock = crate::builtins::plotting::tests::lock_plot_registry();
        state::reset_plot_state();
        {
            let _outputs = crate::output_count::push_output_count(Some(1));
            assert_eq!(
                hold_builtin(vec![]).unwrap_err().identifier(),
                Some("RunMat:hold:TooManyOutputs")
            );
            assert!(state::current_figure_handle_if_exists().is_none());
        }
        let _outputs = crate::output_count::push_output_count(Some(0));
        hold_builtin(vec![Value::from("on")]).unwrap();
        assert!(state::current_hold_enabled());
        for value in [
            Value::from("toggle"),
            Value::Num(0.5),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        ] {
            assert_eq!(
                hold_builtin(vec![value]).unwrap_err().identifier(),
                Some("RunMat:hold:InvalidArgument")
            );
            assert!(state::current_hold_enabled());
        }
        hold_builtin(vec![Value::from("")]).unwrap();
        assert!(!state::current_hold_enabled());
        hold_builtin(vec![]).unwrap();
        assert!(state::current_hold_enabled());
    }

    #[test]
    fn hold_explicit_target_is_axes_local() {
        use crate::builtins::plotting::state;
        let _lock = crate::builtins::plotting::tests::lock_plot_registry();
        state::reset_plot_state();
        let first = state::current_axes_state();
        let (_, second) = state::create_axes_for_figure(Some(first.handle)).unwrap();
        hold_builtin(vec![
            Value::Num(state::encode_axes_handle(first.handle, first.active_index)),
            Value::from("all"),
        ])
        .unwrap();
        assert!(state::current_hold_enabled());
        state::select_axes_for_figure(first.handle, second).unwrap();
        assert!(!state::current_hold_enabled());
        hold_builtin(vec![Value::Bool(true)]).unwrap();
        state::select_axes_for_figure(first.handle, first.active_index).unwrap();
        assert!(state::current_hold_enabled());
    }
}
