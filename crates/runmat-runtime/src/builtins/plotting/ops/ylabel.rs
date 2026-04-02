use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, parse_text_command};
use super::state::set_ylabel_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

#[runtime_builtin(
    name = "ylabel",
    category = "plotting",
    summary = "Set the current axes y-axis label.",
    keywords = "ylabel,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::ylabel"
)]
pub fn ylabel_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let command = parse_text_command("ylabel", &args)?;
    set_ylabel_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("ylabel", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn ylabel_rejects_invalid_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let err = ylabel_builtin(vec![
            Value::Num(crate::builtins::plotting::state::encode_axes_handle(
                crate::builtins::plotting::current_figure_handle(),
                42,
            )),
            Value::String("Amp".into()),
        ])
        .unwrap_err();
        assert!(err.message.contains("invalid axes") || err.message.contains("out of range"));
    }
}
