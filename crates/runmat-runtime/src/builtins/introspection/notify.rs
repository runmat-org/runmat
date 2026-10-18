use runmat_macros::runtime_builtin;
use runmat_value::Value;

/// Callback dispatch is a non-replayable side effect, including with no output.
#[runtime_builtin(
    name = "notify",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::notify"
)]
pub async fn notify_builtin(
    target: Value,
    event_name: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::NOTIFY_DESCRIPTOR,
        &runmat_builtins::NOTIFY_ERROR_OUTPUTS,
        "notify",
    )?;
    if crate::output_count::current_output_count() == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::NOTIFY_STATUS_EXTENSION,
            "notify",
        )?;
    }
    if rest.len() > 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::NOTIFY_EXTRA_ARGS_EXTENSION,
            "notify",
        )?;
    }
    crate::notify_builtin(target, event_name, rest).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn notify_is_integer_inapplicable() {
        assert_eq!(
            runmat_builtins::NOTIFY_INTEGER_AUDIT.kind,
            runmat_builtins::BuiltinIntegerAuditKind::NotApplicable
        );
    }

    #[test]
    fn strict_profile_rejects_status_and_multiple_callback_arguments_before_dispatch() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        {
            let _outputs =
                crate::output_count::push_output_count(Some(1)).expect("output-count scope");
            let error = block_on(notify_builtin(
                Value::Num(1.0),
                "Changed".into(),
                Vec::new(),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                runmat_builtins::NOTIFY_STATUS_EXTENSION.error_identifier
            );
        }
        let error = block_on(notify_builtin(
            Value::Num(1.0),
            "Changed".into(),
            vec![Value::Num(1.0), Value::Num(2.0)],
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::NOTIFY_EXTRA_ARGS_EXTENSION.error_identifier
        );
    }
}
