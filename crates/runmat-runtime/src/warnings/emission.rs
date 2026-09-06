use crate::console::{record_console_line, ConsoleStream};
use crate::warning_store::{self, RuntimeWarning};
use crate::{build_runtime_error, RuntimeError};

use super::policy::{with_policy, WarningAction};

pub(crate) struct WarningRequest<'a> {
    pub(crate) builtin: &'a str,
    pub(crate) identifier: &'a str,
    pub(crate) message: &'a str,
    pub(crate) show_identifier: bool,
}

pub(crate) fn emit(request: WarningRequest<'_>) -> Result<(), RuntimeError> {
    let warning = RuntimeWarning {
        identifier: request.identifier.to_string(),
        message: request.message.to_string(),
    };
    let action = with_policy(|policy| {
        let action = policy.action_for(request.identifier);
        if !matches!(action, WarningAction::Suppress) {
            policy.record_last(warning.clone());
        }
        action
    });

    match action {
        WarningAction::Suppress => Ok(()),
        WarningAction::Display => {
            display(&warning, request.show_identifier);
            warning_store::push(&warning.identifier, &warning.message);
            notify_host(warning);
            Ok(())
        }
        WarningAction::AsError => {
            warning_store::push(&warning.identifier, &warning.message);
            notify_host(warning.clone());
            Err(build_runtime_error(warning.message)
                .with_builtin(request.builtin)
                .with_identifier(warning.identifier)
                .build())
        }
    }
}

fn display(warning: &RuntimeWarning, show_identifier: bool) {
    let (backtrace, verbose) =
        with_policy(|policy| (policy.backtrace_enabled(), policy.verbose_enabled()));
    emit_stderr_line(format!("Warning: {}", warning.message));
    if show_identifier {
        emit_stderr_line(format!("identifier: {}", warning.identifier));
    }
    if verbose {
        emit_stderr_line(format!(
            "(Type \"warning('off','{}')\" to suppress this warning.)",
            warning.identifier
        ));
    }
    if backtrace {
        emit_stderr_line(format!("{}", std::backtrace::Backtrace::force_capture()));
    }
}

fn notify_host(warning: RuntimeWarning) {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(host) = context.service_ports().host() {
            host.warning(warning);
        }
    }
}

fn emit_stderr_line(line: String) {
    tracing::warn!("{line}");
    record_console_line(ConsoleStream::Stderr, line);
}
