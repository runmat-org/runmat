use super::*;
use crate::builtins::common::validation::value_text_scalar_string;
use crate::context::{
    PreferenceServiceError, PreferenceValues, RuntimeCapability, RuntimeCapabilityError,
    RuntimeContext,
};

pub(super) fn sendmail_error(kind: SendmailFailure, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("sendmail")
        .with_identifier(kind.identifier())
        .build()
}

pub(super) fn sendmail_error_with_source(
    kind: SendmailFailure,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("sendmail")
        .with_identifier(kind.identifier())
        .with_source(source)
        .build()
}

pub(super) async fn read_smtp_server(
    runtime: Option<RuntimeContext>,
) -> BuiltinResult<Option<String>> {
    let runtime = runtime.ok_or_else(|| {
        RuntimeCapabilityError::new(RuntimeCapability::Preferences, "read mail preferences")
            .into_runtime_error()
    })?;
    let service = runtime
        .service_ports()
        .require_preferences("read mail preferences")
        .map(Clone::clone)
        .map_err(RuntimeCapabilityError::into_runtime_error)?;
    let mut cooperation = runtime.cpu_cooperation();
    cooperation
        .checkpoint(0)
        .await
        .map_err(|source| source.into_runtime_error())?;
    let values = cooperation
        .await_cancellable(service.group_values(runtime, "Internet".to_string()))
        .await
        .map_err(|source| source.into_runtime_error())?
        .map_err(|source| match source {
            PreferenceServiceError::Execution(source) => source.into_runtime_error(),
            source => sendmail_error_with_source(
                SendmailFailure::Preferences,
                format!("sendmail: unable to read Internet preferences ({source})"),
                source,
            ),
        })?;
    cooperation
        .checkpoint(0)
        .await
        .map_err(|source| source.into_runtime_error())?;
    Ok(smtp_server(values.as_ref()))
}

fn smtp_server(values: Option<&PreferenceValues>) -> Option<String> {
    let values = values?;
    ["SMTP_Server", "SMTPServer"]
        .into_iter()
        .filter_map(|name| values.get(name))
        .find_map(preference_text)
}

fn preference_text(value: &Value) -> Option<String> {
    let mut text = value_text_scalar_string(value)?;
    if matches!(value, Value::CharArray(_)) {
        while text.ends_with(' ') {
            text.pop();
        }
    }
    Some(text)
}

fn mail_text(value: &Value, arg: &str) -> BuiltinResult<String> {
    value_text_scalar_string(value).ok_or_else(|| {
        sendmail_error(
            SendmailFailure::InvalidText,
            format!("sendmail: {arg} must be a string scalar or character vector"),
        )
    })
}

pub(super) fn parse_sendmail_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(Vec<String>, String, String)> {
    if args.len() < 2 || args.len() > 4 {
        return Err(sendmail_error(
            SendmailFailure::Arity,
            "sendmail: recipient, subject, optional message, and optional attachments are expected",
        ));
    }
    let recipients = recipients_from_value(&args[0])?;
    let subject = mail_text(&args[1], "subject")?;
    let body = args
        .get(2)
        .map(|value| mail_text(value, "message"))
        .transpose()?
        .unwrap_or_default();
    if recipients.is_empty() {
        return Err(sendmail_error(
            SendmailFailure::EmptyRecipient,
            "sendmail: recipient must not be empty",
        ));
    }
    if args.len() == 4 {
        return Err(sendmail_error(
            SendmailFailure::Attachments,
            "sendmail: attachments are not supported by the current mail transport",
        ));
    }
    Ok((recipients, subject, body))
}

pub(super) fn value_contains_resident(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(value) => value.data().iter().any(value_contains_resident),
        Value::Struct(value) => value.fields.values().any(value_contains_resident),
        Value::Object(value) => value.properties().values().any(value_contains_resident),
        Value::Closure(value) => value.captures.iter().any(value_contains_resident),
        Value::OutputList(values) => values.iter().any(value_contains_resident),
        _ => false,
    }
}

pub(super) fn recipients_from_value(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => {
            Ok(array
                .data()
                .iter()
                .filter(|value| {
                    !value
                        .as_str()
                        .expect("checked nonmissing string")
                        .trim()
                        .is_empty()
                })
                .map(|value| {
                    value
                        .as_str()
                        .expect("checked nonmissing string")
                        .to_owned()
                })
                .collect())
        }
        Value::Cell(cell) => {
            let mut out = Vec::new();
            for value in cell.data() {
                let recipient = mail_text(value, "recipient")?;
                if !recipient.trim().is_empty() {
                    out.push(recipient);
                }
            }
            Ok(out)
        }
        value => {
            let recipient = mail_text(value, "recipient")?;
            if recipient.trim().is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![recipient])
            }
        }
    }
}

pub(super) fn format_mail_message(recipients: &[String], subject: &str, body: &str) -> String {
    let subject = subject.replace(['\r', '\n'], " ");
    format!(
        "To: {}\nSubject: {subject}\nContent-Type: text/plain; charset=utf-8\n\n{body}\n",
        recipients.join(", ")
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct SelectedSendmailCommand {
    pub executable: PathBuf,
    pub authority: Option<runmat_filesystem::CapturedNativePath>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn selected_sendmail_command() -> BuiltinResult<Option<SelectedSendmailCommand>> {
    let configured = crate::builtins::common::env::optional_var("RUNMAT_SENDMAIL_COMMAND")?;
    if let Some(executable) = configured
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
    {
        return Ok(Some(SelectedSendmailCommand {
            executable,
            authority: None,
        }));
    }
    default_sendmail_command()
}

#[cfg(not(target_arch = "wasm32"))]
fn default_sendmail_command() -> BuiltinResult<Option<SelectedSendmailCommand>> {
    let authority = runmat_filesystem::capture_native_path(
        "/usr/sbin/sendmail",
        runmat_filesystem::NativePathIntent::Read,
    )
    .map_err(|source| {
        sendmail_error_with_source(
            SendmailFailure::Command,
            format!("sendmail: cannot inspect local command: {source}"),
            source,
        )
    })?;
    let Some(authority) = authority else {
        return Ok(None);
    };
    authority.check_current().map_err(|source| {
        sendmail_error_with_source(
            SendmailFailure::Command,
            format!("sendmail: local command authority revoked: {source}"),
            source,
        )
    })?;
    let metadata = std::fs::metadata(authority.path());
    authority.check_current().map_err(|source| {
        sendmail_error_with_source(
            SendmailFailure::Command,
            format!("sendmail: local command authority revoked: {source}"),
            source,
        )
    })?;
    match metadata {
        Ok(_) => {}
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(sendmail_error_with_source(
                SendmailFailure::Command,
                format!("sendmail: cannot inspect local command: {source}"),
                source,
            ));
        }
    }
    Ok(Some(SelectedSendmailCommand {
        executable: authority.path().to_path_buf(),
        authority: Some(authority),
    }))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod command_selection_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn virtual_provider_does_not_probe_ambient_default_executable() {
        let _serial = runmat_filesystem::provider_override_lock();
        let _virtual = runmat_filesystem::replace_provider(Arc::new(
            runmat_filesystem::MemoryFsProvider::with_current_dir("/virtual"),
        ));
        assert!(default_sendmail_command()
            .expect("unsupported native path is not an ambient fallback")
            .is_none());
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) fn selected_sendmail_command() -> Option<PathBuf> {
    crate::builtins::common::env::var("RUNMAT_SENDMAIL_COMMAND")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
}
