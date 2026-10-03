use super::*;
pub(super) fn parse_sendmail_args(args: &[Value]) -> BuiltinResult<(Vec<String>, String, String)> {
    if args.len() < 2 || args.len() > 4 {
        return Err(compat_error(
            "sendmail",
            "sendmail: recipient, subject, optional message, and optional attachments are expected",
        ));
    }
    let recipients = recipients_from_value(&args[0])?;
    let subject = scalar_text(&args[1], "sendmail", "subject")?;
    let body = args
        .get(2)
        .map(|value| scalar_text(value, "sendmail", "message"))
        .transpose()?
        .unwrap_or_default();
    if recipients.is_empty() {
        return Err(compat_error(
            "sendmail",
            "sendmail: recipient must not be empty",
        ));
    }
    if args.len() == 4 {
        return Err(compat_error(
            "sendmail",
            "sendmail: attachments are not supported by the current mail transport",
        ));
    }
    Ok((recipients, subject, body))
}

pub(super) fn value_contains_resident(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(value) => value.data.iter().any(value_contains_resident),
        Value::Struct(value) => value.fields.values().any(value_contains_resident),
        Value::Object(value) => value.properties.values().any(value_contains_resident),
        Value::Closure(value) => value.captures.iter().any(value_contains_resident),
        Value::OutputList(values) => values.iter().any(value_contains_resident),
        _ => false,
    }
}

pub(super) fn recipients_from_value(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::StringArray(array) => Ok(array
            .data
            .iter()
            .filter(|value| !value.trim().is_empty())
            .cloned()
            .collect()),
        Value::Cell(cell) => {
            let mut out = Vec::new();
            for value in &cell.data {
                let recipient = scalar_text(value, "sendmail", "recipient")?;
                if !recipient.trim().is_empty() {
                    out.push(recipient);
                }
            }
            Ok(out)
        }
        value => {
            let recipient = scalar_text(value, "sendmail", "recipient")?;
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
pub(super) fn try_sendmail_command(recipients: &[String], message: &str) -> BuiltinResult<bool> {
    let configured = std::env::var("RUNMAT_SENDMAIL_COMMAND").ok();
    let command = configured
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            let path = PathBuf::from("/usr/sbin/sendmail");
            if path.exists() {
                Some(path)
            } else {
                None
            }
        });
    let Some(command) = command else {
        return Ok(false);
    };
    let mut child = Command::new(&command)
        .arg("-t")
        .arg("-oi")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            compat_error(
                "sendmail",
                format!("sendmail: unable to spawn {} ({err})", command.display()),
            )
        })?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| compat_error("sendmail", "sendmail: unable to open mail stdin"))?;
        use std::io::Write;
        stdin
            .write_all(message.as_bytes())
            .map_err(|err| compat_error("sendmail", format!("sendmail: {err}")))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|err| compat_error("sendmail", format!("sendmail: {err}")))?;
    if output.status.success() {
        return Ok(true);
    }
    let detail = String::from_utf8_lossy(&output.stderr);
    Err(compat_error(
        "sendmail",
        format!(
            "sendmail: {} failed for {} ({})",
            command.display(),
            recipients.join(", "),
            detail.trim()
        ),
    ))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn try_sendmail_command(_recipients: &[String], _message: &str) -> BuiltinResult<bool> {
    Ok(false)
}
