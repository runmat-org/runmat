//! URL, download, and mail compatibility helpers.

#[path = "compat/sendmail_helpers.rs"]
mod sendmail_helpers;
#[path = "compat/websave_helpers.rs"]
mod websave_helpers;
use sendmail_helpers::*;
use websave_helpers::*;

use runmat_builtins::{SendmailFailure, SENDMAIL_DESCRIPTOR, SENDMAIL_ERROR_OUTPUTS};
use runmat_builtins::{
    UrldecodeFailure, UrlencodeFailure, URLDECODE_DESCRIPTOR, URLDECODE_ERROR_TOO_MANY_OUTPUTS,
    URLENCODE_DESCRIPTOR, URLENCODE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_builtins::{WEBSAVE_DESCRIPTOR, WEBSAVE_ERROR_OUTPUTS, WEBSAVE_EXPLICIT_GPU_EXTENSION};
use runmat_value::NumericScalar;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};
use url::Url;

use super::transport::{self, HttpMethod, HttpRequest};
use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
#[cfg(not(target_arch = "wasm32"))]
#[path = "compat/sendmail_process.rs"]
mod sendmail_process;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_TIMEOUT_SECONDS: f64 = 2147.483647;
const USER_AGENT: &str = "RunMat websave/0.0";

fn compat_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn compat_error_with_source(
    name: &str,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_source(source)
        .build()
}

async fn gather_args(
    name: &str,
    args: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, |index| async move {
        gather_if_needed_async(&args[index])
            .await
            .map_err(|err| compat_error(name, format!("{name}: {}", err.message())))
    })
    .await
}

fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::CharArray(array) if array.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                compat_error(
                    name,
                    format!("{name}: {arg} contains malformed UTF-16 text"),
                )
            })
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        _ => Err(compat_error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )),
    }
}

fn char_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[runtime_builtin(
    name = "urlencode",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn urlencode_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &URLENCODE_DESCRIPTOR,
        &URLENCODE_ERROR_TOO_MANY_OUTPUTS,
        "urlencode",
    )?;
    if args.len() != 1 {
        return Err(urlencode_error(UrlencodeFailure::WrongArity));
    }
    // Admit only host text before any provider or aggregate traversal.
    let text = url_text_input(&args[0], || urlencode_error(UrlencodeFailure::InvalidText))?
        .ok_or_else(|| urlencode_error(UrlencodeFailure::InvalidText))?;
    Ok(char_value(&percent_encode(&text)))
}

fn url_text_input(
    value: &Value,
    failure: impl FnOnce() -> RuntimeError,
) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::CharArray(array)
            if array.rows == 1 && array.shape.iter().skip(2).all(|extent| *extent == 1) =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), failure).map(Some)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        _ => Ok(None),
    }
}

fn urlencode_error(failure: UrlencodeFailure) -> RuntimeError {
    build_runtime_error(failure.descriptor().message)
        .with_builtin("urlencode")
        .with_identifier(failure.identifier())
        .build()
}

fn percent_encode(text: &str) -> String {
    let mut out = String::new();
    for byte in text.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            byte => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[runtime_builtin(
    name = "urldecode",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn urldecode_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &URLDECODE_DESCRIPTOR,
        &URLDECODE_ERROR_TOO_MANY_OUTPUTS,
        "urldecode",
    )?;
    if args.len() != 1 {
        return Err(urldecode_error(UrldecodeFailure::WrongArity));
    }
    let text = url_text_input(&args[0], || urldecode_error(UrldecodeFailure::InvalidText))?
        .ok_or_else(|| urldecode_error(UrldecodeFailure::InvalidText))?;
    Ok(char_value(&percent_decode(&text)?))
}

fn urldecode_error(failure: UrldecodeFailure) -> RuntimeError {
    build_runtime_error(failure.descriptor().message)
        .with_builtin("urldecode")
        .with_identifier(failure.identifier())
        .build()
}

fn percent_decode(text: &str) -> BuiltinResult<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] == b'%' {
            if idx + 2 >= bytes.len() {
                return Err(urldecode_error(UrldecodeFailure::IncompleteEscape));
            }
            let hex = std::str::from_utf8(&bytes[idx + 1..idx + 3])
                .map_err(|_| urldecode_error(UrldecodeFailure::InvalidEscape))?;
            let byte = u8::from_str_radix(hex, 16)
                .map_err(|_| urldecode_error(UrldecodeFailure::InvalidEscape))?;
            out.push(byte);
            idx += 3;
        } else if bytes[idx] == b'+' {
            out.push(b' ');
            idx += 1;
        } else {
            out.push(bytes[idx]);
            idx += 1;
        }
    }
    String::from_utf8(out).map_err(|_| urldecode_error(UrldecodeFailure::InvalidUtf8))
}

#[runtime_builtin(
    name = "websave",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn websave_builtin(
    context: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &WEBSAVE_DESCRIPTOR,
        &WEBSAVE_ERROR_OUTPUTS,
        "websave",
    )?;
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WEBSAVE_EXPLICIT_GPU_EXTENSION,
            "websave",
        )?;
    }
    if args.len() < 2 {
        return Err(compat_error(
            "websave",
            "websave: filename and URL are required",
        ));
    }
    let filename = path_from_value(&args[0], "websave")?;
    let url_text = scalar_text(&args[1], "websave", "url")?;
    let mut url = Url::parse(&url_text)
        .map_err(|err| compat_error("websave", format!("websave: invalid URL ({err})")))?;
    let prepared_api_key = match args.get(2) {
        Some(Value::Struct(source)) => Some(super::api_key::PreparedApiKey::parse(source)),
        _ => None,
    };
    preflight_websave_rest(args.view().range(2..)?, prepared_api_key.as_ref())?;
    let tail = args.range(2..)?;
    let rest = gather_args("websave", &tail, context.as_ref()).await?;
    let mut headers = Vec::new();
    let mut timeout = DEFAULT_TIMEOUT;
    let mut user_agent = USER_AGENT.to_string();
    parse_websave_rest(
        &mut url,
        &mut headers,
        &mut timeout,
        &mut user_agent,
        rest.view(),
        prepared_api_key.as_ref(),
    )?;
    let response = transport::send_request(
        HttpRequest {
            url,
            method: HttpMethod::Get,
            headers,
            body: None,
            timeout,
            user_agent,
        },
        context.as_ref(),
    )
    .await
    .map_err(|err| match err {
        transport::TransportCallError::Transport(err) => {
            compat_error_with_source("websave", err.message_with_prefix("websave"), err)
        }
        transport::TransportCallError::Runtime(err) => err,
    })?;
    vfs::write_async(&filename, response.body)
        .await
        .map_err(|err| compat_error("websave", format!("websave: {err}")))?;
    Ok(char_value(&path_to_string(&filename)))
}

#[runtime_builtin(
    name = "sendmail",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn sendmail_builtin(
    context: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &SENDMAIL_DESCRIPTOR,
        &SENDMAIL_ERROR_OUTPUTS,
        "sendmail",
    )?;
    if args.iter().any(|value| {
        crate::builtins::common::validation::value_contains_native_integer_class(value)
            || value_contains_resident(value)
    }) {
        return Err(sendmail_error(
            SendmailFailure::InvalidMailInput,
            "sendmail: mail arguments must be text, not numeric or resident values",
        ));
    }
    let (recipients, subject, body) = parse_sendmail_args(args.view())?;
    let message = format_mail_message(&recipients, &subject, &body);

    let outbox = crate::builtins::common::env::optional_var("RUNMAT_SENDMAIL_OUTBOX")?;
    if let Some(outbox) = outbox {
        let folder = PathBuf::from(outbox);
        vfs::create_dir_all_async(&folder).await.map_err(|err| {
            sendmail_error_with_source(SendmailFailure::Outbox, format!("sendmail: {err}"), err)
        })?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let path = folder.join(format!("runmat-mail-{stamp}.eml"));
        vfs::write_async(path, message.as_bytes())
            .await
            .map_err(|err| {
                sendmail_error_with_source(SendmailFailure::Outbox, format!("sendmail: {err}"), err)
            })?;
        return Ok(Value::Num(0.0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(command) = selected_sendmail_command()? {
        sendmail_process::run(
            context.as_ref(),
            command.executable.clone(),
            command.authority.as_ref(),
            &recipients,
            message,
        )
        .await?;
        return Ok(Value::Num(0.0));
    }
    #[cfg(target_arch = "wasm32")]
    if selected_sendmail_command().is_some() {
        return Err(sendmail_error(
            SendmailFailure::Command,
            "sendmail: local command transport is unavailable in WebAssembly",
        ));
    }

    let smtp_server = if let Some(server) = read_smtp_server(context).await? {
        Some(server)
    } else {
        crate::builtins::common::env::optional_var("RUNMAT_SMTP_SERVER")?
    };
    if let Some(server) = smtp_server.filter(|server| !server.trim().is_empty()) {
        return Err(sendmail_error(
            SendmailFailure::Smtp,
            format!(
                "sendmail: SMTP server '{server}' is configured, but direct SMTP transport is not available in this build; configure RUNMAT_SENDMAIL_COMMAND or RUNMAT_SENDMAIL_OUTBOX"
            ),
        ));
    }

    Err(sendmail_error(
        SendmailFailure::NoTransport,
        "sendmail: no mail transport configured; set RUNMAT_SENDMAIL_COMMAND, RUNMAT_SENDMAIL_OUTBOX, or configure a system sendmail binary",
    ))
}

#[cfg(test)]
#[path = "compat/tests.rs"]
pub(crate) mod tests;
