//! URL, download, and mail compatibility helpers.

#[path = "compat/websave_helpers.rs"]
mod websave_helpers;
#[path = "compat/sendmail_helpers.rs"]
mod sendmail_helpers;
use sendmail_helpers::*;
use websave_helpers::*;

use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};
use runmat_builtins::{WEBSAVE_DESCRIPTOR, WEBSAVE_ERROR_OUTPUTS, WEBSAVE_EXPLICIT_GPU_EXTENSION};
use runmat_value::NumericScalar;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};
use url::Url;

use super::transport::{self, HttpMethod, HttpRequest};
use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::io::repl_fs::compat::session_pref_text;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_TIMEOUT_SECONDS: f64 = 2147.483647;
const USER_AGENT: &str = "RunMat websave/0.0";

const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "input",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input argument.",
}];
const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];

macro_rules! simple_descriptor {
    ($sig:ident, $desc:ident, $label:expr, $inputs:expr) => {
        const $sig: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: &OUTPUT_VALUE,
        }];
        pub const $desc: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$sig,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[],
        };
    };
}

simple_descriptor!(
    URLENCODE_SIGNATURES,
    URLENCODE_DESCRIPTOR,
    "encoded = urlencode(text)",
    &INPUTS_ONE
);
pub const URLENCODE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "urlencode accepts host text only; integer and resident numeric values reject before provider access or encoding.",
    };
simple_descriptor!(
    URLDECODE_SIGNATURES,
    URLDECODE_DESCRIPTOR,
    "decoded = urldecode(text)",
    &INPUTS_ONE
);
pub const URLDECODE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "urldecode accepts host text only; integer and resident numeric values reject before provider access or decoding.",
    };
const SENDMAIL_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "to",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Recipient address or collection of addresses.",
    },
    BuiltinParamDescriptor {
        name: "subject",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Message subject text.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some(""),
        description: "Message body text.",
    },
    BuiltinParamDescriptor {
        name: "attachments",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional attachment path or collection of paths.",
    },
];
const SENDMAIL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = sendmail(to, subject, message, attachments)",
    inputs: &SENDMAIL_INPUTS,
    outputs: &OUTPUT_VALUE,
}];
pub const SENDMAIL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SENDMAIL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub const SENDMAIL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "sendmail accepts textual recipients, subject, message, and attachment paths. Numeric character codes may be embedded in an already constructed character vector, but direct integer scalar, array, nested, or resident numeric arguments are not mail inputs and reject before provider access or transport side effects.",
};

fn compat_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

async fn gather_args(name: &str, args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(|err| compat_error(name, format!("{name}: {}", err.message())))?,
        );
    }
    Ok(out)
}

fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
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
    category = "io/http",
    summary = "Percent-encode text for use in URLs.",
    keywords = "urlencode,url,percent encode,web",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::http::compat::URLENCODE_DESCRIPTOR),
    integer_audit(crate::builtins::io::http::compat::URLENCODE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn urlencode_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 1 {
        return Err(compat_error(
            "urlencode",
            "urlencode: expected exactly one input",
        ));
    }
    if crate::builtins::strings::common::contains_numeric_or_resident_text_input(&args[0]) {
        return Err(compat_error(
            "urlencode",
            "urlencode: text must be a host string scalar or character vector",
        ));
    }
    let args = gather_args("urlencode", &args).await?;
    Ok(char_value(&percent_encode(&scalar_text(
        &args[0],
        "urlencode",
        "text",
    )?)))
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
    category = "io/http",
    summary = "Decode percent-encoded URL text.",
    keywords = "urldecode,url,percent decode,web",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::http::compat::URLDECODE_DESCRIPTOR),
    integer_audit(crate::builtins::io::http::compat::URLDECODE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn urldecode_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 1 {
        return Err(compat_error(
            "urldecode",
            "urldecode: expected exactly one input",
        ));
    }
    if crate::builtins::strings::common::contains_numeric_or_resident_text_input(&args[0]) {
        return Err(compat_error(
            "urldecode",
            "urldecode: text must be a host string scalar or character vector",
        ));
    }
    let args = gather_args("urldecode", &args).await?;
    Ok(char_value(&percent_decode(&scalar_text(
        &args[0],
        "urldecode",
        "text",
    )?)?))
}

fn percent_decode(text: &str) -> BuiltinResult<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] == b'%' {
            if idx + 2 >= bytes.len() {
                return Err(compat_error("urldecode", "urldecode: incomplete escape"));
            }
            let hex = std::str::from_utf8(&bytes[idx + 1..idx + 3])
                .map_err(|_| compat_error("urldecode", "urldecode: invalid escape"))?;
            let byte = u8::from_str_radix(hex, 16)
                .map_err(|_| compat_error("urldecode", "urldecode: invalid escape"))?;
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
    String::from_utf8(out).map_err(|err| compat_error("urldecode", err.to_string()))
}

#[runtime_builtin(
    name = "websave",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn websave_builtin(
    context: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
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
    preflight_websave_rest(&args[2..])?;
    let rest = gather_args("websave", &args[2..]).await?;
    let mut headers = Vec::new();
    let mut timeout = DEFAULT_TIMEOUT;
    let mut user_agent = USER_AGENT.to_string();
    parse_websave_rest(&mut url, &mut headers, &mut timeout, &mut user_agent, &rest)?;
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
            compat_error("websave", err.message_with_prefix("websave"))
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
    category = "io/http",
    summary = "Create an email message using MATLAB sendmail-compatible arguments.",
    keywords = "sendmail,email,mail,notification",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::http::compat::SENDMAIL_DESCRIPTOR),
    integer_audit(crate::builtins::io::http::compat::SENDMAIL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn sendmail_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.iter().any(|value| {
        crate::builtins::common::validation::value_contains_native_integer_class(value)
            || value_contains_resident(value)
    }) {
        return Err(compat_error(
            "sendmail",
            "sendmail: mail arguments must be text, not numeric or resident values",
        ));
    }
    let args = gather_args("sendmail", &args).await?;
    let (recipients, subject, body) = parse_sendmail_args(&args)?;
    let message = format_mail_message(&recipients, &subject, &body);

    let outbox = std::env::var("RUNMAT_SENDMAIL_OUTBOX").ok();
    if let Some(outbox) = outbox {
        let folder = PathBuf::from(outbox);
        vfs::create_dir_all_async(&folder)
            .await
            .map_err(|err| compat_error("sendmail", format!("sendmail: {err}")))?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let path = folder.join(format!("runmat-mail-{stamp}.eml"));
        vfs::write_async(path, message.as_bytes())
            .await
            .map_err(|err| compat_error("sendmail", format!("sendmail: {err}")))?;
        return Ok(Value::Num(0.0));
    }

    if try_sendmail_command(&recipients, &message)? {
        return Ok(Value::Num(0.0));
    }

    let smtp_server = session_pref_text("Internet", "SMTP_Server")
        .or_else(|| session_pref_text("Internet", "SMTPServer"))
        .or_else(|| std::env::var("RUNMAT_SMTP_SERVER").ok());
    if let Some(server) = smtp_server.filter(|server| !server.trim().is_empty()) {
        return Err(compat_error(
            "sendmail",
            format!(
                "sendmail: SMTP server '{server}' is configured, but direct SMTP transport is not available in this build; configure RUNMAT_SENDMAIL_COMMAND or RUNMAT_SENDMAIL_OUTBOX"
            ),
        ));
    }

    Err(compat_error(
        "sendmail",
        "sendmail: no mail transport configured; set RUNMAT_SENDMAIL_COMMAND, RUNMAT_SENDMAIL_OUTBOX, or configure a system sendmail binary",
    ))
}

#[cfg(test)]
#[path = "compat/tests.rs"]
pub(crate) mod tests;
