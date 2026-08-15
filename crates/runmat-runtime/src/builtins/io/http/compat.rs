//! URL, download, and mail compatibility helpers.

use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerAuditDescriptor,
    BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinSignatureDescriptor, CharArray, Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use url::Url;

use super::transport::{self, HttpMethod, HttpRequest};
use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::io::repl_fs::compat::session_pref_text;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const USER_AGENT: &str = "RunMat websave/0.0";

const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "input",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input argument.",
}];
const INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "input1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input argument.",
    },
    BuiltinParamDescriptor {
        name: "input2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input argument.",
    },
];
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
simple_descriptor!(
    URLDECODE_SIGNATURES,
    URLDECODE_DESCRIPTOR,
    "decoded = urldecode(text)",
    &INPUTS_ONE
);
simple_descriptor!(
    WEBSAVE_SIGNATURES,
    WEBSAVE_DESCRIPTOR,
    "filename = websave(filename, url)",
    &INPUTS_TWO
);
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
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn urlencode_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("urlencode", &args).await?;
    if args.len() != 1 {
        return Err(compat_error(
            "urlencode",
            "urlencode: expected exactly one input",
        ));
    }
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
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn urldecode_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("urldecode", &args).await?;
    if args.len() != 1 {
        return Err(compat_error(
            "urldecode",
            "urldecode: expected exactly one input",
        ));
    }
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
    category = "io/http",
    summary = "Download URL content into a file.",
    keywords = "websave,download,url,file,http",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::string_type),
    descriptor(crate::builtins::io::http::compat::WEBSAVE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::http::compat"
)]
async fn websave_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("websave", &args).await?;
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
    let mut headers = Vec::new();
    let mut timeout = DEFAULT_TIMEOUT;
    let mut user_agent = USER_AGENT.to_string();
    parse_websave_rest(
        &mut url,
        &mut headers,
        &mut timeout,
        &mut user_agent,
        &args[2..],
    )?;
    let response = transport::send_request(&HttpRequest {
        url,
        method: HttpMethod::Get,
        headers,
        body: None,
        timeout,
        user_agent,
    })
    .map_err(|err| compat_error("websave", err.message_with_prefix("websave")))?;
    vfs::write_async(&filename, response.body)
        .await
        .map_err(|err| compat_error("websave", format!("websave: {err}")))?;
    Ok(char_value(&path_to_string(&filename)))
}

fn parse_websave_rest(
    url: &mut Url,
    headers: &mut Vec<(String, String)>,
    timeout: &mut Duration,
    user_agent: &mut String,
    args: &[Value],
) -> BuiltinResult<()> {
    let mut idx = 0usize;
    if let Some(Value::Struct(options)) = args.first() {
        apply_websave_options(options, headers, timeout, user_agent)?;
        idx = 1;
    }
    if !(args.len() - idx).is_multiple_of(2) {
        return Err(compat_error(
            "websave",
            "websave: name-value arguments must be paired",
        ));
    }
    while idx < args.len() {
        let name = scalar_text(&args[idx], "websave", "name")?;
        let value = scalar_text(&args[idx + 1], "websave", "value")?;
        match name.to_ascii_lowercase().as_str() {
            "timeout" => *timeout = Duration::from_secs_f64(parse_positive_seconds(&value)?),
            "useragent" => *user_agent = value,
            "headerfields" => {
                return Err(compat_error(
                    "websave",
                    "websave: HeaderFields must be supplied through weboptions",
                ));
            }
            _ => {
                url.query_pairs_mut().append_pair(&name, &value);
            }
        }
        idx += 2;
    }
    Ok(())
}

fn apply_websave_options(
    options: &runmat_builtins::StructValue,
    headers: &mut Vec<(String, String)>,
    timeout: &mut Duration,
    user_agent: &mut String,
) -> BuiltinResult<()> {
    if let Some(value) = options.fields.get("Timeout") {
        *timeout = Duration::from_secs_f64(numeric_seconds(value, "websave Timeout")?);
    }
    if let Some(value) = options.fields.get("UserAgent") {
        let ua = scalar_text(value, "websave", "UserAgent")?;
        if !ua.is_empty() {
            *user_agent = ua;
        }
    }
    if let Some(value) = options.fields.get("HeaderFields") {
        append_header_fields(value, headers)?;
    }
    Ok(())
}

fn numeric_seconds(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(v) if v.is_finite() && *v > 0.0 => Ok(*v),
        Value::Int(v) if v.to_i64() > 0 => Ok(v.to_f64()),
        Value::String(text) => parse_positive_seconds(text),
        Value::CharArray(array) if array.rows == 1 => {
            parse_positive_seconds(&array.data.iter().collect::<String>())
        }
        _ => Err(compat_error(
            "websave",
            format!("websave: {label} must be a positive finite scalar"),
        )),
    }
}

fn parse_positive_seconds(text: &str) -> BuiltinResult<f64> {
    let seconds: f64 = text.trim().parse().map_err(|_| {
        compat_error(
            "websave",
            "websave: Timeout must be a positive finite scalar",
        )
    })?;
    if seconds.is_finite() && seconds > 0.0 {
        Ok(seconds)
    } else {
        Err(compat_error(
            "websave",
            "websave: Timeout must be a positive finite scalar",
        ))
    }
}

fn append_header_fields(value: &Value, headers: &mut Vec<(String, String)>) -> BuiltinResult<()> {
    match value {
        Value::Struct(st) => {
            for (name, value) in &st.fields {
                let header_value = scalar_text(value, "websave", "HeaderFields value")?;
                if !name.is_empty() && !header_value.is_empty() {
                    headers.push((name.clone(), header_value));
                }
            }
            Ok(())
        }
        Value::Cell(cell) if cell.cols == 2 => {
            for row in 0..cell.rows {
                let name = scalar_text(&cell.data[row * cell.cols], "websave", "header name")?;
                let value =
                    scalar_text(&cell.data[row * cell.cols + 1], "websave", "header value")?;
                if !name.is_empty() && !value.is_empty() {
                    headers.push((name, value));
                }
            }
            Ok(())
        }
        Value::Cell(_) => Err(compat_error(
            "websave",
            "websave: HeaderFields cell array must have two columns",
        )),
        _ => Err(compat_error(
            "websave",
            "websave: HeaderFields must be a struct or two-column cell array",
        )),
    }
}

fn path_from_value(value: &Value, name: &str) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, name, "filename")?;
    Ok(PathBuf::from(
        expand_user_path(text.trim(), name).map_err(|err| compat_error(name, err))?,
    ))
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

fn parse_sendmail_args(args: &[Value]) -> BuiltinResult<(Vec<String>, String, String)> {
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

fn value_contains_resident(value: &Value) -> bool {
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

fn recipients_from_value(value: &Value) -> BuiltinResult<Vec<String>> {
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

fn format_mail_message(recipients: &[String], subject: &str, body: &str) -> String {
    let subject = subject.replace(['\r', '\n'], " ");
    format!(
        "To: {}\nSubject: {subject}\nContent-Type: text/plain; charset=utf-8\n\n{body}\n",
        recipients.join(", ")
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn try_sendmail_command(recipients: &[String], message: &str) -> BuiltinResult<bool> {
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
fn try_sendmail_command(_recipients: &[String], _message: &str) -> BuiltinResult<bool> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn url_round_trip_uses_percent_encoding() {
        let encoded = run(urlencode_builtin(vec![Value::String("a b/c".to_string())])).unwrap();
        assert_eq!(encoded, char_value("a%20b%2Fc"));
        let decoded = run(urldecode_builtin(vec![char_value("a%20b%2Fc")])).unwrap();
        assert_eq!(decoded, char_value("a b/c"));
    }

    #[test]
    fn websave_rest_applies_options_and_query_pairs() {
        let mut url = Url::parse("https://example.test/data").unwrap();
        let mut headers = Vec::new();
        let mut timeout = DEFAULT_TIMEOUT;
        let mut user_agent = USER_AGENT.to_string();
        let mut options = runmat_builtins::StructValue::new();
        options.insert("Timeout", Value::Num(7.0));
        options.insert("UserAgent", Value::String("agent".to_string()));
        let mut header_struct = runmat_builtins::StructValue::new();
        header_struct.insert("XRunMat", Value::String("yes".to_string()));
        options.insert("HeaderFields", Value::Struct(header_struct));

        parse_websave_rest(
            &mut url,
            &mut headers,
            &mut timeout,
            &mut user_agent,
            &[
                Value::Struct(options),
                Value::String("q".to_string()),
                Value::String("hello world".to_string()),
            ],
        )
        .unwrap();

        assert_eq!(url.query(), Some("q=hello+world"));
        assert_eq!(headers, vec![("XRunMat".to_string(), "yes".to_string())]);
        assert_eq!(timeout, Duration::from_secs(7));
        assert_eq!(user_agent, "agent");
    }

    #[test]
    fn sendmail_rejects_integer_payloads_before_transport() {
        let error = run(sendmail_builtin(vec![
            Value::String("recipient@example.test".to_string()),
            Value::String("subject".to_string()),
            Value::Int(runmat_builtins::IntValue::U64(9_007_199_254_740_993)),
        ]))
        .expect_err("integer message must reject before transport selection");
        assert!(error.message().contains("must be text"));

        let nested = runmat_builtins::CellArray::new(
            vec![Value::Int(runmat_builtins::IntValue::I8(1))],
            1,
            1,
        )
        .unwrap();
        let error = run(sendmail_builtin(vec![
            Value::Cell(nested),
            Value::String("subject".to_string()),
            Value::String("message".to_string()),
        ]))
        .expect_err("nested integer recipient must reject before transport selection");
        assert!(error.message().contains("must be text"));
    }

    #[test]
    fn sendmail_two_argument_form_uses_an_empty_message() {
        let (recipients, subject, message) = parse_sendmail_args(&[
            Value::String("recipient@example.test".to_string()),
            Value::String("subject".to_string()),
        ])
        .expect("two-argument sendmail form");
        assert_eq!(recipients, ["recipient@example.test"]);
        assert_eq!(subject, "subject");
        assert!(message.is_empty());
    }
}
