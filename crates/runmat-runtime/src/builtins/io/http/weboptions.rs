//! MATLAB-compatible `weboptions` builtin for constructing HTTP client options.

use std::collections::VecDeque;

use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;

#[allow(clippy::too_many_lines)]
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::http::weboptions")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "weboptions",
    op_kind: GpuOpKind::Custom("http-options"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "weboptions validates CPU metadata only; gpuArray inputs are gathered eagerly.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::http::weboptions")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "weboptions",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "weboptions constructs option structs and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "weboptions",
    category = "io/http",
    summary = "Create an options struct that configures webread and webwrite HTTP behaviour.",
    keywords = "weboptions,http options,timeout,headers,rest client",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::weboptions_type),
    builtin_path = "crate::builtins::io::http::weboptions"
)]
async fn weboptions_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let mut gathered = Vec::with_capacity(rest.len());
    for value in rest {
        gathered.push(gather_if_needed_async(&value).await.map_err(|flow| {
            remap_weboptions_flow(flow, |err| format!("weboptions: {}", err.message()))
        })?);
    }
    let mut queue: VecDeque<Value> = gathered.into();
    let mut options = default_options_struct();

    if matches!(queue.front(), Some(Value::Struct(_))) {
        if let Some(Value::Struct(struct_value)) = queue.pop_front() {
            apply_struct_fields(struct_value, &mut options)?;
        }
    }

    while let Some(name_value) = queue.pop_front() {
        let name = expect_string_scalar(
            &name_value,
            "weboptions: option names must be character vectors or string scalars",
        )?;
        let value = queue
            .pop_front()
            .ok_or_else(|| weboptions_error("weboptions: missing value for name-value argument"))?;
        set_option_field(&mut options, &name, &value)?;
    }

    validate_credentials(&options)?;

    Ok(Value::Struct(options))
}

fn weboptions_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("weboptions")
        .build()
}

fn remap_weboptions_flow<F>(err: RuntimeError, message: F) -> RuntimeError
where
    F: FnOnce(&RuntimeError) -> String,
{
    build_runtime_error(message(&err))
        .with_builtin("weboptions")
        .with_source(err)
        .build()
}

fn default_options_struct() -> StructValue {
    let mut out = StructValue::new();
    out.fields
        .insert("ContentType".to_string(), Value::from("auto"));
    out.fields
        .insert("Timeout".to_string(), Value::Num(DEFAULT_TIMEOUT_SECONDS));
    out.fields.insert(
        "HeaderFields".to_string(),
        Value::Struct(StructValue::new()),
    );
    out.fields.insert("UserAgent".to_string(), Value::from(""));
    out.fields.insert("Username".to_string(), Value::from(""));
    out.fields.insert("Password".to_string(), Value::from(""));
    out.fields
        .insert("RequestMethod".to_string(), Value::from("auto"));
    out.fields
        .insert("MediaType".to_string(), Value::from("auto"));
    out.fields.insert(
        "QueryParameters".to_string(),
        Value::Struct(StructValue::new()),
    );
    out
}

fn apply_struct_fields(source: StructValue, target: &mut StructValue) -> BuiltinResult<()> {
    for (key, value) in &source.fields {
        set_option_field(target, key, value)?;
    }
    Ok(())
}

fn set_option_field(options: &mut StructValue, name: &str, value: &Value) -> BuiltinResult<()> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "contenttype" => {
            let canonical = parse_content_type_option(value)?;
            options
                .fields
                .insert("ContentType".to_string(), Value::from(canonical));
            Ok(())
        }
        "timeout" => {
            let seconds = numeric_scalar(
                value,
                "weboptions: Timeout must be a finite, positive scalar",
            )?;
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(weboptions_error(
                    "weboptions: Timeout must be a finite, positive scalar",
                ));
            }
            options
                .fields
                .insert("Timeout".to_string(), Value::Num(seconds));
            Ok(())
        }
        "headerfields" => {
            let canonical = canonical_header_fields(value)?;
            options.fields.insert("HeaderFields".to_string(), canonical);
            Ok(())
        }
        "useragent" => {
            let ua = expect_string_scalar(
                value,
                "weboptions: UserAgent must be a character vector or string scalar",
            )?;
            options
                .fields
                .insert("UserAgent".to_string(), Value::from(ua));
            Ok(())
        }
        "username" => {
            let username = expect_string_scalar(
                value,
                "weboptions: Username must be a character vector or string scalar",
            )?;
            options
                .fields
                .insert("Username".to_string(), Value::from(username));
            Ok(())
        }
        "password" => {
            let password = expect_string_scalar(
                value,
                "weboptions: Password must be a character vector or string scalar",
            )?;
            options
                .fields
                .insert("Password".to_string(), Value::from(password));
            Ok(())
        }
        "requestmethod" => {
            let method = parse_request_method_option(value)?;
            options
                .fields
                .insert("RequestMethod".to_string(), Value::from(method));
            Ok(())
        }
        "mediatype" => {
            let media = expect_string_scalar(
                value,
                "weboptions: MediaType must be a character vector or string scalar",
            )?;
            options
                .fields
                .insert("MediaType".to_string(), Value::from(media));
            Ok(())
        }
        "queryparameters" => {
            let qp = canonical_query_parameters(value)?;
            options.fields.insert("QueryParameters".to_string(), qp);
            Ok(())
        }
        _ => Err(weboptions_error(format!(
            "weboptions: unknown option '{}'",
            name
        ))),
    }
}

fn parse_content_type_option(value: &Value) -> BuiltinResult<String> {
    let text = expect_string_scalar(
        value,
        "weboptions: ContentType must be a character vector or string scalar",
    )?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok("auto".to_string()),
        "json" => Ok("json".to_string()),
        "text" | "char" | "string" => Ok("text".to_string()),
        "binary" | "raw" | "octet-stream" => Ok("binary".to_string()),
        other => Err(weboptions_error(format!(
            "weboptions: unsupported ContentType '{}'; use 'auto', 'json', 'text', or 'binary'",
            other
        ))),
    }
}

fn parse_request_method_option(value: &Value) -> BuiltinResult<String> {
    let text = expect_string_scalar(
        value,
        "weboptions: RequestMethod must be a character vector or string scalar",
    )?;
    let lower = text.trim().to_ascii_lowercase();
    match lower.as_str() {
        "auto" | "get" | "post" | "put" | "patch" | "delete" => Ok(lower),
        _ => Err(weboptions_error(format!(
            "weboptions: unsupported RequestMethod '{}'; expected auto, get, post, put, patch, or delete",
            text
        ))),
    }
}

fn canonical_header_fields(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Struct(struct_value) => {
            let mut out = StructValue::new();
            for (key, val) in &struct_value.fields {
                let header_value = expect_string_scalar(
                    val,
                    "weboptions: HeaderFields values must be character vectors or string scalars",
                )?;
                if header_value.trim().is_empty() {
                    return Err(weboptions_error(
                        "weboptions: header values must not be empty",
                    ));
                }
                if key.trim().is_empty() {
                    return Err(weboptions_error(
                        "weboptions: header names must not be empty",
                    ));
                }
                out.fields.insert(key.clone(), Value::from(header_value));
            }
            Ok(Value::Struct(out))
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(weboptions_error(
                    "weboptions: HeaderFields cell array must have exactly two columns",
                ));
            }
            let mut out = StructValue::new();
            for row in 0..cell.rows {
                let name_val = cell
                    .get(row, 0)
                    .map_err(|err| weboptions_error(format!("weboptions: {err}")))?;
                let value_val = cell
                    .get(row, 1)
                    .map_err(|err| weboptions_error(format!("weboptions: {err}")))?;

                let name = expect_string_scalar(
                    &name_val,
                    "weboptions: header names must be character vectors or string scalars",
                )?;
                if name.trim().is_empty() {
                    return Err(weboptions_error(
                        "weboptions: header names must not be empty",
                    ));
                }
                let header_value = expect_string_scalar(
                    &value_val,
                    "weboptions: header values must be character vectors or string scalars",
                )?;
                if header_value.trim().is_empty() {
                    return Err(weboptions_error(
                        "weboptions: header values must not be empty",
                    ));
                }
                out.fields.insert(name, Value::from(header_value));
            }
            Ok(Value::Struct(out))
        }
        _ => Err(weboptions_error(
            "weboptions: HeaderFields must be a struct or two-column cell array",
        )),
    }
}

fn canonical_query_parameters(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Struct(struct_value) => {
            let mut out = StructValue::new();
            for (key, val) in &struct_value.fields {
                out.fields.insert(key.clone(), val.clone());
            }
            Ok(Value::Struct(out))
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(weboptions_error(
                    "weboptions: QueryParameters cell array must have exactly two columns",
                ));
            }
            let mut out = StructValue::new();
            for row in 0..cell.rows {
                let name_val = cell
                    .get(row, 0)
                    .map_err(|err| weboptions_error(format!("weboptions: {err}")))?;
                let value_val = cell
                    .get(row, 1)
                    .map_err(|err| weboptions_error(format!("weboptions: {err}")))?;
                let name = expect_string_scalar(
                    &name_val,
                    "weboptions: query parameter names must be character vectors or string scalars",
                )?;
                out.fields.insert(name, value_val);
            }
            Ok(Value::Struct(out))
        }
        _ => Err(weboptions_error(
            "weboptions: QueryParameters must be a struct or two-column cell array",
        )),
    }
}

fn validate_credentials(options: &StructValue) -> BuiltinResult<()> {
    let username = string_field(options, "Username").unwrap_or_default();
    let password = string_field(options, "Password").unwrap_or_default();
    if !password.trim().is_empty() && username.trim().is_empty() {
        return Err(weboptions_error(
            "weboptions: Password requires a Username option",
        ));
    }
    Ok(())
}

fn string_field(options: &StructValue, field: &str) -> Option<String> {
    options.fields.get(field).and_then(|value| match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    })
}

fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                Ok(tensor.data[0])
            } else {
                Err(weboptions_error(context))
            }
        }
        _ => Err(weboptions_error(context)),
    }
}

fn expect_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(weboptions_error(context)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;

    use crate::call_builtin_async;
    use runmat_builtins::CellArray;

    fn spawn_server<F>(handler: F) -> String
    where
        F: FnOnce(TcpStream) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                handler(stream);
            }
        });
        format!("http://{}", addr)
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_weboptions(rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(weboptions_builtin(rest))
    }

    fn run_call_builtin(name: &str, args: &[Value]) -> BuiltinResult<Value> {
        futures::executor::block_on(call_builtin_async(name, args))
    }

    fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
        let mut buffer = Vec::new();
        let mut tmp = [0u8; 512];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buffer.extend_from_slice(&tmp[..n]);
                    if buffer.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let header_end = buffer
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap_or(buffer.len());
        let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let body = buffer[header_end..].to_vec();
        (headers, body)
    }

    fn respond_with(mut stream: TcpStream, content_type: &str, body: &[u8]) {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
            body.len(),
            content_type
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(body);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_default_struct_matches_expected_fields() {
        let result = run_weboptions(Vec::new()).expect("weboptions");
        let Value::Struct(options) = result else {
            panic!("expected struct result");
        };
        assert_eq!(
            options.fields.get("ContentType").and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            }),
            Some("auto")
        );
        assert_eq!(
            options.fields.get("Timeout").and_then(|v| match v {
                Value::Num(n) => Some(*n),
                _ => None,
            }),
            Some(DEFAULT_TIMEOUT_SECONDS)
        );
        match options.fields.get("HeaderFields") {
            Some(Value::Struct(headers)) => assert!(headers.fields.is_empty()),
            other => panic!("expected empty HeaderFields struct, got {other:?}"),
        }
        assert_eq!(
            options.fields.get("RequestMethod").and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            }),
            Some("auto")
        );
        assert_eq!(
            options.fields.get("MediaType").and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            }),
            Some("auto")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_overrides_timeout_and_headers() {
        let mut headers = StructValue::new();
        headers
            .fields
            .insert("Accept".to_string(), Value::from("application/json"));
        headers
            .fields
            .insert("X-Client".to_string(), Value::from("RunMat"));
        let args = vec![
            Value::from("Timeout"),
            Value::Num(10.0),
            Value::from("HeaderFields"),
            Value::Struct(headers),
        ];
        let result = run_weboptions(args).expect("weboptions overrides");
        let Value::Struct(opts) = result else {
            panic!("expected struct");
        };
        assert_eq!(
            opts.fields.get("Timeout").and_then(|v| match v {
                Value::Num(n) => Some(*n),
                _ => None,
            }),
            Some(10.0)
        );
        match opts.fields.get("HeaderFields") {
            Some(Value::Struct(headers)) => {
                assert_eq!(
                    headers.fields.get("Accept"),
                    Some(&Value::from("application/json"))
                );
                assert_eq!(headers.fields.get("X-Client"), Some(&Value::from("RunMat")));
            }
            other => panic!("expected header struct, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_updates_existing_struct() {
        let base = run_weboptions(vec![Value::from("ContentType"), Value::from("json")])
            .expect("base weboptions");
        let args = vec![base, Value::from("Timeout"), Value::Num(15.0)];
        let updated = run_weboptions(args).expect("weboptions update");
        let Value::Struct(opts) = updated else {
            panic!("expected struct");
        };
        assert_eq!(
            opts.fields.get("ContentType").and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            }),
            Some("json")
        );
        assert_eq!(
            opts.fields.get("Timeout").and_then(|v| match v {
                Value::Num(n) => Some(*n),
                _ => None,
            }),
            Some(15.0)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_rejects_unknown_option() {
        let err = error_message(
            run_weboptions(vec![Value::from("BogusOption"), Value::Num(1.0)])
                .expect_err("unknown option should fail"),
        );
        assert!(err.contains("unknown option"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_requires_username_when_password_provided() {
        let err = error_message(
            run_weboptions(vec![Value::from("Password"), Value::from("secret")])
                .expect_err("password without username"),
        );
        assert!(
            err.contains("Password requires a Username option"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_rejects_timeout_nonpositive() {
        let err = error_message(
            run_weboptions(vec![Value::from("Timeout"), Value::Num(0.0)])
                .expect_err("timeout should reject nonpositive values"),
        );
        assert!(
            err.contains("Timeout must be a finite, positive scalar"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_rejects_headerfields_bad_cell_shape() {
        let cell = CellArray::new(vec![Value::from("Accept")], 1, 1).expect("cell");
        let err = error_message(
            run_weboptions(vec![Value::from("HeaderFields"), Value::Cell(cell)])
                .expect_err("headerfields cell shape"),
        );
        assert!(
            err.contains("HeaderFields cell array must have exactly two columns"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_uses_weboptions_without_polluting_query() {
        let options = run_weboptions(Vec::new()).expect("weboptions");
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, _) = read_request(&mut stream);
            tx.send(headers).unwrap();
            respond_with(stream, "application/json", br#"{"ok":true}"#);
        });

        let args = vec![Value::from(url.clone()), options];
        let result = run_call_builtin("webread", &args).expect("webread with options");
        match result {
            Value::Struct(reply) => {
                assert!(matches!(reply.fields.get("ok"), Some(Value::Bool(true))));
            }
            other => panic!("expected struct response, got {other:?}"),
        }
        let headers = rx.recv().expect("captured headers");
        assert!(headers.starts_with("GET "));
        assert!(
            !headers.contains("MediaType=auto"),
            "MediaType should not appear in query string"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webwrite_uses_weboptions_auto_request_method() {
        let options = run_weboptions(Vec::new()).expect("weboptions default");
        let payload = Value::from("Hello from RunMat");
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, body) = read_request(&mut stream);
            tx.send((headers, body)).unwrap();
            respond_with(stream, "application/json", br#"{"ack":true}"#);
        });

        let args = vec![Value::from(url), payload, options];
        let result = run_call_builtin("webwrite", &args).expect("webwrite with weboptions");
        match result {
            Value::Struct(reply) => {
                assert!(matches!(reply.fields.get("ack"), Some(Value::Bool(true))));
            }
            other => panic!("expected struct response, got {other:?}"),
        }
        let (headers, body) = rx.recv().expect("request captured");
        assert!(
            headers.starts_with("POST "),
            "expected POST request, got headers: {headers}"
        );
        assert!(
            !body.is_empty(),
            "expected request body to be present when posting form data"
        );
    }
}
