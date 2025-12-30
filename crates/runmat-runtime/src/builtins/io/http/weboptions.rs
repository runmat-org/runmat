//! MATLAB-compatible `weboptions` builtin for constructing HTTP client options.

use std::collections::VecDeque;

use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;

#[allow(clippy::too_many_lines)]
#[runmat_macros::register_doc_text(
    name = "weboptions",
    builtin_path = "crate::builtins::io::http::weboptions"
)]
pub const DOC_MD: &str = r#"---
title: "weboptions"
category: "io/http"
keywords: ["weboptions", "http options", "timeout", "headers", "rest client"]
summary: "Create an options struct that configures webread and webwrite HTTP behaviour."
references:
  - https://www.mathworks.com/help/matlab/ref/weboptions.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "weboptions operates on CPU data structures and gathers gpuArray inputs automatically."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::http::weboptions::tests"
  integration:
    - "builtins::io::http::weboptions::tests::weboptions_default_struct_matches_expected_fields"
    - "builtins::io::http::weboptions::tests::weboptions_overrides_timeout_and_headers"
    - "builtins::io::http::weboptions::tests::weboptions_updates_existing_struct"
    - "builtins::io::http::weboptions::tests::weboptions_rejects_unknown_option"
    - "builtins::io::http::weboptions::tests::weboptions_requires_username_when_password_provided"
    - "builtins::io::http::weboptions::tests::weboptions_rejects_timeout_nonpositive"
    - "builtins::io::http::weboptions::tests::weboptions_rejects_headerfields_bad_cell_shape"
    - "builtins::io::http::weboptions::tests::webread_uses_weboptions_without_polluting_query"
    - "builtins::io::http::weboptions::tests::webwrite_uses_weboptions_auto_request_method"
---

# What does the `weboptions` function do in MATLAB / RunMat?
`weboptions` builds a MATLAB-style options struct that controls HTTP behaviour for
functions such as `webread`, `webwrite`, and `websave`. The struct stores option
fields like `Timeout`, `ContentType`, `HeaderFields`, and `RequestMethod`, all with
MATLAB-compatible defaults.

## How does the `weboptions` function behave in MATLAB / RunMat?
- Returns a struct with canonical field names: `ContentType`, `Timeout`, `HeaderFields`,
  `UserAgent`, `Username`, `Password`, `RequestMethod`, `MediaType`, and `QueryParameters`.
- Defaults mirror MATLAB: `ContentType="auto"`, `Timeout=60`, `UserAgent=""`
  (RunMat substitutes a default agent when this is empty), `RequestMethod="auto"`,
  `MediaType="auto"`, and empty structs for `HeaderFields` and `QueryParameters`.
- Name-value arguments are case-insensitive. Values are validated to ensure MATLAB-compatible
  types (text scalars for string options, positive scalars for `Timeout`,
  structs or two-column cell arrays for `HeaderFields` and `QueryParameters`).
- Passing an existing options struct as the first argument clones it before applying additional
  overrides, matching MATLAB's update pattern `opts = weboptions(opts, "Timeout", 5)`.
- Unknown option names raise descriptive errors.

## `weboptions` Function GPU Execution Behaviour
`weboptions` operates entirely on CPU metadata. It gathers any `gpuArray` inputs back to host
memory before validation, because HTTP requests execute on the CPU regardless of the selected
acceleration provider. No GPU provider hooks are required for this function.

## Examples of using the `weboptions` function in MATLAB / RunMat

### Setting custom timeouts for webread calls
```matlab
opts = weboptions("Timeout", 10);
html = webread("https://example.com", opts);
```
The request aborts after 10 seconds instead of the default 60.

### Providing HTTP basic authentication credentials
```matlab
opts = weboptions("Username", "ada", "Password", "lovelace");
profile = webread("https://api.example.com/me", opts);
```
Credentials are attached automatically; an empty username leaves authentication disabled.

### Sending JSON payloads with webwrite
```matlab
opts = weboptions("ContentType", "json", "MediaType", "application/json");
payload = struct("title", "RunMat", "stars", 5);
reply = webwrite("https://api.example.com/projects", payload, opts);
```
The request posts JSON and expects a JSON response.

### Applying custom headers with struct syntax
```matlab
headers = struct("Accept", "application/json", "X-Client", "RunMat");
opts = weboptions("HeaderFields", headers);
data = webread("https://api.example.com/resources", opts);
```
`HeaderFields` accepts a struct or two-column cell array of header name/value pairs.

### Combining existing options with overrides
```matlab
base = weboptions("ContentType", "json");
opts = weboptions(base, "Timeout", 15, "QueryParameters", struct("verbose", true));
result = webread("https://api.example.com/items", opts);
```
The new struct inherits all fields from `base` and overrides the ones supplied later.

## FAQ

### Which option names are supported in RunMat?
`weboptions` implements the options consumed by `webread` and `webwrite`: `ContentType`,
`Timeout`, `HeaderFields`, `UserAgent`, `Username`, `Password`, `RequestMethod`, `MediaType`,
and `QueryParameters`. Unknown names raise a MATLAB-style error.

### What does `RequestMethod="auto"` mean?
`webread` treats `"auto"` as `"get"` while `webwrite` maps it to `"post"`. Override the method when
you need `put`, `patch`, or `delete`.

### How are empty usernames or passwords handled?
Empty strings leave authentication disabled. A non-empty password without a username raises a
MATLAB-compatible error.

### Can I pass query parameters through the options struct?
Yes. Supply a struct or two-column cell array in the `QueryParameters` option. Values may include
numbers, logicals, or text scalars, and they are percent-encoded when the request is built.

### Do I need to manage GPU residency for options?
No. `weboptions` gathers any GPU-resident values automatically and always returns a host struct.
HTTP builtins ignore GPU residency for metadata.

### Does `weboptions` mutate the input struct?
No. A copy is made before overrides are applied, preserving the original struct you pass in.

### How can I clear headers or query parameters?
Pass an empty struct (`struct()`) or empty cell array (`{}`) to reset the respective option.

## See Also
[webread](./webread), [webwrite](./webwrite), [jsondecode](../json/jsondecode), [jsonencode](../json/jsonencode)
"#;

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
    builtin_path = "crate::builtins::io::http::weboptions"
)]
fn weboptions_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let mut gathered = Vec::with_capacity(rest.len());
    for value in rest {
        gathered.push(gather_if_needed(&value).map_err(|e| format!("weboptions: {e}"))?);
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
            .ok_or_else(|| "weboptions: missing value for name-value argument".to_string())?;
        set_option_field(&mut options, &name, &value)?;
    }

    validate_credentials(&options)?;

    Ok(Value::Struct(options))
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

fn apply_struct_fields(source: StructValue, target: &mut StructValue) -> Result<(), String> {
    for (key, value) in &source.fields {
        set_option_field(target, key, value)?;
    }
    Ok(())
}

fn set_option_field(options: &mut StructValue, name: &str, value: &Value) -> Result<(), String> {
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
                return Err("weboptions: Timeout must be a finite, positive scalar".to_string());
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
        _ => Err(format!("weboptions: unknown option '{}'", name)),
    }
}

fn parse_content_type_option(value: &Value) -> Result<String, String> {
    let text = expect_string_scalar(
        value,
        "weboptions: ContentType must be a character vector or string scalar",
    )?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok("auto".to_string()),
        "json" => Ok("json".to_string()),
        "text" | "char" | "string" => Ok("text".to_string()),
        "binary" | "raw" | "octet-stream" => Ok("binary".to_string()),
        other => Err(format!(
            "weboptions: unsupported ContentType '{}'; use 'auto', 'json', 'text', or 'binary'",
            other
        )),
    }
}

fn parse_request_method_option(value: &Value) -> Result<String, String> {
    let text = expect_string_scalar(
        value,
        "weboptions: RequestMethod must be a character vector or string scalar",
    )?;
    let lower = text.trim().to_ascii_lowercase();
    match lower.as_str() {
        "auto" | "get" | "post" | "put" | "patch" | "delete" => Ok(lower),
        _ => Err(format!(
            "weboptions: unsupported RequestMethod '{}'; expected auto, get, post, put, patch, or delete",
            text
        )),
    }
}

fn canonical_header_fields(value: &Value) -> Result<Value, String> {
    match value {
        Value::Struct(struct_value) => {
            let mut out = StructValue::new();
            for (key, val) in &struct_value.fields {
                let header_value = expect_string_scalar(
                    val,
                    "weboptions: HeaderFields values must be character vectors or string scalars",
                )?;
                if header_value.trim().is_empty() {
                    return Err("weboptions: header values must not be empty".to_string());
                }
                if key.trim().is_empty() {
                    return Err("weboptions: header names must not be empty".to_string());
                }
                out.fields.insert(key.clone(), Value::from(header_value));
            }
            Ok(Value::Struct(out))
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(
                    "weboptions: HeaderFields cell array must have exactly two columns".to_string(),
                );
            }
            let mut out = StructValue::new();
            for row in 0..cell.rows {
                let name_val = cell.get(row, 0).map_err(|e| format!("weboptions: {e}"))?;
                let value_val = cell.get(row, 1).map_err(|e| format!("weboptions: {e}"))?;
                let name = expect_string_scalar(
                    &name_val,
                    "weboptions: header names must be character vectors or string scalars",
                )?;
                if name.trim().is_empty() {
                    return Err("weboptions: header names must not be empty".to_string());
                }
                let header_value = expect_string_scalar(
                    &value_val,
                    "weboptions: header values must be character vectors or string scalars",
                )?;
                if header_value.trim().is_empty() {
                    return Err("weboptions: header values must not be empty".to_string());
                }
                out.fields.insert(name, Value::from(header_value));
            }
            Ok(Value::Struct(out))
        }
        _ => Err("weboptions: HeaderFields must be a struct or two-column cell array".to_string()),
    }
}

fn canonical_query_parameters(value: &Value) -> Result<Value, String> {
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
                return Err(
                    "weboptions: QueryParameters cell array must have exactly two columns"
                        .to_string(),
                );
            }
            let mut out = StructValue::new();
            for row in 0..cell.rows {
                let name_val = cell.get(row, 0).map_err(|e| format!("weboptions: {e}"))?;
                let value_val = cell.get(row, 1).map_err(|e| format!("weboptions: {e}"))?;
                let name = expect_string_scalar(
                    &name_val,
                    "weboptions: query parameter names must be character vectors or string scalars",
                )?;
                out.fields.insert(name, value_val);
            }
            Ok(Value::Struct(out))
        }
        _ => {
            Err("weboptions: QueryParameters must be a struct or two-column cell array".to_string())
        }
    }
}

fn validate_credentials(options: &StructValue) -> Result<(), String> {
    let username = string_field(options, "Username").unwrap_or_default();
    let password = string_field(options, "Password").unwrap_or_default();
    if !password.trim().is_empty() && username.trim().is_empty() {
        return Err("weboptions: Password requires a Username option".to_string());
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

fn numeric_scalar(value: &Value, context: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                Ok(tensor.data[0])
            } else {
                Err(context.to_string())
            }
        }
        _ => Err(context.to_string()),
    }
}

fn expect_string_scalar(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(context.to_string()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;

    use crate::call_builtin;
    use runmat_builtins::CellArray;

    use crate::builtins::common::test_support;

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
        let result = weboptions_builtin(Vec::new()).expect("weboptions");
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
        let result = weboptions_builtin(args).expect("weboptions overrides");
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
        let base = weboptions_builtin(vec![Value::from("ContentType"), Value::from("json")])
            .expect("base weboptions");
        let args = vec![base, Value::from("Timeout"), Value::Num(15.0)];
        let updated = weboptions_builtin(args).expect("weboptions update");
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
        let err = weboptions_builtin(vec![Value::from("BogusOption"), Value::Num(1.0)])
            .expect_err("unknown option should fail");
        assert!(err.contains("unknown option"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_requires_username_when_password_provided() {
        let err = weboptions_builtin(vec![Value::from("Password"), Value::from("secret")])
            .expect_err("password without username");
        assert!(
            err.contains("Password requires a Username option"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_rejects_timeout_nonpositive() {
        let err = weboptions_builtin(vec![Value::from("Timeout"), Value::Num(0.0)])
            .expect_err("timeout should reject nonpositive values");
        assert!(
            err.contains("Timeout must be a finite, positive scalar"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn weboptions_rejects_headerfields_bad_cell_shape() {
        let cell = CellArray::new(vec![Value::from("Accept")], 1, 1).expect("cell");
        let err = weboptions_builtin(vec![Value::from("HeaderFields"), Value::Cell(cell)])
            .expect_err("headerfields cell shape");
        assert!(
            err.contains("HeaderFields cell array must have exactly two columns"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_uses_weboptions_without_polluting_query() {
        let options = weboptions_builtin(Vec::new()).expect("weboptions");
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, _) = read_request(&mut stream);
            tx.send(headers).unwrap();
            respond_with(stream, "application/json", br#"{"ok":true}"#);
        });

        let args = vec![Value::from(url.clone()), options];
        let result = call_builtin("webread", &args).expect("webread with options");
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
        let options = weboptions_builtin(Vec::new()).expect("weboptions default");
        let payload = Value::from("Hello from RunMat");
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, body) = read_request(&mut stream);
            tx.send((headers, body)).unwrap();
            respond_with(stream, "application/json", br#"{"ack":true}"#);
        });

        let args = vec![Value::from(url), payload, options];
        let result = call_builtin("webwrite", &args).expect("webwrite with weboptions");
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
