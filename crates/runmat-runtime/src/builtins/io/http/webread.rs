//! MATLAB-compatible `webread` builtin for HTTP/HTTPS downloads.

use std::collections::VecDeque;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
use base64::Engine;
use runmat_builtins::{CellArray, CharArray, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;
use url::Url;

use super::transport::{
    self, decode_body_as_text, header_value, HttpMethod, HttpRequest, HEADER_CONTENT_TYPE,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::json::jsondecode::decode_json_text;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;
const DEFAULT_USER_AGENT: &str = "RunMat webread/0.0";

#[allow(clippy::too_many_lines)]
#[runmat_macros::register_doc_text(
    name = "webread",
    builtin_path = "crate::builtins::io::http::webread"
)]
pub const DOC_MD: &str = r#"---
title: "webread"
category: "io/http"
keywords: ["webread", "http get", "rest client", "json", "https", "api"]
summary: "Download web content (JSON, text, or binary) over HTTP/HTTPS."
references:
  - https://www.mathworks.com/help/matlab/ref/webread.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "webread always gathers gpuArray inputs and executes on the CPU."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::http::webread::tests"
  integration:
    - "builtins::io::http::webread::tests::webread_fetches_json_response"
    - "builtins::io::http::webread::tests::webread_fetches_text_response"
    - "builtins::io::http::webread::tests::webread_fetches_binary_payload"
    - "builtins::io::http::webread::tests::webread_appends_query_parameters"
    - "builtins::io::http::webread::tests::webread_struct_argument_supports_options_and_query"
    - "builtins::io::http::webread::tests::webread_headerfields_struct_applies_custom_headers"
    - "builtins::io::http::webread::tests::webread_queryparameters_option_struct"
---

# What does the `webread` function do in MATLAB / RunMat?
`webread` issues an HTTP or HTTPS request and returns the response body as a MATLAB-compatible
value. Textual payloads become character vectors, JSON responses are decoded into structs, cells,
and numeric arrays, while binary payloads return numeric vectors of bytes.

## How does the `webread` function behave in MATLAB / RunMat?
- Accepts URLs supplied as character vectors or string scalars; the URL must be absolute.
- Optional `weboptions`-style fields (either through a struct argument or name-value pairs) control
  content decoding (`ContentType`), request timeout (`Timeout`), headers (`HeaderFields`), and
  authentication (`Username`/`Password`). The builtin currently supports the default `GET`
  request method; use `webwrite` for POST/PUT uploads.
- Additional name-value pairs that do not match an option are appended to the query string using
  percent-encoding. A leading struct or cell array argument can also supply query parameters.
- `ContentType 'auto'` (default) inspects the `Content-Type` response header to choose between JSON,
  text, or binary decoding. Explicit `ContentType 'json'`, `'text'`, or `'binary'` override the
  detection logic.
- JSON responses are parsed with the same rules as `jsondecode`, producing doubles, logicals,
  strings, structs, and cell arrays that match MATLAB semantics.
- Text responses preserve the server-provided character encoding (UTF-8 with automatic decoding of
  exotic charsets exposed in the HTTP headers). Binary payloads return `1×N` double arrays whose
  entries store byte values in the range 0–255.
- HTTP errors (non-2xx status codes), timeouts, TLS failures, and parsing problems raise descriptive
  MATLAB-style errors.

## `webread` Function GPU Execution Behaviour
`webread` runs entirely on the CPU. Any `gpuArray` inputs (for example, query parameter values)
are gathered to host memory before building the HTTP request. Results are produced on the host, and
fusion graphs terminate at this builtin via `ResidencyPolicy::GatherImmediately`.

## Examples of using the `webread` function in MATLAB / RunMat

### Reading JSON data from a REST API
```matlab
opts = weboptions("ContentType", "json", "Timeout", 15);
weather = webread("https://api.example.com/weather", opts, "city", "Reykjavik");
disp(weather.temperatureC);
```
Expected output:
```matlab
    2.3
```

### Downloading plain text as a character vector
```matlab
html = webread("https://example.com/index.txt", "Timeout", 5);
extract = html(1:200);
```
`extract` contains the first 200 characters as a `1×200` char vector.

### Retrieving binary payloads such as images
```matlab
bytes = webread("https://example.com/logo.png", "ContentType", "binary");
filewrite("logo.png", uint8(bytes));
```
The PNG file is written locally after converting the returned bytes to `uint8`.

### Supplying custom headers and credentials
```matlab
headers = struct("Accept", "application/json", "X-Client", "RunMat");
data = webread("https://api.example.com/me", ...
               "Username", "ada", "Password", "secret", ...
               "HeaderFields", headers, ...
               "ContentType", "json");
```
Custom headers are merged into the request, and HTTP basic authentication credentials are attached.

### Passing query parameters as a struct
```matlab
query = struct("limit", 25, "sort", "name");
response = webread("https://api.example.com/resources", query, "ContentType", "json");
```
`query` is promoted into the URL query string before the request is sent.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `webread` gathers any GPU-resident values before contacting the network and produces host
results. Keeping inputs on the GPU offers no benefit because HTTP/TLS stacks operate on the CPU.

## FAQ

1. **Can `webread` decode JSON automatically?**  
   Yes. When the server reports a JSON `Content-Type` header (for example `application/json`
   or `application/vnd.api+json`) the builtin decodes it using the same rules as `jsondecode`.
   Override the behaviour with `"ContentType","text"` or `"ContentType","binary"` when needed.

2. **How do I control request timeouts?**  
   Supply `"Timeout", seconds` as a name-value pair or in an options struct. The default timeout
   is 60 seconds. Timeouts raise `webread: request to <url> timed out`.

3. **What headers can I set?**  
   Use `"HeaderFields", struct(...)` or a `cell` array of name/value pairs. Header names must be
   valid HTTP tokens. The builtin automatically sets a RunMat-specific `User-Agent` string unless
   you override it with `"UserAgent", "..."`

4. **Does `webread` follow redirects?**  
   Yes. The underlying HTTP client follows redirects up to the platform default limit while
   preserving headers and authentication.

5. **How do I provide credentials?**  
   Use `"Username", "user", "Password", "pass"` for HTTP basic authentication. Supplying a password
   without a username raises an error.

6. **Can I send POST or PUT requests?**  
   `webread` is designed for read-only requests and currently supports the default `GET` method.
   Use `webwrite` (planned) for requests that include bodies or mutate server state.

7. **How are binary responses represented?**  
   Binary payloads return `1×N` double arrays whose elements are byte values. Convert them to the
   desired integer type (for example `uint8`) before further processing.

8. **What happens when the server returns an error status?**  
   Non-success HTTP status codes raise `webread: request to … failed with HTTP status XYZ`. Inspect
   the remote server logs or response headers for additional diagnostics.

9. **Does `webread` support compressed responses?**  
   Yes. The builtin enables gzip / deflate content decoding through the HTTP client automatically.

10. **Can I pass query parameters as GPU arrays?**  
    Yes. Inputs wrapped in `gpuArray` are gathered before assembling the query string.

## See Also
[webwrite](./webwrite), [weboptions](./weboptions), [jsondecode](./jsondecode), [websave](./filewrite)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::http::webread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "webread",
    op_kind: GpuOpKind::Custom("http-get"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "HTTP requests always execute on the CPU; gpuArray inputs are gathered eagerly.",
};

fn webread_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("webread").build()
}

fn remap_webread_flow<F>(err: RuntimeError, message: F) -> RuntimeError
where
    F: FnOnce(&RuntimeError) -> String,
{
    build_runtime_error(message(&err))
        .with_builtin("webread")
        .with_source(err)
        .build()
}

fn webread_flow_with_context(err: RuntimeError) -> RuntimeError {
    remap_webread_flow(err, |err| format!("webread: {}", err.message()))
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::http::webread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "webread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "webread performs network I/O and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "webread",
    category = "io/http",
    summary = "Download web content (JSON, text, or binary) over HTTP/HTTPS.",
    keywords = "webread,http get,rest client,json,api",
    accel = "sink",
    builtin_path = "crate::builtins::io::http::webread"
)]
async fn webread_builtin(url: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered_url = gather_if_needed_async(&url)
        .await
        .map_err(webread_flow_with_context)?;
    let gathered_args = gather_arguments(rest).await?;
    let url_text = expect_string_scalar(
        &gathered_url,
        "webread: URL must be a character vector or string scalar",
    )?;
    if url_text.trim().is_empty() {
        return Err(webread_error("webread: URL must not be empty"));
    }
    let (options, query_params) = parse_arguments(gathered_args)?;
    execute_request(&url_text, options, &query_params)
}

async fn gather_arguments(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(webread_flow_with_context)?,
        );
    }
    Ok(out)
}

fn parse_arguments(args: Vec<Value>) -> BuiltinResult<(WebReadOptions, Vec<(String, String)>)> {
    let mut queue: VecDeque<Value> = args.into();
    let mut options = WebReadOptions::default();
    let mut query_params = Vec::new();

    if matches!(queue.front(), Some(Value::Struct(_))) {
        if let Some(Value::Struct(struct_value)) = queue.pop_front() {
            process_struct_fields(&struct_value, &mut options, &mut query_params)?;
        }
    } else if matches!(queue.front(), Some(Value::Cell(_))) {
        if let Some(Value::Cell(cell)) = queue.pop_front() {
            append_query_from_cell(&cell, &mut query_params)?
        }
    }

    while let Some(name_value) = queue.pop_front() {
        let name = expect_string_scalar(
            &name_value,
            "webread: parameter names must be character vectors or string scalars",
        )?;
        let value = queue
            .pop_front()
            .ok_or_else(|| webread_error("webread: missing value for name-value argument"))?;
        process_name_value_pair(&name, &value, &mut options, &mut query_params)?;
    }

    Ok((options, query_params))
}

fn process_struct_fields(
    struct_value: &StructValue,
    options: &mut WebReadOptions,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    for (key, value) in &struct_value.fields {
        process_name_value_pair(key, value, options, query_params)?;
    }
    Ok(())
}

fn process_name_value_pair(
    name: &str,
    value: &Value,
    options: &mut WebReadOptions,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "contenttype" => {
            options.content_type = parse_content_type(value)?;
            Ok(())
        }
        "timeout" => {
            options.timeout = parse_timeout(value)?;
            Ok(())
        }
        "headerfields" => {
            let headers = parse_header_fields(value)?;
            options.headers.extend(headers);
            Ok(())
        }
        "useragent" => {
            options.user_agent = Some(expect_string_scalar(
                value,
                "webread: UserAgent must be a character vector or string scalar",
            )?);
            Ok(())
        }
        "username" => {
            options.username = Some(expect_string_scalar(
                value,
                "webread: Username must be a character vector or string scalar",
            )?);
            Ok(())
        }
        "password" => {
            options.password = Some(expect_string_scalar(
                value,
                "webread: Password must be a character vector or string scalar",
            )?);
            Ok(())
        }
        "requestmethod" => {
            options.method = parse_request_method(value)?;
            Ok(())
        }
        "mediatype" => {
            // weboptions exposes MediaType for webwrite; accept and ignore for webread.
            expect_string_scalar(
                value,
                "webread: MediaType must be a character vector or string scalar",
            )?;
            Ok(())
        }
        "queryparameters" => append_query_from_value(value, query_params),
        _ => {
            let param_value = value_to_query_string(value, name)?;
            query_params.push((name.to_string(), param_value));
            Ok(())
        }
    }
}

fn append_query_from_value(
    value: &Value,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    match value {
        Value::Struct(struct_value) => {
            for (key, val) in &struct_value.fields {
                let text = value_to_query_string(val, key)?;
                query_params.push((key.clone(), text));
            }
            Ok(())
        }
        Value::Cell(cell) => append_query_from_cell(cell, query_params),
        _ => Err(webread_error(
            "webread: QueryParameters must be a struct or cell array",
        )),
    }
}

fn append_query_from_cell(
    cell: &CellArray,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    if cell.cols != 2 {
        return Err(webread_error(
            "webread: cell array of query parameters must have two columns",
        ));
    }
    for row in 0..cell.rows {
        let name_value = cell
            .get(row, 0)
            .map_err(|err| webread_error(format!("webread: {err}")))?;
        let value_value = cell
            .get(row, 1)
            .map_err(|err| webread_error(format!("webread: {err}")))?;
        let name = expect_string_scalar(
            &name_value,
            "webread: query parameter names must be text scalars",
        )?;
        let text = value_to_query_string(&value_value, &name)?;
        query_params.push((name, text));
    }
    Ok(())
}

fn execute_request(
    url_text: &str,
    options: WebReadOptions,
    query_params: &[(String, String)],
) -> BuiltinResult<Value> {
    let username_present = options
        .username
        .as_ref()
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let password_present = options
        .password
        .as_ref()
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    if password_present && !username_present {
        return Err(webread_error(
            "webread: Password requires a Username option",
        ));
    }

    let mut url = Url::parse(url_text).map_err(|err| {
        build_runtime_error(format!("webread: invalid URL '{url_text}': {err}"))
            .with_builtin("webread")
            .with_source(err)
            .build()
    })?;
    if !query_params.is_empty() {
        {
            let mut pairs = url.query_pairs_mut();
            for (name, value) in query_params {
                pairs.append_pair(name, value);
            }
        }
    }
    let user_agent = options
        .user_agent
        .as_deref()
        .filter(|ua| !ua.trim().is_empty())
        .unwrap_or(DEFAULT_USER_AGENT)
        .to_string();

    let mut headers = options.headers.clone();
    let has_auth_header = headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("authorization"));
    if !has_auth_header {
        if let Some(username) = options.username.as_ref().filter(|s| !s.is_empty()) {
            let password = options.password.clone().unwrap_or_default();
            let token = BASE64_ENGINE.encode(format!("{username}:{password}"));
            headers.push(("Authorization".to_string(), format!("Basic {token}")));
        }
    }

    let request = HttpRequest {
        url,
        method: HttpMethod::Get,
        headers,
        body: None,
        timeout: options.timeout,
        user_agent,
    };

    let response = transport::send_request(&request).map_err(|err| {
        build_runtime_error(err.message_with_prefix("webread"))
            .with_builtin("webread")
            .with_source(err)
            .build()
    })?;

    let header_content_type =
        header_value(&response.headers, HEADER_CONTENT_TYPE).map(|value| value.to_string());
    let resolved = options.resolve_content_type(header_content_type.as_deref());

    match resolved {
        ResolvedContentType::Json => {
            let body = decode_body_as_text(&response.body, header_content_type.as_deref());
            let value = decode_json_text(&body).map_err(map_json_error)?;
            Ok(value)
        }
        ResolvedContentType::Text => {
            let text = decode_body_as_text(&response.body, header_content_type.as_deref());
            let array = CharArray::new_row(&text);
            Ok(Value::CharArray(array))
        }
        ResolvedContentType::Binary => {
            let data: Vec<f64> = response.body.iter().map(|b| f64::from(*b)).collect();
            let cols = response.body.len();
            let tensor = Tensor::new(data, vec![1, cols])
                .map_err(|err| webread_error(format!("webread: {err}")))?;
            Ok(Value::Tensor(tensor))
        }
    }
}

fn map_json_error(err: RuntimeError) -> RuntimeError {
    let message = if let Some(rest) = err.message().strip_prefix("jsondecode: ") {
        format!("webread: failed to parse JSON response ({rest})")
    } else {
        format!("webread: failed to parse JSON response ({})", err.message())
    };
    build_runtime_error(message)
        .with_builtin("webread")
        .with_source(err)
        .build()
}

fn parse_header_fields(value: &Value) -> BuiltinResult<Vec<(String, String)>> {
    match value {
        Value::Struct(struct_value) => {
            let mut headers = Vec::with_capacity(struct_value.fields.len());
            for (key, val) in &struct_value.fields {
                let header_value = expect_string_scalar(
                    val,
                    "webread: header values must be character vectors or string scalars",
                )?;
                headers.push((key.clone(), header_value));
            }
            Ok(headers)
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(webread_error(
                    "webread: HeaderFields cell array must have exactly two columns",
                ));
            }
            let mut headers = Vec::with_capacity(cell.rows);
            for row in 0..cell.rows {
                let name = cell
                    .get(row, 0)
                    .map_err(|err| webread_error(format!("webread: {err}")))?;
                let value = cell
                    .get(row, 1)
                    .map_err(|err| webread_error(format!("webread: {err}")))?;
                let header_name = expect_string_scalar(
                    &name,
                    "webread: header names must be character vectors or string scalars",
                )?;
                if header_name.trim().is_empty() {
                    return Err(webread_error("webread: header names must not be empty"));
                }
                let header_value = expect_string_scalar(
                    &value,
                    "webread: header values must be character vectors or string scalars",
                )?;
                headers.push((header_name, header_value));
            }
            Ok(headers)
        }
        _ => Err(webread_error(
            "webread: HeaderFields must be provided as a struct or cell array of name/value pairs",
        )),
    }
}

fn parse_content_type(value: &Value) -> BuiltinResult<ContentTypeHint> {
    let text = expect_string_scalar(
        value,
        "webread: ContentType must be a character vector or string scalar",
    )?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(ContentTypeHint::Auto),
        "json" => Ok(ContentTypeHint::Json),
        "text" | "char" | "string" => Ok(ContentTypeHint::Text),
        "binary" | "octet-stream" | "raw" => Ok(ContentTypeHint::Binary),
        other => Err(webread_error(format!(
            "webread: unsupported ContentType '{}'; use 'auto', 'json', 'text', or 'binary'",
            other
        ))),
    }
}

fn parse_timeout(value: &Value) -> BuiltinResult<Duration> {
    let seconds = numeric_scalar(value, "webread: Timeout must be a finite, positive scalar")?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(webread_error(
            "webread: Timeout must be a finite, positive scalar",
        ));
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn parse_request_method(value: &Value) -> BuiltinResult<HttpMethod> {
    let text = expect_string_scalar(
        value,
        "webread: RequestMethod must be a character vector or string scalar",
    )?;
    let lower = text.trim().to_ascii_lowercase();
    match lower.as_str() {
        "get" | "auto" => Ok(HttpMethod::Get),
        other => Err(webread_error(format!(
            "webread: RequestMethod '{}' is not supported; expected 'auto' or 'get'",
            other
        ))),
    }
}

fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                Ok(tensor.data[0])
            } else {
                Err(webread_error(context))
            }
        }
        _ => Err(webread_error(context)),
    }
}

fn expect_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(webread_error(context)),
    }
}

fn value_to_query_string(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::Num(n) => Ok(format!("{}", n)),
        Value::Int(i) => Ok(i.to_i64().to_string()),
        Value::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                Ok(format!("{}", tensor.data[0]))
            } else {
                Err(webread_error(format!(
                    "webread: query parameter '{}' must be scalar",
                    name
                )))
            }
        }
        Value::LogicalArray(array) => {
            if array.len() == 1 {
                Ok(if array.data[0] != 0 {
                    "true".into()
                } else {
                    "false".into()
                })
            } else {
                Err(webread_error(format!(
                    "webread: query parameter '{}' must be scalar",
                    name
                )))
            }
        }
        _ => Err(webread_error(format!(
            "webread: unsupported value type for query parameter '{}'",
            name
        ))),
    }
}

#[derive(Clone, Copy, Debug)]
enum ContentTypeHint {
    Auto,
    Text,
    Json,
    Binary,
}

#[derive(Clone, Copy, Debug)]
enum ResolvedContentType {
    Text,
    Json,
    Binary,
}

#[derive(Clone, Debug)]
struct WebReadOptions {
    content_type: ContentTypeHint,
    timeout: Duration,
    headers: Vec<(String, String)>,
    user_agent: Option<String>,
    username: Option<String>,
    password: Option<String>,
    method: HttpMethod,
}

impl Default for WebReadOptions {
    fn default() -> Self {
        Self {
            content_type: ContentTypeHint::Auto,
            timeout: Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS),
            headers: Vec::new(),
            user_agent: None,
            username: None,
            password: None,
            method: HttpMethod::Get,
        }
    }
}

impl WebReadOptions {
    fn resolve_content_type(&self, header: Option<&str>) -> ResolvedContentType {
        match self.content_type {
            ContentTypeHint::Json => ResolvedContentType::Json,
            ContentTypeHint::Text => ResolvedContentType::Text,
            ContentTypeHint::Binary => ResolvedContentType::Binary,
            ContentTypeHint::Auto => infer_content_type(header),
        }
    }
}

fn infer_content_type(header: Option<&str>) -> ResolvedContentType {
    if let Some(raw) = header {
        let mime = raw
            .split(';')
            .next()
            .map(|part| part.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if mime == "application/json" || mime == "text/json" || mime.ends_with("+json") {
            ResolvedContentType::Json
        } else if mime.starts_with("text/")
            || mime == "application/xml"
            || mime.ends_with("+xml")
            || mime == "application/xhtml+xml"
            || mime == "application/javascript"
            || mime == "application/x-www-form-urlencoded"
        {
            ResolvedContentType::Text
        } else {
            ResolvedContentType::Binary
        }
    } else {
        ResolvedContentType::Text
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;

    use crate::builtins::common::test_support;

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_webread(url: Value, args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(webread_builtin(url, args))
    }

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

    fn respond_with(mut stream: TcpStream, content_type: &str, body: &[u8]) {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
            body.len(),
            content_type
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(body);
    }

    fn read_request_headers(stream: &mut TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 256];
        while let Ok(read) = stream.read(&mut chunk) {
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
            if buffer.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
            if buffer.len() > 16 * 1024 {
                break;
            }
        }
        String::from_utf8_lossy(&buffer).to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_fetches_json_response() {
        let url = spawn_server(|mut stream| {
            let mut buffer = [0u8; 1024];
            let _ = stream.read(&mut buffer);
            respond_with(
                stream,
                "application/json",
                br#"{"message":"hello","value":42}"#,
            );
        });

        let result = run_webread(Value::from(url), vec![]).expect("webread JSON response");

        match result {
            Value::Struct(struct_value) => {
                let message = struct_value.fields.get("message").expect("message field");
                let value = struct_value.fields.get("value").expect("value field");
                match message {
                    Value::CharArray(ca) => {
                        let text: String = ca.data.iter().collect();
                        assert_eq!(text, "hello");
                    }
                    other => panic!("expected char array, got {other:?}"),
                }
                match value {
                    Value::Num(n) => assert_eq!(*n, 42.0),
                    other => panic!("expected numeric value, got {other:?}"),
                }
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_fetches_text_response() {
        let url = spawn_server(|mut stream| {
            let mut buffer = [0u8; 512];
            let _ = stream.read(&mut buffer);
            respond_with(stream, "text/plain; charset=utf-8", b"RunMat webread test");
        });

        let result = run_webread(Value::from(url), vec![]).expect("webread text response");

        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "RunMat webread test");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_fetches_binary_payload() {
        let payload = [1u8, 2, 3, 254, 255];
        let url = spawn_server(move |mut stream| {
            let mut buffer = [0u8; 512];
            let _ = stream.read(&mut buffer);
            respond_with(stream, "application/octet-stream", &payload);
        });

        let args = vec![Value::from("ContentType"), Value::from("binary")];
        let result = run_webread(Value::from(url), args).expect("webread binary response");

        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 5]);
                let bytes: Vec<u8> = tensor.data.iter().map(|v| *v as u8).collect();
                assert_eq!(bytes, payload);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_appends_query_parameters() {
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let request = read_request_headers(&mut stream);
            let _ = tx.send(request);
            respond_with(stream, "application/json", br#"{"ok":true}"#);
        });

        let args = vec![
            Value::from("count"),
            Value::Num(42.0),
            Value::from("ContentType"),
            Value::from("json"),
        ];
        let result = run_webread(Value::from(url.clone()), args).expect("webread query");
        match result {
            Value::Struct(struct_value) => {
                assert!(struct_value.fields.contains_key("ok"));
            }
            other => panic!("expected struct result, got {other:?}"),
        }
        let request = rx.recv().expect("request log");
        assert!(
            request.starts_with("GET /"),
            "unexpected request line: {request}"
        );
        assert!(
            request.contains("count=42"),
            "query parameters missing: {request}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_struct_argument_supports_options_and_query() {
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let request = read_request_headers(&mut stream);
            let _ = tx.send(request);
            respond_with(stream, "application/json", br#"{"value":123}"#);
        });

        let mut fields = StructValue::new();
        fields
            .fields
            .insert("ContentType".to_string(), Value::from("json"));
        fields.fields.insert("limit".to_string(), Value::Num(5.0));

        let result = run_webread(Value::from(url.clone()), vec![Value::Struct(fields)])
            .expect("webread struct arg");

        let request = rx.recv().expect("request log");
        assert!(
            request.contains("GET /?limit=5"),
            "expected limit query parameter: {request}"
        );

        match result {
            Value::Struct(struct_value) => match struct_value.fields.get("value") {
                Some(Value::Num(n)) => assert_eq!(*n, 123.0),
                other => panic!("unexpected JSON decode result: {other:?}"),
            },
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_headerfields_struct_applies_custom_headers() {
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let request = read_request_headers(&mut stream);
            let _ = tx.send(request);
            respond_with(stream, "application/json", br#"{"ok":true}"#);
        });

        let mut headers = StructValue::new();
        headers
            .fields
            .insert("X-Test".to_string(), Value::from("RunMat"));

        let args = vec![
            Value::from("HeaderFields"),
            Value::Struct(headers),
            Value::from("ContentType"),
            Value::from("json"),
        ];

        let result = run_webread(Value::from(url), args).expect("webread header fields");
        assert!(matches!(result, Value::Struct(_)));

        let request = rx.recv().expect("request log");
        assert!(
            request.to_ascii_lowercase().contains("x-test: runmat"),
            "custom header missing: {request}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_queryparameters_option_struct() {
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let request = read_request_headers(&mut stream);
            let _ = tx.send(request);
            respond_with(stream, "application/json", br#"{"ok":true}"#);
        });

        let mut params = StructValue::new();
        params.fields.insert("page".to_string(), Value::Num(2.0));

        let args = vec![
            Value::from("QueryParameters"),
            Value::Struct(params),
            Value::from("ContentType"),
            Value::from("json"),
        ];

        let result =
            run_webread(Value::from(url.clone()), args).expect("webread query parameters");
        assert!(matches!(result, Value::Struct(_)));

        let request = rx.recv().expect("request log");
        assert!(
            request.contains("page=2"),
            "query parameter missing: {request}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_errors_on_missing_name_value_pair() {
        let err = run_webread(
            Value::from("https://example.com"),
            vec![Value::from("Timeout")],
        )
        .expect_err("expected missing value error");
        let err = error_message(err);
        assert!(
            err.contains("missing value"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_rejects_non_positive_timeout() {
        let args = vec![Value::from("Timeout"), Value::Num(0.0)];
        let err =
            run_webread(Value::from("https://example.com"), args).expect_err("timeout error");
        let err = error_message(err);
        assert!(
            err.contains("Timeout must be a finite, positive scalar"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_rejects_password_without_username() {
        let args = vec![Value::from("Password"), Value::from("secret")];
        let err =
            run_webread(Value::from("https://example.com"), args).expect_err("auth error");
        let err = error_message(err);
        assert!(
            err.contains("Password requires a Username"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_rejects_unsupported_content_type() {
        let args = vec![Value::from("ContentType"), Value::from("table")];
        let err =
            run_webread(Value::from("https://example.com"), args).expect_err("format error");
        let err = error_message(err);
        assert!(
            err.contains("unsupported ContentType"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webread_rejects_invalid_headerfields_shape() {
        let cell = crate::make_cell(
            vec![Value::from("A"), Value::from("B"), Value::from("C")],
            1,
            3,
        )
        .expect("make cell");

        let args = vec![Value::from("HeaderFields"), cell];
        let err =
            run_webread(Value::from("https://example.com"), args).expect_err("header error");
        let err = error_message(err);
        assert!(
            err.contains("HeaderFields cell array must have exactly two columns"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
