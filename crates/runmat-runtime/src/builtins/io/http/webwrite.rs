//! MATLAB-compatible `webwrite` builtin for HTTP/HTTPS uploads.

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
use crate::call_builtin_async;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;
const DEFAULT_USER_AGENT: &str = "RunMat webwrite/0.0";

#[allow(clippy::too_many_lines)]
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::http::webwrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "webwrite",
    op_kind: GpuOpKind::Custom("http-write"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "HTTP uploads run on the CPU and gather gpuArray inputs before serialisation.",
};

fn webwrite_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("webwrite")
        .build()
}

fn remap_webwrite_flow<F>(err: RuntimeError, message: F) -> RuntimeError
where
    F: FnOnce(&RuntimeError) -> String,
{
    build_runtime_error(message(&err))
        .with_builtin("webwrite")
        .with_source(err)
        .build()
}

fn webwrite_flow_with_context(err: RuntimeError) -> RuntimeError {
    remap_webwrite_flow(err, |err| format!("webwrite: {}", err.message()))
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::http::webwrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "webwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "webwrite performs network I/O and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "webwrite",
    category = "io/http",
    summary = "Send data to web services using HTTP POST/PUT requests and return the response.",
    keywords = "webwrite,http post,rest client,json upload,form post",
    accel = "sink",
    type_resolver(crate::builtins::io::type_resolvers::webwrite_type),
    builtin_path = "crate::builtins::io::http::webwrite"
)]
async fn webwrite_builtin(url: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered_url = gather_if_needed_async(&url)
        .await
        .map_err(webwrite_flow_with_context)?;
    let url_text = expect_string_scalar(
        &gathered_url,
        "webwrite: URL must be a character vector or string scalar",
    )?;
    if url_text.trim().is_empty() {
        return Err(webwrite_error("webwrite: URL must not be empty"));
    }
    if rest.is_empty() {
        return Err(webwrite_error("webwrite: missing data argument"));
    }

    let mut gathered = Vec::with_capacity(rest.len());
    for value in rest {
        gathered.push(
            gather_if_needed_async(&value)
                .await
                .map_err(webwrite_flow_with_context)?,
        );
    }
    let mut queue: VecDeque<Value> = VecDeque::from(gathered);
    let data_value = queue
        .pop_front()
        .ok_or_else(|| webwrite_error("webwrite: missing data argument"))?;

    let (options, query_params) = parse_arguments(queue)?;
    let body = prepare_request_body(data_value, &options).await?;
    execute_request(&url_text, options, &query_params, body)
}

fn parse_arguments(
    mut queue: VecDeque<Value>,
) -> BuiltinResult<(WebWriteOptions, Vec<(String, String)>)> {
    let mut options = WebWriteOptions::default();
    let mut query_params = Vec::new();

    if matches!(queue.front(), Some(Value::Struct(_))) {
        if let Some(Value::Struct(struct_value)) = queue.pop_front() {
            process_struct_fields(&struct_value, &mut options, &mut query_params)?;
        }
    } else if matches!(queue.front(), Some(Value::Cell(_))) {
        if let Some(Value::Cell(cell)) = queue.pop_front() {
            append_query_from_cell(&cell, &mut query_params)?;
        }
    }

    while let Some(name_value) = queue.pop_front() {
        let name = expect_string_scalar(
            &name_value,
            "webwrite: parameter names must be character vectors or strings",
        )?;
        let value = queue
            .pop_front()
            .ok_or_else(|| webwrite_error("webwrite: missing value for name-value argument"))?;
        process_name_value_pair(&name, &value, &mut options, &mut query_params)?;
    }

    Ok((options, query_params))
}

fn process_struct_fields(
    struct_value: &StructValue,
    options: &mut WebWriteOptions,
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
    options: &mut WebWriteOptions,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "contenttype" => {
            let ct = parse_content_type(value)?;
            options.content_type = ct;
            Ok(())
        }
        "mediatype" => {
            let media = expect_string_scalar(
                value,
                "webwrite: MediaType must be a character vector or string scalar",
            )?;
            let trimmed = media.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                options.media_type = None;
                options.request_format = RequestFormat::Auto;
                options.request_format_explicit = false;
            } else {
                options.media_type = Some(media.clone());
                options.request_format = infer_request_format(&media);
                options.request_format_explicit = true;
            }
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
                "webwrite: UserAgent must be a character vector or string scalar",
            )?);
            Ok(())
        }
        "username" => {
            options.username = Some(expect_string_scalar(
                value,
                "webwrite: Username must be a character vector or string scalar",
            )?);
            Ok(())
        }
        "password" => {
            options.password = Some(expect_string_scalar(
                value,
                "webwrite: Password must be a character vector or string scalar",
            )?);
            Ok(())
        }
        "requestmethod" => {
            options.method = parse_request_method(value)?;
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

fn execute_request(
    url_text: &str,
    options: WebWriteOptions,
    query_params: &[(String, String)],
    body: PreparedBody,
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
        return Err(webwrite_error(
            "webwrite: Password requires a Username option",
        ));
    }

    let mut url = Url::parse(url_text).map_err(|err| {
        build_runtime_error(format!("webwrite: invalid URL '{url_text}': {err}"))
            .with_builtin("webwrite")
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

    let has_ct_header = headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("content-type"));
    if !has_ct_header {
        if let Some(ct) = &body.content_type {
            headers.push(("Content-Type".to_string(), ct.clone()));
        }
    }

    let request = HttpRequest {
        url,
        method: options.method,
        headers,
        body: Some(body.bytes),
        timeout: options.timeout,
        user_agent,
    };

    let response = transport::send_request(&request).map_err(|err| {
        build_runtime_error(err.message_with_prefix("webwrite"))
            .with_builtin("webwrite")
            .with_source(err)
            .build()
    })?;

    let header_content_type =
        header_value(&response.headers, HEADER_CONTENT_TYPE).map(|value| value.to_string());
    let resolved = options.resolve_content_type(header_content_type.as_deref());

    match resolved {
        ResolvedContentType::Json => {
            let body_text = decode_body_as_text(&response.body, header_content_type.as_deref());
            let value = decode_json_text(&body_text).map_err(map_json_error)?;
            Ok(value)
        }
        ResolvedContentType::Text => {
            let body_text = decode_body_as_text(&response.body, header_content_type.as_deref());
            Ok(Value::CharArray(CharArray::new_row(&body_text)))
        }
        ResolvedContentType::Binary => {
            let data: Vec<f64> = response.body.iter().map(|b| f64::from(*b)).collect();
            let cols = data.len();
            let tensor = Tensor::new(data, vec![1, cols])
                .map_err(|err| webwrite_error(format!("webwrite: {err}")))?;
            Ok(Value::Tensor(tensor))
        }
    }
}

async fn prepare_request_body(
    data: Value,
    options: &WebWriteOptions,
) -> BuiltinResult<PreparedBody> {
    let format = match options.request_format {
        RequestFormat::Auto => guess_request_format(&data),
        set => set,
    };
    let content_type = options
        .media_type
        .clone()
        .or_else(|| default_content_type_for(format));
    let bytes = match format {
        RequestFormat::Form => encode_form_payload(&data)?,
        RequestFormat::Json => encode_json_payload(&data).await?,
        RequestFormat::Text => encode_text_payload(&data)?,
        RequestFormat::Binary => encode_binary_payload(&data)?,
        RequestFormat::Auto => encode_json_payload(&data).await?,
    };
    Ok(PreparedBody {
        bytes,
        content_type,
    })
}

fn encode_form_payload(value: &Value) -> BuiltinResult<Vec<u8>> {
    let mut pairs = Vec::new();
    match value {
        Value::Struct(struct_value) => {
            for (key, val) in &struct_value.fields {
                let text = value_to_query_string(val, key)?;
                pairs.push((key.clone(), text));
            }
        }
        Value::Cell(cell) => {
            append_query_from_cell(cell, &mut pairs)?;
        }
        Value::CharArray(_)
        | Value::String(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Tensor(_) => {
            // Allow scalar text/numeric by mapping to a default "data" key.
            let text = scalar_to_string(value)?;
            pairs.push(("data".to_string(), text));
        }
        _ => {
            return Err(webwrite_error(
                "webwrite: form payloads must be structs, two-column cell arrays, or scalars",
            ))
        }
    }

    let encoded = encode_form_pairs(&pairs);
    Ok(encoded.into_bytes())
}

fn encode_form_pairs(pairs: &[(String, String)]) -> String {
    let mut result = String::new();
    for (idx, (name, value)) in pairs.iter().enumerate() {
        if idx > 0 {
            result.push('&');
        }
        result.push_str(&url_encode_component(name));
        result.push('=');
        result.push_str(&url_encode_component(value));
    }
    result
}

fn url_encode_component(input: &str) -> String {
    let mut out = String::new();
    for byte in input.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'*' => {
                out.push(byte as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(hex_digit(byte >> 4));
                out.push(hex_digit(byte & 0xF));
            }
        }
    }
    out
}

fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'A' + (nibble - 10)) as char,
        _ => unreachable!(),
    }
}

async fn encode_json_payload(value: &Value) -> BuiltinResult<Vec<u8>> {
    let encoded = call_builtin_async("jsonencode", std::slice::from_ref(value))
        .await
        .map_err(|flow| remap_webwrite_flow(flow, |err| format!("webwrite: {}", err.message())))?;
    let text = expect_string_scalar(
        &encoded,
        "webwrite: jsonencode returned unexpected value; expected text scalar",
    )?;
    Ok(text.into_bytes())
}

fn encode_text_payload(value: &Value) -> BuiltinResult<Vec<u8>> {
    let text = scalar_to_string(value)?;
    Ok(text.into_bytes())
}

fn encode_binary_payload(value: &Value) -> BuiltinResult<Vec<u8>> {
    match value {
        Value::Tensor(tensor) => tensor_f64_to_bytes(tensor),
        Value::Num(n) => Ok(vec![float_to_byte(*n)?]),
        Value::Int(i) => Ok(vec![int_to_byte(i.to_i64())?]),
        Value::Bool(b) => Ok(vec![if *b { 1 } else { 0 }]),
        Value::LogicalArray(array) => Ok(array.data.clone()),
        Value::CharArray(ca) => {
            let mut bytes = Vec::with_capacity(ca.data.len());
            for ch in &ca.data {
                let code = *ch as u32;
                if code > 0xFF {
                    return Err(webwrite_error(
                        "webwrite: character codes exceed 255 for binary payload",
                    ));
                }
                bytes.push(code as u8);
            }
            Ok(bytes)
        }
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].as_bytes().to_vec())
            } else {
                Err(webwrite_error(
                    "webwrite: binary payload string arrays must be scalar",
                ))
            }
        }
        _ => Err(webwrite_error(
            "webwrite: unsupported value for binary payload",
        )),
    }
}

fn tensor_f64_to_bytes(tensor: &Tensor) -> BuiltinResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(tensor.data.len());
    for value in &tensor.data {
        bytes.push(float_to_byte(*value)?);
    }
    Ok(bytes)
}

fn float_to_byte(value: f64) -> BuiltinResult<u8> {
    if !value.is_finite() {
        return Err(webwrite_error(
            "webwrite: binary payload values must be finite",
        ));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        return Err(webwrite_error(
            "webwrite: binary payload values must be integers in 0..255",
        ));
    }
    let int_val = rounded as i64;
    int_to_byte(int_val)
}

fn int_to_byte(value: i64) -> BuiltinResult<u8> {
    if !(0..=255).contains(&value) {
        return Err(webwrite_error(
            "webwrite: binary payload values must be in the range 0..255",
        ));
    }

    Ok(value as u8)
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
        _ => Err(webwrite_error(
            "webwrite: QueryParameters must be a struct or cell array",
        )),
    }
}

fn append_query_from_cell(
    cell: &CellArray,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    if cell.cols != 2 {
        return Err(webwrite_error(
            "webwrite: cell array of query parameters must have two columns",
        ));
    }
    for row in 0..cell.rows {
        let name_value = cell
            .get(row, 0)
            .map_err(|err| webwrite_error(format!("webwrite: {err}")))?;
        let value_value = cell
            .get(row, 1)
            .map_err(|err| webwrite_error(format!("webwrite: {err}")))?;
        let name = expect_string_scalar(
            &name_value,
            "webwrite: query parameter names must be text scalars",
        )?;
        let text = value_to_query_string(&value_value, &name)?;
        query_params.push((name, text));
    }
    Ok(())
}

fn parse_content_type(value: &Value) -> BuiltinResult<ContentTypeHint> {
    let text = expect_string_scalar(
        value,
        "webwrite: ContentType must be a character vector or string scalar",
    )?;
    let lower = text.trim().to_ascii_lowercase();
    match lower.as_str() {
        "auto" => Ok(ContentTypeHint::Auto),
        "json" => Ok(ContentTypeHint::Json),
        "text" => Ok(ContentTypeHint::Text),
        "binary" => Ok(ContentTypeHint::Binary),
        _ => Err(webwrite_error(
            "webwrite: ContentType must be 'auto', 'json', 'text', or 'binary'",
        )),
    }
}

fn parse_timeout(value: &Value) -> BuiltinResult<Duration> {
    let seconds = numeric_scalar(
        value,
        "webwrite: Timeout must be a finite, non-negative scalar numeric value",
    )?;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(webwrite_error(
            "webwrite: Timeout must be a finite, non-negative scalar numeric value",
        ));
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn parse_request_method(value: &Value) -> BuiltinResult<HttpMethod> {
    let text = expect_string_scalar(
        value,
        "webwrite: RequestMethod must be a character vector or string scalar",
    )?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(HttpMethod::Post),
        "post" => Ok(HttpMethod::Post),
        "put" => Ok(HttpMethod::Put),
        "patch" => Ok(HttpMethod::Patch),
        "delete" => Ok(HttpMethod::Delete),
        other => Err(webwrite_error(format!(
            "webwrite: unsupported RequestMethod '{}'; expected auto, post, put, patch, or delete",
            other
        ))),
    }
}

fn parse_header_fields(value: &Value) -> BuiltinResult<Vec<(String, String)>> {
    match value {
        Value::Struct(struct_value) => {
            let mut headers = Vec::with_capacity(struct_value.fields.len());
            for (key, val) in &struct_value.fields {
                let header_value = expect_string_scalar(
                    val,
                    "webwrite: header values must be character vectors or string scalars",
                )?;
                headers.push((key.clone(), header_value));
            }
            Ok(headers)
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(webwrite_error(
                    "webwrite: HeaderFields cell array must have exactly two columns",
                ));
            }
            let mut headers = Vec::with_capacity(cell.rows);
            for row in 0..cell.rows {
                let name = cell
                    .get(row, 0)
                    .map_err(|err| webwrite_error(format!("webwrite: {err}")))?;
                let value = cell
                    .get(row, 1)
                    .map_err(|err| webwrite_error(format!("webwrite: {err}")))?;
                let header_name = expect_string_scalar(
                    &name,
                    "webwrite: header names must be character vectors or string scalars",
                )?;
                if header_name.trim().is_empty() {
                    return Err(webwrite_error("webwrite: header names must not be empty"));
                }
                let header_value = expect_string_scalar(
                    &value,
                    "webwrite: header values must be character vectors or string scalars",
                )?;
                headers.push((header_name, header_value));
            }
            Ok(headers)
        }
        _ => Err(webwrite_error(
            "webwrite: HeaderFields must be a struct or two-column cell array",
        )),
    }
}

fn map_json_error(err: RuntimeError) -> RuntimeError {
    let message = if let Some(rest) = err.message().strip_prefix("jsondecode: ") {
        format!("webwrite: failed to parse JSON response ({rest})")
    } else {
        format!(
            "webwrite: failed to parse JSON response ({})",
            err.message()
        )
    };
    build_runtime_error(message)
        .with_builtin("webwrite")
        .with_source(err)
        .build()
}

fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                Ok(tensor.data[0])
            } else {
                Err(webwrite_error(context))
            }
        }
        _ => Err(webwrite_error(context)),
    }
}

fn scalar_to_string(value: &Value) -> BuiltinResult<String> {
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
                Err(webwrite_error(
                    "webwrite: expected scalar value for text payload",
                ))
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
                Err(webwrite_error(
                    "webwrite: expected scalar value for text payload",
                ))
            }
        }
        _ => Err(webwrite_error(
            "webwrite: unsupported value type for text payload",
        )),
    }
}

fn expect_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(webwrite_error(context)),
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
                Err(webwrite_error(format!(
                    "webwrite: query parameter '{}' must be scalar",
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
                Err(webwrite_error(format!(
                    "webwrite: query parameter '{}' must be scalar",
                    name
                )))
            }
        }
        _ => Err(webwrite_error(format!(
            "webwrite: unsupported value type for query parameter '{}'",
            name
        ))),
    }
}

fn guess_request_format(value: &Value) -> RequestFormat {
    match value {
        Value::Struct(_) => RequestFormat::Form,
        Value::Cell(cell) if cell.cols == 2 => RequestFormat::Form,
        Value::CharArray(ca) if ca.rows == 1 => RequestFormat::Text,
        Value::String(_) => RequestFormat::Text,
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                RequestFormat::Text
            } else {
                RequestFormat::Json
            }
        }
        Value::Tensor(_) | Value::LogicalArray(_) => RequestFormat::Json,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => RequestFormat::Json,
        _ => RequestFormat::Json,
    }
}

fn infer_request_format(media_type: &str) -> RequestFormat {
    let lower = media_type.trim().to_ascii_lowercase();
    if lower.contains("json") {
        RequestFormat::Json
    } else if lower.starts_with("text/") || lower.contains("xml") {
        RequestFormat::Text
    } else if lower == "application/x-www-form-urlencoded" {
        RequestFormat::Form
    } else {
        RequestFormat::Binary
    }
}

fn default_content_type_for(format: RequestFormat) -> Option<String> {
    match format {
        RequestFormat::Form => Some("application/x-www-form-urlencoded".to_string()),
        RequestFormat::Json => Some("application/json".to_string()),
        RequestFormat::Text => Some("text/plain; charset=utf-8".to_string()),
        RequestFormat::Binary => Some("application/octet-stream".to_string()),
        RequestFormat::Auto => None,
    }
}

#[derive(Clone, Debug)]
struct PreparedBody {
    bytes: Vec<u8>,
    content_type: Option<String>,
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

#[derive(Clone, Copy, Debug)]
enum RequestFormat {
    Auto,
    Form,
    Json,
    Text,
    Binary,
}

#[derive(Clone, Debug)]
struct WebWriteOptions {
    content_type: ContentTypeHint,
    timeout: Duration,
    headers: Vec<(String, String)>,
    user_agent: Option<String>,
    username: Option<String>,
    password: Option<String>,
    method: HttpMethod,
    request_format: RequestFormat,
    request_format_explicit: bool,
    media_type: Option<String>,
}

impl Default for WebWriteOptions {
    fn default() -> Self {
        Self {
            content_type: ContentTypeHint::Auto,
            timeout: Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS),
            headers: Vec::new(),
            user_agent: None,
            username: None,
            password: None,
            method: HttpMethod::Post,
            request_format: RequestFormat::Auto,
            request_format_explicit: false,
            media_type: None,
        }
    }
}

impl WebWriteOptions {
    fn resolve_content_type(&self, header: Option<&str>) -> ResolvedContentType {
        match self.content_type {
            ContentTypeHint::Json => ResolvedContentType::Json,
            ContentTypeHint::Text => ResolvedContentType::Text,
            ContentTypeHint::Binary => ResolvedContentType::Binary,
            ContentTypeHint::Auto => infer_response_content_type(header),
        }
    }
}

fn infer_response_content_type(header: Option<&str>) -> ResolvedContentType {
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
        let mut header_end = None;
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buffer.extend_from_slice(&tmp[..n]);
                    if let Some(idx) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        header_end = Some(idx + 4);
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let header_end = header_end.unwrap_or(buffer.len());
        let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let mut parts = line.splitn(2, ':');
                let name = parts.next()?.trim();
                let value = parts.next()?.trim();
                if name.eq_ignore_ascii_case("content-length") {
                    value.parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let mut body = buffer[header_end..].to_vec();
        while body.len() < content_length {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => body.extend_from_slice(&tmp[..n]),
                Err(_) => break,
            }
        }
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

    fn run_webwrite(url: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(webwrite_builtin(url, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webwrite_posts_form_data_by_default() {
        let payload = {
            let mut st = StructValue::new();
            st.fields.insert("name".to_string(), Value::from("Ada"));
            st.fields.insert("score".to_string(), Value::Num(42.0));
            st
        };
        let opts = {
            let mut st = StructValue::new();
            st.fields
                .insert("ContentType".to_string(), Value::from("json"));
            st
        };

        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, body) = read_request(&mut stream);
            tx.send((headers, body)).unwrap();
            respond_with(
                stream,
                "application/json",
                br#"{"status":"ok","received":true}"#,
            );
        });

        let result = run_webwrite(
            Value::from(url),
            vec![Value::Struct(payload), Value::Struct(opts)],
        )
        .expect("webwrite");

        let (headers, body) = rx.recv().expect("request captured");
        assert!(headers.starts_with("POST "));
        let headers_lower = headers.to_ascii_lowercase();
        assert!(headers_lower.contains("content-type: application/x-www-form-urlencoded"));
        let body_text = String::from_utf8(body).expect("utf8 body");
        assert!(body_text.contains("name=Ada"));
        assert!(body_text.contains("score=42"));

        match result {
            Value::Struct(reply) => {
                assert!(matches!(
                    reply.fields.get("received"),
                    Some(Value::Bool(true))
                ));
            }
            other => panic!("expected struct response, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webwrite_sends_json_when_media_type_json() {
        let payload = {
            let mut st = StructValue::new();
            st.fields.insert("title".to_string(), Value::from("RunMat"));
            st.fields.insert("stars".to_string(), Value::Num(5.0));
            st
        };
        let opts = {
            let mut st = StructValue::new();
            st.fields.insert(
                "MediaType".to_string(),
                Value::from("application/json; charset=utf-8"),
            );
            st.fields
                .insert("ContentType".to_string(), Value::from("json"));
            st
        };

        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, body) = read_request(&mut stream);
            tx.send((headers, body)).unwrap();
            respond_with(stream, "application/json", br#"{"ok":true}"#);
        });

        let result = run_webwrite(
            Value::from(url),
            vec![Value::Struct(payload), Value::Struct(opts)],
        )
        .expect("webwrite");

        let (headers, body) = rx.recv().expect("request");
        let headers_lower = headers.to_ascii_lowercase();
        assert!(headers_lower.contains("content-type: application/json"));
        let body_text = String::from_utf8(body).expect("utf8 body");
        assert!(body_text.contains("\"title\":\"RunMat\""));
        assert!(body_text.contains("\"stars\":5"));

        match result {
            Value::Struct(reply) => {
                assert!(matches!(reply.fields.get("ok"), Some(Value::Bool(true))));
            }
            other => panic!("expected struct response, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webwrite_applies_basic_auth_and_custom_headers() {
        let payload = Value::from("");
        let mut header_struct = StructValue::new();
        header_struct
            .fields
            .insert("X-Test".to_string(), Value::from("yes"));
        header_struct
            .fields
            .insert("Accept".to_string(), Value::from("text/plain"));
        let mut opts_struct = StructValue::new();
        opts_struct
            .fields
            .insert("Username".to_string(), Value::from("ada"));
        opts_struct
            .fields
            .insert("Password".to_string(), Value::from("secret"));
        opts_struct
            .fields
            .insert("HeaderFields".to_string(), Value::Struct(header_struct));
        opts_struct
            .fields
            .insert("ContentType".to_string(), Value::from("text"));
        opts_struct
            .fields
            .insert("MediaType".to_string(), Value::from("text/plain"));

        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, _) = read_request(&mut stream);
            tx.send(headers).unwrap();
            respond_with(stream, "text/plain", b"OK");
        });

        let result = run_webwrite(Value::from(url), vec![payload, Value::Struct(opts_struct)])
            .expect("webwrite");

        let headers = rx.recv().expect("headers");
        let headers_lower = headers.to_ascii_lowercase();
        assert!(headers_lower.contains("authorization: basic"));
        assert!(headers_lower.contains("x-test: yes"));
        assert!(headers_lower.contains("accept: text/plain"));

        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "OK");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webwrite_supports_query_parameters() {
        let payload = Value::Struct(StructValue::new());
        let mut qp_struct = StructValue::new();
        qp_struct.fields.insert("page".to_string(), Value::Num(2.0));
        qp_struct
            .fields
            .insert("verbose".to_string(), Value::Bool(true));
        let mut opts_struct = StructValue::new();
        opts_struct
            .fields
            .insert("QueryParameters".to_string(), Value::Struct(qp_struct));

        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, _) = read_request(&mut stream);
            tx.send(headers).unwrap();
            respond_with(stream, "application/json", br#"{"ok":true}"#);
        });

        let _ = run_webwrite(
            Value::from(url.clone()),
            vec![payload, Value::Struct(opts_struct)],
        )
        .expect("webwrite");

        let headers = rx.recv().expect("headers");
        let first_line = headers.lines().next().unwrap_or("");
        assert!(first_line.starts_with("POST "));
        assert!(first_line.contains("page=2"));
        assert!(first_line.contains("verbose=true"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn webwrite_binary_payload_respected() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 255.0], vec![4, 1]).unwrap();
        let payload = Value::Tensor(tensor);
        let mut opts_struct = StructValue::new();
        opts_struct
            .fields
            .insert("ContentType".to_string(), Value::from("binary"));
        opts_struct.fields.insert(
            "MediaType".to_string(),
            Value::from("application/octet-stream"),
        );

        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, body) = read_request(&mut stream);
            tx.send((headers, body)).unwrap();
            respond_with(stream, "text/plain", b"OK");
        });

        let _ = run_webwrite(Value::from(url), vec![payload, Value::Struct(opts_struct)])
            .expect("webwrite");

        let (headers, body) = rx.recv().expect("request");
        let headers_lower = headers.to_ascii_lowercase();
        assert!(headers_lower.contains("content-type: application/octet-stream"));
        assert_eq!(body, vec![1, 2, 3, 255]);
    }
}
