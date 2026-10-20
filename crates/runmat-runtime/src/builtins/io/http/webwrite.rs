//! MATLAB-compatible `webwrite` builtin for HTTP/HTTPS uploads.

#[path = "webwrite/body.rs"]
mod body;
#[path = "webwrite/model.rs"]
mod model;
#[path = "webwrite/options.rs"]
mod options;
#[path = "webwrite/preflight.rs"]
mod preflight;
use body::*;
use model::*;
use options::*;
use preflight::*;

use runmat_value::NumericScalar;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
use base64::Engine;
use runmat_builtins::{
    parse_http_url, BuiltinErrorDescriptor, HttpResponseDialect, HttpResponseMode, WebOptionField,
    WEBWRITE_DESCRIPTOR, WEBWRITE_ERROR_FLOW, WEBWRITE_ERROR_INVALID_ARGUMENT,
    WEBWRITE_ERROR_INVALID_CREDENTIALS, WEBWRITE_ERROR_INVALID_OPTION_VALUE,
    WEBWRITE_ERROR_INVALID_URL, WEBWRITE_ERROR_MISSING_DATA, WEBWRITE_ERROR_MISSING_OPTION_VALUE,
    WEBWRITE_ERROR_OUTPUT, WEBWRITE_ERROR_OUTPUTS, WEBWRITE_ERROR_RESPONSE_JSON,
    WEBWRITE_ERROR_TRANSPORT, WEBWRITE_EXPLICIT_GPU_EXTENSION as EXPLICIT_GPU_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StructValue, Tensor, Value};
use url::Url;

use super::request_options::{PreparedRequestNames, RequestOptionName};
use super::transport::{
    self, decode_body_as_text, header_value, HttpMethod, HttpRequest, HEADER_CONTENT_TYPE,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::io::json::jsondecode::decode_json_text;
use crate::call_builtin_async;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT_SECONDS: f64 = 5.0;
const MAX_TIMEOUT_SECONDS: f64 = 2147.483647;
const DEFAULT_USER_AGENT: &str = "RunMat webwrite/0.0";
const BUILTIN_NAME: &str = "webwrite";

fn webwrite_error(message: impl Into<String>) -> RuntimeError {
    webwrite_error_with(&WEBWRITE_ERROR_INVALID_ARGUMENT, message)
}

fn webwrite_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn webwrite_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_webwrite_flow<F>(
    error: &'static BuiltinErrorDescriptor,
    err: RuntimeError,
    message: F,
) -> RuntimeError
where
    F: FnOnce(&RuntimeError) -> String,
{
    let mut builder = build_runtime_error(message(&err))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn webwrite_flow_with_context(err: RuntimeError) -> RuntimeError {
    remap_webwrite_flow(&WEBWRITE_ERROR_FLOW, err, |err| {
        format!("webwrite: {}", err.message())
    })
}

#[runtime_builtin(
    name = "webwrite",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::http::webwrite"
)]
async fn webwrite_builtin(
    context: Option<crate::context::RuntimeContext>,
    url: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &WEBWRITE_DESCRIPTOR,
        &WEBWRITE_ERROR_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&url)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let url_text = expect_string_scalar(
        &url,
        "webwrite: URL must be a character vector or string scalar",
    )?;
    if url_text.trim().is_empty() {
        return Err(webwrite_error_with(
            &WEBWRITE_ERROR_INVALID_URL,
            "webwrite: URL must not be empty",
        ));
    }
    if rest.is_empty() {
        return Err(webwrite_error_with(
            &WEBWRITE_ERROR_MISSING_DATA,
            WEBWRITE_ERROR_MISSING_DATA.message,
        ));
    }
    let prepared_api_key = match rest.get(1) {
        Some(Value::Struct(source)) => Some(super::api_key::PreparedApiKey::parse(source)),
        _ => None,
    };
    let prepared_names =
        preflight_host_arguments(rest.view().range(1..)?, prepared_api_key.as_ref())?;
    let parsed_url = parse_http_url(&url_text).map_err(|err| {
        webwrite_error_with_source(
            &WEBWRITE_ERROR_INVALID_URL,
            format!("webwrite: invalid URL: {err}"),
            err,
        )
    })?;

    let source = &rest;
    let mut queue = rest
        .try_map_preserving_roots_async(rest.len(), context.as_ref(), |index| async move {
            gather_if_needed_async(&source[index])
                .await
                .map_err(webwrite_flow_with_context)
        })
        .await?;
    let data_value = queue.pop_front().ok_or_else(|| {
        webwrite_error_with(
            &WEBWRITE_ERROR_MISSING_DATA,
            WEBWRITE_ERROR_MISSING_DATA.message,
        )
    })?;

    let (options, query_params) =
        parse_arguments(&mut queue, prepared_api_key.as_ref(), &prepared_names)?;
    let body = prepare_request_body(data_value, &options).await?;
    execute_request(parsed_url, options, &query_params, body, context.as_ref()).await
}

fn parse_arguments(
    queue: &mut crate::call::BuiltinArguments,
    prepared_api_key: Option<&super::api_key::PreparedApiKey>,
    prepared_names: &PreparedRequestNames,
) -> BuiltinResult<(WebWriteOptions, Vec<(String, String)>)> {
    let mut options = WebWriteOptions::default();
    let mut query_params = Vec::new();

    if matches!(queue.first(), Some(Value::Struct(_))) {
        if let Some(Value::Struct(struct_value)) = queue.pop_front() {
            process_struct_fields(
                &struct_value,
                prepared_api_key,
                prepared_names,
                &mut options,
                &mut query_params,
            )?;
        }
    } else if matches!(queue.first(), Some(Value::Cell(_))) {
        if let Some(Value::Cell(cell)) = queue.pop_front() {
            append_query_from_cell(&cell, &mut query_params)?;
        }
    }

    let mut pair_index = 0;
    while let Some(name_value) = queue.pop_front() {
        let field = match prepared_names
            .pair_fields
            .get(pair_index)
            .and_then(Clone::clone)
        {
            Some(field) => field,
            None => RequestOptionName::decode(&expect_string_scalar(
                &name_value,
                "webwrite: parameter names must be character vectors or strings",
            )?),
        };
        pair_index += 1;
        let value = queue.pop_front().ok_or_else(|| {
            webwrite_error_with(
                &WEBWRITE_ERROR_MISSING_OPTION_VALUE,
                "webwrite: missing value for name-value argument",
            )
        })?;
        process_name_value_pair(&field, &value, &mut options, &mut query_params)?;
    }

    Ok((options, query_params))
}

fn process_struct_fields(
    struct_value: &StructValue,
    prepared_api_key: Option<&super::api_key::PreparedApiKey>,
    prepared_names: &PreparedRequestNames,
    options: &mut WebWriteOptions,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    if let Some(prepared) = prepared_api_key {
        options.api_key_header = prepared.finish(struct_value).map_err(|err| {
            webwrite_error_with_source(
                &WEBWRITE_ERROR_INVALID_OPTION_VALUE,
                format!("webwrite: {err}"),
                err,
            )
        })?;
    }
    for (key, value) in &struct_value.fields {
        if prepared_api_key.is_some_and(|prepared| prepared.contains_field(key)) {
            continue;
        }
        let field = prepared_names
            .struct_fields
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, field)| field.clone())
            .ok_or_else(|| {
                webwrite_error("webwrite: prepared option field changed during gather")
            })?;
        process_name_value_pair(&field, value, options, query_params)?;
    }
    Ok(())
}

fn process_name_value_pair(
    name: &RequestOptionName,
    value: &Value,
    options: &mut WebWriteOptions,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    match name {
        RequestOptionName::Field(WebOptionField::ContentType) => {
            let ct = parse_content_type(value)?;
            options.content_type = ct;
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::MediaType) => {
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
        RequestOptionName::Field(WebOptionField::Timeout) => {
            options.timeout = parse_timeout(value)?;
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::HeaderFields) => {
            let headers = parse_header_fields(value)?;
            options.headers.extend(headers);
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::UserAgent) => {
            options.user_agent = Some(expect_string_scalar(
                value,
                "webwrite: UserAgent must be a character vector or string scalar",
            )?);
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::Username) => {
            options.username = Some(expect_string_scalar(
                value,
                "webwrite: Username must be a character vector or string scalar",
            )?);
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::Password) => {
            options.password = Some(expect_string_scalar(
                value,
                "webwrite: Password must be a character vector or string scalar",
            )?);
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::RequestMethod) => {
            options.method = parse_request_method(value)?;
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::QueryParameters) => {
            append_query_from_value(value, query_params)
        }
        RequestOptionName::Query(name) => {
            let param_value = value_to_query_string(value, name)?;
            query_params.push((name.to_string(), param_value));
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::KeyName | WebOptionField::KeyValue) => Ok(()),
    }
}

async fn execute_request(
    mut url: Url,
    options: WebWriteOptions,
    query_params: &[(String, String)],
    body: PreparedBody,
    context: Option<&crate::context::RuntimeContext>,
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
        return Err(webwrite_error_with(
            &WEBWRITE_ERROR_INVALID_CREDENTIALS,
            "webwrite: Password requires a Username option",
        ));
    }

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
    if let Some((name, value)) = &options.api_key_header {
        headers.push((name.clone(), value.clone()));
    }
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

    let response = transport::send_request(request, context)
        .await
        .map_err(|err| match err {
            transport::TransportCallError::Transport(err) => webwrite_error_with_source(
                &WEBWRITE_ERROR_TRANSPORT,
                err.message_with_prefix("webwrite"),
                err,
            ),
            transport::TransportCallError::Runtime(err) => err,
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
            let tensor = Tensor::new(data, vec![1, cols]).map_err(|err| {
                webwrite_error_with(&WEBWRITE_ERROR_OUTPUT, format!("webwrite: {err}"))
            })?;
            Ok(Value::Tensor(tensor))
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
                query_params.push((key.to_owned(), text));
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

#[cfg(test)]
#[path = "webwrite/tests.rs"]
pub(crate) mod tests;
