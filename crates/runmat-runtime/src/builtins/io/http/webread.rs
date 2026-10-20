//! MATLAB-compatible `webread` builtin for HTTP/HTTPS downloads.

#[path = "webread/options.rs"]
mod options;
#[path = "webread/preflight.rs"]
mod preflight;
use options::*;
use preflight::*;

use runmat_value::NumericScalar;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
use base64::Engine;
use runmat_builtins::{
    parse_http_url, BuiltinErrorDescriptor, HttpResponseDialect, HttpResponseMode, WebOptionField,
    WEBREAD_DESCRIPTOR, WEBREAD_ERROR_FLOW, WEBREAD_ERROR_INVALID_ARGUMENT,
    WEBREAD_ERROR_INVALID_CREDENTIALS, WEBREAD_ERROR_INVALID_OPTION_VALUE,
    WEBREAD_ERROR_INVALID_URL, WEBREAD_ERROR_MISSING_OPTION_VALUE, WEBREAD_ERROR_OUTPUT,
    WEBREAD_ERROR_OUTPUTS, WEBREAD_ERROR_RESPONSE_JSON, WEBREAD_ERROR_TRANSPORT,
    WEBREAD_EXPLICIT_GPU_EXTENSION as EXPLICIT_GPU_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, IntegerStorage, StructValue, Tensor, Value};
use url::Url;

use super::request_options::{PreparedRequestNames, RequestOptionName};
use super::transport::{
    self, decode_body_as_text, header_value, HttpMethod, HttpRequest, HEADER_CONTENT_TYPE,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::io::json::jsondecode::decode_json_text;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT_SECONDS: f64 = 5.0;
const MAX_TIMEOUT_SECONDS: f64 = 2147.483647;
const DEFAULT_USER_AGENT: &str = "RunMat webread/0.0";
const BUILTIN_NAME: &str = "webread";

fn webread_error(message: impl Into<String>) -> RuntimeError {
    webread_error_with(&WEBREAD_ERROR_INVALID_ARGUMENT, message)
}

fn webread_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn webread_error_with_source<E>(
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

fn remap_webread_flow<F>(
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

fn webread_flow_with_context(err: RuntimeError) -> RuntimeError {
    remap_webread_flow(&WEBREAD_ERROR_FLOW, err, |err| {
        format!("webread: {}", err.message())
    })
}

#[runtime_builtin(
    name = "webread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::http::webread"
)]
async fn webread_builtin(
    context: Option<crate::context::RuntimeContext>,
    url: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &WEBREAD_DESCRIPTOR,
        &WEBREAD_ERROR_OUTPUTS,
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
        "webread: URL must be a character vector or string scalar",
    )?;
    if url_text.trim().is_empty() {
        return Err(webread_error_with(
            &WEBREAD_ERROR_INVALID_URL,
            "webread: URL must not be empty",
        ));
    }
    let prepared_api_key = match rest.first() {
        Some(Value::Struct(source)) => Some(super::api_key::PreparedApiKey::parse(source)),
        _ => None,
    };
    let prepared_names = preflight_host_arguments(rest.view(), prepared_api_key.as_ref())?;
    let parsed_url = parse_http_url(&url_text).map_err(|err| {
        webread_error_with_source(
            &WEBREAD_ERROR_INVALID_URL,
            format!("webread: invalid URL: {err}"),
            err,
        )
    })?;
    let mut gathered_args = gather_arguments(&rest, context.as_ref()).await?;
    let (options, query_params) = parse_arguments(
        &mut gathered_args,
        prepared_api_key.as_ref(),
        &prepared_names,
    )?;
    execute_request(parsed_url, options, &query_params, context.as_ref()).await
}

async fn gather_arguments(
    values: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    values
        .try_map_preserving_roots_async(values.len(), runtime, |index| async move {
            gather_if_needed_async(&values[index])
                .await
                .map_err(webread_flow_with_context)
        })
        .await
}

fn parse_arguments(
    queue: &mut crate::call::BuiltinArguments,
    prepared_api_key: Option<&super::api_key::PreparedApiKey>,
    prepared_names: &PreparedRequestNames,
) -> BuiltinResult<(WebReadOptions, Vec<(String, String)>)> {
    let mut options = WebReadOptions::default();
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
            append_query_from_cell(&cell, &mut query_params)?
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
                "webread: parameter names must be character vectors or string scalars",
            )?),
        };
        pair_index += 1;
        let value = queue.pop_front().ok_or_else(|| {
            webread_error_with(
                &WEBREAD_ERROR_MISSING_OPTION_VALUE,
                "webread: missing value for name-value argument",
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
    options: &mut WebReadOptions,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    if let Some(prepared) = prepared_api_key {
        options.api_key_header = prepared.finish(struct_value).map_err(|err| {
            webread_error_with_source(
                &WEBREAD_ERROR_INVALID_OPTION_VALUE,
                format!("webread: {err}"),
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
            .ok_or_else(|| webread_error("webread: prepared option field changed during gather"))?;
        process_name_value_pair(&field, value, options, query_params)?;
    }
    Ok(())
}

fn process_name_value_pair(
    name: &RequestOptionName,
    value: &Value,
    options: &mut WebReadOptions,
    query_params: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    match name {
        RequestOptionName::Field(WebOptionField::ContentType) => {
            options.content_type = parse_content_type(value)?;
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
                "webread: UserAgent must be a character vector or string scalar",
            )?);
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::Username) => {
            options.username = Some(expect_string_scalar(
                value,
                "webread: Username must be a character vector or string scalar",
            )?);
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::Password) => {
            options.password = Some(expect_string_scalar(
                value,
                "webread: Password must be a character vector or string scalar",
            )?);
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::RequestMethod) => {
            options.method = parse_request_method(value)?;
            Ok(())
        }
        RequestOptionName::Field(WebOptionField::MediaType) => {
            // weboptions exposes MediaType for webwrite; accept and ignore for webread.
            expect_string_scalar(
                value,
                "webread: MediaType must be a character vector or string scalar",
            )?;
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
        _ => Err(webread_error_with(
            &WEBREAD_ERROR_INVALID_OPTION_VALUE,
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

async fn execute_request(
    mut url: Url,
    options: WebReadOptions,
    query_params: &[(String, String)],
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
        return Err(webread_error_with(
            &WEBREAD_ERROR_INVALID_CREDENTIALS,
            "webread: Password requires a Username option",
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

    let request = HttpRequest {
        url,
        method: HttpMethod::Get,
        headers,
        body: None,
        timeout: options.timeout,
        user_agent,
    };

    let response = transport::send_request(request, context)
        .await
        .map_err(|err| match err {
            transport::TransportCallError::Transport(err) => webread_error_with_source(
                &WEBREAD_ERROR_TRANSPORT,
                err.message_with_prefix("webread"),
                err,
            ),
            transport::TransportCallError::Runtime(err) => err,
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
            let cols = response.body.len();
            let tensor = Tensor::new_integer(IntegerStorage::U8(response.body), vec![1, cols])
                .map_err(|err| {
                    webread_error_with(&WEBREAD_ERROR_OUTPUT, format!("webread: {err}"))
                })?;
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
    webread_error_with_source(&WEBREAD_ERROR_RESPONSE_JSON, message, err)
}

#[cfg(test)]
#[path = "webread/tests.rs"]
pub(crate) mod tests;
