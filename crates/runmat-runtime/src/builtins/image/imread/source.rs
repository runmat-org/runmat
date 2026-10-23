use super::*;
use runmat_builtins::ImreadSourceRoute;

pub(super) fn string_arg(label: &str, value: &Value) -> BuiltinResult<String> {
    crate::builtins::image::checked_text::scalar_text(value)
        .map_err(|source| {
            let mut error =
                imread_error_with_detail(&IMREAD_ERROR_INVALID_ARGUMENT, source.to_string());
            error.source = Some(Box::new(source));
            error
        })?
        .ok_or_else(|| {
            imread_error_with_detail(
                &IMREAD_ERROR_INVALID_ARGUMENT,
                format!("{label} must be a string scalar or character vector"),
            )
        })
}

pub(super) fn parse_format_hint(value: &str) -> BuiltinResult<ImageFormat> {
    let label = value.trim().trim_start_matches('.').to_ascii_lowercase();
    if label.is_empty() {
        return Err(imread_error_with_detail(
            &IMREAD_ERROR_INVALID_FORMAT,
            "format hint must not be empty",
        ));
    }
    let format = match label.as_str() {
        "jpg" | "jpeg" | "jpe" => ImageFormat::Jpeg,
        "tif" | "tiff" => ImageFormat::Tiff,
        "png" => ImageFormat::Png,
        "bmp" => ImageFormat::Bmp,
        "gif" => ImageFormat::Gif,
        "webp" => ImageFormat::WebP,
        "ico" => ImageFormat::Ico,
        other => ImageFormat::from_extension(other).ok_or_else(|| {
            imread_error_with_detail(
                &IMREAD_ERROR_UNSUPPORTED_FORMAT,
                format!("unsupported image format '{other}'"),
            )
        })?,
    };
    Ok(format)
}

pub(super) async fn read_source_bytes(
    source: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<u8>> {
    match ImreadSourceRoute::parse(source) {
        ImreadSourceRoute::Http(url) => read_url_bytes(url, runtime).await,
        ImreadSourceRoute::FileUrl(url) => {
            let path = file_url_to_path(&url)?;
            read_local_path(&path).await
        }
        ImreadSourceRoute::InvalidFileUrlHost { host } => Err(imread_error_with_detail(
            &IMREAD_ERROR_INVALID_FILE_URL,
            format!("file URL host '{host}' is not local"),
        )),
        ImreadSourceRoute::UnsupportedScheme { scheme } => Err(imread_error_with_detail(
            &IMREAD_ERROR_UNSUPPORTED_SCHEME,
            format!("unsupported URL scheme '{scheme}'"),
        )),
        ImreadSourceRoute::LocalPath => read_local_path(Path::new(source)).await,
    }
}

async fn read_local_path(path: &Path) -> BuiltinResult<Vec<u8>> {
    runmat_filesystem::read_async(path).await.map_err(|err| {
        imread_error_with_source(
            &IMREAD_ERROR_FILE_READ,
            format!("unable to read '{}': {err}", path.display()),
            err,
        )
    })
}

fn file_url_to_path(url: &Url) -> BuiltinResult<PathBuf> {
    let decoded = percent_decode_url_path(url.path())?;

    #[cfg(windows)]
    {
        let path =
            if decoded.len() >= 3 && decoded.as_bytes()[0] == b'/' && decoded.as_bytes()[2] == b':'
            {
                &decoded[1..]
            } else {
                decoded.as_str()
            };
        Ok(PathBuf::from(path))
    }

    #[cfg(not(windows))]
    {
        Ok(PathBuf::from(decoded))
    }
}

fn percent_decode_url_path(input: &str) -> BuiltinResult<String> {
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err(imread_error_with_detail(
                    &IMREAD_ERROR_INVALID_FILE_URL,
                    "invalid percent escape in file URL",
                ));
            }
            let hi = hex_value(bytes[index + 1]).ok_or_else(|| {
                imread_error_with_detail(
                    &IMREAD_ERROR_INVALID_FILE_URL,
                    "invalid percent escape in file URL",
                )
            })?;
            let lo = hex_value(bytes[index + 2]).ok_or_else(|| {
                imread_error_with_detail(
                    &IMREAD_ERROR_INVALID_FILE_URL,
                    "invalid percent escape in file URL",
                )
            })?;
            output.push((hi << 4) | lo);
            index += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }

    String::from_utf8(output).map_err(|err| {
        imread_error_with_source(
            &IMREAD_ERROR_INVALID_FILE_URL,
            format!("file URL path is not valid UTF-8: {err}"),
            err,
        )
    })
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

async fn read_url_bytes(
    url: Url,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<u8>> {
    let request = HttpRequest {
        url,
        method: HttpMethod::Get,
        headers: vec![(
            "Accept".to_string(),
            "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8".to_string(),
        )],
        body: None,
        timeout: Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS),
        user_agent: DEFAULT_USER_AGENT.to_string(),
    };
    let response =
        transport::send_request(request, runtime)
            .await
            .map_err(|error| match error {
                transport::TransportCallError::Transport(error) => imread_transport_error(error),
                transport::TransportCallError::Runtime(error) => error,
            })?;
    Ok(response.body)
}

fn imread_transport_error(err: TransportError) -> RuntimeError {
    let error = match &err.kind {
        TransportErrorKind::Timeout => &IMREAD_ERROR_TIMEOUT,
        TransportErrorKind::Connect => &IMREAD_ERROR_NETWORK,
        TransportErrorKind::Status(_) => &IMREAD_ERROR_HTTP_STATUS,
        TransportErrorKind::InvalidHeader(_) => &IMREAD_ERROR_INVALID_HEADER,
        TransportErrorKind::UnsupportedStreaming
        | TransportErrorKind::BodyLimitExceeded(_)
        | TransportErrorKind::Other => &IMREAD_ERROR_NETWORK,
    };
    let message = err.message_with_prefix(BUILTIN_NAME);
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}
