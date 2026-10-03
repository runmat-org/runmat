use super::*;
use crate::builtins::io::http::transport::{self, HttpMethod, HttpRequest, TransportCallError};
#[cfg(target_arch = "wasm32")]
use std::io::Cursor;
use url::Url;

pub(crate) async fn extract_remote_archive(
    url: &Url,
    output_folder: &Path,
    runtime: Option<&crate::context::RuntimeContext>,
    provider: std::sync::Arc<dyn vfs::FsProvider>,
) -> BuiltinResult<Vec<String>> {
    let request = HttpRequest {
        url: url.clone(),
        method: HttpMethod::Get,
        headers: Vec::new(),
        body: None,
        timeout: DEFAULT_TIMEOUT,
        user_agent: DEFAULT_USER_AGENT.to_string(),
    };
    // A browser without a bounded streaming host rejects here, before any
    // destination directory or file is created. The service owns the body and
    // aborts it when this invocation future is dropped.
    let mut body = transport::open_bounded_request(request, MAX_COMPRESSED_ARCHIVE_BYTES, runtime)
        .await
        .map_err(map_transport)?;
    if let Some(length) = body.head.content_length {
        ensure_compressed_size_within_limit(length)?;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let temp = stream_to_temp(&mut body, runtime).await?;
        super::super::prepare_output_folder(output_folder, runtime, provider.as_ref()).await?;
        super::super::check_cancelled(runtime)?;
        super::super::extract_archive_file_on_worker(
            temp,
            output_folder.to_path_buf(),
            runtime.map(crate::context::RuntimeContext::cancellation),
            provider,
        )
        .await
    }

    #[cfg(target_arch = "wasm32")]
    {
        let bytes = stream_to_memory(&mut body, runtime).await?;
        super::super::prepare_output_folder(output_folder, runtime, provider.as_ref()).await?;
        super::super::check_cancelled(runtime)?;
        super::super::extract_archive(Cursor::new(bytes), output_folder, runtime, provider).await
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn stream_to_temp(
    body: &mut crate::http::HttpBodyStream,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<TempArchive> {
    let mut temp = TempArchive::create()?;
    let mut written = 0u64;
    while let Some(chunk) = transport::next_body_chunk(body, runtime)
        .await
        .map_err(map_transport)?
    {
        written = checked_downloaded_bytes(written, chunk.len())?;
        for part in chunk.chunks(64 * 1024) {
            super::super::check_cancelled(runtime)?;
            temp.write_all(part).map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!(
                        "unzip: unable to write temporary archive '{}': {err}",
                        temp.path().display()
                    ),
                    err,
                )
            })?;
        }
    }
    super::super::check_cancelled(runtime)?;
    temp.flush().map_err(|err| {
        unzip_error_with_source(
            &UNZIP_ERROR_IO,
            format!(
                "unzip: unable to flush temporary archive '{}': {err}",
                temp.path().display()
            ),
            err,
        )
    })?;
    temp.close();
    Ok(temp)
}

#[cfg(target_arch = "wasm32")]
async fn stream_to_memory(
    body: &mut crate::http::HttpBodyStream,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut written = 0u64;
    while let Some(chunk) = transport::next_body_chunk(body, runtime)
        .await
        .map_err(map_transport)?
    {
        written = checked_downloaded_bytes(written, chunk.len())?;
        bytes.try_reserve(chunk.len()).map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_HTTP,
                "unzip: archive buffer allocation is unavailable",
                err,
            )
        })?;
        bytes.extend_from_slice(&chunk);
    }
    super::super::check_cancelled(runtime)?;
    Ok(bytes)
}

fn checked_downloaded_bytes(previous: u64, next_chunk: usize) -> BuiltinResult<u64> {
    let chunk = u64::try_from(next_chunk).map_err(|_| {
        unzip_error_with(&UNZIP_ERROR_HTTP, "unzip: downloaded archive is too large")
    })?;
    let accepted = previous.checked_add(chunk).ok_or_else(|| {
        unzip_error_with(&UNZIP_ERROR_HTTP, "unzip: downloaded archive is too large")
    })?;
    ensure_compressed_size_within_limit(accepted)?;
    Ok(accepted)
}

fn map_transport(error: TransportCallError) -> RuntimeError {
    match error {
        TransportCallError::Transport(error) => {
            unzip_error_with_source(&UNZIP_ERROR_HTTP, error.message_with_prefix("unzip"), error)
        }
        TransportCallError::Runtime(error) => error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumulative_download_cap_rejects_before_write() {
        assert_eq!(checked_downloaded_bytes(4, 3).unwrap(), 7);
        let error = checked_downloaded_bytes(MAX_COMPRESSED_ARCHIVE_BYTES, 1).unwrap_err();
        assert_eq!(error.identifier(), UNZIP_ERROR_HTTP.identifier);
        let overflow = checked_downloaded_bytes(u64::MAX, 1).unwrap_err();
        assert_eq!(overflow.identifier(), UNZIP_ERROR_HTTP.identifier);
    }
}
