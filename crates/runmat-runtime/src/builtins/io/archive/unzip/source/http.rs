use super::*;
use crate::builtins::io::http::transport::{self, HttpMethod, HttpRequest, TransportCallError};
#[cfg(target_arch = "wasm32")]
use std::io::Cursor;
use url::Url;

pub(crate) async fn extract_remote_archive(
    url: &Url,
    output_folder: &Path,
    runtime: Option<&crate::context::RuntimeContext>,
    command_origin: Option<runmat_execution::profiling::CommandProfilingReceipt>,
    provider: std::sync::Arc<dyn vfs::FsProvider>,
    resource_lease: &ArchiveResourceLease,
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
    check_resource_lease(resource_lease)?;
    let mut body = transport::open_bounded_request(request, MAX_COMPRESSED_ARCHIVE_BYTES, runtime)
        .await
        .map_err(map_transport)?;
    check_resource_lease(resource_lease)?;
    if let Some(length) = body.head.content_length {
        ensure_compressed_size_within_limit(length)?;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let temp = stream_to_temp(&mut body, runtime, resource_lease).await?;
        super::super::prepare_output_folder(
            output_folder,
            runtime,
            provider.as_ref(),
            resource_lease,
        )
        .await?;
        super::super::check_cancelled(runtime)?;
        super::super::extract_archive_file_on_worker(
            temp,
            output_folder.to_path_buf(),
            runtime.map(crate::context::RuntimeContext::cancellation),
            command_origin,
            provider,
            resource_lease.clone(),
        )
        .await
    }

    #[cfg(target_arch = "wasm32")]
    {
        let bytes = stream_to_memory(&mut body, runtime, resource_lease).await?;
        super::super::prepare_output_folder(
            output_folder,
            runtime,
            provider.as_ref(),
            resource_lease,
        )
        .await?;
        super::super::check_cancelled(runtime)?;
        super::super::extract_archive(
            Cursor::new(bytes),
            output_folder,
            runtime,
            command_origin,
            provider,
            resource_lease.clone(),
        )
        .await
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn stream_to_temp(
    body: &mut crate::http::HttpBodyStream,
    runtime: Option<&crate::context::RuntimeContext>,
    resource_lease: &ArchiveResourceLease,
) -> BuiltinResult<TempArchive> {
    check_resource_lease(resource_lease)?;
    let mut temp = TempArchive::create()?;
    let mut written = 0u64;
    loop {
        check_resource_lease(resource_lease)?;
        let next = transport::next_body_chunk(body, runtime)
            .await
            .map_err(map_transport)?;
        check_resource_lease(resource_lease)?;
        let Some(chunk) = next else {
            break;
        };
        written = checked_downloaded_bytes(written, chunk.len())?;
        for part in chunk.chunks(64 * 1024) {
            super::super::check_cancelled(runtime)?;
            check_resource_lease(resource_lease)?;
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
    check_resource_lease(resource_lease)?;
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
    check_resource_lease(resource_lease)?;
    temp.close();
    Ok(temp)
}

#[cfg(target_arch = "wasm32")]
async fn stream_to_memory(
    body: &mut crate::http::HttpBodyStream,
    runtime: Option<&crate::context::RuntimeContext>,
    resource_lease: &ArchiveResourceLease,
) -> BuiltinResult<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut written = 0u64;
    loop {
        check_resource_lease(resource_lease)?;
        let next = transport::next_body_chunk(body, runtime)
            .await
            .map_err(map_transport)?;
        check_resource_lease(resource_lease)?;
        let Some(chunk) = next else {
            break;
        };
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
    check_resource_lease(resource_lease)?;
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

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn revoked_pending_remote_chunk_cannot_start_destination_effects() {
        use crate::context::{RuntimeHttpService, RuntimeServiceFuture, RuntimeServicePorts};
        use crate::http::{HttpBodyStream, HttpResponse, HttpResponseHead, TransportError};
        use std::cell::RefCell;
        use std::future::Future;
        use std::rc::Rc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::task::{Context, Poll};

        struct Lease(Arc<AtomicBool>);
        impl crate::context::host_effect::HostEffectResourceLivenessLease for Lease {
            fn is_current(&self) -> bool {
                self.0.load(Ordering::SeqCst)
            }
        }
        struct Service(RefCell<Option<futures::channel::oneshot::Receiver<Vec<u8>>>>);
        impl RuntimeHttpService for Service {
            fn send(
                &self,
                _: crate::context::RuntimeContext,
                _: HttpRequest,
            ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
                unreachable!("archive uses bounded streaming")
            }
            fn open_bounded(
                &self,
                _: crate::context::RuntimeContext,
                _: HttpRequest,
                _: u64,
            ) -> RuntimeServiceFuture<Result<HttpBodyStream, TransportError>> {
                let receiver = self.0.borrow_mut().take().expect("one body owner");
                Box::pin(async move {
                    let stream =
                        futures::stream::once(
                            async move { Ok(receiver.await.expect("released body")) },
                        );
                    Ok(HttpBodyStream::new(
                        HttpResponseHead {
                            headers: Vec::new(),
                            content_length: None,
                        },
                        Box::pin(stream),
                    ))
                })
            }
        }

        let (release, receiver) = futures::channel::oneshot::channel();
        let service = Rc::new(Service(RefCell::new(Some(receiver))));
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_http(service));
        let provider = Arc::new(vfs::MemoryFsProvider::new());
        let live = Arc::new(AtomicBool::new(true));
        let lease: ArchiveResourceLease = Some(Arc::new(Lease(live.clone())));
        let url = Url::parse("https://example.invalid/archive.zip").unwrap();
        let mut call = Box::pin(extract_remote_archive(
            &url,
            Path::new("/output"),
            Some(&runtime),
            None,
            provider.clone(),
            &lease,
        ));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        live.store(false, Ordering::SeqCst);
        release.send(Vec::new()).unwrap();
        let error = futures::executor::block_on(call).expect_err("revoked resumed body");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:StaleResourceAuthority")
        );
        let destination = futures::executor::block_on(vfs::FsProvider::symlink_metadata(
            provider.as_ref(),
            Path::new("/output"),
        ));
        assert!(
            destination.is_err(),
            "no destination mkdir after revoked chunk"
        );
    }
}
