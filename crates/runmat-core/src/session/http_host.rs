use std::rc::Rc;

#[cfg(not(target_arch = "wasm32"))]
use super::host_driver::host_driver;
#[cfg(all(test, not(target_arch = "wasm32")))]
use super::host_driver::HostDriverInitError;
use runmat_runtime::context::RuntimeHttpService;
#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime::context::RuntimeServiceFuture;
#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime::http::{HttpBodyStream, HttpRequest, HttpResponse, TransportError};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct LazyNativeHttpService {
    service: Rc<std::cell::OnceCell<runmat_runtime::http::NativeHttpService>>,
}

#[cfg(not(target_arch = "wasm32"))]
fn session_transport(
    service: &std::cell::OnceCell<runmat_runtime::http::NativeHttpService>,
    request: &HttpRequest,
) -> Result<runmat_runtime::http::NativeHttpService, TransportError> {
    if let Some(transport) = service.get() {
        return Ok(transport.clone());
    }
    let driver = host_driver()
        .map_err(|source| TransportError::host_service_unavailable(&request.url, source))?;
    Ok(service
        .get_or_init(|| runmat_runtime::http::NativeHttpService::new(driver.handle().clone()))
        .clone())
}

#[cfg(not(target_arch = "wasm32"))]
impl RuntimeHttpService for LazyNativeHttpService {
    fn send(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
        request: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        let service = self.service.clone();
        Box::pin(async move {
            let transport = session_transport(&service, &request)?;
            transport.send(runtime, request).await
        })
    }

    fn open_bounded(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
        request: HttpRequest,
        accepted_limit: u64,
    ) -> RuntimeServiceFuture<Result<HttpBodyStream, TransportError>> {
        let service = self.service.clone();
        Box::pin(async move {
            let transport = session_transport(&service, &request)?;
            transport
                .open_bounded(runtime, request, accepted_limit)
                .await
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn default_http_service() -> Rc<dyn RuntimeHttpService> {
    Rc::new(LazyNativeHttpService::default())
}

#[cfg(target_arch = "wasm32")]
pub(super) fn default_http_service() -> Rc<dyn RuntimeHttpService> {
    Rc::new(runmat_runtime::http::BrowserHttpService::new())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use runmat_runtime::context::RuntimeContext;
    use runmat_runtime::execution::RuntimeExecutionService;
    use runmat_runtime::http::HttpMethod;

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn unpolled_host_request_does_not_initialize_session_transport() {
        let service = LazyNativeHttpService::default();
        let request = HttpRequest {
            url: "http://127.0.0.1:1/".parse().unwrap(),
            method: HttpMethod::Get,
            headers: Vec::new(),
            body: None,
            timeout: std::time::Duration::from_secs(1),
            user_agent: "RunMat test".into(),
        };
        drop(service.send(runtime(), request));
        assert!(service.service.get().is_none());
    }

    #[test]
    fn unpolled_stream_does_not_initialize_session_transport() {
        let service = LazyNativeHttpService::default();
        let request = HttpRequest {
            url: "http://127.0.0.1:1/".parse().unwrap(),
            method: HttpMethod::Get,
            headers: Vec::new(),
            body: None,
            timeout: std::time::Duration::from_secs(1),
            user_agent: "RunMat test".into(),
        };
        drop(service.open_bounded(runtime(), request, 1024));
        assert!(service.service.get().is_none());
    }

    #[test]
    fn driver_initialization_error_retains_original_io_cause() {
        use std::error::Error;
        let source = HostDriverInitError(std::sync::Arc::new(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "driver initialization denied",
        )));
        let url = "http://127.0.0.1/".parse().unwrap();
        let error = TransportError::host_service_unavailable(&url, source);
        let original = error
            .source()
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap();
        assert_eq!(original.kind(), std::io::ErrorKind::PermissionDenied);
    }
}
