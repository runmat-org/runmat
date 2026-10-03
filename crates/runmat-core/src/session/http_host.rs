use std::rc::Rc;

use runmat_runtime::context::RuntimeHttpService;
#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime::context::RuntimeServiceFuture;
#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime::http::{HttpRequest, HttpResponse, TransportError};

#[cfg(not(target_arch = "wasm32"))]
struct LazyNativeHttpService;

#[cfg(not(target_arch = "wasm32"))]
impl RuntimeHttpService for LazyNativeHttpService {
    fn send(
        &self,
        request: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        Box::pin(async move {
            let driver = host_driver()
                .map_err(|detail| TransportError::host_service_unavailable(&request.url, detail))?;
            runmat_runtime::http::NativeHttpService::new(driver.handle().clone())
                .send(request)
                .await
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn host_driver() -> Result<&'static tokio::runtime::Runtime, String> {
    use std::sync::OnceLock;

    static DRIVER: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    DRIVER
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn default_http_service() -> Rc<dyn RuntimeHttpService> {
    Rc::new(LazyNativeHttpService)
}

#[cfg(target_arch = "wasm32")]
pub(super) fn default_http_service() -> Rc<dyn RuntimeHttpService> {
    Rc::new(runmat_runtime::http::BrowserHttpService::new())
}
