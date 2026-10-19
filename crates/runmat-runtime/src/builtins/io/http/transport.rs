//! Shared HTTP wire contract and cancellable invocation boundary.

#[cfg(test)]
mod tests;
#[cfg(test)]
mod stream_tests;

use encoding_rs::Encoding;
use futures::future::{select, Either};
use std::future::Future;

use crate::context::RuntimeContext;
use crate::{runtime_error, RuntimeError};

#[cfg(target_arch = "wasm32")]
pub use crate::http::BrowserHttpService;
use crate::http::HttpBodyStream;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::http::NativeHttpService;
pub use crate::http::{HttpMethod, HttpRequest, HttpResponse, TransportError, TransportErrorKind};

pub(crate) const HEADER_CONTENT_TYPE: &str = "content-type";

#[derive(Debug)]
pub enum TransportCallError {
    Transport(TransportError),
    Runtime(RuntimeError),
}

/// The request future has no effects before its first poll. Dropping it after
/// submission drops the host service future, whose contract aborts in-flight
/// I/O (including response-body reception).
pub(crate) async fn send_request(
    request: HttpRequest,
    runtime: Option<&RuntimeContext>,
) -> Result<HttpResponse, TransportCallError> {
    let runtime = require_runtime(runtime)?;
    let service = http_service(runtime)?;
    await_http(service.send(runtime.clone(), request), runtime).await
}

pub(crate) async fn open_bounded_request(
    request: HttpRequest,
    accepted_limit: u64,
    runtime: Option<&RuntimeContext>,
) -> Result<HttpBodyStream, TransportCallError> {
    let runtime = require_runtime(runtime)?;
    let service = http_service(runtime)?;
    await_http(
        service.open_bounded(runtime.clone(), request, accepted_limit),
        runtime,
    )
    .await
}

/// Cancellation drops the body owner, not just the borrowed chunk future.
pub(crate) async fn next_body_chunk(
    body: &mut HttpBodyStream,
    runtime: Option<&RuntimeContext>,
) -> Result<Option<Vec<u8>>, TransportCallError> {
    let runtime = match require_runtime(runtime) {
        Ok(runtime) => runtime,
        Err(error) => {
            body.abort();
            return Err(error);
        }
    };
    let result = await_http(body.next_chunk(), runtime).await;
    if result.is_err() {
        body.abort();
    }
    result
}

fn require_runtime(
    runtime: Option<&RuntimeContext>,
) -> Result<&RuntimeContext, TransportCallError> {
    let runtime = runtime.ok_or_else(|| {
        TransportCallError::Runtime(
            crate::context::RuntimeCapabilityError::new(
                crate::context::RuntimeCapability::Http,
                "HTTP request",
            )
            .into_runtime_error(),
        )
    })?;
    Ok(runtime)
}

fn http_service(
    runtime: &RuntimeContext,
) -> Result<std::rc::Rc<dyn crate::context::RuntimeHttpService>, TransportCallError> {
    if runtime.is_cancelled() {
        return Err(TransportCallError::Runtime(cancelled()));
    }
    runtime
        .service_ports()
        .require_http("HTTP request")
        .map(Clone::clone)
        .map_err(|error| TransportCallError::Runtime(error.into_runtime_error()))
}

async fn await_http<T>(
    future: impl Future<Output = Result<T, TransportError>>,
    runtime: &RuntimeContext,
) -> Result<T, TransportCallError> {
    let cancellation = runtime.cancellation();
    if cancellation.is_cancelled() {
        return Err(TransportCallError::Runtime(cancelled()));
    }
    let response = match select(Box::pin(future), cancellation.cancelled()).await {
        Either::Left((result, _listener)) => result,
        Either::Right((_reason, _request)) => {
            return Err(TransportCallError::Runtime(cancelled()));
        }
    };
    if cancellation.is_cancelled() {
        return Err(TransportCallError::Runtime(cancelled()));
    }
    response.map_err(TransportCallError::Transport)
}

fn cancelled() -> RuntimeError {
    runtime_error::semantic_error("RunMat:Cancelled", "HTTP request was cancelled")
}

pub(crate) fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

pub(crate) fn decode_body_as_text(bytes: &[u8], header: Option<&str>) -> String {
    if let Some(label) = header.and_then(extract_charset) {
        if let Some(encoding) = Encoding::for_label(label.as_bytes()) {
            let (decoded, _, _) = encoding.decode(bytes);
            return decoded.into_owned();
        }
    }
    if let Some((encoding, offset)) = Encoding::for_bom(bytes) {
        let (decoded, _, _) = encoding.decode(&bytes[offset..]);
        return decoded.into_owned();
    }
    String::from_utf8(bytes.to_vec())
        .unwrap_or_else(|_| String::from_utf8_lossy(bytes).into_owned())
}

fn extract_charset(header: &str) -> Option<String> {
    header.split(';').find_map(|part| {
        let mut iter = part.splitn(2, '=');
        let name = iter.next()?.trim().to_ascii_lowercase();
        (name == "charset")
            .then(|| {
                iter.next()
                    .map(|value| value.trim().trim_matches('"').to_ascii_lowercase())
            })
            .flatten()
    })
}

#[cfg(test)]
pub(crate) fn test_runtime_context() -> RuntimeContext {
    use std::rc::Rc;

    use crate::context::RuntimeServicePorts;
    use crate::execution::RuntimeExecutionService;

    #[cfg(not(target_arch = "wasm32"))]
    let service: Rc<dyn crate::context::RuntimeHttpService> = {
        use std::sync::OnceLock;
        static DRIVER: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        let driver =
            DRIVER.get_or_init(|| tokio::runtime::Runtime::new().expect("HTTP test driver"));
        Rc::new(NativeHttpService::new(driver.handle().clone()))
    };
    #[cfg(target_arch = "wasm32")]
    let service: Rc<dyn crate::context::RuntimeHttpService> = Rc::new(BrowserHttpService::new());

    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_http(service))
}
