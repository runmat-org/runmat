use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use futures::channel::oneshot;
use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{DomException, XmlHttpRequest, XmlHttpRequestResponseType};

use super::{HttpBodyStream, HttpRequest, HttpResponse, TransportError, TransportErrorKind};
use crate::context::{RuntimeHttpService, RuntimeServiceFuture};

mod stream;

/// Async XHR is supported in both Window and WebWorker. The browser owns the
/// event loop; this service owns only one request's callbacks and abort guard.
#[derive(Default)]
pub struct BrowserHttpService;

/// Retain the browser exception object on the browser-local error chain.
#[derive(Debug)]
struct BrowserJsError(JsValue);

impl fmt::Display for BrowserJsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

impl std::error::Error for BrowserJsError {}

impl BrowserHttpService {
    pub fn new() -> Self {
        Self
    }
}

impl RuntimeHttpService for BrowserHttpService {
    fn send(
        &self,
        _runtime: crate::context::RuntimeContext,
        request: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        Box::pin(async move { send_impl(request).await })
    }

    fn open_bounded(
        &self,
        _runtime: crate::context::RuntimeContext,
        request: HttpRequest,
        accepted_limit: u64,
    ) -> RuntimeServiceFuture<Result<HttpBodyStream, TransportError>> {
        Box::pin(async move { stream::open_bounded(request, accepted_limit).await })
    }
}

#[derive(Clone, Copy)]
enum Outcome {
    Load,
    Network,
    Timeout,
    Abort,
}

struct ActiveRequest {
    xhr: XmlHttpRequest,
    onload: Closure<dyn FnMut()>,
    onerror: Closure<dyn FnMut()>,
    ontimeout: Closure<dyn FnMut()>,
    onabort: Closure<dyn FnMut()>,
    completed: bool,
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.xhr.set_onload(None);
        self.xhr.set_onerror(None);
        self.xhr.set_ontimeout(None);
        self.xhr.set_onabort(None);
        if !self.completed {
            let _ = self.xhr.abort();
        }
    }
}

async fn send_impl(request: HttpRequest) -> Result<HttpResponse, TransportError> {
    let xhr = XmlHttpRequest::new().map_err(|error| map_js_error(&request, error))?;
    xhr.open_with_async(request.method.as_str(), request.url.as_str(), true)
        .map_err(|error| map_js_error(&request, error))?;
    xhr.set_response_type(XmlHttpRequestResponseType::Arraybuffer);
    xhr.set_timeout(request.timeout.as_millis().min(u32::MAX as u128) as u32);
    for (name, value) in &request.headers {
        xhr.set_request_header(name, value).map_err(|error| {
            TransportError::new(
                TransportErrorKind::InvalidHeader(name.clone()),
                &request.url,
                format!("{error:?}"),
            )
            .with_source(BrowserJsError(error))
        })?;
    }

    let (sender, receiver) = oneshot::channel();
    let sender = Rc::new(RefCell::new(Some(sender)));
    let callback = |outcome| {
        let sender = sender.clone();
        Closure::<dyn FnMut()>::new(move || {
            if let Some(sender) = sender.borrow_mut().take() {
                let _ = sender.send(outcome);
            }
        })
    };
    let active = ActiveRequest {
        xhr,
        onload: callback(Outcome::Load),
        onerror: callback(Outcome::Network),
        ontimeout: callback(Outcome::Timeout),
        onabort: callback(Outcome::Abort),
        completed: false,
    };
    active
        .xhr
        .set_onload(Some(active.onload.as_ref().unchecked_ref()));
    active
        .xhr
        .set_onerror(Some(active.onerror.as_ref().unchecked_ref()));
    active
        .xhr
        .set_ontimeout(Some(active.ontimeout.as_ref().unchecked_ref()));
    active
        .xhr
        .set_onabort(Some(active.onabort.as_ref().unchecked_ref()));

    if let Some(body) = &request.body {
        active
            .xhr
            .send_with_opt_u8_array(Some(body.as_slice()))
            .map_err(|error| map_js_error(&request, error))?;
    } else {
        active
            .xhr
            .send()
            .map_err(|error| map_js_error(&request, error))?;
    }

    let mut active = active;
    let outcome = receiver.await.map_err(|_| {
        TransportError::new(
            TransportErrorKind::Other,
            &request.url,
            "browser request callback ended",
        )
    })?;
    active.completed = true;
    match outcome {
        Outcome::Network => {
            return Err(TransportError::new(
                TransportErrorKind::Connect,
                &request.url,
                wasm_network_detail("browser network error"),
            ))
        }
        Outcome::Timeout => {
            return Err(TransportError::new(
                TransportErrorKind::Timeout,
                &request.url,
                "browser timeout",
            ))
        }
        Outcome::Abort => {
            return Err(TransportError::new(
                TransportErrorKind::Other,
                &request.url,
                "browser request aborted",
            ))
        }
        Outcome::Load => {}
    }

    let status = active
        .xhr
        .status()
        .map_err(|error| map_js_error(&request, error))?;
    if status == 0 {
        return Err(TransportError::new(
            TransportErrorKind::Connect,
            &request.url,
            wasm_network_detail("browser reported status code 0"),
        ));
    }
    if !(200..=299).contains(&status) {
        let status_text = active.xhr.status_text();
        let error = TransportError::new(
            TransportErrorKind::Status(status),
            &request.url,
            status_text.as_deref().unwrap_or_default(),
        );
        return Err(match status_text {
            Ok(_) => error,
            Err(source) => error.with_source(BrowserJsError(source)),
        });
    }
    let headers = parse_response_headers(
        &active
            .xhr
            .get_all_response_headers()
            .map_err(|error| map_js_error(&request, error))?,
    );
    let response = active
        .xhr
        .response()
        .map_err(|error| map_js_error(&request, error))?;
    let array_buffer: ArrayBuffer = response.dyn_into().map_err(|error| {
        TransportError::new(
            TransportErrorKind::Other,
            &request.url,
            "response was not ArrayBuffer",
        )
        .with_source(BrowserJsError(error))
    })?;
    let view = Uint8Array::new(&array_buffer);
    let mut body = vec![0u8; view.length() as usize];
    view.copy_to(&mut body);
    Ok(HttpResponse { headers, body })
}

fn map_js_error(request: &HttpRequest, error: JsValue) -> TransportError {
    if let Some(dom) = error.dyn_ref::<DomException>() {
        let kind = if dom.name().eq_ignore_ascii_case("TimeoutError") {
            TransportErrorKind::Timeout
        } else if dom.name().eq_ignore_ascii_case("NetworkError") {
            TransportErrorKind::Connect
        } else {
            TransportErrorKind::Other
        };
        return TransportError::new(kind, &request.url, dom.message())
            .with_source(BrowserJsError(error));
    }
    TransportError::new(
        TransportErrorKind::Other,
        &request.url,
        format!("{error:?}"),
    )
    .with_source(BrowserJsError(error))
}

fn wasm_network_detail(message: impl AsRef<str>) -> String {
    let message = message.as_ref().trim();
    let suffix = "In browser/WASM, remote URLs must be reachable from the browser and must permit cross-origin access (CORS). Use a CORS-enabled URL, upload/read a local file, or route the image through a same-origin proxy.";
    if message.is_empty() {
        suffix.to_string()
    } else {
        format!("{message}. {suffix}")
    }
}

fn parse_response_headers(raw: &str) -> Vec<(String, String)> {
    raw.lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ':');
            let name = parts.next()?.trim();
            let value = parts.next().unwrap_or("").trim();
            (!name.is_empty()).then(|| (name.to_string(), value.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use std::error::Error;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};
    use wasm_bindgen_test::wasm_bindgen_test;

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    fn blob_request() -> (HttpRequest, String) {
        let parts = js_sys::Array::new();
        parts.push(&JsValue::from_str("RunMat async HTTP"));
        let blob = web_sys::Blob::new_with_str_sequence(&parts).expect("blob");
        let url = web_sys::Url::create_object_url_with_blob(&blob).expect("blob URL");
        let request = HttpRequest {
            url: url::Url::parse(&url).expect("parse blob URL"),
            method: crate::http::HttpMethod::Get,
            headers: Vec::new(),
            body: None,
            timeout: std::time::Duration::from_secs(5),
            user_agent: "RunMat browser test".into(),
        };
        (request, url)
    }

    #[wasm_bindgen_test(async)]
    async fn browser_http_service_fetches_without_synchronous_xhr() {
        let (request, url) = blob_request();
        let response = BrowserHttpService::new()
            .send(runtime(), request)
            .await
            .expect("async XHR");
        web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
        assert_eq!(response.body, b"RunMat async HTTP");
    }

    #[wasm_bindgen_test]
    fn dropping_pending_browser_request_is_safe() {
        let (request, url) = blob_request();
        let mut request = BrowserHttpService::new().send(runtime(), request);
        let waker = futures::task::noop_waker();
        assert!(matches!(
            Pin::as_mut(&mut request).poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        drop(request);
        web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
    }

    #[wasm_bindgen_test]
    fn browser_exception_remains_on_transport_source_chain() {
        let (request, url) = blob_request();
        let exception = DomException::new_with_message("bad header").expect("DOM exception");
        let error = map_js_error(&request, exception.into());
        assert!(error
            .source()
            .and_then(|source| source.downcast_ref::<BrowserJsError>())
            .is_some());
        web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
    }
}
