use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use futures::Stream;
use js_sys::{Array, Promise, Reflect, Uint8Array};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AbortController, Headers, ReadableStreamDefaultReader, Request, RequestInit, Response, Window,
    WorkerGlobalScope,
};

use super::BrowserJsError;
use crate::http::{
    HttpBodyStream, HttpRequest, HttpResponseHead, TransportError, TransportErrorKind,
};

enum HostScope {
    Window(Window),
    Worker(WorkerGlobalScope),
}

impl HostScope {
    fn current(url: &url::Url) -> Result<Self, TransportError> {
        if let Some(window) = web_sys::window() {
            return Ok(Self::Window(window));
        }
        js_sys::global()
            .dyn_into::<WorkerGlobalScope>()
            .map(Self::Worker)
            .map_err(|_| {
                TransportError::new(
                    TransportErrorKind::UnsupportedStreaming,
                    url,
                    "the host is neither Window nor WorkerGlobalScope",
                )
            })
    }

    fn fetch(&self, request: &Request) -> Promise {
        match self {
            Self::Window(window) => window.fetch_with_request(request),
            Self::Worker(worker) => worker.fetch_with_request(request),
        }
    }

    fn set_timeout(&self, callback: &js_sys::Function, timeout_ms: i32) -> Result<i32, JsValue> {
        match self {
            Self::Window(window) => {
                window.set_timeout_with_callback_and_timeout_and_arguments_0(callback, timeout_ms)
            }
            Self::Worker(worker) => {
                worker.set_timeout_with_callback_and_timeout_and_arguments_0(callback, timeout_ms)
            }
        }
    }

    fn clear_timeout(&self, handle: i32) {
        match self {
            Self::Window(window) => window.clear_timeout_with_handle(handle),
            Self::Worker(worker) => worker.clear_timeout_with_handle(handle),
        }
    }
}

/// The active Fetch and timer have one owner before and after the response
/// head. Dropping a pending open or body aborts the same request generation.
struct ActiveFetch {
    scope: HostScope,
    controller: AbortController,
    timeout: Option<TimeoutLease>,
    timed_out: Rc<Cell<bool>>,
    timeout_setup_error: Rc<RefCell<Option<JsValue>>>,
    completed: bool,
}

impl ActiveFetch {
    fn new(scope: HostScope, url: &url::Url) -> Result<Self, TransportError> {
        let controller = AbortController::new().map_err(|source| js_error(url, source))?;
        Ok(Self {
            scope,
            controller,
            timeout: None,
            timed_out: Rc::new(Cell::new(false)),
            timeout_setup_error: Rc::new(RefCell::new(None)),
            completed: false,
        })
    }

    fn arm_timeout(&mut self, request: &HttpRequest) -> Result<(), TransportError> {
        // Match the existing XHR behavior: a zero-millisecond timeout means
        // no browser timer. Clamp at u32::MAX, then arm signed-i32-safe
        // segments so admitted long timeouts are not shortened to 24.8 days.
        let milliseconds = request.timeout.as_millis().min(u32::MAX as u128) as u64;
        if milliseconds == 0 {
            return Ok(());
        }
        self.timeout = Some(TimeoutLease::new(
            self.scope.clone(),
            self.controller.clone(),
            Rc::clone(&self.timed_out),
            Rc::clone(&self.timeout_setup_error),
            milliseconds,
            &request.url,
        )?);
        Ok(())
    }

    fn stop_timer(&mut self) {
        self.timeout.take();
    }

    fn complete(&mut self) {
        self.stop_timer();
        self.completed = true;
    }

    fn abort(&mut self) {
        self.stop_timer();
        if !self.completed {
            self.controller.abort();
            self.completed = true;
        }
    }
}

impl Clone for HostScope {
    fn clone(&self) -> Self {
        match self {
            Self::Window(window) => Self::Window(window.clone()),
            Self::Worker(worker) => Self::Worker(worker.clone()),
        }
    }
}

struct TimeoutState {
    scope: HostScope,
    controller: AbortController,
    timed_out: Rc<Cell<bool>>,
    setup_error: Rc<RefCell<Option<JsValue>>>,
    remaining_ms: u64,
    current_segment_ms: u64,
    handle: Option<i32>,
}

/// Re-arm only when a signed-i32-safe browser timer segment expires. The
/// closure captures a weak self-reference, so dropping the request clears the
/// handle and releases the callback without a retained cycle.
struct TimeoutLease {
    state: Rc<RefCell<TimeoutState>>,
    _callback: Rc<Closure<dyn FnMut()>>,
}

impl TimeoutLease {
    fn new(
        scope: HostScope,
        controller: AbortController,
        timed_out: Rc<Cell<bool>>,
        setup_error: Rc<RefCell<Option<JsValue>>>,
        milliseconds: u64,
        url: &url::Url,
    ) -> Result<Self, TransportError> {
        let state = Rc::new(RefCell::new(TimeoutState {
            scope,
            controller,
            timed_out,
            setup_error,
            remaining_ms: milliseconds,
            current_segment_ms: 0,
            handle: None,
        }));
        let callback_slot = Rc::new(RefCell::new(None::<std::rc::Weak<Closure<dyn FnMut()>>>));
        let callback_state = Rc::clone(&state);
        let callback_weak = Rc::clone(&callback_slot);
        let callback = Rc::new(Closure::<dyn FnMut()>::new(move || {
            let mut state = callback_state.borrow_mut();
            state.handle = None;
            state.remaining_ms -= state.current_segment_ms;
            if state.remaining_ms == 0 {
                state.timed_out.set(true);
                let controller = state.controller.clone();
                drop(state);
                controller.abort();
                return;
            }
            let Some(callback) = callback_weak
                .borrow()
                .as_ref()
                .and_then(std::rc::Weak::upgrade)
            else {
                return;
            };
            if let Err(source) = state.arm(callback.as_ref().as_ref().unchecked_ref()) {
                *state.setup_error.borrow_mut() = Some(source);
                let controller = state.controller.clone();
                drop(state);
                controller.abort();
            }
        }));
        *callback_slot.borrow_mut() = Some(Rc::downgrade(&callback));
        state
            .borrow_mut()
            .arm(callback.as_ref().as_ref().unchecked_ref())
            .map_err(|source| js_error(url, source))?;
        Ok(Self {
            state,
            _callback: callback,
        })
    }
}

impl TimeoutState {
    fn arm(&mut self, callback: &js_sys::Function) -> Result<(), JsValue> {
        let segment = self.remaining_ms.min(i32::MAX as u64);
        let handle = self.scope.set_timeout(callback, segment as i32)?;
        self.current_segment_ms = segment;
        self.handle = Some(handle);
        Ok(())
    }
}

impl Drop for TimeoutLease {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        if let Some(handle) = state.handle.take() {
            state.scope.clear_timeout(handle);
        }
    }
}

impl Drop for ActiveFetch {
    fn drop(&mut self) {
        self.abort();
    }
}

pub(super) async fn open_bounded(
    request: HttpRequest,
    accepted_limit: u64,
) -> Result<HttpBodyStream, TransportError> {
    // Headers and Request construction validate known controls locally before
    // Fetch submission; User-Agent remains browser-controlled as with XHR.
    let headers = Headers::new().map_err(|source| js_error(&request.url, source))?;
    for (name, value) in &request.headers {
        headers.append(name, value).map_err(|source| {
            TransportError::new(
                TransportErrorKind::InvalidHeader(name.clone()),
                &request.url,
                format!("{source:?}"),
            )
            .with_source(BrowserJsError(source))
        })?;
    }
    let scope = HostScope::current(&request.url)?;
    let mut active = ActiveFetch::new(scope, &request.url)?;
    let init = RequestInit::new();
    init.set_method(request.method.as_str());
    init.set_headers_headers(&headers);
    init.set_signal(Some(&active.controller.signal()));
    let body = request.body.as_deref().map(Uint8Array::from);
    init.set_body_opt_u8_array(body.as_ref());
    let fetch_request = Request::new_with_str_and_init(request.url.as_str(), &init)
        .map_err(|source| js_error(&request.url, source))?;
    active.arm_timeout(&request)?;
    let response = JsFuture::from(active.scope.fetch(&fetch_request))
        .await
        .map_err(|source| fetch_error(&request.url, &active, source))?
        .dyn_into::<Response>()
        .map_err(|source| js_error(&request.url, source))?;
    if let Some(error) = active_failure(&request.url, &active) {
        return Err(error);
    }
    if !response.ok() {
        return Err(TransportError::new(
            TransportErrorKind::Status(response.status()),
            &request.url,
            response.status_text(),
        ));
    }
    let response_headers = response.headers();
    let content_length = response_headers
        .get("content-length")
        .map_err(|source| js_error(&request.url, source))?
        .and_then(|text| text.trim().parse::<u64>().ok());
    if content_length.is_some_and(|length| length > accepted_limit) {
        return Err(limit_error(&request.url, accepted_limit));
    }
    let mut header_pairs = Vec::new();
    for entry in response_headers.entries() {
        let entry = entry.map_err(|source| js_error(&request.url, source))?;
        let pair = Array::from(&entry);
        if pair.length() != 2 {
            return Err(js_error(&request.url, entry));
        }
        let (Some(name), Some(value)) = (pair.get(0).as_string(), pair.get(1).as_string()) else {
            return Err(js_error(&request.url, entry));
        };
        header_pairs.push((name, value));
    }
    let head = HttpResponseHead {
        headers: header_pairs,
        content_length,
    };
    let Some(source) = response.body() else {
        if matches!(response.status(), 204 | 205) {
            active.complete();
            return Ok(HttpBodyStream::new(
                head,
                Box::pin(futures::stream::empty()),
            ));
        }
        return Err(TransportError::unsupported_streaming(&request.url));
    };
    let reader = source
        .get_reader()
        .dyn_into::<ReadableStreamDefaultReader>()
        .map_err(|source| js_error(&request.url, source))?;
    Ok(HttpBodyStream::new(
        head,
        Box::pin(BrowserBody {
            active,
            reader,
            pending: None,
            accepted: 0,
            accepted_limit,
            url: request.url,
            finished: false,
        }),
    ))
}

fn limit_error(url: &url::Url, limit: u64) -> TransportError {
    TransportError::new(
        TransportErrorKind::BodyLimitExceeded(limit),
        url,
        "compressed response body is too large",
    )
}

fn js_error(url: &url::Url, source: JsValue) -> TransportError {
    TransportError::new(TransportErrorKind::Other, url, format!("{source:?}"))
        .with_source(BrowserJsError(source))
}

fn fetch_error(url: &url::Url, active: &ActiveFetch, source: JsValue) -> TransportError {
    if let Some(setup_error) = active.timeout_setup_error.borrow_mut().take() {
        return js_error(url, setup_error);
    }
    let kind = if active.timed_out.get() {
        TransportErrorKind::Timeout
    } else {
        TransportErrorKind::Connect
    };
    TransportError::new(kind, url, format!("{source:?}")).with_source(BrowserJsError(source))
}

fn active_failure(url: &url::Url, active: &ActiveFetch) -> Option<TransportError> {
    if let Some(source) = active.timeout_setup_error.borrow_mut().take() {
        return Some(js_error(url, source));
    }
    active.timed_out.get().then(|| {
        TransportError::new(
            TransportErrorKind::Timeout,
            url,
            "browser request timed out",
        )
    })
}

struct BrowserBody {
    active: ActiveFetch,
    reader: ReadableStreamDefaultReader,
    pending: Option<Pin<Box<JsFuture>>>,
    accepted: u64,
    accepted_limit: u64,
    url: url::Url,
    finished: bool,
}

impl BrowserBody {
    fn stop(&mut self) {
        self.finished = true;
        self.pending = None;
        self.active.abort();
        let _ = self.reader.cancel();
    }
}

impl Drop for BrowserBody {
    fn drop(&mut self) {
        if !self.finished {
            self.stop();
        }
    }
}

impl Stream for BrowserBody {
    type Item = Result<Vec<u8>, TransportError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        if let Some(error) = active_failure(&self.url, &self.active) {
            self.stop();
            return Poll::Ready(Some(Err(error)));
        }
        if self.pending.is_none() {
            self.pending = Some(Box::pin(JsFuture::from(self.reader.read())));
        }
        let result = self
            .pending
            .as_mut()
            .expect("pending read")
            .as_mut()
            .poll(cx);
        let value = match result {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Ok(value)) => value,
            Poll::Ready(Err(source)) => {
                let error = fetch_error(&self.url, &self.active, source);
                self.stop();
                return Poll::Ready(Some(Err(error)));
            }
        };
        self.pending = None;
        if let Some(error) = active_failure(&self.url, &self.active) {
            self.stop();
            return Poll::Ready(Some(Err(error)));
        }
        let done = Reflect::get(&value, &JsValue::from_str("done"));
        match done {
            Ok(done) if done.as_bool() == Some(true) => {
                self.finished = true;
                self.active.complete();
                self.reader.release_lock();
                return Poll::Ready(None);
            }
            Ok(done) if done.as_bool() == Some(false) => {}
            Ok(source) | Err(source) => {
                let error = js_error(&self.url, source);
                self.stop();
                return Poll::Ready(Some(Err(error)));
            }
        }
        let chunk = match Reflect::get(&value, &JsValue::from_str("value")) {
            Ok(source) => match source.dyn_into::<Uint8Array>() {
                Ok(chunk) => chunk,
                Err(source) => {
                    let error = js_error(&self.url, source);
                    self.stop();
                    return Poll::Ready(Some(Err(error)));
                }
            },
            Err(source) => {
                let error = js_error(&self.url, source);
                self.stop();
                return Poll::Ready(Some(Err(error)));
            }
        };
        let next = self.accepted.checked_add(u64::from(chunk.length()));
        let Some(next) = next.filter(|next| *next <= self.accepted_limit) else {
            let error = limit_error(&self.url, self.accepted_limit);
            self.stop();
            return Poll::Ready(Some(Err(error)));
        };
        let mut bytes = vec![0u8; chunk.length() as usize];
        chunk.copy_to(&mut bytes);
        self.accepted = next;
        Poll::Ready(Some(Ok(bytes)))
    }
}

#[cfg(test)]
#[path = "stream/tests.rs"]
mod tests;
