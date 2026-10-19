#![cfg(target_arch = "wasm32")]

use std::cell::{Cell, RefCell};
use std::error::Error;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};
use std::time::Duration;

use js_sys::{Array, Object, Promise, Reflect};
use runmat_runtime::context::{RuntimeContext, RuntimeHttpService};
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::http::{BrowserHttpService, HttpMethod, HttpRequest, TransportErrorKind};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::{AbortSignal, ReadableStream, Request, Response, WorkerGlobalScope};

wasm_bindgen_test_configure!(run_in_worker);

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn worker_scope() -> WorkerGlobalScope {
    assert!(web_sys::window().is_none(), "test must execute in a Worker");
    js_sys::global()
        .dyn_into::<WorkerGlobalScope>()
        .expect("WorkerGlobalScope")
}

fn blob_request(text: &str) -> (HttpRequest, String) {
    let parts = Array::new();
    parts.push(&JsValue::from_str(text));
    let blob = web_sys::Blob::new_with_str_sequence(&parts).expect("worker blob");
    let url = web_sys::Url::create_object_url_with_blob(&blob).expect("worker blob URL");
    (
        HttpRequest {
            url: url::Url::parse(&url).expect("parse blob URL"),
            method: HttpMethod::Get,
            headers: Vec::new(),
            body: None,
            timeout: Duration::from_secs(5),
            user_agent: "RunMat worker test".into(),
        },
        url,
    )
}

/// A Worker-only wire fixture. Restore the original property, including
/// inherited-vs-own status, even if an assertion unwinds.
struct FetchOverride {
    global: Object,
    original: JsValue,
    had_own: bool,
    _callback: Closure<dyn FnMut(Request) -> Promise>,
}

impl FetchOverride {
    fn install(callback: Closure<dyn FnMut(Request) -> Promise>) -> Self {
        let global = worker_scope().unchecked_into::<Object>();
        let key = JsValue::from_str("fetch");
        let original = Reflect::get(&global, &key).expect("original Worker fetch");
        let had_own = Object::has_own(&global, &key);
        assert!(
            Reflect::set(&global, &key, callback.as_ref()).expect("set Worker fetch"),
            "Worker fetch must be replaceable for this fixture"
        );
        Self {
            global,
            original,
            had_own,
            _callback: callback,
        }
    }
}

impl Drop for FetchOverride {
    fn drop(&mut self) {
        let key = JsValue::from_str("fetch");
        let restored = if self.had_own {
            Reflect::set(&self.global, &key, &self.original)
        } else {
            Reflect::delete_property(&self.global, &key)
        };
        assert!(matches!(restored, Ok(true)), "restore Worker fetch");
    }
}

#[wasm_bindgen_test(async)]
async fn worker_global_scope_fetches_bounded_response_stream() {
    let _scope = worker_scope();
    let (request, url) = blob_request("RunMat worker stream");
    let mut stream = BrowserHttpService::new()
        .open_bounded(runtime(), request, 1024)
        .await
        .expect("WorkerGlobalScope response stream");
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next_chunk().await.expect("worker body chunk") {
        bytes.extend_from_slice(&chunk);
    }
    web_sys::Url::revoke_object_url(&url).expect("revoke worker blob URL");
    assert_eq!(bytes, b"RunMat worker stream");
}

#[wasm_bindgen_test(async)]
async fn worker_body_cap_rejects_before_excess_bytes_escape() {
    let _scope = worker_scope();
    let (request, url) = blob_request("five!");
    match BrowserHttpService::new()
        .open_bounded(runtime(), request, 4)
        .await
    {
        Err(error) => assert!(matches!(
            error.kind,
            TransportErrorKind::BodyLimitExceeded(4)
        )),
        Ok(mut body) => {
            let mut accepted = 0usize;
            loop {
                match body.next_chunk().await {
                    Ok(Some(bytes)) => {
                        accepted += bytes.len();
                        assert!(accepted <= 4, "over-limit bytes escaped");
                    }
                    Err(error) => {
                        assert!(matches!(
                            error.kind,
                            TransportErrorKind::BodyLimitExceeded(4)
                        ));
                        break;
                    }
                    Ok(None) => panic!("over-limit Worker response completed"),
                }
            }
        }
    }
    web_sys::Url::revoke_object_url(&url).expect("revoke worker blob URL");
}

#[wasm_bindgen_test]
fn worker_unpolled_and_pending_head_drop_have_one_abortable_fetch() {
    let _scope = worker_scope();
    let (request, url) = blob_request("pending head");
    let calls = Rc::new(Cell::new(0usize));
    let signal = Rc::new(RefCell::new(None::<AbortSignal>));
    let calls_for_fetch = Rc::clone(&calls);
    let signal_for_fetch = Rc::clone(&signal);
    let _override = FetchOverride::install(Closure::new(move |request: Request| {
        calls_for_fetch.set(calls_for_fetch.get() + 1);
        *signal_for_fetch.borrow_mut() = Some(request.signal());
        Promise::new(&mut |_, _| {})
    }));
    drop(BrowserHttpService::new().open_bounded(runtime(), request.clone(), 1024));
    assert_eq!(calls.get(), 0, "unpolled call must not Fetch");
    let mut pending = BrowserHttpService::new().open_bounded(runtime(), request, 1024);
    let waker = futures::task::noop_waker();
    assert!(matches!(
        Pin::as_mut(&mut pending).poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(calls.get(), 1, "one invocation must issue one Fetch");
    let signal = signal.borrow().clone().expect("request signal observed");
    assert!(!signal.aborted());
    drop(pending);
    assert!(signal.aborted(), "pending Worker Fetch must abort on drop");
    assert_eq!(calls.get(), 1, "drop must not replay Fetch");
    web_sys::Url::revoke_object_url(&url).expect("revoke worker blob URL");
}

#[wasm_bindgen_test(async)]
async fn worker_pending_body_drop_aborts_and_releases_reader() {
    let _scope = worker_scope();
    let (request, url) = blob_request("pending body");
    let calls = Rc::new(Cell::new(0usize));
    let signal = Rc::new(RefCell::new(None::<AbortSignal>));
    let stream = ReadableStream::new().expect("pending Worker stream");
    let calls_for_fetch = Rc::clone(&calls);
    let signal_for_fetch = Rc::clone(&signal);
    let stream_for_fetch = stream.clone();
    let _override = FetchOverride::install(Closure::new(move |request: Request| {
        calls_for_fetch.set(calls_for_fetch.get() + 1);
        *signal_for_fetch.borrow_mut() = Some(request.signal());
        let response = Response::new_with_opt_readable_stream(Some(&stream_for_fetch))
            .expect("pending Worker response");
        let response: JsValue = response.into();
        Promise::resolve(&response)
    }));
    let mut body = BrowserHttpService::new()
        .open_bounded(runtime(), request, 1024)
        .await
        .expect("Worker response head");
    let mut next = Box::pin(body.next_chunk());
    let waker = futures::task::noop_waker();
    assert!(matches!(
        next.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    drop(next);
    assert!(stream.locked(), "body read owns Worker stream reader");
    let signal = signal.borrow().clone().expect("request signal observed");
    drop(body);
    assert!(signal.aborted(), "pending Worker body must abort on drop");
    assert!(!stream.locked(), "Worker reader lock must be released");
    assert_eq!(calls.get(), 1, "body drop must not replay Fetch");
    web_sys::Url::revoke_object_url(&url).expect("revoke worker blob URL");
}

#[wasm_bindgen_test(async)]
async fn worker_real_timer_aborts_pending_fetch_with_timeout_source() {
    let _scope = worker_scope();
    let (mut request, url) = blob_request("timed out");
    request.timeout = Duration::from_millis(1);
    let calls = Rc::new(Cell::new(0usize));
    let signal = Rc::new(RefCell::new(None::<AbortSignal>));
    let listener = Rc::new(RefCell::new(None::<Closure<dyn FnMut()>>));
    let calls_for_fetch = Rc::clone(&calls);
    let signal_for_fetch = Rc::clone(&signal);
    let listener_for_fetch = Rc::clone(&listener);
    let _override = FetchOverride::install(Closure::new(move |request: Request| {
        calls_for_fetch.set(calls_for_fetch.get() + 1);
        let fetch_signal = request.signal();
        *signal_for_fetch.borrow_mut() = Some(fetch_signal.clone());
        let listener_for_promise = Rc::clone(&listener_for_fetch);
        Promise::new(&mut move |_, reject| {
            let on_abort = Closure::<dyn FnMut()>::new(move || {
                let error = web_sys::DomException::new_with_message("Worker Fetch aborted")
                    .expect("DOM exception");
                let _ = reject.call1(&JsValue::UNDEFINED, &error);
            });
            fetch_signal.set_onabort(Some(on_abort.as_ref().unchecked_ref()));
            *listener_for_promise.borrow_mut() = Some(on_abort);
        })
    }));
    let error = BrowserHttpService::new()
        .open_bounded(runtime(), request, 1024)
        .await
        .err()
        .expect("Worker timer must reject pending Fetch");
    let signal = signal.borrow().clone().expect("request signal observed");
    signal.set_onabort(None);
    listener.borrow_mut().take();
    assert_eq!(calls.get(), 1, "timeout must not replay Fetch");
    assert!(signal.aborted(), "real Worker timeout must abort Fetch");
    assert!(matches!(error.kind, TransportErrorKind::Timeout));
    assert!(
        error.source().is_some(),
        "retain Worker DOM rejection cause"
    );
    web_sys::Url::revoke_object_url(&url).expect("revoke worker blob URL");
}
