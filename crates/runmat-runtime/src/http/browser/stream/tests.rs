use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

use super::*;
use crate::context::RuntimeHttpService;
use crate::http::{BrowserHttpService, HttpMethod};

fn blob_request(text: &str) -> (HttpRequest, String) {
    let parts = Array::new();
    parts.push(&JsValue::from_str(text));
    let blob = web_sys::Blob::new_with_str_sequence(&parts).expect("blob");
    let url = web_sys::Url::create_object_url_with_blob(&blob).expect("blob URL");
    (
        HttpRequest {
            url: url::Url::parse(&url).expect("parse blob URL"),
            method: HttpMethod::Get,
            headers: Vec::new(),
            body: None,
            timeout: Duration::from_secs(5),
            user_agent: "RunMat browser stream test".into(),
        },
        url,
    )
}

#[wasm_bindgen_test(async)]
async fn window_fetch_streams_without_whole_body_xhr() {
    assert!(matches!(
        HostScope::current(&url::Url::parse("https://example.invalid/").unwrap()).unwrap(),
        HostScope::Window(_)
    ));
    let (request, url) = blob_request("RunMat bounded browser HTTP");
    let mut response = BrowserHttpService::new()
        .open_bounded(request, 1024)
        .await
        .expect("Fetch response stream");
    let mut bytes = Vec::new();
    while let Some(chunk) = response.next_chunk().await.expect("body chunk") {
        bytes.extend_from_slice(&chunk);
    }
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
    assert_eq!(bytes, b"RunMat bounded browser HTTP");
}

#[wasm_bindgen_test(async)]
async fn browser_body_limit_never_publishes_excess_bytes() {
    let (request, url) = blob_request("five!");
    match BrowserHttpService::new().open_bounded(request, 4).await {
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
                        assert!(accepted <= 4, "excess chunk escaped the limit");
                    }
                    Err(error) => {
                        assert!(matches!(
                            error.kind,
                            TransportErrorKind::BodyLimitExceeded(4)
                        ));
                        break;
                    }
                    Ok(None) => panic!("oversized response completed without rejection"),
                }
            }
        }
    }
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}

#[wasm_bindgen_test(async)]
async fn invalid_header_precedes_fetch_and_preserves_browser_source() {
    let (mut request, url) = blob_request("not fetched");
    request.headers.push(("bad\nname".into(), "value".into()));
    let error = BrowserHttpService::new()
        .open_bounded(request, 1024)
        .await
        .err()
        .expect("invalid header");
    assert!(matches!(error.kind, TransportErrorKind::InvalidHeader(_)));
    assert!(std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<BrowserJsError>())
        .is_some());
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}

#[wasm_bindgen_test]
fn unpolled_browser_stream_has_no_fetch_or_timeout_effect() {
    let (request, url) = blob_request("unpolled");
    let stream = BrowserHttpService::new().open_bounded(request, 1024);
    drop(stream);
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}

#[wasm_bindgen_test]
fn dropping_active_fetch_guard_aborts_its_controller() {
    let (request, url) = blob_request("cancelled");
    let scope = HostScope::current(&request.url).expect("Window host");
    let mut active = ActiveFetch::new(scope, &request.url).expect("abort controller");
    active.arm_timeout(&request).expect("browser timeout");
    let signal = active.controller.signal();
    assert!(!signal.aborted());
    drop(active);
    assert!(signal.aborted());
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}

#[wasm_bindgen_test]
fn dropping_body_owner_aborts_fetch_and_reader() {
    let (request, url) = blob_request("body owner");
    let scope = HostScope::current(&request.url).expect("Window host");
    let active = ActiveFetch::new(scope, &request.url).expect("abort controller");
    let signal = active.controller.signal();
    let response = Response::new_with_opt_str(Some("pending body")).expect("response fixture");
    let reader = response
        .body()
        .expect("readable body")
        .get_reader()
        .dyn_into::<ReadableStreamDefaultReader>()
        .expect("default reader");
    let body = BrowserBody {
        active,
        reader,
        pending: None,
        accepted: 0,
        accepted_limit: 1024,
        url: request.url,
        finished: false,
    };
    assert!(!signal.aborted());
    drop(body);
    assert!(signal.aborted(), "body drop must abort the fetch owner");
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}

#[wasm_bindgen_test]
fn timeout_error_keeps_browser_exception_source() {
    let (request, url) = blob_request("timeout");
    let active = ActiveFetch::new(
        HostScope::current(&request.url).expect("Window host"),
        &request.url,
    )
    .expect("abort controller");
    active.timed_out.set(true);
    let source =
        web_sys::DomException::new_with_message("abort from timeout").expect("DOM exception");
    let error = fetch_error(&request.url, &active, source.into());
    assert!(matches!(error.kind, TransportErrorKind::Timeout));
    assert!(std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<BrowserJsError>())
        .is_some());
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}

#[wasm_bindgen_test]
fn timed_out_body_cannot_publish_ready_bytes() {
    let (request, url) = blob_request("timeout before read");
    let scope = HostScope::current(&request.url).expect("Window host");
    let active = ActiveFetch::new(scope, &request.url).expect("abort controller");
    let signal = active.controller.signal();
    active.timed_out.set(true);
    let response = Response::new_with_opt_str(Some("must not publish")).expect("response fixture");
    let reader = response
        .body()
        .expect("readable body")
        .get_reader()
        .dyn_into::<ReadableStreamDefaultReader>()
        .expect("default reader");
    let mut body = BrowserBody {
        active,
        reader,
        pending: None,
        accepted: 0,
        accepted_limit: 1024,
        url: request.url,
        finished: false,
    };
    let waker = futures::task::noop_waker();
    let Poll::Ready(Some(Err(error))) =
        Pin::new(&mut body).poll_next(&mut Context::from_waker(&waker))
    else {
        panic!("timed-out body must reject before reader poll");
    };
    assert!(matches!(error.kind, TransportErrorKind::Timeout));
    assert!(signal.aborted());
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}

#[wasm_bindgen_test]
fn long_browser_timeout_is_not_shortened_to_signed_timer_limit() {
    let (mut request, url) = blob_request("long timeout");
    request.timeout = Duration::from_millis(i32::MAX as u64 + 2);
    let scope = HostScope::current(&request.url).expect("Window host");
    let mut active = ActiveFetch::new(scope, &request.url).expect("abort controller");
    active.arm_timeout(&request).expect("first timeout segment");
    let state = active.timeout.as_ref().unwrap().state.borrow();
    assert_eq!(state.remaining_ms, i32::MAX as u64 + 2);
    assert_eq!(state.current_segment_ms, i32::MAX as u64);
    drop(state);
    drop(active);
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}

#[wasm_bindgen_test]
fn browser_stream_future_is_lazy_until_first_poll() {
    let (request, url) = blob_request("pending");
    let mut future = BrowserHttpService::new().open_bounded(request, 1024);
    let waker = futures::task::noop_waker();
    let _ = Pin::as_mut(&mut future).poll(&mut Context::from_waker(&waker));
    drop(future);
    web_sys::Url::revoke_object_url(&url).expect("revoke blob URL");
}
