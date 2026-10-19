//! Browser Fetch head admission. Lifecycle and body reading remain one owner.

mod body;
mod lifecycle;

use js_sys::{Array, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Headers, ReadableStreamDefaultReader, Request, RequestInit, Response};

use self::body::BrowserBody;
use self::lifecycle::{active_failure, fetch_error, ActiveFetch, HostScope};
use super::BrowserJsError;
use crate::http::{
    HttpBodyStream, HttpRequest, HttpResponseHead, TransportError, TransportErrorKind,
};

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
        .map_err(|source| js_error(&request.url, source.into()))?;
    Ok(HttpBodyStream::new(
        head,
        Box::pin(BrowserBody::new(
            active,
            reader,
            accepted_limit,
            request.url,
        )),
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

#[cfg(test)]
#[path = "stream/tests.rs"]
mod tests;
