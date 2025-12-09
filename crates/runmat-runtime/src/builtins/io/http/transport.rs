#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use std::time::Duration;

use encoding_rs::Encoding;
use url::Url;

#[cfg(target_arch = "wasm32")]
use js_sys::{ArrayBuffer, Uint8Array};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use web_sys::{DomException, XmlHttpRequest, XmlHttpRequestResponseType};

pub(crate) const HEADER_CONTENT_TYPE: &str = "content-type";

#[derive(Clone, Copy, Debug)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl HttpMethod {
    #[cfg(target_arch = "wasm32")]
    fn as_str(self) -> &'static str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
            HttpMethod::Put => "PUT",
            HttpMethod::Patch => "PATCH",
            HttpMethod::Delete => "DELETE",
        }
    }
}

#[derive(Clone, Debug)]
pub struct HttpRequest {
    pub url: Url,
    pub method: HttpMethod,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub timeout: Duration,
    pub user_agent: String,
}

#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[derive(Debug)]
pub enum TransportErrorKind {
    Timeout,
    Connect,
    Status(u16),
    InvalidHeader(String),
    Other,
}

#[derive(Debug)]
pub struct TransportError {
    pub kind: TransportErrorKind,
    pub url: String,
    pub detail: String,
}

impl TransportError {
    fn new(kind: TransportErrorKind, url: &Url, detail: impl Into<String>) -> Self {
        Self {
            kind,
            url: url.to_string(),
            detail: detail.into(),
        }
    }

    pub fn into_context(self, prefix: &str) -> String {
        match self.kind {
            TransportErrorKind::Timeout => {
                format!("{prefix}: request to {} timed out", self.url)
            }
            TransportErrorKind::Connect => {
                format!(
                    "{prefix}: unable to connect to {}: {}",
                    self.url, self.detail
                )
            }
            TransportErrorKind::Status(code) => format!(
                "{prefix}: request to {} failed with HTTP status {}",
                self.url, code
            ),
            TransportErrorKind::InvalidHeader(name) => {
                format!("{prefix}: invalid header '{}': {}", name, self.detail)
            }
            TransportErrorKind::Other => {
                format!("{prefix}: failed to contact {}: {}", self.url, self.detail)
            }
        }
    }
}

pub(crate) fn send_request(request: &HttpRequest) -> Result<HttpResponse, TransportError> {
    send_request_impl(request)
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
        if name == "charset" {
            let value = iter
                .next()
                .map(|v| v.trim().trim_matches('"').to_string())?;
            Some(value.to_ascii_lowercase())
        } else {
            None
        }
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn send_request_impl(request: &HttpRequest) -> Result<HttpResponse, TransportError> {
    use reqwest::blocking::Client;
    use reqwest::header::{HeaderName, HeaderValue};

    let client = Client::builder()
        .timeout(request.timeout)
        .user_agent(&request.user_agent)
        .build()
        .map_err(|err| {
            TransportError::new(TransportErrorKind::Other, &request.url, err.to_string())
        })?;

    let mut builder = match request.method {
        HttpMethod::Get => client.get(request.url.clone()),
        HttpMethod::Post => client.post(request.url.clone()),
        HttpMethod::Put => client.put(request.url.clone()),
        HttpMethod::Patch => client.patch(request.url.clone()),
        HttpMethod::Delete => client.delete(request.url.clone()),
    };

    for (name, value) in &request.headers {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|err| {
            TransportError::new(
                TransportErrorKind::InvalidHeader(name.clone()),
                &request.url,
                err.to_string(),
            )
        })?;
        let header_value = HeaderValue::from_str(value).map_err(|err| {
            TransportError::new(
                TransportErrorKind::InvalidHeader(name.clone()),
                &request.url,
                err.to_string(),
            )
        })?;
        builder = builder.header(header_name, header_value);
    }

    if let Some(body) = &request.body {
        builder = builder.body(body.clone());
    }

    let response = builder.send().map_err(|err| {
        if err.is_timeout() {
            TransportError::new(TransportErrorKind::Timeout, &request.url, err.to_string())
        } else if err.is_connect() {
            TransportError::new(TransportErrorKind::Connect, &request.url, err.to_string())
        } else {
            TransportError::new(TransportErrorKind::Other, &request.url, err.to_string())
        }
    })?;

    let status = response.status();
    if !status.is_success() {
        return Err(TransportError::new(
            TransportErrorKind::Status(status.as_u16()),
            &request.url,
            status.canonical_reason().unwrap_or("HTTP error"),
        ));
    }

    let headers = response
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect::<Vec<_>>();

    let body = response
        .bytes()
        .map_err(|err| {
            TransportError::new(TransportErrorKind::Other, &request.url, err.to_string())
        })?
        .to_vec();

    Ok(HttpResponse { headers, body })
}

#[cfg(target_arch = "wasm32")]
fn send_request_impl(request: &HttpRequest) -> Result<HttpResponse, TransportError> {
    let xhr = XmlHttpRequest::new().map_err(|err| {
        TransportError::new(TransportErrorKind::Other, &request.url, format!("{err:?}"))
    })?;

    let async_flag = false;
    let method = request.method.as_str();
    let url_str = request.url.as_str();

    xhr.open_with_async(method, url_str, async_flag)
        .map_err(|err| map_js_error(&request.url, err))?;
    xhr.set_response_type(XmlHttpRequestResponseType::Arraybuffer);

    let timeout_ms = request.timeout.as_millis().min(u32::MAX as u128) as u32;
    xhr.set_timeout(timeout_ms);

    for (name, value) in &request.headers {
        xhr.set_request_header(name, value)
            .map_err(|err| map_js_error(&request.url, err))?;
    }

    if let Some(body) = &request.body {
        xhr.send_with_opt_u8_array(Some(body.as_slice()))
            .map_err(|err| map_js_error(&request.url, err))?;
    } else {
        xhr.send().map_err(|err| map_js_error(&request.url, err))?;
    }

    let status = xhr
        .status()
        .map_err(|err| map_js_error(&request.url, err))? as u16;
    if status == 0 {
        return Err(TransportError::new(
            TransportErrorKind::Connect,
            &request.url,
            "status code 0",
        ));
    }

    let headers_raw = xhr
        .get_all_response_headers()
        .unwrap_or_else(|_| "".to_string());
    let headers = parse_response_headers(&headers_raw);

    if !(200..=299).contains(&status) {
        let status_text = xhr.status_text().unwrap_or_else(|_| String::new());
        return Err(TransportError::new(
            TransportErrorKind::Status(status),
            &request.url,
            status_text,
        ));
    }

    let response = xhr
        .response()
        .map_err(|err| map_js_error(&request.url, err))?;
    let array_buffer: ArrayBuffer = response.dyn_into().map_err(|_| {
        TransportError::new(
            TransportErrorKind::Other,
            &request.url,
            "response was not ArrayBuffer",
        )
    })?;
    let view = Uint8Array::new(&array_buffer);
    let mut body = vec![0u8; view.length() as usize];
    view.copy_to(&mut body);

    Ok(HttpResponse { headers, body })
}

#[cfg(target_arch = "wasm32")]
fn map_js_error(url: &Url, err: JsValue) -> TransportError {
    if let Some(dom) = err.dyn_ref::<DomException>() {
        let name = dom.name();
        if name.eq_ignore_ascii_case("TimeoutError") {
            TransportError::new(TransportErrorKind::Timeout, url, dom.message())
        } else if name.eq_ignore_ascii_case("NetworkError") {
            TransportError::new(TransportErrorKind::Connect, url, dom.message())
        } else {
            TransportError::new(TransportErrorKind::Other, url, dom.message())
        }
    } else {
        TransportError::new(TransportErrorKind::Other, url, format!("{err:?}"))
    }
}

#[cfg(target_arch = "wasm32")]
fn parse_response_headers(raw: &str) -> Vec<(String, String)> {
    raw.lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ':');
            let name = parts.next()?.trim();
            let value = parts.next().unwrap_or("").trim();
            if name.is_empty() {
                None
            } else {
                Some((name.to_string(), value.to_string()))
            }
        })
        .collect()
}
