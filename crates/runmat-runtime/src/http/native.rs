use reqwest::header::{HeaderName, HeaderValue, USER_AGENT};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

use super::{
    HttpBodyStream, HttpMethod, HttpRequest, HttpResponse, TransportError, TransportErrorKind,
};
use crate::context::{RuntimeContext, RuntimeHttpService, RuntimeServiceFuture};

mod stream;

/// The host supplies a reusable Tokio driver; the transport never creates a
/// private runtime or a blocking worker for an individual request.
#[derive(Clone)]
pub struct NativeHttpService {
    driver: tokio::runtime::Handle,
    clients: Arc<OnceLock<Result<reqwest::Client, Arc<reqwest::Error>>>>,
}

impl NativeHttpService {
    pub fn new(driver: tokio::runtime::Handle) -> Self {
        Self {
            driver,
            clients: Arc::new(OnceLock::new()),
        }
    }
}

impl RuntimeHttpService for NativeHttpService {
    fn send(
        &self,
        runtime: RuntimeContext,
        request: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        let driver = self.driver.clone();
        let clients = Arc::clone(&self.clients);
        let profiling = runtime.command_profiling().cloned();
        Box::pin(async move {
            // Spawn only when polled. Dropping this carrier aborts request send
            // and response-body reception rather than detaching a JoinHandle.
            let url = request.url.clone();
            let task = AbortOnDrop(crate::execution::native_task::spawn(
                &driver,
                send_impl(request, clients),
                profiling,
            ));
            task.await.map_err(|error| {
                TransportError::new(TransportErrorKind::Other, &url, error.to_string())
                    .with_source(error)
            })?
        })
    }

    fn open_bounded(
        &self,
        runtime: RuntimeContext,
        request: HttpRequest,
        accepted_limit: u64,
    ) -> RuntimeServiceFuture<Result<HttpBodyStream, TransportError>> {
        stream::open_bounded(
            self.driver.clone(),
            Arc::clone(&self.clients),
            request,
            accepted_limit,
            runtime.command_profiling().cloned(),
        )
    }
}

#[derive(Debug)]
struct SharedClientInitError(Arc<reqwest::Error>);

impl fmt::Display for SharedClientInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0.as_ref(), formatter)
    }
}

impl std::error::Error for SharedClientInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

pub(super) struct AbortOnDrop<T>(pub(super) tokio::task::JoinHandle<T>);

impl<T> Future for AbortOnDrop<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.get_mut().0).poll(context)
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn send_impl(
    mut request: HttpRequest,
    clients: Arc<OnceLock<Result<reqwest::Client, Arc<reqwest::Error>>>>,
) -> Result<HttpResponse, TransportError> {
    let builder = prepare_request(&mut request, &clients)?;
    let response = builder
        .send()
        .await
        .map_err(|error| request_error(&request, error))?;
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
        .collect();
    let body = response
        .bytes()
        .await
        .map_err(|error| request_error(&request, error))?
        .to_vec();
    Ok(HttpResponse { headers, body })
}

fn prepare_request(
    request: &mut HttpRequest,
    clients: &Arc<OnceLock<Result<reqwest::Client, Arc<reqwest::Error>>>>,
) -> Result<reqwest::RequestBuilder, TransportError> {
    // The previous per-request ClientBuilder validated User-Agent before
    // HeaderFields. Preserve that diagnostic order before initializing the
    // shared client or opening a socket.
    let user_agent = HeaderValue::from_str(&request.user_agent).map_err(|error| {
        TransportError::new(TransportErrorKind::Other, &request.url, error.to_string())
            .with_source(error)
    })?;
    let mut headers = Vec::with_capacity(request.headers.len());
    let mut explicit_user_agent = false;
    for (name, value) in &request.headers {
        let key = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            TransportError::new(
                TransportErrorKind::InvalidHeader(name.clone()),
                &request.url,
                error.to_string(),
            )
            .with_source(error)
        })?;
        let value = HeaderValue::from_str(value).map_err(|error| {
            TransportError::new(
                TransportErrorKind::InvalidHeader(name.clone()),
                &request.url,
                error.to_string(),
            )
            .with_source(error)
        })?;
        explicit_user_agent |= key == USER_AGENT;
        headers.push((key, value));
    }
    let client = clients
        .get_or_init(|| reqwest::Client::builder().build().map_err(Arc::new))
        .as_ref()
        .map_err(|error| {
            TransportError::new(TransportErrorKind::Other, &request.url, error.to_string())
                .with_source(SharedClientInitError(Arc::clone(error)))
        })?;
    let mut builder = match request.method {
        HttpMethod::Get => client.get(request.url.clone()),
        HttpMethod::Post => client.post(request.url.clone()),
        HttpMethod::Put => client.put(request.url.clone()),
        HttpMethod::Patch => client.patch(request.url.clone()),
        HttpMethod::Delete => client.delete(request.url.clone()),
    };
    builder = builder.timeout(request.timeout);
    if !explicit_user_agent {
        builder = builder.header(USER_AGENT, user_agent);
    }
    for (key, value) in headers {
        builder = builder.header(key, value);
    }
    if let Some(body) = request.body.take() {
        builder = builder.body(body);
    }
    Ok(builder)
}

pub(super) fn request_error(request: &HttpRequest, error: reqwest::Error) -> TransportError {
    let kind = if error.is_timeout() {
        TransportErrorKind::Timeout
    } else if error.is_connect() {
        TransportErrorKind::Connect
    } else {
        TransportErrorKind::Other
    };
    TransportError::new(kind, &request.url, error.to_string()).with_source(error)
}

#[cfg(test)]
#[path = "native/tests.rs"]
mod tests;
