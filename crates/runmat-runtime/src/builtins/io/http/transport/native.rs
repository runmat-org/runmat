use reqwest::header::{HeaderName, HeaderValue};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use super::{HttpMethod, HttpRequest, HttpResponse, TransportError, TransportErrorKind};
use crate::context::{RuntimeHttpService, RuntimeServiceFuture};

/// The host supplies a reusable Tokio driver; the transport never creates a
/// private runtime or a blocking worker for an individual request.
pub struct NativeHttpService {
    driver: tokio::runtime::Handle,
}

impl NativeHttpService {
    pub fn new(driver: tokio::runtime::Handle) -> Self {
        Self { driver }
    }
}

impl RuntimeHttpService for NativeHttpService {
    fn send(
        &self,
        request: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        let driver = self.driver.clone();
        Box::pin(async move {
            // Spawn only when polled. Dropping this carrier aborts request send
            // and response-body reception rather than detaching a JoinHandle.
            let task = AbortOnDrop(driver.spawn(send_impl(request.clone())));
            task.await.map_err(|error| {
                TransportError::new(TransportErrorKind::Other, &request.url, error.to_string())
            })?
        })
    }
}

struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

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

async fn send_impl(request: HttpRequest) -> Result<HttpResponse, TransportError> {
    let client = reqwest::Client::builder()
        .timeout(request.timeout)
        .user_agent(&request.user_agent)
        .build()
        .map_err(|error| {
            TransportError::new(TransportErrorKind::Other, &request.url, error.to_string())
        })?;
    let mut builder = match request.method {
        HttpMethod::Get => client.get(request.url.clone()),
        HttpMethod::Post => client.post(request.url.clone()),
        HttpMethod::Put => client.put(request.url.clone()),
        HttpMethod::Patch => client.patch(request.url.clone()),
        HttpMethod::Delete => client.delete(request.url.clone()),
    };
    for (name, value) in &request.headers {
        let key = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            TransportError::new(
                TransportErrorKind::InvalidHeader(name.clone()),
                &request.url,
                error.to_string(),
            )
        })?;
        let value = HeaderValue::from_str(value).map_err(|error| {
            TransportError::new(
                TransportErrorKind::InvalidHeader(name.clone()),
                &request.url,
                error.to_string(),
            )
        })?;
        builder = builder.header(key, value);
    }
    if let Some(body) = &request.body {
        builder = builder.body(body.clone());
    }
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

fn request_error(request: &HttpRequest, error: reqwest::Error) -> TransportError {
    let kind = if error.is_timeout() {
        TransportErrorKind::Timeout
    } else if error.is_connect() {
        TransportErrorKind::Connect
    } else {
        TransportErrorKind::Other
    };
    TransportError::new(kind, &request.url, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    struct DropFlag(Arc<AtomicUsize>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn dropping_submitted_task_aborts_instead_of_detaching() {
        let driver = tokio::runtime::Runtime::new().expect("native HTTP test driver");
        let dropped = Arc::new(AtomicUsize::new(0));
        let (started, ready) = std::sync::mpsc::channel();
        let marker = dropped.clone();
        let task = AbortOnDrop(driver.spawn(async move {
            let _guard = DropFlag(marker);
            started.send(()).expect("signal task start");
            futures::future::pending::<()>().await;
        }));
        ready
            .recv_timeout(Duration::from_secs(2))
            .expect("task started");
        drop(task);
        driver.block_on(async {
            tokio::time::timeout(Duration::from_secs(2), async {
                while dropped.load(Ordering::SeqCst) == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("task aborted");
        });
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
