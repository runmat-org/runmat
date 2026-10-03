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
            let url = request.url.clone();
            let task = AbortOnDrop(driver.spawn(send_impl(request)));
            task.await.map_err(|error| {
                TransportError::new(TransportErrorKind::Other, &url, error.to_string())
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

async fn send_impl(mut request: HttpRequest) -> Result<HttpResponse, TransportError> {
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
    if let Some(body) = request.body.take() {
        builder = builder.body(body);
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
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use url::Url;

    fn request(url: Url) -> HttpRequest {
        HttpRequest {
            url,
            method: HttpMethod::Get,
            headers: Vec::new(),
            body: None,
            timeout: Duration::from_secs(2),
            user_agent: "RunMat HTTP test".into(),
        }
    }

    fn server() -> (TcpListener, Url) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("local HTTP listener");
        let url = Url::parse(&format!(
            "http://{}/resource",
            listener.local_addr().unwrap()
        ))
        .expect("local URL");
        (listener, url)
    }

    #[test]
    fn unpolled_and_invalid_header_requests_do_not_open_a_socket() {
        let driver = tokio::runtime::Runtime::new().expect("native HTTP test driver");
        let service = NativeHttpService::new(driver.handle().clone());
        let (listener, url) = server();
        listener.set_nonblocking(true).unwrap();
        drop(service.send(request(url.clone())));
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );

        let mut invalid = request(url);
        invalid.headers.push(("bad\nname".into(), "value".into()));
        let error = futures::executor::block_on(service.send(invalid)).expect_err("invalid header");
        assert!(matches!(error.kind, TransportErrorKind::InvalidHeader(_)));
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    }

    #[test]
    fn post_preserves_auth_body_response_headers_and_status_error() {
        let driver = tokio::runtime::Runtime::new().expect("native HTTP test driver");
        let service = NativeHttpService::new(driver.handle().clone());
        let (listener, url) = server();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("request");
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0u8; 1024];
            while !bytes.windows(4).any(|part| part == b"\r\n\r\n") || !bytes.ends_with(b"ping") {
                let count = socket.read(&mut buffer).expect("read request");
                assert!(count > 0, "request ended early");
                bytes.extend_from_slice(&buffer[..count]);
            }
            let text = String::from_utf8(bytes).expect("HTTP request text");
            assert!(text.starts_with("POST /resource HTTP/1.1"));
            assert!(text
                .to_ascii_lowercase()
                .contains("authorization: basic dhj1c3rzzwnyzxq="));
            assert!(text.ends_with("ping"));
            socket.write_all(b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\nX-RunMat: yes\r\nConnection: close\r\n\r\nok").unwrap();
        });
        let mut post = request(url);
        post.method = HttpMethod::Post;
        post.headers
            .push(("Authorization".into(), "Basic dHJ1c3RzZWNyZXQ=".into()));
        post.body = Some(b"ping".to_vec());
        let response = futures::executor::block_on(service.send(post)).expect("POST response");
        assert_eq!(response.body, b"ok");
        assert!(response
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("x-runmat") && value == "yes"));
        server.join().unwrap();

        let (listener, url) = server();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("status request");
            let mut buffer = [0u8; 1024];
            let _ = socket.read(&mut buffer);
            socket
                .write_all(
                    b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });
        let error = futures::executor::block_on(service.send(request(url))).expect_err("HTTP 403");
        assert!(matches!(error.kind, TransportErrorKind::Status(403)));
        server.join().unwrap();
    }

    #[test]
    fn timeout_covers_delayed_response_headers() {
        let driver = tokio::runtime::Runtime::new().expect("native HTTP test driver");
        let service = NativeHttpService::new(driver.handle().clone());
        let (listener, url) = server();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("timeout request");
            let mut buffer = [0u8; 1024];
            let _ = socket.read(&mut buffer);
            std::thread::sleep(Duration::from_millis(150));
        });
        let mut timed = request(url);
        timed.timeout = Duration::from_millis(20);
        let error = futures::executor::block_on(service.send(timed)).expect_err("header timeout");
        assert!(matches!(error.kind, TransportErrorKind::Timeout));
        server.join().unwrap();
    }

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
