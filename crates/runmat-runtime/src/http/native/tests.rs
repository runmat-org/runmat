use super::*;
use crate::execution::RuntimeExecutionService;
use runmat_execution::profiling::{CommandProfilingRoot, DeferredWorkCounts};
use std::error::Error;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use url::Url;

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

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
    drop(service.send(runtime(), request(url.clone())));
    assert!(
        service.clients.get().is_none(),
        "unpolled send has no client"
    );
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );

    let mut invalid = request(url.clone());
    invalid.headers.push(("bad\nname".into(), "value".into()));
    let error =
        futures::executor::block_on(service.send(runtime(), invalid)).expect_err("invalid header");
    assert!(matches!(error.kind, TransportErrorKind::InvalidHeader(_)));
    assert!(error
        .source()
        .and_then(|source| source.downcast_ref::<reqwest::header::InvalidHeaderName>())
        .is_some());
    assert!(
        service.clients.get().is_none(),
        "invalid header has no client"
    );
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );

    let mut invalid_agent = request(url);
    invalid_agent.user_agent = "bad\nagent".into();
    let error = futures::executor::block_on(service.send(runtime(), invalid_agent))
        .expect_err("invalid User-Agent");
    assert!(matches!(error.kind, TransportErrorKind::Other));
    assert!(error.source().is_some());
    assert!(
        service.clients.get().is_none(),
        "invalid User-Agent has no client"
    );
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn cloned_service_reuses_client_and_honors_request_user_agent_override() {
    let driver = tokio::runtime::Runtime::new().expect("native HTTP test driver");
    let service = NativeHttpService::new(driver.handle().clone());
    let clone = service.clone();
    assert!(Arc::ptr_eq(&service.clients, &clone.clients));
    let (listener, url) = server();
    let (sent, received) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            let (mut socket, _) = listener.accept().expect("HTTP request");
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0u8; 512];
            while !bytes.windows(4).any(|part| part == b"\r\n\r\n") {
                let count = socket.read(&mut buffer).expect("read headers");
                assert!(count > 0, "request ended before headers");
                bytes.extend_from_slice(&buffer[..count]);
            }
            sent.send(String::from_utf8(bytes).expect("HTTP headers"))
                .expect("captured headers");
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .expect("response");
        }
    });

    let mut first = request(url.clone());
    first.user_agent = "RunMat first".into();
    let command = CommandProfilingRoot::new();
    assert_eq!(
        futures::executor::block_on(
            service.send(runtime().with_command_profiling(command.receipt()), first)
        )
        .expect("first response")
        .body,
        b"ok"
    );
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            completed: 1,
            ..DeferredWorkCounts::default()
        })
    );
    let first_headers = received.recv().expect("first request headers");
    assert!(first_headers
        .to_ascii_lowercase()
        .contains("user-agent: runmat first"));
    assert!(service.clients.get().is_some());

    let mut second = request(url);
    second.user_agent = "RunMat fallback".into();
    second
        .headers
        .push(("User-Agent".into(), "RunMat override".into()));
    assert_eq!(
        futures::executor::block_on(clone.send(runtime(), second))
            .expect("second response")
            .body,
        b"ok"
    );
    let second_headers = received.recv().expect("second request headers");
    assert!(second_headers
        .to_ascii_lowercase()
        .contains("user-agent: runmat override"));
    assert!(!second_headers
        .to_ascii_lowercase()
        .contains("runmat fallback"));
    server.join().unwrap();
}

#[test]
fn keep_alive_reuses_one_socket_and_applies_each_request_timeout() {
    fn read_headers(socket: &mut std::net::TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buffer = [0u8; 512];
        while !bytes.windows(4).any(|part| part == b"\r\n\r\n") {
            let count = socket.read(&mut buffer).expect("read request headers");
            assert!(count > 0, "keep-alive socket closed before next request");
            bytes.extend_from_slice(&buffer[..count]);
            assert!(bytes.len() <= 16 * 1024, "unexpected large request headers");
        }
        String::from_utf8(bytes).expect("ASCII request headers")
    }

    let driver = tokio::runtime::Runtime::new().expect("native HTTP test driver");
    let service = NativeHttpService::new(driver.handle().clone());
    let cloned = service.clone();
    let (listener, url) = server();
    let (sent, received) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("one TCP connection");
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read deadline");
        for index in 0..3 {
            sent.send(read_headers(&mut socket))
                .expect("capture request");
            if index == 2 {
                std::thread::sleep(Duration::from_millis(200));
            }
            let _ = socket.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok",
            );
        }
    });

    let mut first = request(url.clone());
    first.user_agent = "RunMat first pooled".into();
    assert_eq!(
        futures::executor::block_on(service.send(runtime(), first))
            .expect("first keep-alive response")
            .body,
        b"ok"
    );
    assert!(received
        .recv_timeout(Duration::from_secs(2))
        .expect("first request")
        .to_ascii_lowercase()
        .contains("user-agent: runmat first pooled"));

    let mut second = request(url.clone());
    second.user_agent = "RunMat second pooled".into();
    second.timeout = Duration::from_secs(1);
    assert_eq!(
        futures::executor::block_on(cloned.send(runtime(), second))
            .expect("second response on the same socket")
            .body,
        b"ok"
    );
    assert!(received
        .recv_timeout(Duration::from_secs(2))
        .expect("second request on accepted socket")
        .to_ascii_lowercase()
        .contains("user-agent: runmat second pooled"));

    let mut third = request(url);
    third.timeout = Duration::from_millis(40);
    let error = futures::executor::block_on(service.send(runtime(), third))
        .expect_err("third request must use its shorter timeout");
    assert!(matches!(error.kind, TransportErrorKind::Timeout));
    received
        .recv_timeout(Duration::from_secs(2))
        .expect("third request on accepted socket");
    server.join().expect("single-socket server");
}

#[test]
fn shared_client_initialization_failure_retains_typed_cause_without_network() {
    let driver = tokio::runtime::Runtime::new().expect("native HTTP test driver");
    let service = NativeHttpService::new(driver.handle().clone());
    let init_error = reqwest::Client::new()
        .get("://invalid")
        .build()
        .expect_err("invalid request URL fixture");
    assert!(service.clients.set(Err(Arc::new(init_error))).is_ok());
    let (listener, url) = server();
    listener.set_nonblocking(true).unwrap();
    let error = futures::executor::block_on(service.send(runtime(), request(url)))
        .expect_err("shared client initialization failure");
    let shared = error
        .source()
        .and_then(|source| source.downcast_ref::<SharedClientInitError>())
        .expect("typed shared initialization source");
    assert!(shared.source().unwrap().is::<reqwest::Error>());
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
    let post_server = std::thread::spawn(move || {
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
    let response =
        futures::executor::block_on(service.send(runtime(), post)).expect("POST response");
    assert_eq!(response.body, b"ok");
    assert!(response
        .headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("x-runmat") && value == "yes"));
    post_server.join().unwrap();

    let (listener, url) = server();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("status request");
        let mut buffer = [0u8; 1024];
        let _ = socket.read(&mut buffer);
        socket
            .write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });
    let error =
        futures::executor::block_on(service.send(runtime(), request(url))).expect_err("HTTP 403");
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
    let error =
        futures::executor::block_on(service.send(runtime(), timed)).expect_err("header timeout");
    assert!(matches!(error.kind, TransportErrorKind::Timeout));
    assert!(error
        .source()
        .and_then(|source| source.downcast_ref::<reqwest::Error>())
        .is_some());
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
