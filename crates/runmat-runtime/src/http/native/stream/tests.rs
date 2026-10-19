use std::io::{Read, Write};
use std::net::TcpListener;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use super::*;
use crate::context::RuntimeContext;
use crate::context::RuntimeHttpService;
use crate::execution::RuntimeExecutionService;
use crate::http::{HttpMethod, NativeHttpService};
use runmat_execution::profiling::{CommandProfilingRoot, DeferredWorkCounts};

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn fixture() -> (TcpListener, HttpRequest) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("loopback listener");
    let url = Url::parse(&format!(
        "http://{}/archive.zip",
        listener.local_addr().unwrap()
    ))
    .expect("loopback URL");
    (
        listener,
        HttpRequest {
            url,
            method: HttpMethod::Get,
            headers: Vec::new(),
            body: None,
            timeout: Duration::from_secs(2),
            user_agent: "RunMat bounded test".into(),
        },
    )
}

fn read_request(socket: &mut std::net::TcpStream) {
    socket
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("read deadline");
    let mut bytes = Vec::new();
    let mut part = [0u8; 512];
    while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = socket.read(&mut part).expect("read headers");
        assert!(count > 0, "request ended before headers");
        bytes.extend_from_slice(&part[..count]);
        assert!(bytes.len() < 16 * 1024, "unexpected large request");
    }
}

#[test]
fn unpolled_open_has_no_client_and_no_network_effect() {
    let driver = tokio::runtime::Runtime::new().expect("host driver");
    let service = NativeHttpService::new(driver.handle().clone());
    let (listener, request) = fixture();
    listener.set_nonblocking(true).unwrap();
    drop(service.open_bounded(runtime(), request, 4));
    assert!(service.clients.get().is_none());
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn invalid_stream_request_is_rejected_before_client_or_socket() {
    let driver = tokio::runtime::Runtime::new().expect("host driver");
    let service = NativeHttpService::new(driver.handle().clone());
    let (listener, mut request) = fixture();
    listener.set_nonblocking(true).unwrap();
    request.headers.push(("bad\nname".into(), "value".into()));
    let error = futures::executor::block_on(service.open_bounded(runtime(), request, 4))
        .err()
        .expect("invalid header");
    assert!(matches!(error.kind, TransportErrorKind::InvalidHeader(_)));
    assert!(service.clients.get().is_none());
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn advertised_length_is_rejected_before_body_publication() {
    let driver = tokio::runtime::Runtime::new().expect("host driver");
    let service = NativeHttpService::new(driver.handle().clone());
    let (listener, request) = fixture();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("request");
        read_request(&mut socket);
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\n")
            .expect("head");
    });
    let error = futures::executor::block_on(service.open_bounded(runtime(), request, 4))
        .err()
        .expect("advertised length rejected");
    assert!(matches!(
        error.kind,
        TransportErrorKind::BodyLimitExceeded(4)
    ));
    server.join().unwrap();
}

#[test]
fn unknown_length_crosses_limit_without_publishing_excess_chunk() {
    let driver = tokio::runtime::Runtime::new().expect("host driver");
    let service = NativeHttpService::new(driver.handle().clone());
    let (listener, request) = fixture();
    let (first_sent, first_received) = mpsc::channel();
    let (continue_sent, continue_received) = mpsc::channel();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("request");
        read_request(&mut socket);
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\nab\r\n")
            .expect("first chunk");
        first_sent.send(()).unwrap();
        continue_received
            .recv_timeout(Duration::from_secs(2))
            .unwrap();
        socket
            .write_all(b"3\r\ncde\r\n0\r\n\r\n")
            .expect("excess chunk");
    });
    let mut body = futures::executor::block_on(service.open_bounded(runtime(), request, 4))
        .expect("unknown-length head");
    assert_eq!(body.head.content_length, None);
    first_received.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(
        futures::executor::block_on(body.next_chunk())
            .expect("first bounded chunk")
            .expect("body chunk"),
        b"ab"
    );
    continue_sent.send(()).unwrap();
    let error = futures::executor::block_on(body.next_chunk()).expect_err("limit crossed");
    assert!(matches!(
        error.kind,
        TransportErrorKind::BodyLimitExceeded(4)
    ));
    assert!(futures::executor::block_on(body.next_chunk())
        .expect("stream ended after rejection")
        .is_none());
    server.join().unwrap();
}

#[test]
fn body_timeout_after_head_retains_reqwest_cause() {
    let driver = tokio::runtime::Runtime::new().expect("host driver");
    let service = NativeHttpService::new(driver.handle().clone());
    let (listener, mut request) = fixture();
    request.timeout = Duration::from_millis(150);
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("request");
        read_request(&mut socket);
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n")
            .expect("head");
        std::thread::sleep(Duration::from_millis(350));
    });
    let command = CommandProfilingRoot::new();
    let mut body = futures::executor::block_on(service.open_bounded(
        runtime().with_command_profiling(command.receipt()),
        request,
        4,
    ))
    .expect("head arrived before timeout");
    assert_eq!(body.head.content_length, Some(2));
    let error = futures::executor::block_on(body.next_chunk()).expect_err("body timeout");
    assert!(matches!(error.kind, TransportErrorKind::Timeout));
    assert!(std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<reqwest::Error>())
        .is_some());
    assert!(futures::executor::block_on(body.next_chunk())
        .expect("worker retired after timeout")
        .is_none());
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            completed: 1,
            ..DeferredWorkCounts::default()
        })
    );
    server.join().unwrap();
}

#[test]
fn dropping_open_body_aborts_pending_socket_read() {
    for observe_pending_head in [true, false] {
        let driver = tokio::runtime::Runtime::new().expect("host driver");
        let service = NativeHttpService::new(driver.handle().clone());
        let (listener, request) = fixture();
        let (closed_tx, closed_rx) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("request");
            read_request(&mut socket);
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .expect("head");
            let mut byte = [0u8; 1];
            closed_tx
                .send(socket.read(&mut byte).expect("closed socket"))
                .unwrap();
        });
        let command = CommandProfilingRoot::new();
        let body = futures::executor::block_on(service.open_bounded(
            runtime().with_command_profiling(command.receipt()),
            request,
            4,
        ))
        .expect("open body awaiting server chunk");
        let command = if observe_pending_head {
            assert_eq!(
                command.seal().deferred_work,
                Some(DeferredWorkCounts {
                    admitted: 1,
                    pending: 1,
                    ..DeferredWorkCounts::default()
                }),
                "publishing the response head is not worker retirement"
            );
            None
        } else {
            Some(command)
        };
        drop(body);
        assert_eq!(
            closed_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            0,
            "dropping the stream must close the pending response"
        );
        server.join().unwrap();
        if let Some(command) = command {
            driver.shutdown_timeout(Duration::from_secs(2));
            assert_eq!(
                command.seal().deferred_work,
                Some(DeferredWorkCounts {
                    admitted: 1,
                    missing: 1,
                    ..DeferredWorkCounts::default()
                }),
                "aborted body task retires with Missing after runtime shutdown"
            );
        }
    }
}
