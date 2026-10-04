//! MATLAB-compatible `readline` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, READLINE_ERROR_INTERNAL, READLINE_ERROR_INVALID_ARGUMENTS,
    READLINE_ERROR_INVALID_CLIENT, READLINE_ERROR_NOT_CONNECTED,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Tensor, Value};
use std::io::{self, Read};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::client_state::{client_handle, CLIENT_HANDLE_FIELD};
use super::control::configure_stream;

const BUILTIN_NAME: &str = "readline";

fn readline_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn readline_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    readline_error_with_message(error.message, error)
}

fn readline_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("readline: ").unwrap_or(detail);
    readline_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn readline_flow(error: &'static BuiltinErrorDescriptor, message: impl AsRef<str>) -> RuntimeError {
    readline_error_with_detail(error, message)
}

fn map_readline_flow(err: RuntimeError, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "readline",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::net::readline"
)]
pub(super) async fn readline_builtin(
    client: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(readline_error(&READLINE_ERROR_INVALID_ARGUMENTS));
    }
    super::transport::require_native_tcp(BUILTIN_NAME)?;

    let client = gather_if_needed_async(&client)
        .await
        .map_err(|err| map_readline_flow(err, &READLINE_ERROR_INVALID_CLIENT))?;
    let client_struct = match &client {
        Value::Struct(st) => st,
        _ => {
            return Err(readline_flow(
                &READLINE_ERROR_INVALID_CLIENT,
                "readline: expected tcpclient struct as first argument",
            ))
        }
    };

    let client_id = extract_client_id(client_struct)?;
    let handle = client_handle(client_id).ok_or_else(|| {
        readline_flow(
            &READLINE_ERROR_INVALID_CLIENT,
            "readline: tcpclient handle is no longer valid",
        )
    })?;

    let (mut stream, timeout, mut buffer) = {
        let mut guard = handle.lock().unwrap_or_else(|poison| poison.into_inner());
        if !guard.connected {
            return Err(readline_error(&READLINE_ERROR_NOT_CONNECTED));
        }
        let stream = guard.stream.try_clone().map_err(|err| {
            super::error::with_source(
                BUILTIN_NAME,
                &READLINE_ERROR_INTERNAL,
                format!("readline: unable to clone socket ({err})"),
                err,
            )
        })?;
        let timeout = guard.timeout;
        let buffer = std::mem::take(&mut guard.readline_buffer);
        (stream, timeout, buffer)
    };

    if let Err(err) = configure_stream(&stream, timeout) {
        if let Ok(mut guard) = handle.lock() {
            guard.readline_buffer = buffer;
        }
        return Err(super::error::with_source(
            BUILTIN_NAME,
            &READLINE_ERROR_INTERNAL,
            format!("readline: unable to configure socket timeout ({err})"),
            err,
        ));
    }

    let timeout = if timeout.is_infinite() || timeout == 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(timeout))
    };
    let outcome = match read_line(&mut stream, &mut buffer, timeout) {
        Ok(outcome) => outcome,
        Err(err) => {
            if let Ok(mut guard) = handle.lock() {
                guard.readline_buffer = buffer;
            }
            return Err(super::error::with_source(
                BUILTIN_NAME,
                &READLINE_ERROR_INTERNAL,
                format!("readline: socket error ({err})"),
                err,
            ));
        }
    };

    {
        let mut guard = handle.lock().unwrap_or_else(|poison| poison.into_inner());
        if matches!(outcome, LineReadResult::Closed(_)) {
            guard.connected = false;
        }
        guard.readline_buffer = buffer;
    }

    let value = match outcome {
        LineReadResult::Complete(bytes) => value_from_bytes(bytes),
        LineReadResult::Timeout => empty_double_matrix(),
        LineReadResult::Closed(bytes) => value_from_bytes(bytes),
    };

    Ok(value)
}

enum LineReadResult {
    Complete(Vec<u8>),
    Timeout,
    Closed(Vec<u8>),
}

fn read_line(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
    timeout: Option<Duration>,
) -> Result<LineReadResult, io::Error> {
    if let Some(line) = extract_line(buffer) {
        return Ok(LineReadResult::Complete(line));
    }

    let mut byte = [0u8; 1];
    let start = Instant::now();
    loop {
        if let Some(timeout) = timeout {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Ok(LineReadResult::Timeout);
            }
            stream.set_read_timeout(Some(timeout - elapsed))?;
        }
        match stream.read(&mut byte) {
            Ok(0) => {
                if buffer.is_empty() {
                    return Ok(LineReadResult::Closed(Vec::new()));
                }
                let bytes = std::mem::take(buffer);
                return Ok(LineReadResult::Closed(bytes));
            }
            Ok(_) => {
                let b = byte[0];
                buffer.push(b);
                if let Some(line) = extract_line(buffer) {
                    return Ok(LineReadResult::Complete(line));
                }
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) if err.kind() == io::ErrorKind::TimedOut => {
                return Ok(LineReadResult::Timeout);
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                if let Some(timeout) = timeout {
                    if start.elapsed() >= timeout {
                        return Ok(LineReadResult::Timeout);
                    }
                    continue;
                }
                return Err(err);
            }
            Err(err) => return Err(err),
        }
    }
}

fn extract_line(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    if let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
        let mut segment: Vec<u8> = buffer.drain(..=pos).collect();
        if segment.last() == Some(&b'\n') {
            segment.pop();
        }
        if segment.last() == Some(&b'\r') {
            segment.pop();
        }
        return Some(segment);
    }
    None
}

fn value_from_bytes(bytes: Vec<u8>) -> Value {
    if bytes.is_empty() {
        return Value::String(String::new());
    }
    match String::from_utf8(bytes) {
        Ok(text) => Value::String(text),
        Err(err) => {
            let lossy = err.into_bytes();
            let mapped: String = lossy.into_iter().map(|b| b as char).collect();
            Value::String(mapped)
        }
    }
}

fn empty_double_matrix() -> Value {
    Value::Tensor(Tensor::new(vec![], vec![0, 0]).expect("valid 0x0 tensor"))
}

fn extract_client_id(struct_value: &StructValue) -> BuiltinResult<u64> {
    let id_value = struct_field(struct_value, CLIENT_HANDLE_FIELD).ok_or_else(|| {
        readline_flow(
            &READLINE_ERROR_INVALID_CLIENT,
            "readline: tcpclient struct is missing internal handle",
        )
    })?;
    match id_value {
        Value::Int(iv) => iv.try_to_u64().ok_or_else(|| {
            readline_flow(
                &READLINE_ERROR_INVALID_CLIENT,
                "readline: tcpclient struct has invalid handle field",
            )
        }),
        _ => Err(readline_flow(
            &READLINE_ERROR_INVALID_CLIENT,
            "readline: tcpclient struct has invalid handle field",
        )),
    }
}

fn struct_field<'a>(value: &'a StructValue, name: &str) -> Option<&'a Value> {
    value.fields.get(name)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::net::client_state::{
        client_handle, insert_client, remove_client_for_test,
    };
    use crate::builtins::io::net::control::configure_stream;
    use runmat_builtins::READLINE_DESCRIPTOR;
    use runmat_value::{IntValue, StructValue, Value};
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::Duration;

    fn make_client(stream: TcpStream, timeout: f64) -> Value {
        let peer_addr = stream.peer_addr().expect("peer addr");
        configure_stream(&stream, timeout).expect("configure stream");
        let client_id = insert_client(stream, 0, peer_addr, timeout, "little-endian".to_string());
        let mut st = StructValue::new();
        st.fields.insert(
            CLIENT_HANDLE_FIELD.to_string(),
            Value::Int(IntValue::U64(client_id)),
        );
        Value::Struct(st)
    }

    fn client_id(client: &Value) -> u64 {
        match client {
            Value::Struct(st) => match st.fields.get(CLIENT_HANDLE_FIELD) {
                Some(Value::Int(IntValue::U64(id))) => *id,
                Some(Value::Int(iv)) => iv.to_i64() as u64,
                other => panic!("unexpected id field {other:?}"),
            },
            other => panic!("expected struct, got {other:?}"),
        }
    }

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    #[test]
    fn typed_negative_client_handle_is_rejected() {
        let mut client = StructValue::new();
        client.fields.insert(
            CLIENT_HANDLE_FIELD.to_string(),
            Value::Int(IntValue::I8(-1)),
        );
        let err = extract_client_id(&client).unwrap_err();
        assert_error_identifier(err, READLINE_ERROR_INVALID_CLIENT.identifier.unwrap());
    }

    fn run_readline(client: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(readline_builtin(client, rest))
    }

    fn net_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::builtins::io::net::client_state::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = READLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"line = readline(client)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_returns_line_without_terminator() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"hello world\n").expect("write");
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);

        let line = run_readline(client.clone(), Vec::new()).expect("readline");
        match line {
            Value::String(text) => assert_eq!(text, "hello world"),
            other => panic!("expected string result, got {other:?}"),
        }

        handle.join().expect("server thread");
        remove_client_for_test(client_id(&client));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_strips_crlf_pairs() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"status OK\r\n").expect("write");
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);

        let line = run_readline(client.clone(), Vec::new()).expect("readline");
        match line {
            Value::String(text) => assert_eq!(text, "status OK"),
            other => panic!("expected string result, got {other:?}"),
        }

        handle.join().expect("server thread");
        remove_client_for_test(client_id(&client));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_returns_empty_matrix_on_timeout() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let _handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            // keep connection open without sending anything until client times out
            std::thread::sleep(Duration::from_millis(300));
            drop(stream);
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 0.1);

        let value = run_readline(client.clone(), Vec::new()).expect("readline");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty 0x0 double, got {other:?}"),
        }

        remove_client_for_test(client_id(&client));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_buffers_partial_data_across_timeouts() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"partial ").expect("write partial prefix");
            stream.flush().ok();
            std::thread::sleep(Duration::from_millis(150));
            stream
                .write_all(b"payload\n")
                .expect("write remaining payload");
            stream.flush().ok();
            std::thread::sleep(Duration::from_millis(50));
            drop(stream);
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 0.05);
        let id = client_id(&client);

        let first = run_readline(client.clone(), Vec::new()).expect("readline");
        match first {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected timeout as empty 0x0 double, got {other:?}"),
        }

        // Wait for the newline to arrive before attempting again.
        std::thread::sleep(Duration::from_millis(200));

        let second = run_readline(client.clone(), Vec::new()).expect("readline");
        match second {
            Value::String(text) => assert_eq!(text, "partial payload"),
            other => panic!("expected buffered payload after newline, got {other:?}"),
        }

        let handle_state = client_handle(id).expect("handle");
        let guard = handle_state.lock().expect("lock");
        assert!(
            guard.connected,
            "client should remain connected after completing buffered line"
        );
        drop(guard);

        handle.join().expect("server thread");
        remove_client_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_returns_partial_line_on_connection_close() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"incomplete line").expect("write");
            // close without newline
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);
        let id = client_id(&client);

        let value = run_readline(client.clone(), Vec::new()).expect("readline");
        match value {
            Value::String(text) => assert_eq!(text, "incomplete line"),
            other => panic!("expected partial string, got {other:?}"),
        }

        let handle_state = client_handle(id).expect("handle");
        let guard = handle_state.lock().expect("lock");
        assert!(!guard.connected);

        drop(guard);
        handle.join().expect("server thread");
        remove_client_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_errors_on_additional_arguments() {
        let _guard = net_guard();
        let err = run_readline(Value::Num(42.0), vec![Value::Num(1.0)])
            .expect_err("expected invalid argument error");
        assert_error_identifier(err, READLINE_ERROR_INVALID_ARGUMENTS.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_rejects_non_struct_argument() {
        let _guard = net_guard();
        let err =
            run_readline(Value::Num(5.0), Vec::new()).expect_err("expected invalid client error");
        assert_error_identifier(err, READLINE_ERROR_INVALID_CLIENT.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_errors_when_not_connected() {
        let _guard = net_guard();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            // Hold the stream open briefly, then drop.
            std::thread::sleep(Duration::from_millis(100));
            drop(stream);
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);
        let id = client_id(&client);

        let state = client_handle(id).expect("handle");
        {
            let mut guard = state.lock().expect("lock");
            guard.connected = false;
        }

        let err =
            run_readline(client.clone(), Vec::new()).expect_err("expected not-connected error");
        assert_error_identifier(err, READLINE_ERROR_NOT_CONNECTED.identifier.unwrap());

        handle.join().expect("server");
        remove_client_for_test(id);
    }
}
