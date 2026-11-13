//! MATLAB-compatible `readline` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, READLINE_ERROR_INTERNAL, READLINE_ERROR_INVALID_ARGUMENTS,
    READLINE_ERROR_INVALID_CLIENT, READLINE_ERROR_NOT_CONNECTED,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::context::{RuntimeTcpInvocation, RuntimeTcpReadMode};
use crate::tcp::TcpTransportErrorKind;
use crate::{build_runtime_error, RuntimeError};

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

#[runtime_builtin(
    name = "readline",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::net::readline"
)]
pub(super) async fn readline_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    client: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(readline_error(&READLINE_ERROR_INVALID_ARGUMENTS));
    }
    let authority = super::receiver::authority(&client, runmat_builtins::TCPCLIENT_CLASS)
        .ok_or_else(|| {
            readline_flow(
                &READLINE_ERROR_INVALID_CLIENT,
                "readline: expected a live tcpclient object",
            )
        })?;
    let service = super::transport::service(runtime.as_ref(), BUILTIN_NAME)?;
    let mut work = super::transport::NetCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    let outcome = work
        .await_service(service.read(
            RuntimeTcpInvocation::from_context(runtime.as_ref()),
            authority,
            RuntimeTcpReadMode::Line,
        ))
        .await
        .map_err(|failure| failure.into_runtime_error(BUILTIN_NAME))?;
    let receipt = match outcome {
        Ok(receipt) => receipt,
        Err(err) if err.kind() == TcpTransportErrorKind::Timeout => {
            work.completed(0).await?;
            return Ok(empty_double_matrix());
        }
        Err(err) => {
            let descriptor = match err.kind() {
                TcpTransportErrorKind::InvalidAuthority => &READLINE_ERROR_INVALID_CLIENT,
                TcpTransportErrorKind::Closed => &READLINE_ERROR_NOT_CONNECTED,
                _ => &READLINE_ERROR_INTERNAL,
            };
            return Err(super::error::with_source(
                BUILTIN_NAME,
                descriptor,
                format!("readline: {err}"),
                err,
            ));
        }
    };
    let bytes = receipt
        .bytes()
        .strip_suffix(b"\n")
        .unwrap_or(receipt.bytes());
    let bytes = bytes.strip_suffix(b"\r").unwrap_or(bytes);
    let value = value_from_bytes(bytes.to_vec());
    work.completed(0).await?;
    receipt.publish();
    Ok(value)
}

fn value_from_bytes(bytes: Vec<u8>) -> Value {
    if bytes.is_empty() {
        return Value::string(String::new());
    }
    match String::from_utf8(bytes) {
        Ok(text) => Value::string(text),
        Err(err) => {
            let lossy = err.into_bytes();
            let mapped: String = lossy.into_iter().map(|b| b as char).collect();
            Value::string(mapped)
        }
    }
}

fn empty_double_matrix() -> Value {
    Value::Tensor(Tensor::new(vec![], vec![0, 0]).expect("valid 0x0 tensor"))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::net::test_support::TcpTestHost;
    use crate::context::RuntimeTcpByteOrder;
    use runmat_builtins::READLINE_DESCRIPTOR;
    use runmat_value::{ObjectInstance, Value};
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    #[test]
    fn detached_client_object_is_rejected() {
        let host = TcpTestHost::new();
        let client = Value::Object(ObjectInstance::new(runmat_builtins::TCPCLIENT_CLASS));
        let err = run_readline(&host, client, Vec::new()).unwrap_err();
        assert_error_identifier(err, READLINE_ERROR_INVALID_CLIENT.identifier.unwrap());
    }

    fn run_readline(
        host: &TcpTestHost,
        client: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(readline_builtin(
            Some(host.runtime.clone()),
            client,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
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
        let host = TcpTestHost::new();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"hello world\n").expect("write");
        });

        let client = host.connect(port, 1.0, RuntimeTcpByteOrder::Little);

        let line = run_readline(&host, client.clone(), Vec::new()).expect("readline");
        match line {
            Value::String(runmat_value::StringElement::Text(text)) => {
                assert_eq!(text, "hello world")
            }
            other => panic!("expected string result, got {other:?}"),
        }

        handle.join().expect("server thread");
        host.close(&client);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_strips_crlf_pairs() {
        let host = TcpTestHost::new();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"status OK\r\n").expect("write");
        });

        let client = host.connect(port, 1.0, RuntimeTcpByteOrder::Little);

        let line = run_readline(&host, client.clone(), Vec::new()).expect("readline");
        match line {
            Value::String(runmat_value::StringElement::Text(text)) => assert_eq!(text, "status OK"),
            other => panic!("expected string result, got {other:?}"),
        }

        handle.join().expect("server thread");
        host.close(&client);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_returns_empty_matrix_on_timeout() {
        let host = TcpTestHost::new();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let _handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            // keep connection open without sending anything until client times out
            std::thread::sleep(Duration::from_millis(300));
            drop(stream);
        });

        let client = host.connect(port, 0.1, RuntimeTcpByteOrder::Little);

        let value = run_readline(&host, client.clone(), Vec::new()).expect("readline");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty 0x0 double, got {other:?}"),
        }

        host.close(&client);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_buffers_partial_data_across_timeouts() {
        let host = TcpTestHost::new();
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

        let client = host.connect(port, 0.05, RuntimeTcpByteOrder::Little);

        let first = run_readline(&host, client.clone(), Vec::new()).expect("readline");
        match first {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected timeout as empty 0x0 double, got {other:?}"),
        }

        // Wait for the newline to arrive before attempting again.
        std::thread::sleep(Duration::from_millis(200));

        let second = run_readline(&host, client.clone(), Vec::new()).expect("readline");
        match second {
            Value::String(runmat_value::StringElement::Text(text)) => {
                assert_eq!(text, "partial payload")
            }
            other => panic!("expected buffered payload after newline, got {other:?}"),
        }

        handle.join().expect("server thread");
        host.close(&client);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_returns_partial_line_on_connection_close() {
        let host = TcpTestHost::new();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"incomplete line").expect("write");
            // close without newline
        });

        let client = host.connect(port, 1.0, RuntimeTcpByteOrder::Little);

        let value = run_readline(&host, client.clone(), Vec::new()).expect("readline");
        match value {
            Value::String(runmat_value::StringElement::Text(text)) => {
                assert_eq!(text, "incomplete line")
            }
            other => panic!("expected partial string, got {other:?}"),
        }

        let err = run_readline(&host, client.clone(), Vec::new())
            .expect_err("known EOF closes subsequent read");
        assert_error_identifier(err, READLINE_ERROR_NOT_CONNECTED.identifier.unwrap());
        handle.join().expect("server thread");
        host.close(&client);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_errors_on_additional_arguments() {
        let host = TcpTestHost::new();
        let err = run_readline(&host, Value::Num(42.0), vec![Value::Num(1.0)])
            .expect_err("expected invalid argument error");
        assert_error_identifier(err, READLINE_ERROR_INVALID_ARGUMENTS.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_rejects_non_struct_argument() {
        let host = TcpTestHost::new();
        let err = run_readline(&host, Value::Num(5.0), Vec::new())
            .expect_err("expected invalid client error");
        assert_error_identifier(err, READLINE_ERROR_INVALID_CLIENT.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_errors_when_not_connected() {
        let host = TcpTestHost::new();
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            // Hold the stream open briefly, then drop.
            std::thread::sleep(Duration::from_millis(100));
            drop(stream);
        });

        let client = host.connect(port, 1.0, RuntimeTcpByteOrder::Little);
        handle.join().expect("server");
        assert_eq!(
            run_readline(&host, client.clone(), Vec::new()).expect("EOF read"),
            Value::String(String::new().into())
        );
        let err = run_readline(&host, client.clone(), Vec::new())
            .expect_err("expected not-connected error");
        assert_error_identifier(err, READLINE_ERROR_NOT_CONNECTED.identifier.unwrap());

        host.close(&client);
    }
}
