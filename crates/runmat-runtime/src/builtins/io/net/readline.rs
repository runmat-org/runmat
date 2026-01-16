//! MATLAB-compatible `readline` builtin for TCP/IP clients in RunMat.

use runmat_builtins::{IntValue, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::io::{self, Read};
use std::net::TcpStream;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{gather_if_needed, build_runtime_error, BuiltinResult, RuntimeControlFlow, RuntimeError};

use super::accept::{client_handle, configure_stream, CLIENT_HANDLE_FIELD};

const MESSAGE_ID_INVALID_CLIENT: &str = "MATLAB:readline:InvalidTcpClient";
const MESSAGE_ID_NOT_CONNECTED: &str = "MATLAB:readline:NotConnected";
const MESSAGE_ID_INVALID_ARGUMENTS: &str = "MATLAB:readline:InvalidArguments";
const MESSAGE_ID_INTERNAL: &str = "MATLAB:readline:InternalError";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "readline",
        builtin_path = "crate::builtins::io::net::readline"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "readline"
category: "io/net"
keywords: ["readline", "tcpclient", "networking", "terminator", "text"]
summary: "Read ASCII text until the terminator from a MATLAB-compatible tcpclient struct."
references:
  - https://www.mathworks.com/help/matlab/ref/tcpclient.readline.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Socket reads execute on the host CPU. GPU-resident structs are gathered automatically before reading."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::net::readline::tests"
  integration: "builtins::io::net::readline::tests::readline_buffers_partial_data_across_timeouts"
---

# What does the `readline` function do in MATLAB / RunMat?
`readline(t)` reads ASCII text from the remote host associated with the MATLAB-compatible `tcpclient`
struct `t`. It consumes data until it encounters the line terminator (RunMat currently honours the MATLAB
default line feed (LF, `"\n"`)) and returns the characters **without** the terminator. If the terminator is
not observed before the timeout elapses, the builtin returns a 0-by-0 double (`[]`) just like MATLAB and
preserves the buffered bytes for the next call. When the peer closes the connection, any buffered characters
are returned and the internal connection flag is cleared.

## How does the `readline` function behave in MATLAB / RunMat?
- `readline(t)` waits for data and reads character bytes until the first line feed. The returned value is a
  MATLAB string scalar that excludes the terminator, mirroring MATLAB’s `tcpclient.readline`.
- If the peer uses carriage-return/line-feed pairs (`"\r\n"`), RunMat automatically strips both bytes so the
  returned string matches MATLAB’s behaviour for default CR/LF terminators.
- When the terminator is not encountered before the client’s `Timeout` expires, the builtin returns a
  0-by-0 double (`[]`). Any bytes that arrived remain buffered internally so the next `readline` call
  continues assembling the same logical line.
- If the remote host closes the connection before a terminator is observed, the builtin returns the buffered
  characters (which might be empty) and marks the underlying client as disconnected so subsequent network
  calls can react accordingly.
- Inputs passed from the GPU are gathered automatically. Networking always runs on the CPU and the returned
  string is CPU-resident.
- Invalid structs, missing handles, or disconnected clients raise MATLAB-style diagnostics (for example,
  `MATLAB:readline:InvalidTcpClient`).

## `readline` Function GPU Execution Behaviour
Networking is exclusively a host-side activity. When a `tcpclient` struct resides on the GPU, RunMat gathers
it back to the CPU before touching the socket. Acceleration providers are not consulted and no GPU buffers
are produced. Future GPU-aware networking features will continue to gather metadata automatically so sockets
remain purely CPU-bound.

## Examples of using the `readline` function in MATLAB / RunMat

### Reading a newline-terminated response from an echo server
```matlab
client = tcpclient("127.0.0.1", 55000);
write(client, "ping\n");
reply = readline(client);
```
Expected output:
```matlab
reply = "ping"
```

### Handling CRLF-terminated lines from legacy services
```matlab
client = tcpclient("legacy.example.com", 1600);
write(client, "HELLO\r\n");
line = readline(client);  % CRLF is removed automatically
```
Expected output:
```matlab
line = "OK HELLO"
```

### Looping over streaming telemetry one line at a time
```matlab
client = tcpclient("telemetry.example.com", 9000, "Timeout", 5);
while true
    payload = readline(client);
    if isequal(payload, [])
        continue   % timeout, try again once more data arrives
    end
    disp(payload)
end
```

### Detecting closed connections without losing buffered text
```matlab
client = tcpclient("sensor.example.com", 7000);
try
    line = readline(client);        % returns partial line if peer closed without LF
    disp(line)
    readline(client);               % second call raises MATLAB:readline:NotConnected
catch err
    if err.identifier == "MATLAB:readline:NotConnected"
        disp("Sensor disconnected")
    else
        rethrow(err)
    end
end
```

### Responding gracefully to quiet periods with finite timeouts
```matlab
client = tcpclient("chat.example.com", 4040, "Timeout", 0.5);
msg = readline(client);    % returns [] when no line arrives within 0.5 s
if isequal(msg, [])
    disp("No complete chat messages yet.")
end
```

### Parsing numeric payloads that arrive as text lines
```matlab
client = tcpclient("metrics.example.com", 5050);
line = readline(client);
values = str2double(split(line, ","));  % convert comma-separated numbers
```

### Logging line-delimited JSON messages from a service
```matlab
client = tcpclient("json.example.com", 6000);
while true
    payload = readline(client);
    if isequal(payload, [])
        break
    end
    decoded = jsondecode(payload);
    disp(decoded.status)
end
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. RunMat automatically gathers GPU-resident structs before reading from sockets. The returned string lives
on the CPU and there is no benefit to calling `gpuArray` for networking builtins such as `readline`.

## FAQ
- **Which terminator does RunMat use?** RunMat currently honours MATLAB’s default line feed (LF, `"\n"`).
  Support for `configureTerminator` will allow other terminators in a future update.
- **Does `readline` return the terminator?** No. The terminator bytes are stripped and the MATLAB string
  contains only the payload characters.
- **What happens on timeout?** If the terminator is not observed before the configured `Timeout`, the
  builtin returns a 0-by-0 double (`[]`) and retains any buffered bytes so the next call can complete the
  line once the terminator arrives.
- **How are CR/LF pairs handled?** When the payload ends with `"\r\n"`, both bytes are removed so the result
  matches MATLAB’s CR/LF behaviour.
- **What if the connection closes mid-line?** Any buffered text is returned and subsequent calls raise
  `MATLAB:readline:NotConnected`, signalling that the socket closed.
- **Does the builtin perform Unicode validation?** Bytes are decoded using UTF-8. Invalid sequences fall
  back to lossless byte-to-char mapping so no data is lost.
- **Can I call `readline` after gathering a GPU tensor?** Yes. `readline` gathers arguments automatically
  and always returns host values.
- **Will future GPU providers change the return type?** No. Networking results remain CPU strings even when
  GPUs are available.

## See Also
[tcpclient](./tcpclient), [accept](./accept), [read](./read), [write](./write)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/io/net/readline.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/net/readline.rs)
- Issues & feature requests: https://github.com/runmat-org/runmat/issues/new/choose
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::net::readline")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "readline",
    op_kind: GpuOpKind::Custom("network"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Networking occurs on the host CPU; GPU providers are not involved.",
};

fn readline_error(message_id: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(message_id)
        .with_builtin("readline")
        .build()
}

fn readline_flow(message_id: &'static str, message: impl Into<String>) -> RuntimeControlFlow {
    readline_error(message_id, message).into()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::net::readline")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "readline",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

#[runtime_builtin(
    name = "readline",
    category = "io/net",
    summary = "Read ASCII text until the terminator from a TCP/IP client.",
    keywords = "readline,tcpclient,networking",
    builtin_path = "crate::builtins::io::net::readline"
)]
fn readline_builtin(client: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(readline_flow(
            MESSAGE_ID_INVALID_ARGUMENTS,
            "readline: expected only the tcpclient argument",
        ));
    }

    let client = gather_if_needed(&client)?;
    let client_struct = match &client {
        Value::Struct(st) => st,
        _ => {
            return Err(readline_flow(
                MESSAGE_ID_INVALID_CLIENT,
                "readline: expected tcpclient struct as first argument",
            ))
        }
    };

    let client_id = extract_client_id(client_struct)?;
    let handle = client_handle(client_id).ok_or_else(|| {
        readline_flow(
            MESSAGE_ID_INVALID_CLIENT,
            "readline: tcpclient handle is no longer valid",
        )
    })?;

    let (mut stream, timeout, mut buffer) = {
        let mut guard = handle.lock().unwrap_or_else(|poison| poison.into_inner());
        if !guard.connected {
            return Err(readline_flow(
                MESSAGE_ID_NOT_CONNECTED,
                "readline: tcpclient is disconnected",
            ));
        }
        let stream = guard.stream.try_clone().map_err(|err| {
            readline_flow(
                MESSAGE_ID_INTERNAL,
                format!("readline: unable to clone socket ({err})"),
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
        return Err(readline_flow(
            MESSAGE_ID_INTERNAL,
            format!("readline: unable to configure socket timeout ({err})"),
        ));
    }

    let outcome = match read_line(&mut stream, &mut buffer) {
        Ok(outcome) => outcome,
        Err(err) => {
            if let Ok(mut guard) = handle.lock() {
                guard.readline_buffer = buffer;
            }
            return Err(readline_flow(
                MESSAGE_ID_INTERNAL,
                format!("readline: socket error ({err})"),
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

fn read_line(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> Result<LineReadResult, io::Error> {
    if let Some(line) = extract_line(buffer) {
        return Ok(LineReadResult::Complete(line));
    }

    let mut byte = [0u8; 1];
    loop {
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
            Err(err)
                if err.kind() == io::ErrorKind::TimedOut
                    || err.kind() == io::ErrorKind::WouldBlock =>
            {
                return Ok(LineReadResult::Timeout);
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
            MESSAGE_ID_INVALID_CLIENT,
            "readline: tcpclient struct is missing internal handle",
        )
    })?;
    match id_value {
        Value::Int(IntValue::U64(id)) => Ok(*id),
        Value::Int(iv) => Ok(iv.to_i64() as u64),
        _ => Err(readline_flow(
            MESSAGE_ID_INVALID_CLIENT,
            "readline: tcpclient struct has invalid handle field",
        )),
    }
}

fn struct_field<'a>(value: &'a StructValue, name: &str) -> Option<&'a Value> {
    value.fields.get(name)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::net::accept::{
        client_handle, configure_stream, insert_client, remove_client_for_test,
    };
    use runmat_builtins::{IntValue, StructValue, Value};
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

    fn assert_error_identifier(flow: RuntimeControlFlow, expected: &str) {
        match flow {
            RuntimeControlFlow::Error(err) => {
                assert_eq!(err.identifier(), Some(expected));
            }
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_returns_line_without_terminator() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"hello world\n").expect("write");
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);

        let line = readline_builtin(client.clone(), Vec::new()).expect("readline");
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
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            stream.write_all(b"status OK\r\n").expect("write");
        });

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        let client = make_client(stream, 1.0);

        let line = readline_builtin(client.clone(), Vec::new()).expect("readline");
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

        let value = readline_builtin(client.clone(), Vec::new()).expect("readline");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty 0x0 double, got {other:?}"),
        }

        remove_client_for_test(client_id(&client));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_buffers_partial_data_across_timeouts() {
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

        let first = readline_builtin(client.clone(), Vec::new()).expect("readline");
        match first {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected timeout as empty 0x0 double, got {other:?}"),
        }

        // Wait for the newline to arrive before attempting again.
        std::thread::sleep(Duration::from_millis(200));

        let second = readline_builtin(client.clone(), Vec::new()).expect("readline");
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

        let value = readline_builtin(client.clone(), Vec::new()).expect("readline");
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
        let err = readline_builtin(Value::Num(42.0), vec![Value::Num(1.0)])
            .expect_err("expected invalid argument error");
        assert_error_identifier(err, MESSAGE_ID_INVALID_ARGUMENTS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_rejects_non_struct_argument() {
        let err = readline_builtin(Value::Num(5.0), Vec::new())
            .expect_err("expected invalid client error");
        assert_error_identifier(err, MESSAGE_ID_INVALID_CLIENT);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn readline_errors_when_not_connected() {
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
            readline_builtin(client.clone(), Vec::new()).expect_err("expected not-connected error");
        assert_error_identifier(err, MESSAGE_ID_NOT_CONNECTED);

        handle.join().expect("server");
        remove_client_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
