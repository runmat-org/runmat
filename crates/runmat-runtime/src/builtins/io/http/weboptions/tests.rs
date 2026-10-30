use super::*;
async fn weboptions_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    super::weboptions_builtin(
        runtime,
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
}
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::net::{TcpListener, TcpStream};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

#[cfg(not(target_arch = "wasm32"))]
use crate::call_builtin_async_in_context;
use runmat_value::CellArray;

#[cfg(not(target_arch = "wasm32"))]
fn spawn_server<F>(handler: F) -> String
where
    F: FnOnce(TcpStream) + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            handler(stream);
        }
    });
    format!("http://{}", addr)
}

fn error_message(err: crate::RuntimeError) -> String {
    err.message().to_string()
}

fn run_weboptions(rest: Vec<Value>) -> BuiltinResult<crate::execution::RootedValue> {
    futures::executor::block_on(weboptions_builtin(None, rest))
}

#[cfg(not(target_arch = "wasm32"))]
fn run_call_builtin(name: &str, args: &[Value]) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = crate::builtins::io::http::transport::test_runtime_context();
    futures::executor::block_on(call_builtin_async_in_context(&runtime, name, args))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut buffer = Vec::new();
    let mut tmp = [0u8; 512];
    loop {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => {
                buffer.extend_from_slice(&tmp[..n]);
                if buffer.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let header_end = buffer
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .unwrap_or(buffer.len());
    let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let body = buffer[header_end..].to_vec();
    (headers, body)
}

#[cfg(not(target_arch = "wasm32"))]
fn respond_with(mut stream: TcpStream, content_type: &str, body: &[u8]) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
        body.len(),
        content_type
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.write_all(body);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = WEBOPTIONS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"options = weboptions()"));
    assert!(labels.contains(&"options = weboptions(name, value, ...)"));
    assert!(labels.contains(&"options = weboptions(optionsStruct, name, value, ...)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_default_struct_matches_expected_fields() {
    let result = run_weboptions(Vec::new()).expect("weboptions");
    let Value::Struct(options) = result.value() else {
        panic!("expected struct result");
    };
    assert_eq!(
        options.fields.get("ContentType").and_then(|v| match v {
            Value::String(s) => s.as_str(),
            _ => None,
        }),
        Some("auto")
    );
    assert_eq!(
        options.fields.get("Timeout").and_then(|v| match v {
            Value::Num(n) => Some(*n),
            _ => None,
        }),
        Some(DEFAULT_TIMEOUT_SECONDS)
    );
    match options.fields.get("HeaderFields") {
        Some(Value::Struct(headers)) => assert!(headers.fields.is_empty()),
        other => panic!("expected empty HeaderFields struct, got {other:?}"),
    }
    assert_eq!(
        options.fields.get("RequestMethod").and_then(|v| match v {
            Value::String(s) => s.as_str(),
            _ => None,
        }),
        Some("auto")
    );
    assert_eq!(
        options.fields.get("MediaType").and_then(|v| match v {
            Value::String(s) => s.as_str(),
            _ => None,
        }),
        Some("auto")
    );
}

#[test]
fn weboptions_timeout_reads_typed_integer_tensor_storage_exactly() {
    let timeout = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::U16(vec![2026]),
        vec![1, 1],
    )
    .expect("typed timeout");

    let result = run_weboptions(vec![Value::from("Timeout"), Value::Tensor(timeout)])
        .expect("weboptions timeout");
    let Value::Struct(options) = result.value() else {
        panic!("expected options struct");
    };
    assert_eq!(
        options.fields.get("Timeout").and_then(|v| match v {
            Value::Num(n) => Some(*n),
            _ => None,
        }),
        Some(2026.0)
    );
}

#[test]
fn weboptions_preserves_exact_integer_key_value() {
    let value = Value::Int(runmat_value::IntValue::U64(u64::MAX));
    let result = run_weboptions(vec![
        Value::from("KeyName"),
        Value::from("sequence"),
        Value::from("KeyValue"),
        value.clone(),
    ])
    .expect("integer key value");
    let Value::Struct(options) = result.value() else {
        panic!("expected options struct");
    };
    assert_eq!(options.fields.get("KeyValue"), Some(&value));
}

#[test]
fn weboptions_rejects_nonscalar_text_key_values_before_other_resident_readback() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident unrelated option");
    let invalid_text = [
        Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars(vec!['a', 'b'], 2, 1)
                .expect("char matrix"),
        ),
        Value::StringArray(
            runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2])
                .expect("string array"),
        ),
    ];
    for invalid in invalid_text {
        let error = run_weboptions(vec![
            Value::from("KeyValue"),
            invalid,
            Value::from("Timeout"),
            Value::GpuTensor(allocation.handle.clone()),
        ])
        .expect_err("KeyValue requires scalar text");
        assert_eq!(
            error.identifier(),
            Some("RunMat:weboptions:InvalidOptionValue")
        );
        assert!(!error.message().contains("provider"));
    }
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn weboptions_runmat_only_forms_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let query_error = run_weboptions(vec![
        Value::from("QueryParameters"),
        Value::Struct(StructValue::new()),
    ])
    .expect_err("query property gate");
    assert_eq!(
        query_error.identifier(),
        Some("RunMat:compatibility:WeboptionsQueryParametersExtension")
    );

    let copy_error =
        run_weboptions(vec![Value::Struct(StructValue::new())]).expect_err("struct copy gate");
    assert_eq!(
        copy_error.identifier(),
        Some("RunMat:compatibility:WeboptionsStructCopyExtension")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_overrides_timeout_and_headers() {
    let mut headers = runmat_value::UnboundedStructValueBuilder::new();
    headers
        .fields
        .insert("Accept".to_string(), Value::from("application/json"));
    headers
        .fields
        .insert("X-Client".to_string(), Value::from("RunMat"));
    let args = vec![
        Value::from("Timeout"),
        Value::Num(10.0),
        Value::from("HeaderFields"),
        Value::Struct(headers.finish()),
    ];
    let result = run_weboptions(args).expect("weboptions overrides");
    let Value::Struct(opts) = result.value() else {
        panic!("expected struct");
    };
    assert_eq!(
        opts.fields.get("Timeout").and_then(|v| match v {
            Value::Num(n) => Some(*n),
            _ => None,
        }),
        Some(10.0)
    );
    match opts.fields.get("HeaderFields") {
        Some(Value::Struct(headers)) => {
            assert_eq!(
                headers.fields.get("Accept"),
                Some(&Value::from("application/json"))
            );
            assert_eq!(headers.fields.get("X-Client"), Some(&Value::from("RunMat")));
        }
        other => panic!("expected header struct, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_updates_existing_struct() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = run_weboptions(vec![Value::from("ContentType"), Value::from("json")])
        .expect("base weboptions");
    let args = vec![
        base.value().clone(),
        Value::from("Timeout"),
        Value::Num(15.0),
    ];
    let updated = run_weboptions(args).expect("weboptions update");
    let Value::Struct(opts) = updated.value() else {
        panic!("expected struct");
    };
    assert_eq!(
        opts.fields.get("ContentType").and_then(|v| match v {
            Value::String(s) => s.as_str(),
            _ => None,
        }),
        Some("json")
    );
    assert_eq!(
        opts.fields.get("Timeout").and_then(|v| match v {
            Value::Num(n) => Some(*n),
            _ => None,
        }),
        Some(15.0)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_rejects_unknown_option() {
    let err = error_message(
        run_weboptions(vec![Value::from("BogusOption"), Value::Num(1.0)])
            .expect_err("unknown option should fail"),
    );
    assert!(err.contains("unknown option"), "unexpected error: {err}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_preflights_all_literal_names_and_host_values_before_gather() {
    let unknown = run_weboptions(vec![
        Value::from("Timeout"),
        Value::Num(10.0),
        Value::from("CharacterEncoding"),
        Value::from("UTF-8"),
    ])
    .expect_err("unsupported field");
    assert_eq!(
        unknown.identifier(),
        Some("RunMat:weboptions:UnknownOption")
    );

    let invalid = run_weboptions(vec![
        Value::from("Timeout"),
        Value::Num(0.0),
        Value::from("KeyValue"),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
    ])
    .expect_err("invalid known Timeout");
    assert_eq!(
        invalid.identifier(),
        Some("RunMat:weboptions:InvalidOptionValue")
    );

    let missing = run_weboptions(vec![Value::from("Timeout")]).expect_err("unpaired option");
    assert_eq!(
        missing.identifier(),
        Some("RunMat:weboptions:MissingOptionValue")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_rejects_excess_outputs_before_constructing_options() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = run_weboptions(vec![]).expect_err("fixed one-output contract");
    assert_eq!(error.identifier(), Some("RunMat:weboptions:TooManyOutputs"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_requires_username_when_password_provided() {
    let err = error_message(
        run_weboptions(vec![Value::from("Password"), Value::from("secret")])
            .expect_err("password without username"),
    );
    assert!(
        err.contains("Password requires a Username option"),
        "unexpected error: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_rejects_timeout_nonpositive() {
    let err = error_message(
        run_weboptions(vec![Value::from("Timeout"), Value::Num(0.0)])
            .expect_err("timeout should reject nonpositive values"),
    );
    assert!(
        err.contains("Timeout must be a positive numeric scalar"),
        "unexpected error: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn weboptions_rejects_headerfields_bad_cell_shape() {
    let cell = CellArray::new(vec![Value::from("Accept")], 1, 1).expect("cell");
    let err = error_message(
        run_weboptions(vec![Value::from("HeaderFields"), Value::Cell(cell)])
            .expect_err("headerfields cell shape"),
    );
    assert!(
        err.contains("HeaderFields cell array must have exactly two columns"),
        "unexpected error: {err}"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn webread_uses_weboptions_without_polluting_query() {
    let options = run_weboptions(Vec::new()).expect("weboptions");
    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let (headers, _) = read_request(&mut stream);
        tx.send(headers).unwrap();
        respond_with(stream, "application/json", br#"{"ok":true}"#);
    });

    let args = vec![Value::from(url.clone()), options.value().clone()];
    let result = run_call_builtin("webread", &args).expect("webread with options");
    match result.value() {
        Value::Struct(reply) => {
            assert!(matches!(reply.fields.get("ok"), Some(Value::Bool(true))));
        }
        other => panic!("expected struct response, got {other:?}"),
    }
    let headers = rx.recv().expect("captured headers");
    assert!(headers.starts_with("GET "));
    assert!(!headers.contains("KeyName="));
    assert!(!headers.contains("KeyValue="));
    assert!(
        !headers.contains("MediaType=auto"),
        "MediaType should not appear in query string"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn webread_and_webwrite_apply_exact_integer_api_key_header() {
    let key = run_weboptions(vec![
        Value::from("KeyName"),
        Value::from("X-Sequence"),
        Value::from("KeyValue"),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
    ])
    .expect("API key options");
    for (builtin, payload) in [("webread", None), ("webwrite", Some(Value::from("body")))] {
        let (tx, rx) = mpsc::channel();
        let url = spawn_server(move |mut stream| {
            let (headers, _) = read_request(&mut stream);
            tx.send(headers).expect("captured request");
            respond_with(stream, "text/plain", b"ok");
        });
        let mut args = vec![Value::from(url)];
        if let Some(body) = payload {
            args.push(body);
        }
        args.push(key.value().clone());
        run_call_builtin(builtin, &args).expect("HTTP request with API key");
        let headers = rx.recv().expect("captured headers");
        assert!(headers
            .to_ascii_lowercase()
            .contains(&format!("x-sequence: {}", u64::MAX)));
        assert!(!headers.contains("KeyName="));
        assert!(!headers.contains("KeyValue="));
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn webwrite_uses_weboptions_auto_request_method() {
    let options = run_weboptions(Vec::new()).expect("weboptions default");
    let payload = Value::from("Hello from RunMat");
    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let (headers, body) = read_request(&mut stream);
        tx.send((headers, body)).unwrap();
        respond_with(stream, "application/json", br#"{"ack":true}"#);
    });

    let args = vec![Value::from(url), payload, options.value().clone()];
    let result = run_call_builtin("webwrite", &args).expect("webwrite with weboptions");
    match result.value() {
        Value::Struct(reply) => {
            assert!(matches!(reply.fields.get("ack"), Some(Value::Bool(true))));
        }
        other => panic!("expected struct response, got {other:?}"),
    }
    let (headers, body) = rx.recv().expect("request captured");
    assert!(
        headers.starts_with("POST "),
        "expected POST request, got headers: {headers}"
    );
    assert!(
        !body.is_empty(),
        "expected request body to be present when posting form data"
    );
}
