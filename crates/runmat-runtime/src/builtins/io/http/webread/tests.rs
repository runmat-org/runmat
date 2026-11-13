use super::*;
async fn webread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    url: Value,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    super::webread_builtin(
        runtime,
        url,
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
}
use runmat_builtins::WEBREAD_DESCRIPTOR;

#[test]
fn webread_rejects_excess_outputs_before_input_or_http_service() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = futures::executor::block_on(webread_builtin(None, Value::Num(1.0), vec![]))
        .expect_err("excess outputs");
    assert_eq!(error.identifier(), Some("RunMat:webread:TooManyOutputs"));
}

#[test]
fn webread_rejects_known_invalid_timeout_before_resident_readback() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident query value");
    let error = futures::executor::block_on(webread_builtin(
        None,
        Value::from("https://example.test"),
        vec![
            Value::from("Timeout"),
            Value::Num(-1.0),
            Value::from("q"),
            Value::GpuTensor(allocation.handle.clone()),
        ],
    ))
    .expect_err("invalid host timeout");
    assert!(error.message().contains("Timeout"));
    assert!(!error.message().contains("provider"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn webread_rejects_non_http_url_before_resident_query_readback() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident query value");
    let error = futures::executor::block_on(webread_builtin(
        None,
        Value::from("mailto:alice@example.test"),
        vec![
            Value::from("q"),
            Value::GpuTensor(allocation.handle.clone()),
        ],
    ))
    .expect_err("non-HTTP URL must fail before readback");
    assert_eq!(error.identifier(), Some("RunMat:webread:InvalidUrl"));
    assert!(error.message().contains("unsupported URL scheme"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn webread_rejects_invalid_known_key_name_before_resident_key_value_readback() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident key value");
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options
        .fields
        .insert("KeyName".into(), Value::from("bad\nname"));
    options.fields.insert(
        "KeyValue".into(),
        Value::GpuTensor(allocation.handle.clone()),
    );
    let error = futures::executor::block_on(webread_builtin(
        None,
        Value::from("https://example.test"),
        vec![Value::Struct(options.finish())],
    ))
    .expect_err("invalid header");
    assert_eq!(
        error.identifier(),
        Some("RunMat:webread:InvalidOptionValue")
    );
    assert!(error.message().contains("valid HTTP header name"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn direct_keyname_pair_remains_a_query_parameter() {
    let mut options = WebReadOptions::default();
    let mut query = Vec::new();
    process_name_value_pair(
        &RequestOptionName::decode("KeyName"),
        &Value::from("ordinary-query"),
        &mut options,
        &mut query,
    )
    .expect("direct query pair");
    assert_eq!(query, vec![("KeyName".into(), "ordinary-query".into())]);
    assert!(options.api_key_header.is_none());
}

#[test]
fn request_names_are_prepared_once_across_option_and_query_overloads() {
    let args = [
        Value::from("tImEoUt"),
        Value::Num(3.0),
        Value::from("KeyName"),
        Value::from("ordinary-query"),
    ];
    let prepared =
        preflight_host_arguments(crate::call::FunctionArgumentView::from_slice(&args), None)
            .unwrap();
    assert_eq!(
        prepared.pair_fields,
        vec![
            Some(RequestOptionName::Field(WebOptionField::Timeout)),
            Some(RequestOptionName::Query("KeyName".into())),
        ]
    );
    let mut owned = crate::execution::RootedValues::new(args.to_vec())
        .unwrap()
        .into();
    let (_, query) = parse_arguments(&mut owned, None, &prepared).unwrap();
    assert_eq!(query, vec![("KeyName".into(), "ordinary-query".into())]);
}
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;

#[test]
fn query_text_preserves_exact_uint64() {
    assert_eq!(
        value_to_query_string(&Value::Int(runmat_value::IntValue::U64(u64::MAX)), "id")
            .expect("query text"),
        "18446744073709551615"
    );
}

#[test]
fn query_text_and_numeric_scalar_read_typed_integer_storage_exactly() {
    let query = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX]),
        vec![1, 1],
    )
    .expect("typed query tensor");
    assert_eq!(
        value_to_query_string(&Value::Tensor(query), "id").expect("query text"),
        "18446744073709551615"
    );

    let timeout = Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![2026]), vec![1, 1])
        .expect("typed timeout tensor");
    assert_eq!(
        numeric_scalar(&Value::Tensor(timeout), "timeout").expect("numeric scalar"),
        2026.0
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_query_geometry_accepts_low_rank_values_and_rejects_matrices() {
    let rank_one = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![2],
    )
    .expect("rank-one query");
    assert_eq!(
        value_to_query_string(&Value::Tensor(rank_one), "id").expect("rank-one query text"),
        "9007199254740993,18446744073709551615"
    );
    let rank_zero = Value::Tensor(Tensor::new(vec![7.0], Vec::new()).expect("rank-zero query"));
    assert_eq!(value_to_query_string(&rank_zero, "id").unwrap(), "7");
    let empty_matrix =
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty query matrix"));
    assert_eq!(value_to_query_string(&empty_matrix, "id").unwrap(), "");
    let matrix = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).expect("query matrix"));
    assert!(value_to_query_string(&matrix, "id").is_err());
}

fn error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_webread(url: Value, args: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(webread_builtin(
        Some(transport::test_runtime_context()),
        url,
        args,
    ))
}

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

fn respond_with(mut stream: TcpStream, content_type: &str, body: &[u8]) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
        body.len(),
        content_type
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.write_all(body);
}

fn read_request_headers(stream: &mut TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 256];
    while let Ok(read) = stream.read(&mut chunk) {
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buffer.len() > 16 * 1024 {
            break;
        }
    }
    String::from_utf8_lossy(&buffer).to_string()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = WEBREAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"data = webread(url)"));
    assert!(labels.contains(&"data = webread(url, optionsStruct)"));
    assert!(labels.contains(&"data = webread(url, name, value, ...)"));
}

#[test]
fn webread_query_values_preserve_exact_integer_vectors() {
    let value = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![1, 2],
        )
        .expect("integer query vector"),
    );
    assert_eq!(
        value_to_query_string(&value, "id").expect("query encoding"),
        "9007199254740993,18446744073709551615"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_fetches_json_response() {
    let url = spawn_server(|mut stream| {
        let mut buffer = [0u8; 1024];
        let _ = stream.read(&mut buffer);
        respond_with(
            stream,
            "application/json",
            br#"{"message":"hello","value":42}"#,
        );
    });

    let result = run_webread(Value::from(url), vec![]).expect("webread JSON response");

    match result {
        Value::Struct(struct_value) => {
            let message = struct_value.fields.get("message").expect("message field");
            let value = struct_value.fields.get("value").expect("value field");
            match message {
                Value::CharArray(ca) => {
                    let text: String = runmat_value::decode_utf16_code_units(ca.data())
                        .expect("valid UTF-16 fixture");
                    assert_eq!(text, "hello");
                }
                other => panic!("expected char array, got {other:?}"),
            }
            match value {
                Value::Num(n) => assert_eq!(*n, 42.0),
                other => panic!("expected numeric value, got {other:?}"),
            }
        }
        other => panic!("expected struct, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_fetches_text_response() {
    let url = spawn_server(|mut stream| {
        let mut buffer = [0u8; 512];
        let _ = stream.read(&mut buffer);
        respond_with(stream, "text/plain; charset=utf-8", b"RunMat webread test");
    });

    let result = run_webread(Value::from(url), vec![]).expect("webread text response");

    match result {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "RunMat webread test");
        }
        other => panic!("expected char array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_fetches_binary_payload() {
    let payload = [1u8, 2, 3, 254, 255];
    let url = spawn_server(move |mut stream| {
        let mut buffer = [0u8; 512];
        let _ = stream.read(&mut buffer);
        respond_with(stream, "application/octet-stream", &payload);
    });

    let args = vec![Value::from("ContentType"), Value::from("binary")];
    let result = run_webread(Value::from(url), args).expect("webread binary response");

    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 5]);
            assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U8(payload.to_vec()))
            );

            // Binary consumers must use exact uint8 backing storage rather
            // than the compatibility f64 mirror.
            let persisted = crate::data::DataArrayPayload::from_value(
                "uint8".to_string(),
                &Value::Tensor(tensor),
            )
            .expect("persist binary payload");
            let bytes = serde_json::to_vec(&persisted).expect("encode payload");
            let decoded: crate::data::DataArrayPayload =
                serde_json::from_slice(&bytes).expect("decode payload");
            assert_eq!(
                decoded.values,
                crate::data::DataArrayValues::U8(payload.to_vec())
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_appends_query_parameters() {
    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let request = read_request_headers(&mut stream);
        let _ = tx.send(request);
        respond_with(stream, "application/json", br#"{"ok":true}"#);
    });

    let args = vec![
        Value::from("count"),
        Value::Num(42.0),
        Value::from("ContentType"),
        Value::from("json"),
    ];
    let result = run_webread(Value::from(url.clone()), args).expect("webread query");
    match result {
        Value::Struct(struct_value) => {
            assert!(struct_value.fields.contains_key("ok"));
        }
        other => panic!("expected struct result, got {other:?}"),
    }
    let request = rx.recv().expect("request log");
    assert!(
        request.starts_with("GET /"),
        "unexpected request line: {request}"
    );
    assert!(
        request.contains("count=42"),
        "query parameters missing: {request}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_struct_argument_supports_options_and_query() {
    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let request = read_request_headers(&mut stream);
        let _ = tx.send(request);
        respond_with(stream, "application/json", br#"{"value":123}"#);
    });

    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields
        .fields
        .insert("ContentType".to_string(), Value::from("json"));
    fields.fields.insert("limit".to_string(), Value::Num(5.0));

    let result = run_webread(
        Value::from(url.clone()),
        vec![Value::Struct(fields.finish())],
    )
    .expect("webread struct arg");

    let request = rx.recv().expect("request log");
    assert!(
        request.contains("GET /?limit=5"),
        "expected limit query parameter: {request}"
    );

    match result {
        Value::Struct(struct_value) => match struct_value.fields.get("value") {
            Some(Value::Num(n)) => assert_eq!(*n, 123.0),
            other => panic!("unexpected JSON decode result: {other:?}"),
        },
        other => panic!("expected struct, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_headerfields_struct_applies_custom_headers() {
    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let request = read_request_headers(&mut stream);
        let _ = tx.send(request);
        respond_with(stream, "application/json", br#"{"ok":true}"#);
    });

    let mut headers = runmat_value::UnboundedStructValueBuilder::new();
    headers
        .fields
        .insert("X-Test".to_string(), Value::from("RunMat"));

    let args = vec![
        Value::from("HeaderFields"),
        Value::Struct(headers.finish()),
        Value::from("ContentType"),
        Value::from("json"),
    ];

    let result = run_webread(Value::from(url), args).expect("webread header fields");
    assert!(matches!(result, Value::Struct(_)));

    let request = rx.recv().expect("request log");
    assert!(
        request.to_ascii_lowercase().contains("x-test: runmat"),
        "custom header missing: {request}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_queryparameters_option_struct() {
    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let request = read_request_headers(&mut stream);
        let _ = tx.send(request);
        respond_with(stream, "application/json", br#"{"ok":true}"#);
    });

    let mut params = runmat_value::UnboundedStructValueBuilder::new();
    params.fields.insert("page".to_string(), Value::Num(2.0));

    let args = vec![
        Value::from("QueryParameters"),
        Value::Struct(params.finish()),
        Value::from("ContentType"),
        Value::from("json"),
    ];

    let result = run_webread(Value::from(url.clone()), args).expect("webread query parameters");
    assert!(matches!(result, Value::Struct(_)));

    let request = rx.recv().expect("request log");
    assert!(
        request.contains("page=2"),
        "query parameter missing: {request}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_errors_on_missing_name_value_pair() {
    let err = run_webread(
        Value::from("https://example.com"),
        vec![Value::from("Timeout")],
    )
    .expect_err("expected missing value error");
    let err = error_message(err);
    assert!(
        err.contains("missing value"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_rejects_non_positive_timeout() {
    let args = vec![Value::from("Timeout"), Value::Num(0.0)];
    let err = run_webread(Value::from("https://example.com"), args).expect_err("timeout error");
    let err = error_message(err);
    assert!(
        err.contains("Timeout must be a positive scalar"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_rejects_password_without_username() {
    let args = vec![Value::from("Password"), Value::from("secret")];
    let err = run_webread(Value::from("https://example.com"), args).expect_err("auth error");
    let err = error_message(err);
    assert!(
        err.contains("Password requires a Username"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_rejects_unsupported_content_type() {
    let args = vec![Value::from("ContentType"), Value::from("table")];
    let err = run_webread(Value::from("https://example.com"), args).expect_err("format error");
    let err = error_message(err);
    assert!(
        err.contains("unsupported ContentType"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webread_rejects_invalid_headerfields_shape() {
    let cell = crate::make_cell(
        vec![Value::from("A"), Value::from("B"), Value::from("C")],
        1,
        3,
    )
    .expect("make cell");

    let args = vec![Value::from("HeaderFields"), cell];
    let err = run_webread(Value::from("https://example.com"), args).expect_err("header error");
    let err = error_message(err);
    assert!(
        err.contains("HeaderFields cell array must have exactly two columns"),
        "unexpected error message: {err}"
    );
}
