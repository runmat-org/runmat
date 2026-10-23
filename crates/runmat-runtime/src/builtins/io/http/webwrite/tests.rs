use super::*;
async fn webwrite_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    url: Value,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    super::webwrite_builtin(
        runtime,
        url,
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
}
use runmat_builtins::WEBWRITE_DESCRIPTOR;

#[test]
fn webwrite_rejects_excess_outputs_before_input_or_http_service() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = futures::executor::block_on(webwrite_builtin(None, Value::Num(1.0), vec![]))
        .expect_err("excess outputs");
    assert_eq!(error.identifier(), Some("RunMat:webwrite:TooManyOutputs"));
}

#[test]
fn webwrite_rejects_known_invalid_timeout_before_resident_body_readback() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident request body");
    let error = futures::executor::block_on(webwrite_builtin(
        None,
        Value::from("https://example.test"),
        vec![
            Value::GpuTensor(allocation.handle.clone()),
            Value::from("Timeout"),
            Value::Num(-1.0),
        ],
    ))
    .expect_err("invalid host timeout");
    assert!(error.message().contains("Timeout"));
    assert!(!error.message().contains("provider"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn webwrite_rejects_non_http_url_before_resident_body_readback() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident request body");
    let error = futures::executor::block_on(webwrite_builtin(
        None,
        Value::from("ftp://example.test/item"),
        vec![Value::GpuTensor(allocation.handle.clone())],
    ))
    .expect_err("non-HTTP URL must fail before body readback");
    assert_eq!(error.identifier(), Some("RunMat:webwrite:InvalidUrl"));
    assert!(error.message().contains("unsupported URL scheme"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn webwrite_rejects_invalid_known_key_name_before_resident_key_value_or_body_readback() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident request body");
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options
        .fields
        .insert("KeyName".into(), Value::from("bad\nname"));
    options.fields.insert(
        "KeyValue".into(),
        Value::GpuTensor(allocation.handle.clone()),
    );
    let error = futures::executor::block_on(webwrite_builtin(
        None,
        Value::from("https://example.test"),
        vec![
            Value::GpuTensor(allocation.handle.clone()),
            Value::Struct(options.finish()),
        ],
    ))
    .expect_err("invalid header");
    assert_eq!(
        error.identifier(),
        Some("RunMat:webwrite:InvalidOptionValue")
    );
    assert!(error.message().contains("valid HTTP header name"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn direct_keyname_pair_remains_a_request_parameter() {
    let mut options = WebWriteOptions::default();
    let mut query = Vec::new();
    process_name_value_pair(
        &RequestOptionName::decode("KeyName"),
        &Value::from("ordinary-query"),
        &mut options,
        &mut query,
    )
    .expect("direct request pair");
    assert_eq!(query, vec![("KeyName".into(), "ordinary-query".into())]);
    assert!(options.api_key_header.is_none());
}

#[test]
fn request_names_are_prepared_once_across_option_and_query_overloads() {
    let args = [
        Value::from("ReQuEsTmEtHoD"),
        Value::from("post"),
        Value::from("KeyValue"),
        Value::from("ordinary-query"),
    ];
    let prepared =
        preflight_host_arguments(crate::call::FunctionArgumentView::from_slice(&args), None)
            .unwrap();
    assert_eq!(
        prepared.pair_fields,
        vec![
            Some(RequestOptionName::Field(WebOptionField::RequestMethod)),
            Some(RequestOptionName::Query("KeyValue".into())),
        ]
    );
    let mut owned = crate::execution::RootedValues::new(args.to_vec())
        .unwrap()
        .into();
    let (_, query) = parse_arguments(&mut owned, None, &prepared).unwrap();
    assert_eq!(query, vec![("KeyValue".into(), "ordinary-query".into())]);
}
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;

#[test]
fn body_and_query_text_preserve_exact_uint64() {
    let value = Value::Int(runmat_value::IntValue::U64(u64::MAX));
    assert_eq!(
        scalar_to_string(&value).expect("body text"),
        "18446744073709551615"
    );
    assert_eq!(
        value_to_query_string(&value, "id").expect("query text"),
        "18446744073709551615"
    );
}

#[test]
fn body_query_and_numeric_scalar_read_typed_integer_storage_exactly() {
    let text = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX]),
        vec![1, 1],
    )
    .expect("typed text tensor");
    let value = Value::Tensor(text.clone());
    assert_eq!(
        scalar_to_string(&value).expect("body text"),
        "18446744073709551615"
    );
    assert_eq!(
        value_to_query_string(&value, "id").expect("query text"),
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

fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut buffer = Vec::new();
    let mut tmp = [0u8; 512];
    let mut header_end = None;
    loop {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => {
                buffer.extend_from_slice(&tmp[..n]);
                if let Some(idx) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                    header_end = Some(idx + 4);
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let header_end = header_end.unwrap_or(buffer.len());
    let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let mut parts = line.splitn(2, ':');
            let name = parts.next()?.trim();
            let value = parts.next()?.trim();
            if name.eq_ignore_ascii_case("content-length") {
                value.parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    let mut body = buffer[header_end..].to_vec();
    while body.len() < content_length {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => body.extend_from_slice(&tmp[..n]),
            Err(_) => break,
        }
    }
    (headers, body)
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

fn run_webwrite(url: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(webwrite_builtin(
        Some(transport::test_runtime_context()),
        url,
        rest,
    ))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webwrite_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = WEBWRITE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"response = webwrite(url, data)"));
    assert!(labels.contains(&"response = webwrite(url, data, optionsStruct)"));
    assert!(labels.contains(&"response = webwrite(url, data, name, value, ...)"));
}

#[test]
fn webwrite_post_values_preserve_exact_integer_vectors() {
    let value = Value::Tensor(
        Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MIN, (1_i64 << 53) + 1]),
            vec![1, 2],
        )
        .expect("integer post vector"),
    );
    assert_eq!(
        value_to_query_string(&value, "id").expect("post encoding"),
        "-9223372036854775808,9007199254740993"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webwrite_posts_form_data_by_default() {
    let payload = {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert("name".to_string(), Value::from("Ada"));
        st.fields.insert("score".to_string(), Value::Num(42.0));
        st.finish()
    };
    let opts = {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields
            .insert("ContentType".to_string(), Value::from("json"));
        st.finish()
    };

    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let (headers, body) = read_request(&mut stream);
        tx.send((headers, body)).unwrap();
        respond_with(
            stream,
            "application/json",
            br#"{"status":"ok","received":true}"#,
        );
    });

    let result = run_webwrite(
        Value::from(url),
        vec![Value::Struct(payload), Value::Struct(opts)],
    )
    .expect("webwrite");

    let (headers, body) = rx.recv().expect("request captured");
    assert!(headers.starts_with("POST "));
    let headers_lower = headers.to_ascii_lowercase();
    assert!(headers_lower.contains("content-type: application/x-www-form-urlencoded"));
    let body_text = String::from_utf8(body).expect("utf8 body");
    assert!(body_text.contains("name=Ada"));
    assert!(body_text.contains("score=42"));

    match result {
        Value::Struct(reply) => {
            assert!(matches!(
                reply.fields.get("received"),
                Some(Value::Bool(true))
            ));
        }
        other => panic!("expected struct response, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webwrite_sends_json_when_media_type_json() {
    let payload = {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert("title".to_string(), Value::from("RunMat"));
        st.fields.insert("stars".to_string(), Value::Num(5.0));
        st.finish()
    };
    let opts = {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert(
            "MediaType".to_string(),
            Value::from("application/json; charset=utf-8"),
        );
        st.fields
            .insert("ContentType".to_string(), Value::from("json"));
        st.finish()
    };

    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let (headers, body) = read_request(&mut stream);
        tx.send((headers, body)).unwrap();
        respond_with(stream, "application/json", br#"{"ok":true}"#);
    });

    let result = run_webwrite(
        Value::from(url),
        vec![Value::Struct(payload), Value::Struct(opts)],
    )
    .expect("webwrite");

    let (headers, body) = rx.recv().expect("request");
    let headers_lower = headers.to_ascii_lowercase();
    assert!(headers_lower.contains("content-type: application/json"));
    let body_text = String::from_utf8(body).expect("utf8 body");
    assert!(body_text.contains("\"title\":\"RunMat\""));
    assert!(body_text.contains("\"stars\":5"));

    match result {
        Value::Struct(reply) => {
            assert!(matches!(reply.fields.get("ok"), Some(Value::Bool(true))));
        }
        other => panic!("expected struct response, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webwrite_applies_basic_auth_and_custom_headers() {
    let payload = Value::from("");
    let mut header_struct = runmat_value::UnboundedStructValueBuilder::new();
    header_struct
        .fields
        .insert("X-Test".to_string(), Value::from("yes"));
    header_struct
        .fields
        .insert("Accept".to_string(), Value::from("text/plain"));
    let mut opts_struct = runmat_value::UnboundedStructValueBuilder::new();
    opts_struct
        .fields
        .insert("Username".to_string(), Value::from("ada"));
    opts_struct
        .fields
        .insert("Password".to_string(), Value::from("secret"));
    opts_struct.fields.insert(
        "HeaderFields".to_string(),
        Value::Struct(header_struct.finish()),
    );
    opts_struct
        .fields
        .insert("ContentType".to_string(), Value::from("text"));
    opts_struct
        .fields
        .insert("MediaType".to_string(), Value::from("text/plain"));

    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let (headers, _) = read_request(&mut stream);
        tx.send(headers).unwrap();
        respond_with(stream, "text/plain", b"OK");
    });

    let result = run_webwrite(
        Value::from(url),
        vec![payload, Value::Struct(opts_struct.finish())],
    )
    .expect("webwrite");

    let headers = rx.recv().expect("headers");
    let headers_lower = headers.to_ascii_lowercase();
    assert!(headers_lower.contains("authorization: basic"));
    assert!(headers_lower.contains("x-test: yes"));
    assert!(headers_lower.contains("accept: text/plain"));

    match result {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "OK");
        }
        other => panic!("expected char array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webwrite_supports_query_parameters() {
    let payload = Value::Struct(StructValue::new());
    let mut qp_struct = runmat_value::UnboundedStructValueBuilder::new();
    qp_struct.fields.insert("page".to_string(), Value::Num(2.0));
    qp_struct
        .fields
        .insert("verbose".to_string(), Value::Bool(true));
    let mut opts_struct = runmat_value::UnboundedStructValueBuilder::new();
    opts_struct.fields.insert(
        "QueryParameters".to_string(),
        Value::Struct(qp_struct.finish()),
    );

    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let (headers, _) = read_request(&mut stream);
        tx.send(headers).unwrap();
        respond_with(stream, "application/json", br#"{"ok":true}"#);
    });

    let _ = run_webwrite(
        Value::from(url.clone()),
        vec![payload, Value::Struct(opts_struct.finish())],
    )
    .expect("webwrite");

    let headers = rx.recv().expect("headers");
    let first_line = headers.lines().next().unwrap_or("");
    assert!(first_line.starts_with("POST "));
    assert!(first_line.contains("page=2"));
    assert!(first_line.contains("verbose=true"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn webwrite_binary_payload_respected() {
    let tensor = Tensor::new_integer(
        runmat_value::IntegerStorage::U8(vec![1, 2, 3, 255]),
        vec![4, 1],
    )
    .unwrap();
    let payload = Value::Tensor(tensor);
    let mut opts_struct = runmat_value::UnboundedStructValueBuilder::new();
    opts_struct
        .fields
        .insert("ContentType".to_string(), Value::from("binary"));
    opts_struct.fields.insert(
        "MediaType".to_string(),
        Value::from("application/octet-stream"),
    );

    let (tx, rx) = mpsc::channel();
    let url = spawn_server(move |mut stream| {
        let (headers, body) = read_request(&mut stream);
        tx.send((headers, body)).unwrap();
        respond_with(stream, "text/plain", b"OK");
    });

    let _ = run_webwrite(
        Value::from(url),
        vec![payload, Value::Struct(opts_struct.finish())],
    )
    .expect("webwrite");

    let (headers, body) = rx.recv().expect("request");
    let headers_lower = headers.to_ascii_lowercase();
    assert!(headers_lower.contains("content-type: application/octet-stream"));
    assert_eq!(body, vec![1, 2, 3, 255]);
}
