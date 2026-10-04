use super::{accept, read, readline, tcpclient, tcpserver, write};
use runmat_value::{CharArray, IntValue, StructValue, Value};

fn handle(field: &str) -> Value {
    let mut value = StructValue::new();
    value
        .fields
        .insert(field.to_string(), Value::Int(IntValue::U64(1)));
    Value::Struct(value)
}

fn identifier(result: crate::BuiltinResult<Value>) -> Option<String> {
    result
        .err()
        .and_then(|error| error.identifier().map(str::to_owned))
}

#[wasm_bindgen_test::wasm_bindgen_test]
fn valid_native_tcp_forms_reject_at_one_target_boundary() {
    let client = handle(super::client_state::CLIENT_HANDLE_FIELD);
    let server = handle(super::server_state::HANDLE_ID_FIELD);
    let cases = [
        identifier(futures::executor::block_on(tcpclient::tcpclient_builtin(
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![],
        ))),
        identifier(futures::executor::block_on(tcpserver::tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![],
        ))),
        identifier(futures::executor::block_on(accept::accept_builtin(
            server,
            vec![],
        ))),
        identifier(futures::executor::block_on(read::read_builtin(
            client.clone(),
            vec![],
        ))),
        identifier(futures::executor::block_on(readline::readline_builtin(
            client.clone(),
            vec![],
        ))),
        identifier(futures::executor::block_on(write::write_builtin(
            client,
            Value::Num(1.0),
            vec![],
        ))),
    ];
    for actual in cases {
        assert_eq!(actual.as_deref(), Some("RunMat:network:UnsupportedTarget"));
    }
}

#[wasm_bindgen_test::wasm_bindgen_test]
fn malformed_host_controls_keep_their_diagnostics_before_target_rejection() {
    let client = handle(super::client_state::CLIENT_HANDLE_FIELD);
    let server = handle(super::server_state::HANDLE_ID_FIELD);
    let cases = [
        identifier(futures::executor::block_on(tcpclient::tcpclient_builtin(
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![Value::from("Timeout")],
        ))),
        identifier(futures::executor::block_on(tcpserver::tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![Value::from("Timeout")],
        ))),
        identifier(futures::executor::block_on(accept::accept_builtin(
            server,
            vec![Value::from("Timeout")],
        ))),
        identifier(futures::executor::block_on(read::read_builtin(
            client.clone(),
            vec![Value::Num(-1.0)],
        ))),
        identifier(futures::executor::block_on(readline::readline_builtin(
            client.clone(),
            vec![Value::Num(1.0)],
        ))),
        identifier(futures::executor::block_on(write::write_builtin(
            client,
            Value::Num(1.0),
            vec![Value::from("not-a-datatype")],
        ))),
    ];
    for actual in cases {
        assert!(actual.is_some());
        assert_ne!(actual.as_deref(), Some("RunMat:network:UnsupportedTarget"));
    }
}

#[wasm_bindgen_test::wasm_bindgen_test]
fn known_invalid_primary_operands_reject_before_native_target_access() {
    let fake_client = handle(super::client_state::CLIENT_HANDLE_FIELD);
    let cases = [
        (
            identifier(futures::executor::block_on(tcpclient::tcpclient_builtin(
                Value::Num(4.0),
                Value::Num(80.0),
                vec![],
            ))),
            runmat_builtins::TCPCLIENT_ERROR_INVALID_ADDRESS.identifier,
        ),
        (
            identifier(futures::executor::block_on(tcpserver::tcpserver_builtin(
                Value::from("127.0.0.1"),
                Value::Num(-1.0),
                vec![],
            ))),
            runmat_builtins::TCPSERVER_ERROR_INVALID_PORT.identifier,
        ),
        (
            identifier(futures::executor::block_on(accept::accept_builtin(
                Value::Num(1.0),
                vec![],
            ))),
            runmat_builtins::ACCEPT_ERROR_INVALID_SERVER.identifier,
        ),
        (
            identifier(futures::executor::block_on(read::read_builtin(
                Value::Num(1.0),
                vec![],
            ))),
            runmat_builtins::READ_ERROR_INVALID_CLIENT.identifier,
        ),
        (
            identifier(futures::executor::block_on(readline::readline_builtin(
                Value::Num(1.0),
                vec![],
            ))),
            runmat_builtins::READLINE_ERROR_INVALID_CLIENT.identifier,
        ),
        (
            identifier(futures::executor::block_on(write::write_builtin(
                fake_client,
                Value::Struct(StructValue::new()),
                vec![],
            ))),
            runmat_builtins::WRITE_ERROR_INVALID_DATA.identifier,
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual.as_deref(), expected);
    }
}

#[wasm_bindgen_test::wasm_bindgen_test]
fn scalar_text_controls_reject_pages_but_keep_trailing_singletons() {
    let paged_option = Value::CharArray(
        CharArray::new_with_shape("Timeout".chars().collect(), vec![1, 1, 7]).unwrap(),
    );
    let trailing_option = Value::CharArray(
        CharArray::new_with_shape("Timeout".chars().collect(), vec![1, 7, 1]).unwrap(),
    );
    let fake_client = handle(super::client_state::CLIENT_HANDLE_FIELD);
    assert_eq!(
        identifier(futures::executor::block_on(tcpclient::tcpclient_builtin(
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![paged_option, Value::Num(1.0)],
        )))
        .as_deref(),
        runmat_builtins::TCPCLIENT_ERROR_INVALID_NAME_VALUE.identifier,
    );
    assert_eq!(
        identifier(futures::executor::block_on(tcpserver::tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![trailing_option, Value::Num(1.0)],
        )))
        .as_deref(),
        Some("RunMat:network:UnsupportedTarget"),
    );
    let paged_datatype = Value::CharArray(
        CharArray::new_with_shape("uint8".chars().collect(), vec![1, 1, 5]).unwrap(),
    );
    assert_eq!(
        identifier(futures::executor::block_on(read::read_builtin(
            fake_client.clone(),
            vec![Value::Num(1.0), paged_datatype],
        )))
        .as_deref(),
        runmat_builtins::READ_ERROR_INVALID_DATATYPE.identifier,
    );
    let trailing_datatype = Value::CharArray(
        CharArray::new_with_shape("uint8".chars().collect(), vec![1, 5, 1]).unwrap(),
    );
    assert_eq!(
        identifier(futures::executor::block_on(write::write_builtin(
            fake_client,
            Value::Num(1.0),
            vec![trailing_datatype],
        )))
        .as_deref(),
        Some("RunMat:network:UnsupportedTarget"),
    );
}
