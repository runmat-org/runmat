use super::{accept, read, readline, tcpclient, tcpserver, write};
use runmat_value::{IntValue, StructValue, Value};

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
