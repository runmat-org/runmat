use super::{accept, read, readline, tcpclient, tcpserver, write};
use crate::context::RuntimeContext;
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{CharArray, ForeignRef, ForeignResourceKey, ObjectInstance, Value};
use std::rc::Rc;

fn context_without_tcp() -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}

// A sealed receiver passes class/authority admission. Browser target rejection
// happens before its detached test token can reach any socket service.
fn receiver(class: runmat_types::StaticClassIdentity, name: &str) -> Value {
    let authority = ForeignRef::detached(
        ForeignResourceKey {
            host_identity: "test.browser.tcp".into(),
            handle: 1,
            generation: 1,
        },
        ForeignTypeIdentity {
            family: "runmat.tcp".into(),
            name: name.into(),
            version: 1,
        },
        ForeignOwnership::Shared,
        ForeignAffinity::OriginProcess,
        ForeignLifetime::Persistent,
    );
    Value::Object(ObjectInstance::new(class).with_tcp_authority(authority))
}

fn client() -> Value {
    receiver(runmat_builtins::TCPCLIENT_CLASS, "tcpclient")
}

fn server() -> Value {
    receiver(runmat_builtins::TCPSERVER_CLASS, "tcpserver")
}

fn identifier(result: crate::BuiltinResult<Value>) -> Option<String> {
    result
        .err()
        .and_then(|error| error.identifier().map(str::to_owned))
}

#[wasm_bindgen_test::wasm_bindgen_test]
fn valid_native_tcp_forms_reject_at_one_target_boundary() {
    let runtime = context_without_tcp();
    let cases = [
        identifier(futures::executor::block_on(tcpclient::tcpclient_builtin(
            Some(runtime.clone()),
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![],
        ))),
        identifier(futures::executor::block_on(tcpserver::tcpserver_builtin(
            Some(runtime.clone()),
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![],
        ))),
        identifier(futures::executor::block_on(accept::accept_builtin(
            Some(runtime.clone()),
            server(),
            vec![],
        ))),
        identifier(futures::executor::block_on(read::read_builtin(
            Some(runtime.clone()),
            client(),
            vec![],
        ))),
        identifier(futures::executor::block_on(readline::readline_builtin(
            Some(runtime.clone()),
            client(),
            vec![],
        ))),
        identifier(futures::executor::block_on(write::write_builtin(
            Some(runtime),
            client(),
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
    let runtime = context_without_tcp();
    let cases = [
        (
            identifier(futures::executor::block_on(tcpclient::tcpclient_builtin(
                Some(runtime.clone()),
                Value::from("127.0.0.1"),
                Value::Num(80.0),
                vec![Value::from("Timeout")],
            ))),
            runmat_builtins::TCPCLIENT_ERROR_INVALID_NAME_VALUE.identifier,
        ),
        (
            identifier(futures::executor::block_on(tcpserver::tcpserver_builtin(
                Some(runtime.clone()),
                Value::from("127.0.0.1"),
                Value::Num(80.0),
                vec![Value::from("Timeout")],
            ))),
            runmat_builtins::TCPSERVER_ERROR_INVALID_NAME_VALUE.identifier,
        ),
        (
            identifier(futures::executor::block_on(accept::accept_builtin(
                Some(runtime.clone()),
                server(),
                vec![Value::from("Timeout")],
            ))),
            runmat_builtins::ACCEPT_ERROR_INVALID_NAME_VALUE.identifier,
        ),
        (
            identifier(futures::executor::block_on(read::read_builtin(
                Some(runtime.clone()),
                client(),
                vec![Value::Num(-1.0)],
            ))),
            runmat_builtins::READ_ERROR_INVALID_COUNT.identifier,
        ),
        (
            identifier(futures::executor::block_on(readline::readline_builtin(
                Some(runtime.clone()),
                client(),
                vec![Value::Num(1.0)],
            ))),
            runmat_builtins::READLINE_ERROR_INVALID_ARGUMENTS.identifier,
        ),
        (
            identifier(futures::executor::block_on(write::write_builtin(
                Some(runtime),
                client(),
                Value::Num(1.0),
                vec![Value::from("not-a-datatype")],
            ))),
            runmat_builtins::WRITE_ERROR_INVALID_DATATYPE.identifier,
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual.as_deref(), expected);
    }
}

#[wasm_bindgen_test::wasm_bindgen_test]
fn known_invalid_primary_operands_reject_before_native_target_access() {
    let runtime = context_without_tcp();
    let cases = [
        (
            identifier(futures::executor::block_on(tcpclient::tcpclient_builtin(
                Some(runtime.clone()),
                Value::Num(4.0),
                Value::Num(80.0),
                vec![],
            ))),
            runmat_builtins::TCPCLIENT_ERROR_INVALID_ADDRESS.identifier,
        ),
        (
            identifier(futures::executor::block_on(tcpserver::tcpserver_builtin(
                Some(runtime.clone()),
                Value::from("127.0.0.1"),
                Value::Num(-1.0),
                vec![],
            ))),
            runmat_builtins::TCPSERVER_ERROR_INVALID_PORT.identifier,
        ),
        (
            identifier(futures::executor::block_on(accept::accept_builtin(
                Some(runtime.clone()),
                Value::Num(1.0),
                vec![],
            ))),
            runmat_builtins::ACCEPT_ERROR_INVALID_SERVER.identifier,
        ),
        (
            identifier(futures::executor::block_on(read::read_builtin(
                Some(runtime.clone()),
                Value::Num(1.0),
                vec![],
            ))),
            runmat_builtins::READ_ERROR_INVALID_CLIENT.identifier,
        ),
        (
            identifier(futures::executor::block_on(readline::readline_builtin(
                Some(runtime.clone()),
                Value::Num(1.0),
                vec![],
            ))),
            runmat_builtins::READLINE_ERROR_INVALID_CLIENT.identifier,
        ),
        (
            identifier(futures::executor::block_on(write::write_builtin(
                Some(runtime),
                client(),
                Value::Object(ObjectInstance::new(runmat_builtins::TCPCLIENT_CLASS)),
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
    let runtime = context_without_tcp();
    let paged_option = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("Timeout".chars().collect(), vec![1, 1, 7])
            .unwrap(),
    );
    let trailing_option = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("Timeout".chars().collect(), vec![1, 7, 1])
            .unwrap(),
    );
    assert_eq!(
        identifier(futures::executor::block_on(tcpclient::tcpclient_builtin(
            Some(runtime.clone()),
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![paged_option, Value::Num(1.0)],
        )))
        .as_deref(),
        runmat_builtins::TCPCLIENT_ERROR_INVALID_NAME_VALUE.identifier,
    );
    assert_eq!(
        identifier(futures::executor::block_on(tcpserver::tcpserver_builtin(
            Some(runtime.clone()),
            Value::from("127.0.0.1"),
            Value::Num(80.0),
            vec![trailing_option, Value::Num(1.0)],
        )))
        .as_deref(),
        Some("RunMat:network:UnsupportedTarget"),
    );
    let paged_datatype = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("uint8".chars().collect(), vec![1, 1, 5])
            .unwrap(),
    );
    assert_eq!(
        identifier(futures::executor::block_on(read::read_builtin(
            Some(runtime.clone()),
            client(),
            vec![Value::Num(1.0), paged_datatype],
        )))
        .as_deref(),
        runmat_builtins::READ_ERROR_INVALID_DATATYPE.identifier,
    );
    let trailing_datatype = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("uint8".chars().collect(), vec![1, 5, 1])
            .unwrap(),
    );
    assert_eq!(
        identifier(futures::executor::block_on(write::write_builtin(
            Some(runtime),
            client(),
            Value::Num(1.0),
            vec![trailing_datatype],
        )))
        .as_deref(),
        Some("RunMat:network:UnsupportedTarget"),
    );
}
