//! MATLAB-compatible `close` builtin for networking resources in RunMat.

use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

use super::accept::{
    close_all_clients, close_client, close_clients_for_server, CLIENT_HANDLE_FIELD,
};
use super::tcpserver::{close_all_servers, close_server, HANDLE_ID_FIELD};

const MESSAGE_ID_INVALID_ARGUMENT: &str = "MATLAB:close:InvalidArgument";
const MESSAGE_ID_INVALID_HANDLE: &str = "MATLAB:close:InvalidHandle";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::net::close")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "close",
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
    notes:
        "Networking resources are host-only; the builtin gathers GPU values before closing handles.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::net::close")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "close",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtins execute eagerly on the CPU; close participates only in host bookkeeping.",
};

#[runtime_builtin(
    name = "close",
    category = "io/net",
    summary = "Close TCP clients or servers created by tcpclient, tcpserver, or accept.",
    keywords = "close,tcpclient,tcpserver,networking",
    builtin_path = "crate::builtins::io::net::close"
)]
fn close_builtin(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        let closed = close_everything();
        return Ok(Value::Num(if closed { 1.0 } else { 0.0 }));
    }

    let mut any_closed = false;
    for raw in args {
        let gathered = gather_if_needed(&raw)
            .map_err(|err| runtime_error(MESSAGE_ID_INVALID_ARGUMENT, err))?;
        any_closed |= close_value(&gathered)?;
    }

    Ok(Value::Num(if any_closed { 1.0 } else { 0.0 }))
}

fn close_value(value: &Value) -> Result<bool, String> {
    match value {
        Value::Struct(st) => close_struct(st),
        Value::String(text) => close_command(text),
        Value::CharArray(chars) => {
            if chars.data.is_empty() {
                Ok(false)
            } else if chars.rows == 1 {
                let text: String = chars.data.iter().collect();
                close_command(&text)
            } else {
                Err(runtime_error(
                    MESSAGE_ID_INVALID_ARGUMENT,
                    "close: character arrays must be a single row of text",
                ))
            }
        }
        Value::StringArray(sa) => match sa.data.len() {
            0 => Ok(false),
            1 => close_command(&sa.data[0]),
            _ => Err(runtime_error(
                MESSAGE_ID_INVALID_ARGUMENT,
                "close: string array inputs must be scalar",
            )),
        },
        Value::Cell(cell) => {
            let mut closed = false;
            for element in &cell.data {
                let inner = unsafe { &*element.as_raw() };
                closed |= close_value(inner)?;
            }
            Ok(closed)
        }
        Value::Tensor(tensor) if tensor.data.is_empty() => Ok(false),
        Value::LogicalArray(logical) if logical.is_empty() => Ok(false),
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_ARGUMENT,
            format!("close: unsupported argument {value:?}"),
        )),
    }
}

fn close_struct(struct_value: &StructValue) -> Result<bool, String> {
    if let Some(id_value) = struct_value.fields.get(CLIENT_HANDLE_FIELD) {
        let id = value_to_u64(id_value).ok_or_else(|| {
            runtime_error(
                MESSAGE_ID_INVALID_HANDLE,
                "close: tcpclient identifier is missing or invalid",
            )
        })?;
        return Ok(close_client(id));
    }

    if let Some(id_value) = struct_value.fields.get(HANDLE_ID_FIELD) {
        let id = value_to_u64(id_value).ok_or_else(|| {
            runtime_error(
                MESSAGE_ID_INVALID_HANDLE,
                "close: tcpserver identifier is missing or invalid",
            )
        })?;
        let clients_closed = close_clients_for_server(id);
        let server_closed = close_server(id);
        return Ok(server_closed || clients_closed > 0);
    }

    Err(runtime_error(
        MESSAGE_ID_INVALID_HANDLE,
        "close: expected tcpclient or tcpserver struct",
    ))
}

fn close_command(raw: &str) -> Result<bool, String> {
    let token = raw.trim().to_ascii_lowercase();
    match token.as_str() {
        "all" => Ok(close_everything()),
        "clients" | "client" => Ok(close_all_clients() > 0),
        "servers" | "server" => Ok(close_all_servers() > 0),
        "" => Ok(false),
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_ARGUMENT,
            format!("close: unrecognised option '{raw}'"),
        )),
    }
}

fn close_everything() -> bool {
    let clients = close_all_clients();
    let servers = close_all_servers();
    clients > 0 || servers > 0
}

fn value_to_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Int(int) => {
            let raw = int.to_i64();
            if raw >= 0 {
                Some(raw as u64)
            } else {
                None
            }
        }
        Value::Num(num) => {
            if num.is_finite() && *num >= 0.0 && num.fract() == 0.0 {
                Some(*num as u64)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn runtime_error(id: &'static str, message: impl Into<String>) -> String {
    format!("{id}: {}", message.into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::net::accept::{accept_builtin, client_handle};
    use crate::builtins::io::net::tcpclient::tcpclient_builtin;
    use crate::builtins::io::net::tcpserver::{server_handle, tcpserver_builtin};
    use once_cell::sync::Lazy;
    use runmat_builtins::{
        CellArray, CharArray, IntValue, StringArray, StructValue, Tensor, Value,
    };
    use std::net::{TcpListener, TcpStream};
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    static TEST_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn client_id(value: &Value) -> u64 {
        match value {
            Value::Struct(st) => match st.fields.get(CLIENT_HANDLE_FIELD) {
                Some(Value::Int(iv)) => iv.to_i64() as u64,
                Some(Value::Num(n)) => *n as u64,
                other => panic!("unexpected client id {other:?}"),
            },
            other => panic!("expected tcpclient struct, got {other:?}"),
        }
    }

    fn server_id(value: &Value) -> u64 {
        match value {
            Value::Struct(st) => match st.fields.get(HANDLE_ID_FIELD) {
                Some(Value::Int(iv)) => iv.to_i64() as u64,
                Some(Value::Num(n)) => *n as u64,
                other => panic!("unexpected server id {other:?}"),
            },
            other => panic!("expected tcpserver struct, got {other:?}"),
        }
    }

    fn server_address(value: &Value) -> (String, u16) {
        match value {
            Value::Struct(st) => {
                let address = match st.fields.get("ServerAddress") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::CharArray(ca)) if ca.rows == 1 => ca.data.iter().collect(),
                    other => panic!("unexpected ServerAddress {other:?}"),
                };
                let port = match st.fields.get("ServerPort") {
                    Some(Value::Int(iv)) => iv.to_i64() as u16,
                    other => panic!("unexpected ServerPort {other:?}"),
                };
                (address, port)
            }
            other => panic!("expected tcpserver struct, got {other:?}"),
        }
    }

    fn spawn_loopback_client() -> Value {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let accept_thread = thread::spawn(move || {
            let (_stream, _) = listener.accept().expect("accept");
            thread::sleep(Duration::from_millis(10));
        });

        let client = tcpclient_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(port as i32)),
            Vec::new(),
        )
        .expect("tcpclient");

        accept_thread.join().expect("join accept thread");
        client
    }

    fn spawn_tcp_server() -> Value {
        tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver")
    }

    fn accept_from_server(server: &Value) -> Value {
        let (host, port) = server_address(server);
        let connector = thread::spawn(move || {
            let _stream = TcpStream::connect((host.as_str(), port)).expect("connect client");
        });
        let accepted = accept_builtin(server.clone(), Vec::new()).expect("accept client");
        connector.join().expect("join connector");
        accepted
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_tcpclient_releases_handle() {
        let _lock = TEST_GUARD.lock().unwrap();

        let client = spawn_loopback_client();

        let cid = client_id(&client);
        let status = close_builtin(vec![client.clone()]).expect("close");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(cid).is_none());

        let second = close_builtin(vec![client]).expect("close again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_tcpserver_releases_listener_and_clients() {
        let _lock = TEST_GUARD.lock().unwrap();

        let server = spawn_tcp_server();
        let sid = server_id(&server);
        let accepted = accept_from_server(&server);
        let accepted_id = client_id(&accepted);

        let status = close_builtin(vec![server.clone()]).expect("close server");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(accepted_id).is_none());
        assert!(server_handle(sid).is_none());

        let second = close_builtin(vec![server]).expect("close server again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_multiple_handles_in_single_call() {
        let _lock = TEST_GUARD.lock().unwrap();

        let standalone_client = spawn_loopback_client();
        let standalone_id = client_id(&standalone_client);

        let server = spawn_tcp_server();
        let sid = server_id(&server);
        let accepted = accept_from_server(&server);
        let accepted_id = client_id(&accepted);

        let status = close_builtin(vec![
            standalone_client.clone(),
            accepted.clone(),
            server.clone(),
        ])
        .expect("close resources");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(standalone_id).is_none());
        assert!(client_handle(accepted_id).is_none());
        assert!(server_handle(sid).is_none());

        let second = close_builtin(vec![standalone_client, accepted, server])
            .expect("close resources again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_returns_zero_when_no_resources() {
        let _lock = TEST_GUARD.lock().unwrap();

        let _ = close_builtin(Vec::new()).expect("initial cleanup");
        let status = close_builtin(Vec::new()).expect("close without resources");
        assert_eq!(status, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_string_clients_option() {
        let _lock = TEST_GUARD.lock().unwrap();

        let client_a = spawn_loopback_client();
        let client_b = spawn_loopback_client();
        let id_a = client_id(&client_a);
        let id_b = client_id(&client_b);

        let status = close_builtin(vec![Value::from("clients")]).expect("close clients");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(id_a).is_none());
        assert!(client_handle(id_b).is_none());

        let second = close_builtin(vec![Value::from("clients")]).expect("close clients again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_string_servers_option() {
        let _lock = TEST_GUARD.lock().unwrap();

        let server = spawn_tcp_server();
        let sid = server_id(&server);

        let status = close_builtin(vec![Value::from("servers")]).expect("close servers");
        assert_eq!(status, Value::Num(1.0));
        assert!(server_handle(sid).is_none());

        let second = close_builtin(vec![Value::from("servers")]).expect("close servers again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_string_all_option() {
        let _lock = TEST_GUARD.lock().unwrap();

        let standalone = spawn_loopback_client();
        let standalone_id = client_id(&standalone);

        let server = spawn_tcp_server();
        let sid = server_id(&server);
        let accepted = accept_from_server(&server);
        let accepted_id = client_id(&accepted);

        let status = close_builtin(vec![Value::from("all")]).expect("close all");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(standalone_id).is_none());
        assert!(client_handle(accepted_id).is_none());
        assert!(server_handle(sid).is_none());

        let second = close_builtin(vec![Value::from("all")]).expect("close all again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_char_array_command() {
        let _lock = TEST_GUARD.lock().unwrap();

        let client = spawn_loopback_client();
        let cid = client_id(&client);
        let status = close_builtin(vec![Value::CharArray(CharArray::new_row("clients"))])
            .expect("close char command");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(cid).is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_cell_array_arguments() {
        let _lock = TEST_GUARD.lock().unwrap();

        let client = spawn_loopback_client();
        let client_id_value = client_id(&client);
        let server = spawn_tcp_server();
        let server_id_value = server_id(&server);
        let accepted = accept_from_server(&server);
        let accepted_id = client_id(&accepted);

        let cell =
            CellArray::new(vec![client, accepted, server, Value::from("clients")], 1, 4).unwrap();
        let status = close_builtin(vec![Value::Cell(cell)]).expect("close cell inputs");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(client_id_value).is_none());
        assert!(client_handle(accepted_id).is_none());
        assert!(server_handle(server_id_value).is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_invalid_argument_errors() {
        let _lock = TEST_GUARD.lock().unwrap();

        let err = close_builtin(vec![Value::Num(13.0)]).unwrap_err();
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_ARGUMENT),
            "error did not include id: {err}"
        );

        let multi = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let err = close_builtin(vec![Value::CharArray(multi)]).unwrap_err();
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_ARGUMENT),
            "char array error missing id: {err}"
        );

        let strings =
            StringArray::new(vec!["clients".to_string(), "servers".to_string()], vec![2]).unwrap();
        let err = close_builtin(vec![Value::StringArray(strings)]).unwrap_err();
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_ARGUMENT),
            "string array error missing id: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_invalid_struct_errors() {
        let _lock = TEST_GUARD.lock().unwrap();

        let st = StructValue::new();
        let err = close_builtin(vec![Value::Struct(st)]).unwrap_err();
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_HANDLE),
            "error did not include id: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_empty_array_argument_returns_zero() {
        let _lock = TEST_GUARD.lock().unwrap();

        let empty = Tensor::new(vec![], vec![0]).expect("empty tensor");
        let status = close_builtin(vec![Value::Tensor(empty)]).expect("close empty tensor");
        assert_eq!(status, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn close_with_wgpu_provider_active() {
        let _lock = TEST_GUARD.lock().unwrap();

        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let client = spawn_loopback_client();
        let cid = client_id(&client);

        let status = close_builtin(vec![client]).expect("close with provider active");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(cid).is_none());
    }
}
