//! MATLAB-compatible `tcpserver` builtin for RunMat.

use once_cell::sync::OnceCell;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use thiserror::Error;

use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};

const BUILTIN_NAME: &str = "tcpserver";

const TCPSERVER_OUTPUT_SERVER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "server",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tcpserver handle struct for accept/close operations.",
}];
const TCPSERVER_INPUTS_ADDRESS_PORT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "address",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bind address or hostname.",
    },
    BuiltinParamDescriptor {
        name: "port",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Listening TCP port (0..65535).",
    },
];
const TCPSERVER_INPUTS_ADDRESS_PORT_NAME_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "address",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bind address or hostname.",
    },
    BuiltinParamDescriptor {
        name: "port",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Listening TCP port (0..65535).",
    },
    BuiltinParamDescriptor {
        name: "name_value_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/Value options such as Timeout, UserData, Name, and ByteOrder.",
    },
];
const TCPSERVER_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "server = tcpserver(address, port)",
        inputs: &TCPSERVER_INPUTS_ADDRESS_PORT,
        outputs: &TCPSERVER_OUTPUT_SERVER,
    },
    BuiltinSignatureDescriptor {
        label: "server = tcpserver(address, port, Name, Value, ...)",
        inputs: &TCPSERVER_INPUTS_ADDRESS_PORT_NAME_VALUE,
        outputs: &TCPSERVER_OUTPUT_SERVER,
    },
];

const TCPSERVER_ERROR_INVALID_ADDRESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.INVALID_ADDRESS",
    identifier: Some("RunMat:tcpserver:InvalidAddress"),
    when: "Address argument is not a valid string scalar.",
    message: "tcpserver: invalid address argument",
};
const TCPSERVER_ERROR_INVALID_PORT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.INVALID_PORT",
    identifier: Some("RunMat:tcpserver:InvalidPort"),
    when: "Port argument is non-scalar, non-integer, non-finite, or out of range.",
    message: "tcpserver: invalid port argument",
};
const TCPSERVER_ERROR_INVALID_NAME_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.INVALID_NAME_VALUE",
    identifier: Some("RunMat:tcpserver:InvalidNameValue"),
    when: "Name/Value arguments are malformed, unsupported, or have invalid values.",
    message: "tcpserver: invalid name-value arguments",
};
const TCPSERVER_ERROR_BIND_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.BIND_FAILED",
    identifier: Some("RunMat:tcpserver:BindFailed"),
    when: "Socket bind operation fails.",
    message: "tcpserver: unable to bind listener",
};
const TCPSERVER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.INTERNAL",
    identifier: Some("RunMat:tcpserver:InternalError"),
    when: "Internal listener metadata retrieval fails.",
    message: "tcpserver: internal error",
};
const TCPSERVER_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TCPSERVER_ERROR_INVALID_ADDRESS,
    TCPSERVER_ERROR_INVALID_PORT,
    TCPSERVER_ERROR_INVALID_NAME_VALUE,
    TCPSERVER_ERROR_BIND_FAILED,
    TCPSERVER_ERROR_INTERNAL,
];
pub const TCPSERVER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TCPSERVER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TCPSERVER_ERRORS,
};

pub(crate) const DEFAULT_TIMEOUT_SECONDS: f64 = 10.0;
pub(crate) const HANDLE_ID_FIELD: &str = "__tcpserver_id";

type SharedTcpServer = Arc<Mutex<TcpServerState>>;

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct TcpServerState {
    pub(crate) id: u64,
    pub(crate) listener: TcpListener,
    pub(crate) requested_address: String,
    pub(crate) local_addr: SocketAddr,
    pub(crate) timeout: f64,
    pub(crate) name: Option<String>,
    #[allow(dead_code)]
    pub(crate) byte_order: String,
}

#[derive(Default)]
struct TcpServerRegistry {
    next_id: u64,
    servers: HashMap<u64, SharedTcpServer>,
}

static TCP_SERVER_REGISTRY: OnceCell<Mutex<TcpServerRegistry>> = OnceCell::new();

fn registry() -> &'static Mutex<TcpServerRegistry> {
    TCP_SERVER_REGISTRY.get_or_init(|| Mutex::new(TcpServerRegistry::default()))
}

fn insert_server(
    listener: TcpListener,
    requested_address: String,
    local_addr: SocketAddr,
    options: &ParsedOptions,
) -> u64 {
    let mut guard = registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    guard.next_id = guard.next_id.wrapping_add(1);
    let id = guard.next_id;
    let state = TcpServerState {
        id,
        listener,
        requested_address,
        local_addr,
        timeout: options.timeout,
        name: options.name.clone(),
        byte_order: options.byte_order.clone(),
    };
    guard.servers.insert(id, Arc::new(Mutex::new(state)));
    id
}

#[allow(dead_code)]
pub(crate) fn server_handle(id: u64) -> Option<SharedTcpServer> {
    registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .servers
        .get(&id)
        .cloned()
}

pub(crate) fn close_server(id: u64) -> bool {
    let entry = {
        let mut guard = registry()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.servers.remove(&id)
    };
    entry.is_some()
}

pub(crate) fn close_all_servers() -> usize {
    let entries = {
        let mut guard = registry()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.servers.drain().collect::<Vec<_>>()
    };
    entries.len()
}

#[cfg(test)]
pub(super) fn remove_server_for_test(id: u64) {
    if let Some(entry) = registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .servers
        .remove(&id)
    {
        drop(entry);
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(super) fn clear_registry_for_test() {
    let mut guard = registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.servers.clear();
    guard.next_id = 0;
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::net::tcpserver")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tcpserver",
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
    notes: "Host networking only. GPU-resident scalars are gathered prior to socket binding.",
};

fn tcpserver_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tcpserver_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("tcpserver: ").unwrap_or(detail);
    tcpserver_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn tcpserver_flow(
    error: &'static BuiltinErrorDescriptor,
    message: impl AsRef<str>,
) -> RuntimeError {
    tcpserver_error_with_detail(error, message)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::net::tcpserver")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tcpserver",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

#[runtime_builtin(
    name = "tcpserver",
    category = "io/net",
    summary = "Create a TCP server that listens for MATLAB-compatible client connections.",
    keywords = "tcpserver,tcp,network,server",
    type_resolver(crate::builtins::io::type_resolvers::tcpserver_type),
    descriptor(crate::builtins::io::net::tcpserver::TCPSERVER_DESCRIPTOR),
    builtin_path = "crate::builtins::io::net::tcpserver"
)]
pub(crate) async fn tcpserver_builtin(
    address: Value,
    port: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let address = gather_if_needed_async(&address).await?;
    let port = gather_if_needed_async(&port).await?;

    let host = string_scalar(&address, "tcpserver address")
        .map_err(|err| tcpserver_flow(&TCPSERVER_ERROR_INVALID_ADDRESS, err.to_string()))?;
    let port = parse_port(&port)
        .map_err(|err| tcpserver_flow(&TCPSERVER_ERROR_INVALID_PORT, err.to_string()))?;

    let options = parse_name_value_pairs(rest).await?;

    let listener = TcpListener::bind((host.as_str(), port)).map_err(|err| {
        tcpserver_flow(
            &TCPSERVER_ERROR_BIND_FAILED,
            format!("tcpserver: unable to bind {host}:{port} ({err})"),
        )
    })?;
    let local_addr = listener
        .local_addr()
        .map_err(|err| tcpserver_flow(&TCPSERVER_ERROR_INTERNAL, format!("tcpserver: {err}")))?;

    let id = insert_server(listener, host.clone(), local_addr, &options);
    Ok(build_tcpserver_struct(id, &host, local_addr, &options))
}

#[derive(Clone)]
struct ParsedOptions {
    timeout: f64,
    user_data: Value,
    name: Option<String>,
    byte_order: String,
}

impl Default for ParsedOptions {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT_SECONDS,
            user_data: default_user_data(),
            name: None,
            byte_order: "little-endian".to_string(),
        }
    }
}

async fn parse_name_value_pairs(rest: Vec<Value>) -> BuiltinResult<ParsedOptions> {
    if rest.is_empty() {
        return Ok(ParsedOptions::default());
    }
    if !rest.len().is_multiple_of(2) {
        return Err(tcpserver_flow(
            &TCPSERVER_ERROR_INVALID_NAME_VALUE,
            "tcpserver: name-value arguments must appear in pairs",
        ));
    }

    let mut options = ParsedOptions::default();
    let mut iter = rest.into_iter();
    while let Some(name_raw) = iter.next() {
        let value_raw = iter
            .next()
            .expect("even-length vector ensures paired name/value");
        let name_value = gather_if_needed_async(&name_raw)
            .await
            .map_err(|err| tcpserver_flow(&TCPSERVER_ERROR_INTERNAL, err.message()))?;

        let name = string_scalar(&name_value, "OptionName").map_err(|err| {
            tcpserver_flow(
                &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                format!("tcpserver: invalid option name: {err}"),
            )
        })?;
        let lower = name.to_ascii_lowercase();
        match lower.as_str() {
            "timeout" => {
                let timeout_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpserver_flow(&TCPSERVER_ERROR_INTERNAL, err.message()))?;
                options.timeout = parse_timeout(&timeout_value).map_err(|err| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid Timeout value: {err}"),
                    )
                })?
            }
            "userdata" => options.user_data = value_raw,
            "name" => {
                let name_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpserver_flow(&TCPSERVER_ERROR_INTERNAL, err.message()))?;
                let text = string_scalar(&name_value, "Name").map_err(|err| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid Name value: {err}"),
                    )
                })?;
                options.name = Some(text);
            }
            "byteorder" => {
                let order_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpserver_flow(&TCPSERVER_ERROR_INTERNAL, err.message()))?;
                let raw_order = string_scalar(&order_value, "ByteOrder").map_err(|err| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid ByteOrder value: {err}"),
                    )
                })?;
                let canon = canonicalize_byte_order(&raw_order).ok_or_else(|| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: unsupported ByteOrder '{raw_order}'"),
                    )
                })?;
                options.byte_order = canon.to_string();
            }
            _ => {
                return Err(tcpserver_flow(
                    &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                    format!("tcpserver: unsupported option '{name}'"),
                ));
            }
        }
    }

    Ok(options)
}

fn build_tcpserver_struct(
    id: u64,
    requested_address: &str,
    local_addr: SocketAddr,
    options: &ParsedOptions,
) -> Value {
    let mut st = StructValue::new();

    let server_address = local_addr.ip().to_string();
    let server_port = local_addr.port();
    let name = options
        .name
        .clone()
        .unwrap_or_else(|| format!("tcpserver:{server_address}:{server_port}"));

    st.fields
        .insert("Type".to_string(), Value::String("tcpserver".to_string()));
    st.fields.insert("Name".to_string(), Value::String(name));
    st.fields.insert(
        "ServerAddress".to_string(),
        Value::String(server_address.clone()),
    );
    st.fields.insert(
        "ServerPort".to_string(),
        Value::Int(IntValue::U16(server_port)),
    );
    st.fields
        .insert("Port".to_string(), Value::Int(IntValue::U16(server_port)));
    st.fields.insert(
        "RequestedAddress".to_string(),
        Value::String(requested_address.to_string()),
    );
    st.fields
        .insert("Connected".to_string(), Value::Bool(false));
    st.fields
        .insert("ClientAddress".to_string(), Value::String(String::new()));
    st.fields
        .insert("ClientPort".to_string(), Value::Int(IntValue::I32(0)));
    st.fields.insert(
        "NumBytesAvailable".to_string(),
        Value::Int(IntValue::I32(0)),
    );
    st.fields
        .insert("BytesAvailableFcn".to_string(), default_user_data());
    st.fields.insert(
        "BytesAvailableFcnMode".to_string(),
        Value::String("byte".to_string()),
    );
    st.fields.insert(
        "BytesAvailableFcnCount".to_string(),
        Value::Int(IntValue::I32(1)),
    );
    st.fields
        .insert("ConnectionChangedFcn".to_string(), default_user_data());
    st.fields.insert(
        "ByteOrder".to_string(),
        Value::String(options.byte_order.clone()),
    );
    st.fields
        .insert("EnableBroadcast".to_string(), Value::Bool(false));
    st.fields
        .insert("EnableMulticast".to_string(), Value::Bool(false));
    st.fields
        .insert("EnableReuseAddress".to_string(), Value::Bool(false));
    st.fields
        .insert("KeepAlive".to_string(), Value::Bool(false));
    st.fields
        .insert(HANDLE_ID_FIELD.to_string(), Value::Int(IntValue::U64(id)));
    st.fields
        .insert("Timeout".to_string(), Value::Num(options.timeout));
    st.fields
        .insert("UserData".to_string(), options.user_data.clone());

    Value::Struct(st)
}

#[derive(Debug, Error)]
#[error("{message}")]
pub(crate) struct StringScalarError {
    message: String,
}

impl StringScalarError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

pub(crate) fn string_scalar(value: &Value, context: &str) -> Result<String, StringScalarError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        other => Err(StringScalarError::new(format!(
            "{context} must be a string scalar or character vector (got {other:?})"
        ))),
    }
}

#[derive(Debug, Error)]
#[error("{message}")]
pub(crate) struct PortParseError {
    message: String,
}

impl PortParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

pub(crate) fn parse_port(value: &Value) -> Result<u16, PortParseError> {
    let port = match value {
        Value::Int(int) => int.to_i64(),
        Value::Num(num) => {
            if !num.is_finite() {
                return Err(PortParseError::new("port must be finite"));
            }
            if num.fract() != 0.0 {
                return Err(PortParseError::new("port must be an integer value"));
            }
            *num as i64
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let raw = t.data[0];
            if !raw.is_finite() {
                return Err(PortParseError::new("port must be finite"));
            }
            if raw.fract() != 0.0 {
                return Err(PortParseError::new("port must be an integer value"));
            }
            raw as i64
        }
        Value::Tensor(_) => {
            return Err(PortParseError::new("port must be a numeric scalar"));
        }
        _ => {
            return Err(PortParseError::new("port must be a numeric scalar"));
        }
    };

    if !(0..=65_535).contains(&port) {
        return Err(PortParseError::new(format!(
            "port {port} is outside the valid range 0–65535"
        )));
    }
    Ok(port as u16)
}

#[derive(Debug, Error)]
enum TimeoutParseError {
    #[error("Timeout must be a scalar")]
    NonScalar,
    #[error("Timeout must be a numeric scalar")]
    NonNumeric,
    #[error("Timeout must be a finite, non-negative scalar")]
    NonFinite,
}

fn parse_timeout(value: &Value) -> Result<f64, TimeoutParseError> {
    let timeout = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        Value::Tensor(_) => return Err(TimeoutParseError::NonScalar),
        _ => {
            return Err(TimeoutParseError::NonNumeric);
        }
    };

    if !timeout.is_finite() || timeout < 0.0 {
        return Err(TimeoutParseError::NonFinite);
    }
    Ok(timeout)
}

pub(crate) fn canonicalize_byte_order(raw: &str) -> Option<&'static str> {
    let mut compact = String::with_capacity(raw.len());
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            compact.push(ch.to_ascii_lowercase());
        }
    }
    match compact.as_str() {
        "littleendian" | "little" => Some("little-endian"),
        "bigendian" | "big" => Some("big-endian"),
        _ => None,
    }
}

pub(crate) fn default_user_data() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{Tensor, Value};
    use std::net::TcpStream;
    use std::time::Duration;

    fn struct_field<'a>(value: &'a Value, name: &str) -> &'a Value {
        match value {
            Value::Struct(struct_value) => struct_value
                .fields
                .get(name)
                .unwrap_or_else(|| panic!("missing field {name}")),
            _ => panic!("expected struct result"),
        }
    }

    fn server_id(value: &Value) -> u64 {
        match struct_field(value, HANDLE_ID_FIELD) {
            Value::Int(IntValue::U64(id)) => *id,
            Value::Int(iv) => iv.to_i64() as u64,
            other => panic!("unexpected id representation {other:?}"),
        }
    }

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    fn run_tcpserver(address: Value, port: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tcpserver_builtin(address, port, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TCPSERVER_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"server = tcpserver(address, port)"));
        assert!(labels.contains(&"server = tcpserver(address, port, Name, Value, ...)"));
    }

    fn net_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::builtins::io::net::accept::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_accepts_loopback_connection() {
        let _guard = net_guard();

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let id = server_id(&result);
        let address = match struct_field(&result, "ServerAddress") {
            Value::String(s) => s.clone(),
            other => panic!("expected ServerAddress string, got {other:?}"),
        };
        let port = match struct_field(&result, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };

        TcpStream::connect((address.as_str(), port)).expect("connect to loopback server");
        remove_server_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_applies_timeout_option() {
        let _guard = net_guard();

        let result = run_tcpserver(
            Value::from("localhost"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("Timeout"), Value::Num(5.0)],
        )
        .expect("tcpserver");

        let timeout = match struct_field(&result, "Timeout") {
            Value::Num(n) => *n,
            other => panic!("expected numeric timeout, got {other:?}"),
        };
        assert!((timeout - 5.0).abs() < f64::EPSILON);

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_supports_custom_name() {
        let _guard = net_guard();

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("Name"), Value::from("CustomListener")],
        )
        .expect("tcpserver");

        let name = match struct_field(&result, "Name") {
            Value::String(s) => s.clone(),
            other => panic!("expected Name string, got {other:?}"),
        };
        assert_eq!(name, "CustomListener");

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_accepts_byte_order_option() {
        let _guard = net_guard();

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("ByteOrder"), Value::from("big-endian")],
        )
        .expect("tcpserver");

        let order = match struct_field(&result, "ByteOrder") {
            Value::String(s) => s.clone(),
            other => panic!("expected ByteOrder string, got {other:?}"),
        };
        assert_eq!(order, "big-endian");

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_rejects_invalid_byte_order() {
        let _guard = net_guard();
        let err = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(8000)),
            vec![Value::from("ByteOrder"), Value::from("middle-endian")],
        )
        .unwrap_err();
        assert_error_identifier(err, TCPSERVER_ERROR_INVALID_NAME_VALUE.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_accepts_scalar_tensor_port() {
        let _guard = net_guard();

        let tensor_port = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Tensor(tensor_port),
            Vec::new(),
        )
        .expect("tcpserver");

        let port = match struct_field(&result, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };
        assert!(port > 0);

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_rejects_invalid_port() {
        let err = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(-1)),
            Vec::new(),
        )
        .unwrap_err();
        assert_error_identifier(err, TCPSERVER_ERROR_INVALID_PORT.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_requires_name_value_pairs() {
        let _guard = net_guard();
        let err = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(9000)),
            vec![Value::from("Timeout")],
        )
        .unwrap_err();
        assert_error_identifier(err, TCPSERVER_ERROR_INVALID_NAME_VALUE.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_stores_userdata() {
        let _guard = net_guard();

        let mut user_struct_value = StructValue::new();
        user_struct_value
            .fields
            .insert("tag".to_string(), Value::from("demo"));
        let user_struct = Value::Struct(user_struct_value);

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("UserData"), user_struct.clone()],
        )
        .expect("tcpserver");

        let stored = struct_field(&result, "UserData");
        assert_eq!(stored, &user_struct);

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_times_out_connect_attempt() {
        let _guard = net_guard();

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("Timeout"), Value::Num(1.5)],
        )
        .expect("tcpserver");
        let id = server_id(&result);
        let port = match struct_field(&result, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_millis(10)))
            .expect("set timeout");

        remove_server_for_test(id);
    }
}
