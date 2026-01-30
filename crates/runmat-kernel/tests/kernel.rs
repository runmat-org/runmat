use futures::executor::block_on;
use runmat_kernel::{
    execution::ExecutionStatus,
    protocol::{ExecuteRequest, JupyterMessage, MessageType},
    ConnectionInfo, ExecutionEngine, KernelConfig, KernelServer,
};
use std::collections::HashMap;
use tempfile::NamedTempFile;

fn assign_ports_or_skip(conn: &mut ConnectionInfo) -> bool {
    match conn.assign_ports() {
        Ok(()) => true,
        Err(err) if err.to_string().contains("Operation not permitted") => {
            eprintln!("skipping port assignment: {err}");
            false
        }
        Err(err) => panic!("{err}"),
    }
}

#[test]
fn test_connection_info_roundtrip() {
    let mut conn = ConnectionInfo::default();
    if !assign_ports_or_skip(&mut conn) {
        return;
    }

    // Test file I/O
    let temp_file = NamedTempFile::new().unwrap();
    conn.write_to_file(temp_file.path()).unwrap();

    let loaded = ConnectionInfo::from_file(temp_file.path()).unwrap();
    assert_eq!(conn.key, loaded.key);
    assert_eq!(conn.shell_port, loaded.shell_port);
    assert_eq!(conn.iopub_port, loaded.iopub_port);
}

#[test]
fn test_jupyter_message_protocol() {
    let execute_req = ExecuteRequest {
        code: "x = magic(5)".to_string(),
        silent: false,
        store_history: true,
        user_expressions: HashMap::new(),
        allow_stdin: false,
        stop_on_error: true,
    };

    let content = serde_json::to_value(&execute_req).unwrap();
    let message = JupyterMessage::new(MessageType::ExecuteRequest, "test-session", content);

    // Test message serialization
    let json = message.to_json().unwrap();
    let parsed = JupyterMessage::from_json(&json).unwrap();

    assert_eq!(message.header.msg_type, parsed.header.msg_type);
    assert_eq!(message.header.session, parsed.header.session);
    assert_eq!(message.content, parsed.content);
}

#[test]
fn test_execution_engine_integration() {
    let mut engine = ExecutionEngine::new();

    // Test successful execution
    let result = block_on(engine.execute("a = 1; b = 2; c = a + b")).unwrap();
    assert_eq!(result.status, ExecutionStatus::Success);
    assert_eq!(engine.execution_count(), 1);

    // Test parse error
    let result = block_on(engine.execute("invalid syntax here")).unwrap();
    assert_eq!(result.status, ExecutionStatus::Error);
    assert_eq!(engine.execution_count(), 2);

    // Test runtime error
    let result = block_on(engine.execute("x = unknown_variable")).unwrap();
    assert_eq!(result.status, ExecutionStatus::Error);
    assert_eq!(engine.execution_count(), 3);
}

#[test]
fn test_kernel_server_lifecycle() {
    let mut config = KernelConfig::default();
    if !assign_ports_or_skip(&mut config.connection) {
        return;
    }

    let server = KernelServer::new(config);
    let kernel_info = server.kernel_info();

    assert_eq!(kernel_info.implementation, "runmat");
    assert_eq!(kernel_info.language_info.name, "matlab");
    assert_eq!(kernel_info.protocol_version, "5.3");
}

#[tokio::test]
async fn test_execution_engine_async() {
    let mut engine = ExecutionEngine::new();

    // Test multiple sequential executions (each independent for now)
    let codes = [
        "x = 1",
        "y = 2",
        "z = 1 + 2",      // Use literal values since x,y don't persist
        "result = 3 * 2", // Use literal values
    ];

    for (i, code) in codes.iter().enumerate() {
        let result = block_on(engine.execute(code)).unwrap();
        assert_eq!(
            result.status,
            ExecutionStatus::Success,
            "Failed for code: {code}"
        );
        assert_eq!(engine.execution_count(), (i + 1) as u64);
    }

    let stats = engine.stats();
    assert_eq!(stats.execution_count, 4);
    assert!(stats.timeout_seconds.is_some());
}

#[test]
fn test_matlab_syntax_execution() {
    let mut engine = ExecutionEngine::new();

    // Test syntax that currently works with the interpreter
    let test_cases = vec![
        ("x = 1", ExecutionStatus::Success),
        ("y = 5 + 2", ExecutionStatus::Success),
        ("z = 1 * 3", ExecutionStatus::Success),
        ("a = 2 - 1", ExecutionStatus::Success),
        ("b = 8 / 4", ExecutionStatus::Success),
        // Control flow using arithmetic conditions (like the interpreter tests)
        (
            "if 1-1; result = 0; else; result = 1; end",
            ExecutionStatus::Success,
        ),
        ("for i = 1:3; sum = i; end", ExecutionStatus::Success),
    ];

    for (code, expected_status) in test_cases {
        let result = block_on(engine.execute(code)).unwrap();
        assert_eq!(result.status, expected_status, "Failed for code: {code}");
    }
}

#[test]
fn test_unsupported_syntax_errors() {
    let mut engine = ExecutionEngine::new();

    // Test syntax that should currently fail (as expected)
    let test_cases = vec![
        ("if x > 0; y = 1; end", ExecutionStatus::Error), // Greater than not implemented yet
        ("result = matrix(1)", ExecutionStatus::Error),   // Indexing not supported yet
                                                          // Range with step is now supported: ("x = 1:2:5", ExecutionStatus::Error),
    ];

    for (code, expected_status) in test_cases {
        let result = block_on(engine.execute(code)).unwrap();
        assert_eq!(
            result.status, expected_status,
            "Failed for code: {code} (should have failed)"
        );
    }
}

#[test]
fn test_matrix_syntax_support() {
    let mut engine = ExecutionEngine::new();

    // Test that matrix syntax is properly supported
    let test_cases = vec![
        ("matrix = [1, 2, 3]", ExecutionStatus::Success),
        ("zeros_mat = [0, 0]", ExecutionStatus::Success),
        ("single = [42]", ExecutionStatus::Success),
    ];

    for (code, expected_status) in test_cases {
        let result = block_on(engine.execute(code)).unwrap();
        assert_eq!(result.status, expected_status, "Failed for code: {code}");
    }
}

#[test]
fn test_error_handling_details() {
    let mut engine = ExecutionEngine::new();

    // Test parse error details
    let result = block_on(engine.execute("x = 1 +")).unwrap();
    assert_eq!(result.status, ExecutionStatus::Error);
    assert!(result.error.is_some());

    let error = result.error.unwrap();
    assert_eq!(error.error_type, "SyntaxError");
    assert!(!error.message.is_empty());
    assert!(!error.traceback.is_empty());

    // Test undefined variable error
    let result = block_on(engine.execute("y = undefined_var")).unwrap();
    assert_eq!(result.status, ExecutionStatus::Error);
    assert!(result.error.is_some());
}

#[test]
fn test_connection_validation() {
    let mut conn = ConnectionInfo::default();

    // Should fail with unassigned ports
    assert!(conn.validate().is_err());

    // Should pass after port assignment
    if !assign_ports_or_skip(&mut conn) {
        return;
    }
    conn.validate().unwrap();

    // Should fail with empty key
    conn.key.clear();
    assert!(conn.validate().is_err());

    // Should fail with empty IP
    conn.key = "test-key".to_string();
    conn.ip.clear();
    assert!(conn.validate().is_err());
}

#[test]
fn test_kernel_config_defaults() {
    let config = KernelConfig::default();

    assert_eq!(config.connection.ip, "127.0.0.1");
    assert_eq!(config.connection.transport, "tcp");
    assert_eq!(config.connection.signature_scheme, "hmac-sha256");
    assert!(!config.connection.key.is_empty());
    assert!(!config.debug);
    assert!(config.execution_timeout.is_some());
    assert_eq!(config.execution_timeout.unwrap(), 300);
}
