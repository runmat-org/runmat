use runmat_kernel::{
    protocol::{ExecuteRequest, JupyterMessage, MessageType},
    transport::{recv_jupyter_message, send_jupyter_message},
    KernelConfig, KernelServer,
};
use runmat_time::Instant;
use std::collections::HashMap;

fn assign_ports_or_skip(config: &mut KernelConfig) -> bool {
    match config.connection.assign_ports() {
        Ok(()) => true,
        Err(err) if err.to_string().contains("Operation not permitted") => {
            eprintln!("skipping ZMQ integration test: {err}");
            false
        }
        Err(err) => panic!("{err}"),
    }
}

fn poll_readable(socket: &zmq::Socket, timeout_ms: i64) -> bool {
    // Fallback polling via get_events since PollItem construction is limited
    let start = Instant::now();
    loop {
        if let Ok(ev) = socket.get_events() {
            if ev.contains(zmq::POLLIN) {
                return true;
            }
        }
        if start.elapsed().as_millis() as i64 >= timeout_ms {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zmq_kernel_info_roundtrip() {
    let mut config = KernelConfig::default();
    if !assign_ports_or_skip(&mut config) {
        return;
    }

    let mut server = KernelServer::new(config.clone());
    server.start().await.unwrap();

    // Setup client sockets
    let ctx = zmq::Context::new();
    let shell = ctx.socket(zmq::DEALER).unwrap();
    shell.set_rcvtimeo(5000).unwrap();
    shell.set_sndtimeo(5000).unwrap();
    shell.connect(&config.connection.shell_url()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Build and send KernelInfoRequest
    let req = JupyterMessage::new(
        MessageType::KernelInfoRequest,
        &config.session_id,
        serde_json::json!({}),
    );
    send_jupyter_message(
        &shell,
        &[],
        &config.connection.key,
        &config.connection.signature_scheme,
        &req,
    )
    .unwrap();

    // Wait for reply
    assert!(poll_readable(&shell, 5000));
    let (_ids, msg) = recv_jupyter_message(
        &shell,
        &config.connection.key,
        &config.connection.signature_scheme,
    )
    .expect("kernel_info reply");
    assert_eq!(msg.header.msg_type, MessageType::KernelInfoReply);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zmq_execute_request_and_iopub() {
    let mut config = KernelConfig::default();
    if !assign_ports_or_skip(&mut config) {
        return;
    }

    let mut server = KernelServer::new(config.clone());
    server.start().await.unwrap();

    let ctx = zmq::Context::new();
    let shell = ctx.socket(zmq::DEALER).unwrap();
    shell.set_rcvtimeo(5000).unwrap();
    shell.set_sndtimeo(5000).unwrap();
    shell.connect(&config.connection.shell_url()).unwrap();

    let iopub = ctx.socket(zmq::SUB).unwrap();
    iopub.set_rcvtimeo(5000).unwrap();
    iopub.set_subscribe(b"").unwrap();
    iopub.connect(&config.connection.iopub_url()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Heartbeat check
    let hb = ctx.socket(zmq::REQ).unwrap();
    hb.set_rcvtimeo(2000).unwrap();
    hb.set_sndtimeo(2000).unwrap();
    hb.connect(&config.connection.heartbeat_url()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(100));
    hb.send("ping", 0).unwrap();
    let pong = hb.recv_string(0).unwrap().unwrap();
    assert_eq!(pong, "ping");

    let exec_req = ExecuteRequest {
        code: "a = 10".to_string(),
        silent: false,
        store_history: true,
        user_expressions: HashMap::new(),
        allow_stdin: false,
        stop_on_error: false,
    };
    let req = JupyterMessage::new(
        MessageType::ExecuteRequest,
        &config.session_id,
        serde_json::to_value(&exec_req).unwrap(),
    );
    send_jupyter_message(
        &shell,
        &[],
        &config.connection.key,
        &config.connection.signature_scheme,
        &req,
    )
    .unwrap();

    // Expect a shell reply and some IOPub traffic
    assert!(poll_readable(&shell, 5000));
    let (_ids, reply) = recv_jupyter_message(
        &shell,
        &config.connection.key,
        &config.connection.signature_scheme,
    )
    .expect("execute reply");
    assert_eq!(reply.header.msg_type, MessageType::ExecuteReply);

    // Drain IOPub until we see either ExecuteResult or Error
    let mut saw_any = false;
    let mut attempts = 0;
    while attempts < 50 {
        if poll_readable(&iopub, 200) {
            let (_id, msg) = recv_jupyter_message(
                &iopub,
                &config.connection.key,
                &config.connection.signature_scheme,
            )
            .expect("iopub message");
            match msg.header.msg_type {
                MessageType::ExecuteResult | MessageType::Error | MessageType::Status => {
                    saw_any = true;
                    break;
                }
                _ => {}
            }
        }
        attempts += 1;
    }
    assert!(saw_any, "expected ExecuteResult or Error on IOPub");
}
