//! Jupyter kernel server implementation
//!
//! Manages ZMQ sockets and handles the complete Jupyter messaging protocol
//! with async execution and proper error handling.

use crate::{
    execution::{ExecutionStats, ExecutionStatus},
    protocol::{
        ErrorContent, ExecuteReply, ExecuteRequest, ExecuteResult, ExecutionState, JupyterMessage,
        MessageType, Status,
    },
    transport::{recv_jupyter_message, send_jupyter_message},
    ConnectionInfo, ExecutionEngine, KernelConfig, KernelError, KernelInfo, Result,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};

/// Main kernel server managing all communication channels
pub struct KernelServer {
    /// Kernel configuration
    config: KernelConfig,
    /// ZMQ context (must outlive all sockets)
    ctx: Option<zmq::Context>,
    /// Execution engine
    engine: Arc<tokio::sync::Mutex<ExecutionEngine>>,
    /// Broadcast channel for status updates
    status_tx: broadcast::Sender<ExecutionState>,
    /// Shutdown signal
    shutdown_tx: mpsc::Sender<()>,
    /// Server task handles
    tasks: Vec<std::thread::JoinHandle<Result<()>>>,
    /// Message router for handling Jupyter protocol messages
    router: Option<Arc<MessageRouter>>,
}

/// Message router for handling different message types
struct MessageRouter {
    engine: Arc<tokio::sync::Mutex<ExecutionEngine>>,
    session_id: String,
    status_tx: broadcast::Sender<ExecutionState>,
}

impl KernelServer {
    /// Create a new kernel server
    pub fn new(config: KernelConfig) -> Self {
        let engine = Arc::new(tokio::sync::Mutex::new(ExecutionEngine::new()));
        let (status_tx, _) = broadcast::channel(16);
        let (shutdown_tx, _) = mpsc::channel(1);

        Self {
            config,
            ctx: None,
            engine,
            status_tx,
            shutdown_tx,
            tasks: Vec::new(),
            router: None,
        }
    }

    /// Start the kernel server
    pub async fn start(&mut self) -> Result<()> {
        log::info!("Starting RunMat kernel server");

        // Validate connection info
        self.config.connection.validate()?;

        // Create ZMQ context and keep it alive on self
        let ctx = zmq::Context::new();
        self.ctx = Some(ctx.clone());

        // Precompute URLs; sockets will be opened/bound in their own threads
        let shell_url = self.config.connection.shell_url();
        let iopub_url = self.config.connection.iopub_url();
        let stdin_url = self.config.connection.stdin_url();
        let control_url = self.config.connection.control_url();
        let heartbeat_url = self.config.connection.heartbeat_url();

        log::info!(
            "Kernel bound to ports: shell={}, iopub={}, stdin={}, control={}, hb={}",
            self.config.connection.shell_port,
            self.config.connection.iopub_port,
            self.config.connection.stdin_port,
            self.config.connection.control_port,
            self.config.connection.hb_port
        );

        // Create message router
        let router = Arc::new(MessageRouter::new(
            Arc::clone(&self.engine),
            self.config.session_id.clone(),
            self.status_tx.clone(),
        ));

        // Log router initialization
        log::info!(
            "Message router initialized for session: {}",
            router.session_id()
        );

        // Store router for the server to use
        self.router = Some(Arc::clone(&router));

        // Channel to serialize IOPub publishing onto a single ZMQ socket/thread
        let (iopub_tx, mut iopub_rx) = tokio::sync::mpsc::unbounded_channel::<JupyterMessage>();

        // Spawn IOPub publisher task
        let session_for_iopub = self.config.session_id.clone();
        let key_for_iopub = self.config.connection.key.clone();
        let scheme_for_iopub = self.config.connection.signature_scheme.clone();
        let ctx_iopub = ctx.clone();
        let iopub_task = std::thread::spawn(move || -> Result<()> {
            let socket = ctx_iopub.socket(zmq::PUB).map_err(KernelError::Zmq)?;
            socket.bind(&iopub_url).map_err(KernelError::Zmq)?;
            while let Some(mut msg) = iopub_rx.blocking_recv() {
                msg.header.session = session_for_iopub.clone();
                if msg.parent_header.is_none() {
                    msg.parent_header = Some(crate::protocol::MessageHeader::new(
                        MessageType::Status,
                        &session_for_iopub,
                    ));
                }
                if let Err(e) =
                    send_jupyter_message(&socket, &[], &key_for_iopub, &scheme_for_iopub, &msg)
                {
                    log::error!("Failed to publish IOPub message: {e}");
                }
            }
            Ok(())
        });
        self.tasks.push(iopub_task);

        // Spawn Heartbeat echo task (REP -> echo request)
        let ctx_hb = ctx.clone();
        let hb_task = std::thread::spawn(move || -> Result<()> {
            let socket = ctx_hb.socket(zmq::REP).map_err(KernelError::Zmq)?;
            socket.bind(&heartbeat_url).map_err(KernelError::Zmq)?;
            loop {
                let msg = socket.recv_multipart(0).map_err(KernelError::Zmq)?;
                socket.send_multipart(msg, 0).map_err(KernelError::Zmq)?;
            }
        });
        self.tasks.push(hb_task);

        // Spawn Shell loop
        let engine_shell = Arc::clone(&self.engine);
        let router_shell = Arc::clone(&router);
        let session_id_shell = self.config.session_id.clone();
        let key_shell = self.config.connection.key.clone();
        let scheme_shell = self.config.connection.signature_scheme.clone();
        let iopub_tx_shell = iopub_tx.clone();
        let ctx_shell = ctx.clone();
        let shell_task = std::thread::spawn(move || -> Result<()> {
            let shell_socket = ctx_shell.socket(zmq::ROUTER).map_err(KernelError::Zmq)?;
            shell_socket.bind(&shell_url).map_err(KernelError::Zmq)?;
            loop {
                let (ids, msg) = recv_jupyter_message(&shell_socket, &key_shell, &scheme_shell)?;

                match msg.header.msg_type.clone() {
                    MessageType::KernelInfoRequest => {
                        // IOPub busy
                        let status_busy = Status {
                            execution_state: ExecutionState::Busy,
                        };
                        let mut status_msg = JupyterMessage::reply(
                            &msg,
                            MessageType::Status,
                            serde_json::to_value(status_busy)?,
                        );
                        status_msg.header.session = session_id_shell.clone();
                        let _ = iopub_tx_shell.send(status_msg);

                        let mut reply = futures::executor::block_on(
                            router_shell.handle_kernel_info_request(&msg),
                        )?;
                        reply.header.session = session_id_shell.clone();
                        send_jupyter_message(
                            &shell_socket,
                            &ids,
                            &key_shell,
                            &scheme_shell,
                            &reply,
                        )?;

                        // IOPub idle
                        let status_idle = Status {
                            execution_state: ExecutionState::Idle,
                        };
                        let mut status_msg = JupyterMessage::reply(
                            &msg,
                            MessageType::Status,
                            serde_json::to_value(status_idle)?,
                        );
                        status_msg.header.session = session_id_shell.clone();
                        let _ = iopub_tx_shell.send(status_msg);
                    }
                    MessageType::ExecuteRequest => {
                        let exec_req: ExecuteRequest = serde_json::from_value(msg.content.clone())?;

                        // IOPub busy
                        let mut status_msg = JupyterMessage::reply(
                            &msg,
                            MessageType::Status,
                            serde_json::to_value(Status {
                                execution_state: ExecutionState::Busy,
                            })?,
                        );
                        status_msg.header.session = session_id_shell.clone();
                        let _ = iopub_tx_shell.send(status_msg);

                        // Predict next count and publish execute_input
                        let predicted = {
                            let eng = futures::executor::block_on(engine_shell.lock());
                            eng.execution_count() + 1
                        };
                        let mut input_msg = JupyterMessage::reply(
                            &msg,
                            MessageType::ExecuteInput,
                            serde_json::json!({"code": exec_req.code, "execution_count": predicted}),
                        );
                        input_msg.header.session = session_id_shell.clone();
                        let _ = iopub_tx_shell.send(input_msg);

                        let exec_result = {
                            let mut eng = futures::executor::block_on(engine_shell.lock());
                            let req_again: ExecuteRequest =
                                serde_json::from_value(msg.content.clone())?;
                            futures::executor::block_on(eng.execute(&req_again.code))
                                .map_err(|e| KernelError::Execution(e.to_string()))?
                        };

                        let status = match exec_result.status {
                            ExecutionStatus::Success => crate::protocol::ExecutionStatus::Ok,
                            ExecutionStatus::Error => crate::protocol::ExecutionStatus::Error,
                            ExecutionStatus::Interrupted | ExecutionStatus::Timeout => {
                                crate::protocol::ExecutionStatus::Abort
                            }
                        };

                        let exec_count = {
                            let eng = futures::executor::block_on(engine_shell.lock());
                            eng.execution_count()
                        };

                        match exec_result.status {
                            ExecutionStatus::Success => {
                                if let Some(val) = exec_result.result {
                                    let mut data = std::collections::HashMap::new();
                                    data.insert(
                                        "text/plain".to_string(),
                                        serde_json::json!(val.to_string()),
                                    );
                                    let res = ExecuteResult {
                                        execution_count: exec_count,
                                        data,
                                        metadata: std::collections::HashMap::new(),
                                    };
                                    let mut res_msg = JupyterMessage::reply(
                                        &msg,
                                        MessageType::ExecuteResult,
                                        serde_json::to_value(res)?,
                                    );
                                    res_msg.header.session = session_id_shell.clone();
                                    let _ = iopub_tx_shell.send(res_msg);
                                }
                            }
                            ExecutionStatus::Error => {
                                if let Some(err) = exec_result.error {
                                    let ec = ErrorContent {
                                        ename: err.error_type,
                                        evalue: err.message,
                                        traceback: err.traceback,
                                    };
                                    let mut err_msg = JupyterMessage::reply(
                                        &msg,
                                        MessageType::Error,
                                        serde_json::to_value(ec)?,
                                    );
                                    err_msg.header.session = session_id_shell.clone();
                                    let _ = iopub_tx_shell.send(err_msg);
                                }
                            }
                            _ => {}
                        }

                        let reply = ExecuteReply {
                            status,
                            execution_count: exec_count,
                            user_expressions: HashMap::new(),
                            payload: Vec::new(),
                        };
                        let mut reply_msg = JupyterMessage::reply(
                            &msg,
                            MessageType::ExecuteReply,
                            serde_json::to_value(reply)?,
                        );
                        reply_msg.header.session = session_id_shell.clone();
                        send_jupyter_message(
                            &shell_socket,
                            &ids,
                            &key_shell,
                            &scheme_shell,
                            &reply_msg,
                        )?;

                        // IOPub idle
                        let mut status_msg = JupyterMessage::reply(
                            &msg,
                            MessageType::Status,
                            serde_json::to_value(Status {
                                execution_state: ExecutionState::Idle,
                            })?,
                        );
                        status_msg.header.session = session_id_shell.clone();
                        let _ = iopub_tx_shell.send(status_msg);
                    }
                    other => {
                        log::warn!("Unhandled shell message: {:?}", other);
                        if let Ok(Some(reply)) = futures::executor::block_on(async {
                            router_shell.route_message(&msg).await
                        }) {
                            send_jupyter_message(
                                &shell_socket,
                                &ids,
                                &key_shell,
                                &scheme_shell,
                                &reply,
                            )?;
                        }
                    }
                }
            }
        });
        self.tasks.push(shell_task);

        // Control loop
        let router_ctrl = Arc::clone(&router);
        let key_ctrl = self.config.connection.key.clone();
        let scheme_ctrl = self.config.connection.signature_scheme.clone();
        let session_ctrl = self.config.session_id.clone();
        let iopub_tx_ctrl = iopub_tx.clone();
        let ctx_ctrl = ctx.clone();
        let control_task = std::thread::spawn(move || -> Result<()> {
            let control_socket = ctx_ctrl.socket(zmq::ROUTER).map_err(KernelError::Zmq)?;
            control_socket
                .bind(&control_url)
                .map_err(KernelError::Zmq)?;
            loop {
                let (ids, msg) = recv_jupyter_message(&control_socket, &key_ctrl, &scheme_ctrl)?;
                match msg.header.msg_type.clone() {
                    MessageType::ShutdownRequest | MessageType::InterruptRequest => {
                        let mut status_msg = JupyterMessage::reply(
                            &msg,
                            MessageType::Status,
                            serde_json::to_value(Status {
                                execution_state: ExecutionState::Busy,
                            })?,
                        );
                        status_msg.header.session = session_ctrl.clone();
                        let _ = iopub_tx_ctrl.send(status_msg);

                        let mut reply =
                            futures::executor::block_on(router_ctrl.route_message(&msg))?
                                .unwrap_or_else(|| {
                                    JupyterMessage::reply(
                                        &msg,
                                        MessageType::InterruptReply,
                                        serde_json::json!({"status":"ok"}),
                                    )
                                });
                        reply.header.session = session_ctrl.clone();
                        send_jupyter_message(
                            &control_socket,
                            &ids,
                            &key_ctrl,
                            &scheme_ctrl,
                            &reply,
                        )?;

                        let mut status_msg = JupyterMessage::reply(
                            &msg,
                            MessageType::Status,
                            serde_json::to_value(Status {
                                execution_state: ExecutionState::Idle,
                            })?,
                        );
                        status_msg.header.session = session_ctrl.clone();
                        let _ = iopub_tx_ctrl.send(status_msg);
                    }
                    _ => {}
                }
            }
        });
        self.tasks.push(control_task);

        // Stdin loop
        let key_stdin = self.config.connection.key.clone();
        let scheme_stdin = self.config.connection.signature_scheme.clone();
        let session_stdin = self.config.session_id.clone();
        let ctx_stdin = ctx.clone();
        let stdin_task = std::thread::spawn(move || -> Result<()> {
            let stdin_socket = ctx_stdin.socket(zmq::ROUTER).map_err(KernelError::Zmq)?;
            stdin_socket.bind(&stdin_url).map_err(KernelError::Zmq)?;
            loop {
                let (ids, msg) = recv_jupyter_message(&stdin_socket, &key_stdin, &scheme_stdin)?;
                if matches!(msg.header.msg_type, MessageType::InputRequest) {
                    let mut reply = JupyterMessage::reply(
                        &msg,
                        MessageType::InputReply,
                        serde_json::json!({"value": ""}),
                    );
                    reply.header.session = session_stdin.clone();
                    send_jupyter_message(&stdin_socket, &ids, &key_stdin, &scheme_stdin, &reply)?;
                }
            }
        });
        self.tasks.push(stdin_task);

        // Initial status via IOPub: starting -> idle
        let mut start_msg = JupyterMessage::new(
            MessageType::Status,
            &self.config.session_id,
            serde_json::to_value(Status {
                execution_state: ExecutionState::Starting,
            })?,
        );
        start_msg.parent_header = None;
        let _ = iopub_tx.send(start_msg);

        let mut idle_msg = JupyterMessage::new(
            MessageType::Status,
            &self.config.session_id,
            serde_json::to_value(Status {
                execution_state: ExecutionState::Idle,
            })?,
        );
        idle_msg.parent_header = None;
        let _ = iopub_tx.send(idle_msg);

        log::info!("RunMat kernel is ready for connections");

        Ok(())
    }

    /// Stop the kernel server
    pub async fn stop(&mut self) -> Result<()> {
        log::info!("Stopping kernel server");

        // Send shutdown signal
        if (self.shutdown_tx.send(()).await).is_err() {
            log::warn!("Failed to send shutdown signal");
        }

        // Wait for all tasks to complete
        for task in self.tasks.drain(..) {
            match task.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => log::error!("Task failed during shutdown: {e:?}"),
                Err(e) => log::error!("Task panicked: {e:?}"),
            }
        }

        log::info!("Kernel server stopped");
        Ok(())
    }

    /// Get kernel information for Jupyter frontend
    pub fn kernel_info(&self) -> KernelInfo {
        KernelInfo::default()
    }

    /// Get connection information
    pub fn connection_info(&self) -> &ConnectionInfo {
        &self.config.connection
    }

    /// Get execution engine statistics
    pub async fn stats(&self) -> Result<ExecutionStats> {
        let engine = self.engine.lock().await;
        Ok(engine.stats())
    }

    /// Handle a Jupyter message using the router
    pub async fn handle_message(&self, message: &JupyterMessage) -> Result<Option<JupyterMessage>> {
        if let Some(ref router) = self.router {
            router.route_message(message).await
        } else {
            Err(crate::KernelError::Internal(
                "Message router not initialized".to_string(),
            ))
        }
    }

    /// Get the current session ID from the router
    pub fn session_id(&self) -> Option<&str> {
        self.router.as_ref().map(|r| r.session_id())
    }

    /// Send a status update
    pub async fn send_status(&self, status: ExecutionState) -> Result<()> {
        if let Some(ref router) = self.router {
            router.send_status(status).await
        } else {
            self.status_tx
                .send(status)
                .map_err(|e| crate::KernelError::Internal(format!("Failed to send status: {e}")))?;
            Ok(())
        }
    }
}

impl MessageRouter {
    /// Create a new message router
    pub fn new(
        engine: Arc<tokio::sync::Mutex<ExecutionEngine>>,
        session_id: String,
        status_tx: broadcast::Sender<ExecutionState>,
    ) -> Self {
        Self {
            engine,
            session_id,
            status_tx,
        }
    }

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Send status update
    pub async fn send_status(&self, status: ExecutionState) -> Result<()> {
        self.status_tx
            .send(status)
            .map_err(|e| KernelError::Internal(format!("Failed to send status: {e}")))?;
        Ok(())
    }

    /// Route an incoming message to the appropriate handler
    pub async fn route_message(&self, msg: &JupyterMessage) -> Result<Option<JupyterMessage>> {
        // Update status to busy
        let _ = self.send_status(ExecutionState::Busy).await;

        let result = match msg.header.msg_type {
            MessageType::KernelInfoRequest => Ok(Some(self.handle_kernel_info_request(msg).await?)),
            MessageType::ExecuteRequest => Ok(Some(self.handle_execute_request(msg).await?)),
            MessageType::ShutdownRequest => Ok(Some(self.handle_shutdown_request(msg).await?)),
            MessageType::InterruptRequest => Ok(Some(self.handle_interrupt_request(msg).await?)),
            _ => {
                log::warn!("Unhandled message type: {:?}", msg.header.msg_type);
                Ok(None)
            }
        };

        // Update status back to idle
        let _ = self.send_status(ExecutionState::Idle).await;

        result
    }

    /// Handle kernel info request
    async fn handle_kernel_info_request(&self, msg: &JupyterMessage) -> Result<JupyterMessage> {
        let kernel_info = KernelInfo::default();
        let content = serde_json::to_value(&kernel_info)?;
        Ok(JupyterMessage::reply(
            msg,
            MessageType::KernelInfoReply,
            content,
        ))
    }

    /// Handle execute request
    async fn handle_execute_request(&self, msg: &JupyterMessage) -> Result<JupyterMessage> {
        // Parse execute request
        let execute_req: ExecuteRequest = serde_json::from_value(msg.content.clone())?;

        // Update status to busy
        let _ = self.status_tx.send(ExecutionState::Busy);

        // Execute the code
        let mut engine = self.engine.lock().await;
        let exec_result = engine.execute(&execute_req.code).await?;

        // Create execute reply
        let status = match exec_result.status {
            ExecutionStatus::Success => crate::protocol::ExecutionStatus::Ok,
            ExecutionStatus::Error => crate::protocol::ExecutionStatus::Error,
            ExecutionStatus::Interrupted => crate::protocol::ExecutionStatus::Abort,
            ExecutionStatus::Timeout => crate::protocol::ExecutionStatus::Abort,
        };

        let reply = ExecuteReply {
            status,
            execution_count: engine.execution_count(),
            user_expressions: HashMap::new(),
            payload: Vec::new(),
        };

        // Update status back to idle
        let _ = self.status_tx.send(ExecutionState::Idle);

        let content = serde_json::to_value(&reply)?;
        Ok(JupyterMessage::reply(
            msg,
            MessageType::ExecuteReply,
            content,
        ))
    }

    /// Handle shutdown request
    async fn handle_shutdown_request(&self, msg: &JupyterMessage) -> Result<JupyterMessage> {
        let shutdown_reply = serde_json::json!({
            "restart": false
        });
        Ok(JupyterMessage::reply(
            msg,
            MessageType::ShutdownReply,
            shutdown_reply,
        ))
    }

    /// Handle interrupt request
    async fn handle_interrupt_request(&self, msg: &JupyterMessage) -> Result<JupyterMessage> {
        let interrupt_reply = serde_json::json!({
            "status": "ok"
        });
        Ok(JupyterMessage::reply(
            msg,
            MessageType::InterruptReply,
            interrupt_reply,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_server_creation() {
        let config = KernelConfig::default();
        let server = KernelServer::new(config);
        assert!(server.tasks.is_empty());
    }

    #[tokio::test]
    async fn test_message_router_kernel_info() {
        let engine = Arc::new(tokio::sync::Mutex::new(ExecutionEngine::new()));
        let (status_tx, _) = broadcast::channel(16);

        let router = MessageRouter::new(engine, "test".to_string(), status_tx);

        let request = JupyterMessage::new(
            MessageType::KernelInfoRequest,
            "test",
            serde_json::json!({}),
        );

        let reply = router.handle_kernel_info_request(&request).await.unwrap();
        assert_eq!(reply.header.msg_type, MessageType::KernelInfoReply);
        assert!(reply.parent_header.is_some());
    }

    #[tokio::test]
    async fn test_message_router_execute() {
        let engine = Arc::new(tokio::sync::Mutex::new(ExecutionEngine::new()));
        let (status_tx, _) = broadcast::channel(16);

        let router = MessageRouter::new(engine, "test".to_string(), status_tx);

        let execute_req = ExecuteRequest {
            code: "x = 1 + 2".to_string(),
            silent: false,
            store_history: true,
            user_expressions: HashMap::new(),
            allow_stdin: false,
            stop_on_error: false,
        };

        let content = serde_json::to_value(&execute_req).unwrap();
        let request = JupyterMessage::new(MessageType::ExecuteRequest, "test", content);

        let reply = router.handle_execute_request(&request).await.unwrap();
        assert_eq!(reply.header.msg_type, MessageType::ExecuteReply);

        let reply_content: ExecuteReply = serde_json::from_value(reply.content).unwrap();
        assert_eq!(reply_content.execution_count, 1);
    }

    #[test]
    fn test_kernel_info_default() {
        let info = KernelInfo::default();
        assert_eq!(info.implementation, "runmat");
        assert_eq!(info.language_info.name, "matlab");
        assert_eq!(info.protocol_version, "5.3");
    }
}
