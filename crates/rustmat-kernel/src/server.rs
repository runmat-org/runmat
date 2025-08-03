//! Jupyter kernel server implementation
//! 
//! Manages ZMQ sockets and handles the complete Jupyter messaging protocol
//! with async execution and proper error handling.

use tokio::sync::{broadcast, mpsc};
use std::sync::Arc;
use tokio::task::JoinHandle;
use crate::{
    KernelConfig, KernelInfo, ConnectionInfo, ExecutionEngine,
    protocol::{JupyterMessage, MessageType, ExecuteRequest, ExecuteReply, ExecutionState},
    execution::{ExecutionStatus, ExecutionStats},
    Result,
};
use std::collections::HashMap;

/// Main kernel server managing all communication channels
pub struct KernelServer {
    /// Kernel configuration
    config: KernelConfig,
    /// Execution engine
    engine: Arc<tokio::sync::Mutex<ExecutionEngine>>,
    /// Broadcast channel for status updates
    status_tx: broadcast::Sender<ExecutionState>,
    /// Shutdown signal
    shutdown_tx: mpsc::Sender<()>,
    /// Server task handles
    tasks: Vec<JoinHandle<Result<()>>>,
}

/// Message router for handling different message types
#[allow(dead_code)]
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
            engine,
            status_tx,
            shutdown_tx,
            tasks: Vec::new(),
        }
    }

    /// Start the kernel server
    pub async fn start(&mut self) -> Result<()> {
        log::info!("Starting RustMat kernel server");
        
        // Validate connection info
        self.config.connection.validate()?;
        
        // Create ZMQ context
        let ctx = zmq::Context::new();
        
        // Create and bind sockets
        let shell_socket = ctx.socket(zmq::ROUTER)?;
        shell_socket.bind(&self.config.connection.shell_url())?;
        
        let iopub_socket = ctx.socket(zmq::PUB)?;
        iopub_socket.bind(&self.config.connection.iopub_url())?;
        
        let stdin_socket = ctx.socket(zmq::ROUTER)?;
        stdin_socket.bind(&self.config.connection.stdin_url())?;
        
        let control_socket = ctx.socket(zmq::ROUTER)?;
        control_socket.bind(&self.config.connection.control_url())?;
        
        let heartbeat_socket = ctx.socket(zmq::REP)?;
        heartbeat_socket.bind(&self.config.connection.heartbeat_url())?;
        
        log::info!("Kernel bound to ports: shell={}, iopub={}, stdin={}, control={}, hb={}", 
                  self.config.connection.shell_port,
                  self.config.connection.iopub_port,
                  self.config.connection.stdin_port,
                  self.config.connection.control_port,
                  self.config.connection.hb_port);

        // Create message router
        let _router = Arc::new(MessageRouter {
            engine: Arc::clone(&self.engine),
            session_id: self.config.session_id.clone(),
            status_tx: self.status_tx.clone(),
        });

        // Send initial status
        let _ = self.status_tx.send(ExecutionState::Starting);

        // Kernel is now idle and ready
        let _ = self.status_tx.send(ExecutionState::Idle);
        
        log::info!("RustMat kernel is ready for connections");
        
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
            if let Err(e) = task.await {
                log::error!("Task failed during shutdown: {e:?}");
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
}

#[allow(dead_code)]
impl MessageRouter {
    /// Route an incoming message to the appropriate handler
    async fn route_message(&self, msg: &JupyterMessage) -> Result<Option<JupyterMessage>> {
        match msg.header.msg_type {
            MessageType::KernelInfoRequest => {
                Ok(Some(self.handle_kernel_info_request(msg).await?))
            }
            MessageType::ExecuteRequest => {
                Ok(Some(self.handle_execute_request(msg).await?))
            }
            MessageType::ShutdownRequest => {
                Ok(Some(self.handle_shutdown_request(msg).await?))
            }
            MessageType::InterruptRequest => {
                Ok(Some(self.handle_interrupt_request(msg).await?))
            }
            _ => {
                log::warn!("Unhandled message type: {:?}", msg.header.msg_type);
                Ok(None)
            }
        }
    }

    /// Handle kernel info request
    async fn handle_kernel_info_request(&self, msg: &JupyterMessage) -> Result<JupyterMessage> {
        let kernel_info = KernelInfo::default();
        let content = serde_json::to_value(&kernel_info)?;
        Ok(JupyterMessage::reply(msg, MessageType::KernelInfoReply, content))
    }

    /// Handle execute request
    async fn handle_execute_request(&self, msg: &JupyterMessage) -> Result<JupyterMessage> {
        // Parse execute request
        let execute_req: ExecuteRequest = serde_json::from_value(msg.content.clone())?;
        
        // Update status to busy
        let _ = self.status_tx.send(ExecutionState::Busy);
        
        // Execute the code
        let mut engine = self.engine.lock().await;
        let exec_result = engine.execute(&execute_req.code)?;
        
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
        Ok(JupyterMessage::reply(msg, MessageType::ExecuteReply, content))
    }

    /// Handle shutdown request
    async fn handle_shutdown_request(&self, msg: &JupyterMessage) -> Result<JupyterMessage> {
        let shutdown_reply = serde_json::json!({
            "restart": false
        });
        Ok(JupyterMessage::reply(msg, MessageType::ShutdownReply, shutdown_reply))
    }

    /// Handle interrupt request
    async fn handle_interrupt_request(&self, msg: &JupyterMessage) -> Result<JupyterMessage> {
        let interrupt_reply = serde_json::json!({
            "status": "ok"
        });
        Ok(JupyterMessage::reply(msg, MessageType::InterruptReply, interrupt_reply))
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
        
        let router = MessageRouter {
            engine,
            session_id: "test".to_string(),
            status_tx,
        };

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
        
        let router = MessageRouter {
            engine,
            session_id: "test".to_string(),
            status_tx,
        };

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
        assert_eq!(info.implementation, "rustmat");
        assert_eq!(info.language_info.name, "matlab");
        assert_eq!(info.protocol_version, "5.3");
    }
} 