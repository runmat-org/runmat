//! Jupyter messaging protocol implementation
//!
//! Implements the Jupyter kernel protocol v5.3 for communication between
//! the kernel and Jupyter frontends.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use uuid::Uuid;

/// Jupyter message types as defined in the protocol
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    // Shell channel
    ExecuteRequest,
    ExecuteReply,
    InspectRequest,
    InspectReply,
    CompleteRequest,
    CompleteReply,
    HistoryRequest,
    HistoryReply,
    IsCompleteRequest,
    IsCompleteReply,
    KernelInfoRequest,
    KernelInfoReply,

    // Control channel
    ShutdownRequest,
    ShutdownReply,
    InterruptRequest,
    InterruptReply,

    // IOPub channel
    Status,
    Stream,
    DisplayData,
    ExecuteInput,
    ExecuteResult,
    Error,

    // Stdin channel
    InputRequest,
    InputReply,
}

/// Complete Jupyter message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterMessage {
    /// Message header containing metadata
    pub header: MessageHeader,
    /// Parent message header (if this is a reply)
    pub parent_header: Option<MessageHeader>,
    /// Message metadata
    pub metadata: HashMap<String, JsonValue>,
    /// Message content (varies by message type)
    pub content: JsonValue,
    /// Buffer data for binary content
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub buffers: Vec<Vec<u8>>,
}

/// Message header with identification and routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    /// Unique message identifier
    pub msg_id: String,
    /// Message type
    pub msg_type: MessageType,
    /// Jupyter session identifier
    pub session: String,
    /// Message creation timestamp (ISO 8601)
    pub date: String,
    /// Jupyter protocol version
    pub version: String,
    /// Username who sent the message
    pub username: String,
}

impl MessageHeader {
    /// Create a new message header
    pub fn new(msg_type: MessageType, session: &str) -> Self {
        Self {
            msg_id: Uuid::new_v4().to_string(),
            msg_type,
            session: session.to_string(),
            date: chrono::Utc::now().to_rfc3339(),
            version: "5.3".to_string(),
            username: "kernel".to_string(),
        }
    }
}

impl JupyterMessage {
    /// Create a new message
    pub fn new(msg_type: MessageType, session: &str, content: JsonValue) -> Self {
        Self {
            header: MessageHeader::new(msg_type, session),
            parent_header: None,
            metadata: HashMap::new(),
            content,
            buffers: Vec::new(),
        }
    }

    /// Create a reply message to a parent
    pub fn reply(parent: &JupyterMessage, msg_type: MessageType, content: JsonValue) -> Self {
        Self {
            header: MessageHeader::new(msg_type, &parent.header.session),
            parent_header: Some(parent.header.clone()),
            metadata: HashMap::new(),
            content,
            buffers: Vec::new(),
        }
    }

    /// Serialize message to JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Deserialize message from JSON
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

/// Execute request content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteRequest {
    /// Source code to execute
    pub code: String,
    /// Whether to store this execution in history
    pub silent: bool,
    /// Whether to store intermediate results
    pub store_history: bool,
    /// User variables to be made available
    #[serde(default)]
    pub user_expressions: HashMap<String, String>,
    /// Whether to allow stdin requests
    pub allow_stdin: bool,
    /// Whether to stop on error
    #[serde(default)]
    pub stop_on_error: bool,
}

/// Execute reply content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteReply {
    /// Execution status
    pub status: ExecutionStatus,
    /// Execution counter
    pub execution_count: u64,
    /// User expressions results (if requested)
    #[serde(default)]
    pub user_expressions: HashMap<String, JsonValue>,
    /// Payload for additional actions
    #[serde(default)]
    pub payload: Vec<JsonValue>,
}

/// Execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionStatus {
    Ok,
    Error,
    Abort,
}

/// Kernel info reply content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfoReply {
    /// Protocol version
    pub protocol_version: String,
    /// Implementation name
    pub implementation: String,
    /// Implementation version
    pub implementation_version: String,
    /// Language information
    pub language_info: LanguageInfo,
    /// Kernel banner
    pub banner: String,
    /// Debugging support
    #[serde(default)]
    pub debugger: bool,
    /// Help links
    #[serde(default)]
    pub help_links: Vec<HelpLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    /// Language name
    pub name: String,
    /// Language version
    pub version: String,
    /// MIME type for code
    pub mimetype: String,
    /// File extension
    pub file_extension: String,
    /// Pygments lexer name
    pub pygments_lexer: String,
    /// CodeMirror mode
    pub codemirror_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelpLink {
    /// Link text
    pub text: String,
    /// Link URL
    pub url: String,
}

/// Kernel status message content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Status {
    /// Current execution state
    pub execution_state: ExecutionState,
}

/// Kernel execution state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionState {
    Starting,
    Idle,
    Busy,
}

/// Stream output content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stream {
    /// Stream name (stdout, stderr)
    pub name: String,
    /// Stream text content
    pub text: String,
}

/// Error content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContent {
    /// Error name/type
    pub ename: String,
    /// Error value/message
    pub evalue: String,
    /// Error traceback
    pub traceback: Vec<String>,
}

/// Execute result content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteResult {
    /// Execution counter
    pub execution_count: u64,
    /// Result data in various formats
    pub data: HashMap<String, JsonValue>,
    /// Result metadata
    #[serde(default)]
    pub metadata: HashMap<String, JsonValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let content = serde_json::json!({"code": "x = 1 + 2"});
        let msg = JupyterMessage::new(MessageType::ExecuteRequest, "test-session", content);

        assert_eq!(msg.header.msg_type, MessageType::ExecuteRequest);
        assert_eq!(msg.header.session, "test-session");
        assert!(!msg.header.msg_id.is_empty());
        assert!(msg.parent_header.is_none());
    }

    #[test]
    fn test_reply_message() {
        let request_content = serde_json::json!({"code": "x = 1"});
        let request = JupyterMessage::new(MessageType::ExecuteRequest, "test", request_content);

        let reply_content = serde_json::json!({"status": "ok"});
        let reply = JupyterMessage::reply(&request, MessageType::ExecuteReply, reply_content);

        assert_eq!(reply.header.msg_type, MessageType::ExecuteReply);
        assert_eq!(reply.header.session, "test");
        assert!(reply.parent_header.is_some());
        assert_eq!(reply.parent_header.unwrap().msg_id, request.header.msg_id);
    }

    #[test]
    fn test_execute_request_serialization() {
        let execute_req = ExecuteRequest {
            code: "disp('hello')".to_string(),
            silent: false,
            store_history: true,
            user_expressions: HashMap::new(),
            allow_stdin: false,
            stop_on_error: true,
        };

        let json = serde_json::to_string(&execute_req).unwrap();
        let parsed: ExecuteRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(execute_req.code, parsed.code);
        assert_eq!(execute_req.silent, parsed.silent);
    }

    #[test]
    fn test_message_json_roundtrip() {
        let content = serde_json::json!({
            "code": "x = magic(3)",
            "silent": false
        });

        let original = JupyterMessage::new(MessageType::ExecuteRequest, "test", content);
        let json = original.to_json().unwrap();
        let parsed = JupyterMessage::from_json(&json).unwrap();

        assert_eq!(original.header.msg_type, parsed.header.msg_type);
        assert_eq!(original.header.session, parsed.header.session);
        assert_eq!(original.content, parsed.content);
    }

    #[test]
    fn test_status_message() {
        let status = Status {
            execution_state: ExecutionState::Busy,
        };

        let content = serde_json::to_value(&status).unwrap();
        let msg = JupyterMessage::new(MessageType::Status, "test", content);

        assert_eq!(msg.header.msg_type, MessageType::Status);
    }
}
