//! Jupyter kernel connection management
//! 
//! Handles connection file parsing and ZMQ socket configuration compatible
//! with the Jupyter protocol.

use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::{KernelError, Result};

/// Connection information for Jupyter kernel communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    /// IP address to bind to (usually 127.0.0.1)
    pub ip: String,
    /// Transport protocol (usually "tcp")
    pub transport: String,
    /// Signature scheme for message authentication (usually "hmac-sha256")
    pub signature_scheme: String,
    /// HMAC key for message signing
    pub key: String,
    /// Shell socket port (handles execute requests)
    pub shell_port: u16,
    /// IOPub socket port (publishes execution results)
    pub iopub_port: u16,
    /// Stdin socket port (handles input requests)
    pub stdin_port: u16,
    /// Control socket port (handles kernel control)
    pub control_port: u16,
    /// Heartbeat socket port (kernel liveness check)
    pub hb_port: u16,
}

impl Default for ConnectionInfo {
    fn default() -> Self {
        Self {
            ip: "127.0.0.1".to_string(),
            transport: "tcp".to_string(),
            signature_scheme: "hmac-sha256".to_string(),
            key: uuid::Uuid::new_v4().to_string(),
            shell_port: 0,    // Let OS assign
            iopub_port: 0,    // Let OS assign
            stdin_port: 0,    // Let OS assign
            control_port: 0,  // Let OS assign
            hb_port: 0,       // Let OS assign
        }
    }
}

impl ConnectionInfo {
    /// Create connection info from a Jupyter connection file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| KernelError::Connection(format!("Failed to read connection file: {e}")))?;
        
        Self::from_json(&content)
    }

    /// Parse connection info from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| KernelError::Connection(format!("Invalid connection JSON: {e}")))
    }

    /// Serialize connection info to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(KernelError::Json)
    }

    /// Write connection info to a file
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)
            .map_err(|e| KernelError::Connection(format!("Failed to write connection file: {e}")))
    }

    /// Generate a connection URL for a given socket type
    pub fn socket_url(&self, port: u16) -> String {
        format!("{}://{}:{}", self.transport, self.ip, port)
    }

    /// Get shell socket URL
    pub fn shell_url(&self) -> String {
        self.socket_url(self.shell_port)
    }

    /// Get IOPub socket URL
    pub fn iopub_url(&self) -> String {
        self.socket_url(self.iopub_port)
    }

    /// Get stdin socket URL
    pub fn stdin_url(&self) -> String {
        self.socket_url(self.stdin_port)
    }

    /// Get control socket URL
    pub fn control_url(&self) -> String {
        self.socket_url(self.control_port)
    }

    /// Get heartbeat socket URL
    pub fn heartbeat_url(&self) -> String {
        self.socket_url(self.hb_port)
    }

    /// Validate that all required fields are present and valid
    pub fn validate(&self) -> Result<()> {
        if self.ip.is_empty() {
            return Err(KernelError::Connection("IP address cannot be empty".to_string()));
        }
        
        if self.transport.is_empty() {
            return Err(KernelError::Connection("Transport cannot be empty".to_string()));
        }

        if self.key.is_empty() {
            return Err(KernelError::Connection("Key cannot be empty".to_string()));
        }

        // Validate ports are non-zero (indicating they've been assigned)
        let ports = [
            ("shell", self.shell_port),
            ("iopub", self.iopub_port),
            ("stdin", self.stdin_port),
            ("control", self.control_port),
            ("hb", self.hb_port),
        ];

        for (name, port) in ports {
            if port == 0 {
                return Err(KernelError::Connection(format!("{name} port must be assigned")));
            }
        }

        Ok(())
    }

    /// Assign random available ports to all sockets
    pub fn assign_ports(&mut self) -> Result<()> {
        use std::net::TcpListener;

        // Helper to find an available port
        fn find_available_port() -> Result<u16> {
            let listener = TcpListener::bind("127.0.0.1:0")
                .map_err(|e| KernelError::Connection(format!("Failed to find available port: {e}")))?;
            Ok(listener.local_addr()
                .map_err(|e| KernelError::Connection(format!("Failed to get port: {e}")))?
                .port())
        }

        self.shell_port = find_available_port()?;
        self.iopub_port = find_available_port()?;
        self.stdin_port = find_available_port()?;
        self.control_port = find_available_port()?;
        self.hb_port = find_available_port()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_connection() {
        let conn = ConnectionInfo::default();
        assert_eq!(conn.ip, "127.0.0.1");
        assert_eq!(conn.transport, "tcp");
        assert_eq!(conn.signature_scheme, "hmac-sha256");
        assert!(!conn.key.is_empty());
    }

    #[test]
    fn test_connection_json_roundtrip() {
        let mut conn = ConnectionInfo::default();
        conn.shell_port = 12345;
        conn.iopub_port = 12346;
        conn.stdin_port = 12347;
        conn.control_port = 12348;
        conn.hb_port = 12349;

        let json = conn.to_json().unwrap();
        let parsed = ConnectionInfo::from_json(&json).unwrap();

        assert_eq!(conn.shell_port, parsed.shell_port);
        assert_eq!(conn.iopub_port, parsed.iopub_port);
        assert_eq!(conn.key, parsed.key);
    }

    #[test]
    fn test_connection_file_io() {
        let mut conn = ConnectionInfo::default();
        conn.shell_port = 12345;
        conn.iopub_port = 12346;
        conn.stdin_port = 12347;
        conn.control_port = 12348;
        conn.hb_port = 12349;

        let temp_file = NamedTempFile::new().unwrap();
        conn.write_to_file(temp_file.path()).unwrap();

        let loaded = ConnectionInfo::from_file(temp_file.path()).unwrap();
        assert_eq!(conn.shell_port, loaded.shell_port);
        assert_eq!(conn.key, loaded.key);
    }

    #[test]
    fn test_socket_urls() {
        let mut conn = ConnectionInfo::default();
        conn.shell_port = 12345;
        conn.iopub_port = 12346;

        assert_eq!(conn.shell_url(), "tcp://127.0.0.1:12345");
        assert_eq!(conn.iopub_url(), "tcp://127.0.0.1:12346");
    }

    #[test]
    fn test_port_assignment() {
        let mut conn = ConnectionInfo::default();
        conn.assign_ports().unwrap();

        assert_ne!(conn.shell_port, 0);
        assert_ne!(conn.iopub_port, 0);
        assert_ne!(conn.stdin_port, 0);
        assert_ne!(conn.control_port, 0);
        assert_ne!(conn.hb_port, 0);

        conn.validate().unwrap();
    }

    #[test]
    fn test_validation() {
        let mut conn = ConnectionInfo::default();
        
        // Should fail with unassigned ports
        assert!(conn.validate().is_err());

        // Should pass after port assignment
        conn.assign_ports().unwrap();
        conn.validate().unwrap();

        // Should fail with empty key
        conn.key.clear();
        assert!(conn.validate().is_err());
    }
} 