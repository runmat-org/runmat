//! ZMQ transport and Jupyter v5 framing utilities
//!
//! Implements multipart message encoding/decoding with HMAC signatures
//! according to the Jupyter messaging protocol. Used by the kernel server
//! to read requests from the shell/control channels and publish results on
//! the IOPub channel.

use crate::protocol::{JupyterMessage, MessageHeader};
use crate::{KernelError, Result};
use hmac::{Hmac, Mac};
use serde_json::Value as JsonValue;
use sha2::Sha256;
use std::env;

const DELIM: &str = "<IDS|MSG>";

/// Signature algorithm supported by the kernel
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignatureAlg {
    None,
    HmacSha256,
}

impl SignatureAlg {
    pub fn from_scheme(scheme: &str) -> Self {
        match scheme.to_ascii_lowercase().as_str() {
            "hmac-sha256" => SignatureAlg::HmacSha256,
            _ => SignatureAlg::None,
        }
    }
}

/// Compute Jupyter message signature for the 4 JSON frames
fn compute_signature(alg: SignatureAlg, key: &[u8], frames: &[Vec<u8>]) -> String {
    match alg {
        SignatureAlg::None => String::new(),
        SignatureAlg::HmacSha256 => {
            let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
            for frame in frames {
                mac.update(frame);
            }
            let bytes = mac.finalize().into_bytes();
            hex::encode(bytes)
        }
    }
}

/// Decode a multipart message received from a ZMQ socket into routing identities
/// and a structured `JupyterMessage`. Verifies the HMAC signature if a key exists.
pub fn recv_jupyter_message(
    socket: &zmq::Socket,
    key: &str,
    scheme: &str,
) -> Result<(Vec<Vec<u8>>, JupyterMessage)> {
    let trace = env::var("RUNMAT_KERNEL_ZMQ_TRACE").is_ok();
    // Receive all frames for one message
    let frames = socket.recv_multipart(0).map_err(KernelError::Zmq)?;

    // Split identities and content frames
    let mut ids: Vec<Vec<u8>> = Vec::new();
    let mut idx = 0usize;
    while idx < frames.len() {
        if frames[idx] == DELIM.as_bytes() {
            idx += 1; // skip delim
            break;
        }
        ids.push(frames[idx].clone());
        idx += 1;
    }

    if idx >= frames.len() {
        return Err(KernelError::Protocol(
            "Missing <IDS|MSG> delimiter".to_string(),
        ));
    }

    // We require at least signature + 4 JSON frames
    if frames.len() - idx < 5 {
        return Err(KernelError::Protocol(
            "Incomplete message (expected signature + 4 JSON frames)".to_string(),
        ));
    }

    let signature = &frames[idx];
    let header = &frames[idx + 1];
    let parent_header = &frames[idx + 2];
    let metadata = &frames[idx + 3];
    let content = &frames[idx + 4];
    let buffers: Vec<Vec<u8>> = frames[idx + 5..].to_vec();

    // Validate signature if key present
    let alg = if key.is_empty() {
        SignatureAlg::None
    } else {
        SignatureAlg::from_scheme(scheme)
    };

    if !matches!(alg, SignatureAlg::None) {
        let expected = compute_signature(
            alg,
            key.as_bytes(),
            &[
                header.clone(),
                parent_header.clone(),
                metadata.clone(),
                content.clone(),
            ],
        );
        let provided = String::from_utf8_lossy(signature).to_string();
        if expected != provided {
            if trace {
                eprintln!(
                    "[ZMQ-TRACE] signature mismatch: expected {} provided {}",
                    expected, provided
                );
            }
            return Err(KernelError::Protocol("Invalid HMAC signature".to_string()));
        }
    }

    // Build structured message
    let header: MessageHeader = serde_json::from_slice(header)?;
    // Parent header can be {} or null. Treat both as None.
    let parent_val: JsonValue = serde_json::from_slice(parent_header)?;
    let parent_header: Option<MessageHeader> = match parent_val {
        JsonValue::Null => None,
        JsonValue::Object(ref m) if m.is_empty() => None,
        other => Some(serde_json::from_value(other).map_err(KernelError::Json)?),
    };
    let metadata_map: serde_json::Map<String, JsonValue> = serde_json::from_slice(metadata)?;
    let metadata: std::collections::HashMap<String, JsonValue> = metadata_map.into_iter().collect();
    let content: JsonValue = serde_json::from_slice(content)?;

    let msg = JupyterMessage {
        header,
        parent_header,
        metadata,
        content,
        buffers,
    };

    if trace {
        eprintln!(
            "[ZMQ-TRACE] RECV type={:?} session={}",
            msg.header.msg_type, msg.header.session
        );
    }

    Ok((ids, msg))
}

/// Encode and send a `JupyterMessage` with given routing identities on a ZMQ socket.
pub fn send_jupyter_message(
    socket: &zmq::Socket,
    ids: &[Vec<u8>],
    key: &str,
    scheme: &str,
    msg: &JupyterMessage,
) -> Result<()> {
    let trace = env::var("RUNMAT_KERNEL_ZMQ_TRACE").is_ok();
    let alg = if key.is_empty() {
        SignatureAlg::None
    } else {
        SignatureAlg::from_scheme(scheme)
    };

    // Serialize frames
    let header = serde_json::to_vec(&msg.header)?;
    let parent_header = if let Some(ref p) = msg.parent_header {
        serde_json::to_vec(p)?
    } else {
        // Parent header can be an empty JSON object according to protocol
        serde_json::to_vec(&serde_json::json!({}))?
    };
    let metadata = serde_json::to_vec(&msg.metadata)?;
    let content = serde_json::to_vec(&msg.content)?;

    let signature = compute_signature(
        alg,
        key.as_bytes(),
        &[
            header.clone(),
            parent_header.clone(),
            metadata.clone(),
            content.clone(),
        ],
    );

    // Assemble multipart frames
    let mut frames: Vec<Vec<u8>> = Vec::new();
    frames.extend_from_slice(ids);
    frames.push(DELIM.as_bytes().to_vec());
    frames.push(signature.into_bytes());
    frames.push(header);
    frames.push(parent_header);
    frames.push(metadata);
    frames.push(content);
    frames.extend_from_slice(&msg.buffers);

    socket.send_multipart(frames, 0).map_err(KernelError::Zmq)?;

    if trace {
        eprintln!(
            "[ZMQ-TRACE] SEND type={:?} session={}",
            msg.header.msg_type, msg.header.session
        );
    }

    Ok(())
}
