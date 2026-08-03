use std::collections::HashSet;

use rand::RngCore as _;
use runmat_execution_artifact::encryption::{
    open_transfer_frame, seal_transfer_frame, RunKeyMaterial, TransferFrameAuthority,
    TRANSFER_FRAME_ENCRYPTION_SUITE,
};

use super::{FrameKind, FrameLimits, ReplayWindow, WireFrame};
use crate::{TransportError, TransportResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueFramePayload {
    pub encryption_suite: String,
    pub key_epoch: u32,
    pub ciphertext: Vec<u8>,
}

impl OpaqueFramePayload {
    pub fn contains_only_ciphertext_metadata(&self) -> bool {
        self.encryption_suite == TRANSFER_FRAME_ENCRYPTION_SUITE
            && self.key_epoch > 0
            && !self.ciphertext.is_empty()
    }
}

/// One direction of an application-encrypted overlay session.
///
/// QUIC/WebSocket protects transport metadata and availability. This layer
/// independently binds every plaintext to the run, session, direction, frame
/// kind, sequence, and key epoch, so a relay sees only opaque bytes.
pub struct EncryptedFrameSession {
    run_identity: String,
    session_id: [u8; 16],
    direction: String,
    key_epoch: u32,
    key: RunKeyMaterial,
    next_sequence: u64,
    replay: ReplayWindow,
    used_salts: HashSet<[u8; 32]>,
}

impl EncryptedFrameSession {
    pub const SUITE: &'static str = TRANSFER_FRAME_ENCRYPTION_SUITE;

    pub fn new(
        run_identity: impl Into<String>,
        session_id: [u8; 16],
        direction: impl Into<String>,
        key_epoch: u32,
        key: RunKeyMaterial,
    ) -> TransportResult<Self> {
        let run_identity = run_identity.into();
        let direction = direction.into();
        // The portable codec performs the same validation; reject early so a
        // malformed session can never be retained as live transport state.
        if run_identity.is_empty()
            || run_identity.len() > 256
            || direction.is_empty()
            || direction.len() > 64
            || !run_identity.is_ascii()
            || !direction.is_ascii()
            || key_epoch == 0
        {
            return Err(TransportError::Encryption(
                "encrypted frame authority is malformed".into(),
            ));
        }
        Ok(Self {
            run_identity,
            session_id,
            direction,
            key_epoch,
            key,
            next_sequence: 0,
            replay: ReplayWindow::default(),
            used_salts: HashSet::new(),
        })
    }

    pub fn seal(
        &mut self,
        kind: FrameKind,
        plaintext: &[u8],
        limits: FrameLimits,
    ) -> TransportResult<WireFrame> {
        let mut salt = [0_u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut salt);
        self.seal_with_entropy(kind, plaintext, salt, limits)
    }

    pub fn seal_with_entropy(
        &mut self,
        kind: FrameKind,
        plaintext: &[u8],
        salt: [u8; 32],
        limits: FrameLimits,
    ) -> TransportResult<WireFrame> {
        if !self.used_salts.insert(salt) {
            return Err(TransportError::Encryption(
                "frame derivation salt was reused".into(),
            ));
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(TransportError::Overflow)?;
        let payload = seal_transfer_frame(
            &self.key,
            &self.authority(kind, sequence),
            salt,
            plaintext,
            limits.maximum_payload_bytes,
        )
        .map_err(encryption)?;
        Ok(WireFrame {
            session_id: self.session_id,
            sequence,
            kind,
            payload,
        })
    }

    pub fn open(&mut self, frame: &WireFrame, limits: FrameLimits) -> TransportResult<Vec<u8>> {
        if frame.session_id != self.session_id {
            return Err(TransportError::Integrity);
        }
        let opened = open_transfer_frame(
            &self.key,
            &self.authority(frame.kind, frame.sequence),
            &frame.payload,
            limits.maximum_payload_bytes,
        )
        .map_err(encryption)?;
        if self.used_salts.contains(&opened.derivation_salt) {
            return Err(TransportError::Integrity);
        }
        // Authenticate before advancing replay state, so corrupt high sequence
        // numbers cannot consume the receive window.
        self.replay.accept(frame.sequence)?;
        self.used_salts.insert(opened.derivation_salt);
        Ok(opened.plaintext)
    }

    fn authority(&self, kind: FrameKind, sequence: u64) -> TransferFrameAuthority<'_> {
        TransferFrameAuthority {
            run_identity: &self.run_identity,
            session_id: self.session_id,
            direction: &self.direction,
            frame_kind: kind as u8,
            sequence,
            key_epoch: self.key_epoch,
        }
    }
}

fn encryption(error: impl std::fmt::Display) -> TransportError {
    TransportError::Encryption(error.to_string())
}
