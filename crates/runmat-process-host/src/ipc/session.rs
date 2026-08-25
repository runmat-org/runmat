use std::fmt;

use hmac::{Hmac, Mac};
use rand::{rngs::OsRng, RngCore};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::Sha256;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use zeroize::Zeroize;

use crate::{ProcessHostError, ProcessHostResult};

use super::{negotiate_handshake, read_payload, write_payload, FrameLimits, HostHandshake};

pub const INITIAL_SESSION_LIMITS: FrameLimits = FrameLimits {
    max_message_bytes: 64 * 1024,
};

const SECRET_BYTES: usize = 32;
const CHALLENGE_BYTES: usize = 32;
const PROOF_BYTES: usize = 32;

#[derive(Clone)]
pub struct SessionSecret([u8; SECRET_BYTES]);

impl Drop for SessionSecret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl SessionSecret {
    pub fn generate() -> Self {
        let mut bytes = [0_u8; SECRET_BYTES];
        OsRng.fill_bytes(&mut bytes);
        Self(bytes)
    }

    pub fn from_hex(encoded: &str) -> ProcessHostResult<Self> {
        if encoded.len() != SECRET_BYTES * 2 || !encoded.is_ascii() {
            return Err(authentication_error());
        }
        let mut bytes = [0_u8; SECRET_BYTES];
        for (index, pair) in encoded.as_bytes().chunks_exact(2).enumerate() {
            bytes[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
        }
        Ok(Self(bytes))
    }

    pub fn expose_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(SECRET_BYTES * 2);
        for byte in self.0 {
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 0x0f) as usize] as char);
        }
        encoded
    }

    fn proof(
        &self,
        role: SessionRole,
        handshake: &HostHandshake,
        challenge: &[u8; CHALLENGE_BYTES],
    ) -> [u8; PROOF_BYTES] {
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.0)
            .expect("HMAC accepts every fixed-width session secret");
        mac.update(b"runmat-process-host-session-v1\0");
        mac.update(role.label());
        mac.update(&handshake.schema_version.to_be_bytes());
        mac.update(handshake.protocol.as_bytes());
        mac.update(&[0]);
        mac.update(challenge);
        mac.finalize().into_bytes().into()
    }

    fn verify(
        &self,
        role: SessionRole,
        handshake: &HostHandshake,
        challenge: &[u8; CHALLENGE_BYTES],
        proof: &[u8; PROOF_BYTES],
    ) -> ProcessHostResult<()> {
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.0)
            .expect("HMAC accepts every fixed-width session secret");
        mac.update(b"runmat-process-host-session-v1\0");
        mac.update(role.label());
        mac.update(&handshake.schema_version.to_be_bytes());
        mac.update(handshake.protocol.as_bytes());
        mac.update(&[0]);
        mac.update(challenge);
        mac.verify_slice(proof).map_err(|_| authentication_error())
    }
}

impl fmt::Debug for SessionSecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SessionSecret([REDACTED])")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedSession {
    pub remote: HostHandshake,
    pub limits: FrameLimits,
}

#[derive(Clone, Copy)]
enum SessionRole {
    Driver,
    Host,
}

impl SessionRole {
    const fn label(self) -> &'static [u8] {
        match self {
            Self::Driver => b"driver",
            Self::Host => b"host",
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DriverChallenge {
    challenge: [u8; CHALLENGE_BYTES],
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HostProof {
    proof: [u8; PROOF_BYTES],
    challenge: [u8; CHALLENGE_BYTES],
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DriverProof {
    proof: [u8; PROOF_BYTES],
}

pub async fn authenticate_driver(
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
    local: HostHandshake,
    secret: &SessionSecret,
) -> ProcessHostResult<AuthenticatedSession> {
    write_message(writer, &local, INITIAL_SESSION_LIMITS).await?;
    let remote: HostHandshake = read_message(reader, INITIAL_SESSION_LIMITS).await?;
    let limits = negotiate_handshake(&local, &remote)?;

    let challenge = random_challenge();
    write_message(writer, &DriverChallenge { challenge }, limits).await?;
    let response: HostProof = read_message(reader, limits).await?;
    if let Err(error) = secret.verify(SessionRole::Host, &remote, &challenge, &response.proof) {
        let _ = writer.shutdown().await;
        return Err(error);
    }
    let proof = secret.proof(SessionRole::Driver, &local, &response.challenge);
    write_message(writer, &DriverProof { proof }, limits).await?;

    Ok(AuthenticatedSession { remote, limits })
}

pub async fn authenticate_host(
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
    local: HostHandshake,
    secret: &SessionSecret,
) -> ProcessHostResult<AuthenticatedSession> {
    let remote: HostHandshake = read_message(reader, INITIAL_SESSION_LIMITS).await?;
    write_message(writer, &local, INITIAL_SESSION_LIMITS).await?;
    let limits = negotiate_handshake(&local, &remote)?;

    let request: DriverChallenge = read_message(reader, limits).await?;
    let challenge = random_challenge();
    let proof = secret.proof(SessionRole::Host, &local, &request.challenge);
    write_message(writer, &HostProof { proof, challenge }, limits).await?;
    let response: DriverProof = read_message(reader, limits).await?;
    secret.verify(SessionRole::Driver, &remote, &challenge, &response.proof)?;

    Ok(AuthenticatedSession { remote, limits })
}

async fn write_message<T: Serialize>(
    writer: &mut (impl AsyncWrite + Unpin),
    value: &T,
    limits: FrameLimits,
) -> ProcessHostResult<()> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        ProcessHostError::Protocol(format!("could not encode IPC record: {error}"))
    })?;
    write_payload(writer, &encoded, limits).await
}

async fn read_message<T: DeserializeOwned>(
    reader: &mut (impl AsyncRead + Unpin),
    limits: FrameLimits,
) -> ProcessHostResult<T> {
    let encoded = read_payload(reader, limits).await?;
    serde_json::from_slice(&encoded).map_err(|error| {
        ProcessHostError::Protocol(format!("could not decode IPC record: {error}"))
    })
}

fn random_challenge() -> [u8; CHALLENGE_BYTES] {
    let mut challenge = [0_u8; CHALLENGE_BYTES];
    OsRng.fill_bytes(&mut challenge);
    challenge
}

fn hex_nibble(byte: u8) -> ProcessHostResult<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(authentication_error()),
    }
}

fn authentication_error() -> ProcessHostError {
    ProcessHostError::Protocol("IPC peer authentication failed".into())
}
