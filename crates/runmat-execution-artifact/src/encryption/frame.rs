use minicbor::{Decoder, Encoder};
use runmat_execution::Digest;

use super::{
    EncryptedRunObject, EncryptionContext, EncryptionPurpose, RunKeyMaterial, RunObjectEncryption,
    RunObjectEncryptionSuite,
};
use crate::{ArtifactError, ArtifactResult};

pub const TRANSFER_FRAME_ENCRYPTION_SUITE: &str = "hkdf-sha256-aes256-gcm-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferFrameAuthority<'a> {
    pub run_identity: &'a str,
    pub session_id: [u8; 16],
    pub direction: &'a str,
    pub frame_kind: u8,
    pub sequence: u64,
    pub key_epoch: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenedTransferFrame {
    pub derivation_salt: [u8; 32],
    pub plaintext: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferWireFrame {
    pub session_id: [u8; 16],
    pub sequence: u64,
    pub frame_kind: u8,
    pub encrypted_payload: Vec<u8>,
}

pub fn seal_transfer_frame(
    run_key: &RunKeyMaterial,
    authority: &TransferFrameAuthority<'_>,
    derivation_salt: [u8; 32],
    plaintext: &[u8],
    maximum_payload_bytes: usize,
) -> ArtifactResult<Vec<u8>> {
    let context = context(authority, Digest::sha256(plaintext), plaintext.len())?;
    let encrypted =
        RunObjectEncryption.seal_with_entropy(run_key, derivation_salt, context, plaintext)?;
    encode(authority.key_epoch, &encrypted, maximum_payload_bytes)
}

pub fn open_transfer_frame(
    run_key: &RunKeyMaterial,
    authority: &TransferFrameAuthority<'_>,
    bytes: &[u8],
    maximum_payload_bytes: usize,
) -> ArtifactResult<OpenedTransferFrame> {
    let payload = decode(bytes, maximum_payload_bytes)?;
    if payload.key_epoch != authority.key_epoch {
        return Err(ArtifactError::Identity(
            "transfer frame key epoch does not match its authority".into(),
        ));
    }
    let object = EncryptedRunObject {
        schema_version: 1,
        suite: RunObjectEncryptionSuite::HkdfSha256Aes256GcmV1,
        context: context(authority, payload.digest, payload.plaintext_size)?,
        derivation_salt: payload.salt.to_vec(),
        ciphertext: payload.ciphertext,
    };
    Ok(OpenedTransferFrame {
        derivation_salt: payload.salt,
        plaintext: RunObjectEncryption.open(run_key, &object)?,
    })
}

pub fn encode_transfer_wire_frame(
    frame: &TransferWireFrame,
    maximum_frame_bytes: usize,
) -> ArtifactResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(frame.encrypted_payload.len() + 64);
    Encoder::new(&mut bytes)
        .map(5)
        .and_then(|encoder| encoder.u8(0))
        .and_then(|encoder| encoder.u16(1))
        .and_then(|encoder| encoder.u8(1))
        .and_then(|encoder| encoder.bytes(&frame.session_id))
        .and_then(|encoder| encoder.u8(2))
        .and_then(|encoder| encoder.u64(frame.sequence))
        .and_then(|encoder| encoder.u8(3))
        .and_then(|encoder| encoder.u8(frame.frame_kind))
        .and_then(|encoder| encoder.u8(4))
        .and_then(|encoder| encoder.bytes(&frame.encrypted_payload))
        .map_err(encoding)?;
    if bytes.len() > maximum_frame_bytes {
        return Err(ArtifactError::Invalid(
            "transfer wire frame exceeds its bound".into(),
        ));
    }
    Ok(bytes)
}

pub fn decode_transfer_wire_frame(
    bytes: &[u8],
    maximum_frame_bytes: usize,
) -> ArtifactResult<TransferWireFrame> {
    if bytes.len() > maximum_frame_bytes {
        return Err(ArtifactError::Invalid(
            "transfer wire frame exceeds its bound".into(),
        ));
    }
    let mut decoder = Decoder::new(bytes);
    if decoder
        .map()
        .map_err(decoding)?
        .ok_or_else(|| ArtifactError::Encoding("indefinite transfer wire frame".into()))?
        != 5
    {
        return Err(ArtifactError::Encoding(
            "transfer wire frame field count is invalid".into(),
        ));
    }
    expect_key(&mut decoder, 0)?;
    if decoder.u16().map_err(decoding)? != 1 {
        return Err(ArtifactError::Invalid(
            "unsupported transfer wire frame schema".into(),
        ));
    }
    expect_key(&mut decoder, 1)?;
    let session_id = decoder
        .bytes()
        .map_err(decoding)?
        .try_into()
        .map_err(|_| ArtifactError::Encoding("invalid transfer session identifier".into()))?;
    expect_key(&mut decoder, 2)?;
    let sequence = decoder.u64().map_err(decoding)?;
    expect_key(&mut decoder, 3)?;
    let frame_kind = decoder.u8().map_err(decoding)?;
    expect_key(&mut decoder, 4)?;
    let encrypted_payload = decoder.bytes().map_err(decoding)?.to_vec();
    if decoder.position() != bytes.len() {
        return Err(ArtifactError::Encoding(
            "transfer wire frame has trailing bytes".into(),
        ));
    }
    Ok(TransferWireFrame {
        session_id,
        sequence,
        frame_kind,
        encrypted_payload,
    })
}

fn context(
    authority: &TransferFrameAuthority<'_>,
    digest: Digest,
    plaintext_size: usize,
) -> ArtifactResult<EncryptionContext> {
    if authority.run_identity.is_empty()
        || authority.run_identity.len() > 256
        || authority.direction.is_empty()
        || authority.direction.len() > 64
        || !authority.run_identity.is_ascii()
        || !authority.direction.is_ascii()
        || authority.key_epoch == 0
    {
        return Err(ArtifactError::Invalid(
            "transfer frame authority is malformed".into(),
        ));
    }
    Ok(EncryptionContext {
        schema_version: 1,
        run_identity: authority.run_identity.to_string(),
        purpose: EncryptionPurpose::TransferFrame,
        object_digest: digest,
        task_identity: Some(format!(
            "overlay:{}:{}:{}",
            hex_session(authority.session_id),
            authority.direction,
            authority.frame_kind
        )),
        attempt_identity: None,
        chunk_index: authority.sequence,
        total_length: u64::try_from(plaintext_size)
            .map_err(|_| ArtifactError::Invalid("transfer frame length overflowed".into()))?,
        key_epoch: authority.key_epoch,
    })
}

fn hex_session(session_id: [u8; 16]) -> String {
    let mut encoded = String::with_capacity(32);
    for byte in session_id {
        use std::fmt::Write as _;
        write!(encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

struct DecodedPayload {
    key_epoch: u32,
    salt: [u8; 32],
    digest: Digest,
    plaintext_size: usize,
    ciphertext: Vec<u8>,
}

fn encode(
    key_epoch: u32,
    encrypted: &EncryptedRunObject,
    maximum_payload_bytes: usize,
) -> ArtifactResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(encrypted.ciphertext.len() + 96);
    Encoder::new(&mut bytes)
        .array(6)
        .and_then(|encoder| encoder.u16(1))
        .and_then(|encoder| encoder.u32(key_epoch))
        .and_then(|encoder| encoder.bytes(&encrypted.derivation_salt))
        .and_then(|encoder| encoder.bytes(encrypted.context.object_digest.bytes()))
        .and_then(|encoder| encoder.u64(encrypted.context.total_length))
        .and_then(|encoder| encoder.bytes(&encrypted.ciphertext))
        .map_err(encoding)?;
    if bytes.len() > maximum_payload_bytes {
        return Err(ArtifactError::Invalid(
            "encrypted transfer frame exceeds its bound".into(),
        ));
    }
    Ok(bytes)
}

fn decode(bytes: &[u8], maximum_payload_bytes: usize) -> ArtifactResult<DecodedPayload> {
    if bytes.len() > maximum_payload_bytes {
        return Err(ArtifactError::Invalid(
            "encrypted transfer frame exceeds its bound".into(),
        ));
    }
    let mut decoder = Decoder::new(bytes);
    if decoder
        .array()
        .map_err(decoding)?
        .ok_or_else(|| ArtifactError::Encoding("indefinite transfer frame".into()))?
        != 6
        || decoder.u16().map_err(decoding)? != 1
    {
        return Err(ArtifactError::Invalid(
            "unsupported encrypted transfer frame".into(),
        ));
    }
    let key_epoch = decoder.u32().map_err(decoding)?;
    let salt: [u8; 32] = decoder
        .bytes()
        .map_err(decoding)?
        .try_into()
        .map_err(|_| ArtifactError::Encoding("invalid transfer frame salt".into()))?;
    let digest = Digest::from_bytes(
        decoder
            .bytes()
            .map_err(decoding)?
            .try_into()
            .map_err(|_| ArtifactError::Encoding("invalid transfer frame digest".into()))?,
    );
    let plaintext_size = usize::try_from(decoder.u64().map_err(decoding)?)
        .map_err(|_| ArtifactError::Invalid("transfer frame length overflowed".into()))?;
    let ciphertext = decoder.bytes().map_err(decoding)?.to_vec();
    if decoder.position() != bytes.len() || ciphertext.len() != plaintext_size.saturating_add(16) {
        return Err(ArtifactError::Identity(
            "transfer frame ciphertext length is invalid".into(),
        ));
    }
    Ok(DecodedPayload {
        key_epoch,
        salt,
        digest,
        plaintext_size,
        ciphertext,
    })
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}

fn decoding(error: minicbor::decode::Error) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}

fn expect_key(decoder: &mut Decoder<'_>, expected: u8) -> ArtifactResult<()> {
    if decoder.u8().map_err(decoding)? != expected {
        return Err(ArtifactError::Encoding(
            "transfer wire frame key order is non-canonical".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_and_wasm_hosts_share_one_canonical_frame_ciphertext_vector() {
        let key = RunKeyMaterial::from_entropy([9; 32]).unwrap();
        let authority = TransferFrameAuthority {
            run_identity: "run-vector",
            session_id: [7; 16],
            direction: "submitter-to-driver",
            frame_kind: 0,
            sequence: 5,
            key_epoch: 1,
        };
        let encoded = seal_transfer_frame(&key, &authority, [3; 32], b"secret", 1024).unwrap();
        let opened = open_transfer_frame(&key, &authority, &encoded, 1024).unwrap();
        assert_eq!(opened.plaintext, b"secret");
        assert_eq!(opened.derivation_salt, [3; 32]);
    }
}
