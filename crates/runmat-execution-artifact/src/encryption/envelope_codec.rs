use minicbor::{Decoder, Encoder};
use runmat_execution::Digest;

use super::{
    EncryptedArtifact, EncryptionContext, EncryptionPurpose, ExecutionHpkeSuite, RunKeyEnvelope,
};
use crate::{ArtifactError, ArtifactResult};

const PREFIX: &[u8] = b"runmat-run-key-envelope-v1\0";
const FIELD_COUNT: u64 = 15;

pub fn encode_run_key_envelope(envelope: &RunKeyEnvelope) -> ArtifactResult<Vec<u8>> {
    envelope.encrypted_key.context.validate()?;
    validate_shape(envelope)?;
    let mut bytes = PREFIX.to_vec();
    let mut encoder = Encoder::new(&mut bytes);
    encoder
        .array(FIELD_COUNT)
        .and_then(|encoder| encoder.u16(envelope.schema_version))
        .and_then(|encoder| encoder.str(&envelope.recipient_fingerprint))
        .and_then(|encoder| encoder.u16(envelope.encrypted_key.schema_version))
        .and_then(|encoder| encoder.u8(suite_code(envelope.encrypted_key.suite)))
        .and_then(|encoder| encoder.u16(envelope.encrypted_key.context.schema_version))
        .and_then(|encoder| encoder.str(&envelope.encrypted_key.context.run_identity))
        .and_then(|encoder| encoder.u8(envelope.encrypted_key.context.purpose as u8))
        .and_then(|encoder| encoder.bytes(envelope.encrypted_key.context.object_digest.bytes()))
        .map_err(encoding)?;
    encode_optional_text(
        &mut encoder,
        envelope.encrypted_key.context.task_identity.as_deref(),
    )?;
    encode_optional_text(
        &mut encoder,
        envelope.encrypted_key.context.attempt_identity.as_deref(),
    )?;
    encoder
        .u64(envelope.encrypted_key.context.chunk_index)
        .and_then(|encoder| encoder.u64(envelope.encrypted_key.context.total_length))
        .and_then(|encoder| encoder.u32(envelope.encrypted_key.context.key_epoch))
        .and_then(|encoder| encoder.bytes(&envelope.encrypted_key.encapsulated_key))
        .and_then(|encoder| encoder.bytes(&envelope.encrypted_key.ciphertext))
        .map_err(encoding)?;
    Ok(bytes)
}

pub fn decode_run_key_envelope(
    bytes: &[u8],
    maximum_ciphertext_bytes: usize,
) -> ArtifactResult<RunKeyEnvelope> {
    let encoded = bytes
        .strip_prefix(PREFIX)
        .ok_or_else(|| ArtifactError::Encoding("run key envelope prefix is invalid".into()))?;
    let mut decoder = Decoder::new(encoded);
    if decoder
        .array()
        .map_err(decoding)?
        .ok_or_else(|| ArtifactError::Encoding("indefinite run key envelope".into()))?
        != FIELD_COUNT
    {
        return Err(ArtifactError::Encoding(
            "run key envelope field count is invalid".into(),
        ));
    }
    let schema_version = decoder.u16().map_err(decoding)?;
    let recipient_fingerprint = bounded_text(&mut decoder, 256)?;
    let encrypted_schema_version = decoder.u16().map_err(decoding)?;
    let suite = match decoder.u8().map_err(decoding)? {
        0 => ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1,
        _ => {
            return Err(ArtifactError::Invalid(
                "unknown execution HPKE suite".into(),
            ))
        }
    };
    let context_schema_version = decoder.u16().map_err(decoding)?;
    let run_identity = bounded_text(&mut decoder, 256)?;
    let purpose = match decoder.u8().map_err(decoding)? {
        0 => EncryptionPurpose::Bundle,
        1 => EncryptionPurpose::Input,
        2 => EncryptionPurpose::Result,
        3 => EncryptionPurpose::DetailedEvent,
        4 => EncryptionPurpose::Log,
        5 => EncryptionPurpose::Checkpoint,
        6 => EncryptionPurpose::TransferFrame,
        7 => EncryptionPurpose::RunKeyEnvelope,
        _ => return Err(ArtifactError::Invalid("unknown encryption purpose".into())),
    };
    let object_digest = Digest::from_bytes(
        decoder
            .bytes()
            .map_err(decoding)?
            .try_into()
            .map_err(|_| ArtifactError::Encoding("invalid run key object digest".into()))?,
    );
    let task_identity = optional_text(&mut decoder, 256)?;
    let attempt_identity = optional_text(&mut decoder, 256)?;
    let chunk_index = decoder.u64().map_err(decoding)?;
    let total_length = decoder.u64().map_err(decoding)?;
    let key_epoch = decoder.u32().map_err(decoding)?;
    let encapsulated_key = decoder.bytes().map_err(decoding)?.to_vec();
    let ciphertext = decoder.bytes().map_err(decoding)?.to_vec();
    if decoder.position() != encoded.len()
        || encapsulated_key.len() > 128
        || ciphertext.len() > maximum_ciphertext_bytes
    {
        return Err(ArtifactError::Invalid(
            "run key envelope is oversized or has trailing bytes".into(),
        ));
    }
    let context = EncryptionContext {
        schema_version: context_schema_version,
        run_identity,
        purpose,
        object_digest,
        task_identity,
        attempt_identity,
        chunk_index,
        total_length,
        key_epoch,
    };
    context.validate()?;
    let envelope = RunKeyEnvelope {
        schema_version,
        recipient_fingerprint,
        encrypted_key: EncryptedArtifact {
            schema_version: encrypted_schema_version,
            suite,
            context,
            encapsulated_key,
            ciphertext,
        },
    };
    validate_shape(&envelope)?;
    Ok(envelope)
}

fn validate_shape(envelope: &RunKeyEnvelope) -> ArtifactResult<()> {
    if envelope.schema_version != 1
        || envelope.encrypted_key.schema_version != 1
        || envelope.recipient_fingerprint.is_empty()
        || envelope.encrypted_key.context.purpose != EncryptionPurpose::RunKeyEnvelope
        || envelope.encrypted_key.context.task_identity.is_some()
        || envelope.encrypted_key.context.attempt_identity.is_some()
        || envelope.encrypted_key.context.chunk_index != 0
        || envelope.encrypted_key.context.total_length != 32
        || envelope.encrypted_key.context.key_epoch == 0
        || envelope.encrypted_key.encapsulated_key.len() != 32
        || envelope.encrypted_key.ciphertext.is_empty()
    {
        return Err(ArtifactError::Invalid(
            "run key envelope version or authority is invalid".into(),
        ));
    }
    Ok(())
}

fn suite_code(suite: ExecutionHpkeSuite) -> u8 {
    match suite {
        ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1 => 0,
    }
}

fn encode_optional_text(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: Option<&str>,
) -> ArtifactResult<()> {
    match value {
        Some(value) => encoder.str(value),
        None => encoder.null(),
    }
    .map(|_| ())
    .map_err(encoding)
}

fn bounded_text(decoder: &mut Decoder<'_>, maximum: usize) -> ArtifactResult<String> {
    let value = decoder.str().map_err(decoding)?;
    if value.len() > maximum {
        return Err(ArtifactError::Invalid(
            "run key envelope text exceeds its bound".into(),
        ));
    }
    Ok(value.to_string())
}

fn optional_text(decoder: &mut Decoder<'_>, maximum: usize) -> ArtifactResult<Option<String>> {
    if decoder.datatype().map_err(decoding)? == minicbor::data::Type::Null {
        decoder.null().map_err(decoding)?;
        Ok(None)
    } else {
        bounded_text(decoder, maximum).map(Some)
    }
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}

fn decoding(error: minicbor::decode::Error) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}
