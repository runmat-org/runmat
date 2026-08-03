use minicbor::{Decoder, Encoder};
use runmat_execution::Digest;

use super::{EncryptedRunObject, EncryptionContext, EncryptionPurpose, RunObjectEncryptionSuite};
use crate::{ArtifactError, ArtifactResult};

pub fn encode_encrypted_run_object(object: &EncryptedRunObject) -> ArtifactResult<Vec<u8>> {
    object.context.validate()?;
    let mut bytes = b"runmat-encrypted-run-object-v1\0".to_vec();
    let mut encoder = Encoder::new(&mut bytes);
    encoder
        .array(13)
        .and_then(|encoder| encoder.u16(object.schema_version))
        .and_then(|encoder| encoder.u8(object.suite as u8))
        .and_then(|encoder| encoder.str(&object.context.run_identity))
        .and_then(|encoder| encoder.u8(object.context.purpose as u8))
        .and_then(|encoder| encoder.bytes(object.context.object_digest.bytes()))
        .map_err(encoding)?;
    encode_optional_text(&mut encoder, object.context.task_identity.as_deref())?;
    encode_optional_text(&mut encoder, object.context.attempt_identity.as_deref())?;
    encoder
        .u64(object.context.chunk_index)
        .and_then(|encoder| encoder.u64(object.context.total_length))
        .and_then(|encoder| encoder.u32(object.context.key_epoch))
        .and_then(|encoder| encoder.bytes(&object.derivation_salt))
        .and_then(|encoder| encoder.bytes(&object.ciphertext))
        .and_then(|encoder| encoder.u16(object.context.schema_version))
        .map_err(encoding)?;
    Ok(bytes)
}

pub fn decode_encrypted_run_object(
    bytes: &[u8],
    maximum_ciphertext_bytes: usize,
) -> ArtifactResult<EncryptedRunObject> {
    const PREFIX: &[u8] = b"runmat-encrypted-run-object-v1\0";
    let encoded = bytes
        .strip_prefix(PREFIX)
        .ok_or_else(|| ArtifactError::Encoding("encrypted run object prefix is invalid".into()))?;
    let mut decoder = Decoder::new(encoded);
    if decoder
        .array()
        .map_err(decoding)?
        .ok_or_else(|| ArtifactError::Encoding("indefinite encrypted run object".into()))?
        != 13
    {
        return Err(ArtifactError::Encoding(
            "encrypted run object field count is invalid".into(),
        ));
    }
    let schema_version = decoder.u16().map_err(decoding)?;
    let suite = match decoder.u8().map_err(decoding)? {
        0 => RunObjectEncryptionSuite::HkdfSha256Aes256GcmV1,
        _ => {
            return Err(ArtifactError::Invalid(
                "unsupported encrypted run object suite".into(),
            ))
        }
    };
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
            .map_err(|_| ArtifactError::Encoding("invalid encrypted object digest".into()))?,
    );
    let task_identity = optional_text(&mut decoder, 256)?;
    let attempt_identity = optional_text(&mut decoder, 256)?;
    let chunk_index = decoder.u64().map_err(decoding)?;
    let total_length = decoder.u64().map_err(decoding)?;
    let key_epoch = decoder.u32().map_err(decoding)?;
    let derivation_salt = decoder.bytes().map_err(decoding)?.to_vec();
    let ciphertext = decoder.bytes().map_err(decoding)?.to_vec();
    let context_schema_version = decoder.u16().map_err(decoding)?;
    if decoder.position() != encoded.len() || ciphertext.len() > maximum_ciphertext_bytes {
        return Err(ArtifactError::Invalid(
            "encrypted run object is oversized or has trailing bytes".into(),
        ));
    }
    let object = EncryptedRunObject {
        schema_version,
        suite,
        context: EncryptionContext {
            schema_version: context_schema_version,
            run_identity,
            purpose,
            object_digest,
            task_identity,
            attempt_identity,
            chunk_index,
            total_length,
            key_epoch,
        },
        derivation_salt,
        ciphertext,
    };
    object.context.validate()?;
    Ok(object)
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
            "encrypted run object text exceeds its bound".into(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::{RunKeyMaterial, RunObjectEncryption};

    #[test]
    fn canonical_object_codec_roundtrips_and_rejects_trailing_data() {
        let plaintext = b"portable";
        let object = RunObjectEncryption
            .seal_with_entropy(
                &RunKeyMaterial::from_entropy([5; 32]).unwrap(),
                [7; 32],
                EncryptionContext {
                    schema_version: 1,
                    run_identity: "run-codec".into(),
                    purpose: EncryptionPurpose::Result,
                    object_digest: Digest::sha256(plaintext),
                    task_identity: Some("task-a".into()),
                    attempt_identity: Some("attempt-a".into()),
                    chunk_index: 0,
                    total_length: plaintext.len() as u64,
                    key_epoch: 1,
                },
                plaintext,
            )
            .unwrap();
        let encoded = encode_encrypted_run_object(&object).unwrap();
        assert_eq!(decode_encrypted_run_object(&encoded, 1024).unwrap(), object);
        let mut trailing = encoded;
        trailing.push(0);
        assert!(decode_encrypted_run_object(&trailing, 1024).is_err());
    }
}
