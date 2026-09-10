use minicbor::{Decoder, Encoder};
use runmat_execution::{host::NativeObjectFormat, Digest};
use serde::{Deserialize, Serialize};

use crate::{ArtifactError, ArtifactResult};

pub const NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION: u16 = 1;
const PREFIX: &[u8] = b"runmat-native-object-payload-v1\0";
const MAX_METADATA_BYTES: usize = 8 * 1024 * 1024;
const MAX_OBJECT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeObjectPayload {
    pub schema_version: u16,
    pub object_format: NativeObjectFormat,
    pub metadata_digest: Digest,
    pub object_digest: Digest,
    pub metadata: Vec<u8>,
    pub object: Vec<u8>,
}

impl NativeObjectPayload {
    pub fn new(
        object_format: NativeObjectFormat,
        metadata: Vec<u8>,
        object: Vec<u8>,
    ) -> ArtifactResult<Self> {
        let payload = Self {
            schema_version: NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION,
            object_format,
            metadata_digest: Digest::sha256(&metadata),
            object_digest: Digest::sha256(&object),
            metadata,
            object,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION
            || self.metadata.is_empty()
            || self.metadata.len() > MAX_METADATA_BYTES
            || self.object.is_empty()
            || self.object.len() > MAX_OBJECT_BYTES
            || self.metadata_digest != Digest::sha256(&self.metadata)
            || self.object_digest != Digest::sha256(&self.object)
        {
            return Err(ArtifactError::Invalid(
                "native object payload is invalid or exceeds its bounds".into(),
            ));
        }
        Ok(())
    }

    pub fn to_canonical_bytes(&self) -> ArtifactResult<Vec<u8>> {
        self.validate()?;
        let mut bytes = PREFIX.to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .array(6)
            .and_then(|encoder| encoder.u16(self.schema_version))
            .and_then(|encoder| encoder.str(self.object_format.token()))
            .and_then(|encoder| encoder.bytes(self.metadata_digest.bytes()))
            .and_then(|encoder| encoder.bytes(self.object_digest.bytes()))
            .and_then(|encoder| encoder.bytes(&self.metadata))
            .and_then(|encoder| encoder.bytes(&self.object))
            .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> ArtifactResult<Self> {
        let encoded = bytes.strip_prefix(PREFIX).ok_or_else(|| {
            ArtifactError::Invalid("native object payload prefix is invalid".into())
        })?;
        let mut decoder = Decoder::new(encoded);
        if decoder.array().map_err(decoding)? != Some(6) {
            return Err(ArtifactError::Invalid(
                "native object payload field count is invalid".into(),
            ));
        }
        let schema_version = decoder.u16().map_err(decoding)?;
        if schema_version != NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(format!(
                "native object payload schema mismatch: actual {schema_version}, expected {NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION}"
            )));
        }
        let object_format = NativeObjectFormat::from_token(decoder.str().map_err(decoding)?)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        let metadata_digest = decode_digest(&mut decoder)?;
        let object_digest = decode_digest(&mut decoder)?;
        let metadata = decode_bounded_bytes(&mut decoder, MAX_METADATA_BYTES, "metadata")?;
        let object = decode_bounded_bytes(&mut decoder, MAX_OBJECT_BYTES, "object")?;
        if decoder.position() != encoded.len() {
            return Err(ArtifactError::Invalid(
                "native object payload has trailing data".into(),
            ));
        }
        let payload = Self {
            schema_version,
            object_format,
            metadata_digest,
            object_digest,
            metadata,
            object,
        };
        payload.validate()?;
        Ok(payload)
    }
}

fn decode_digest(decoder: &mut Decoder<'_>) -> ArtifactResult<Digest> {
    let bytes = decoder.bytes().map_err(decoding)?;
    let bytes: [u8; 32] = bytes.try_into().map_err(|_| {
        ArtifactError::Invalid("native object payload digest length is invalid".into())
    })?;
    Ok(Digest::from_bytes(bytes))
}

fn decode_bounded_bytes(
    decoder: &mut Decoder<'_>,
    maximum: usize,
    label: &str,
) -> ArtifactResult<Vec<u8>> {
    let bytes = decoder.bytes().map_err(decoding)?;
    if bytes.is_empty() || bytes.len() > maximum {
        return Err(ArtifactError::Invalid(format!(
            "native object {label} exceeds its payload bound"
        )));
    }
    Ok(bytes.to_vec())
}

fn decoding(error: minicbor::decode::Error) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        NativeObjectFormat, NativeObjectPayload, NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION, PREFIX,
    };

    #[test]
    fn canonical_payload_round_trips_and_rejects_tampering() {
        let payload = NativeObjectPayload::new(
            NativeObjectFormat::MachO,
            b"metadata".to_vec(),
            b"object".to_vec(),
        )
        .unwrap();
        let bytes = payload.to_canonical_bytes().unwrap();
        assert_eq!(
            NativeObjectPayload::from_canonical_bytes(&bytes).unwrap(),
            payload
        );

        let mut tampered = payload;
        tampered.object.push(0);
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn stale_schema_is_rejected_before_later_fields_are_decoded() {
        let mut bytes = PREFIX.to_vec();
        let mut encoder = minicbor::Encoder::new(&mut bytes);
        encoder.array(6).unwrap().u16(0).unwrap();

        let error = NativeObjectPayload::from_canonical_bytes(&bytes).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "invalid execution artifact: native object payload schema mismatch: actual 0, expected {NATIVE_OBJECT_PAYLOAD_SCHEMA_VERSION}"
            )
        );
    }
}
