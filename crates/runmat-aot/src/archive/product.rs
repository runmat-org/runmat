use runmat_execution::{Digest, ProgramEnvironment};

use crate::{AotError, AotResult};

use super::{
    RuntimeArchiveEncoding, RuntimeArchiveManifest, MAX_RUNTIME_ARCHIVE_BYTES,
    MAX_RUNTIME_PAYLOAD_BYTES, RUNTIME_ARCHIVE_SCHEMA_VERSION,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeArchive {
    pub manifest: RuntimeArchiveManifest,
    payload: Vec<u8>,
}

impl RuntimeArchive {
    pub fn new(manifest: RuntimeArchiveManifest, payload: Vec<u8>) -> AotResult<Self> {
        manifest.validate()?;
        let payload_bytes = u64::try_from(payload.len()).map_err(|_| {
            AotError::contract(
                "aot.archive.payload",
                "runtime archive payload is too large",
            )
        })?;
        if payload_bytes != manifest.payload_bytes
            || Digest::sha256(&payload) != manifest.payload_digest
        {
            return Err(AotError::contract(
                "aot.archive.payload",
                "runtime archive payload does not match its manifest",
            ));
        }
        Ok(Self { manifest, payload })
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn decode(&self) -> AotResult<Vec<u8>> {
        let archive = match self.manifest.payload_encoding {
            RuntimeArchiveEncoding::Raw => self.payload.clone(),
            RuntimeArchiveEncoding::Zstd => zstd::stream::decode_all(self.payload.as_slice())
                .map_err(|error| AotError::contract("aot.archive.decode", error.to_string()))?,
        };
        let bytes = u64::try_from(archive.len()).map_err(|_| {
            AotError::contract("aot.archive.decode", "decoded runtime archive is too large")
        })?;
        if bytes != self.manifest.archive_bytes
            || bytes > MAX_RUNTIME_ARCHIVE_BYTES
            || Digest::sha256(&archive) != self.manifest.archive_digest
        {
            return Err(AotError::contract(
                "aot.archive.identity",
                "decoded runtime archive does not match its manifest",
            ));
        }
        Ok(archive)
    }
}

pub fn build_runtime_archive(
    archive: &[u8],
    environment: &ProgramEnvironment,
    native_link_tokens: Vec<String>,
    encoding: RuntimeArchiveEncoding,
) -> AotResult<RuntimeArchive> {
    let archive_bytes = u64::try_from(archive.len()).map_err(|_| {
        AotError::contract("aot.archive.size", "runtime archive exceeds the host size")
    })?;
    if archive_bytes == 0 || archive_bytes > MAX_RUNTIME_ARCHIVE_BYTES {
        return Err(AotError::contract(
            "aot.archive.size",
            "runtime archive is empty or exceeds its bound",
        ));
    }
    let payload = match encoding {
        RuntimeArchiveEncoding::Raw => archive.to_vec(),
        RuntimeArchiveEncoding::Zstd => zstd::stream::encode_all(archive, 9)
            .map_err(|error| AotError::contract("aot.archive.encode", error.to_string()))?,
    };
    let payload_bytes = u64::try_from(payload.len()).map_err(|_| {
        AotError::contract(
            "aot.archive.payload",
            "runtime payload exceeds the host size",
        )
    })?;
    if payload_bytes == 0 || payload_bytes > MAX_RUNTIME_PAYLOAD_BYTES {
        return Err(AotError::contract(
            "aot.archive.payload",
            "runtime payload is empty or exceeds its bound",
        ));
    }
    let manifest = RuntimeArchiveManifest {
        schema_version: RUNTIME_ARCHIVE_SCHEMA_VERSION,
        runmat_version: env!("CARGO_PKG_VERSION").to_string(),
        target_triple: target_lexicon::HOST.to_string(),
        native_target: runmat_native_codegen::NativeTarget::current(),
        native_ir_schema_version: runmat_native_codegen::NATIVE_IR_SCHEMA_VERSION,
        native_object_schema_version: runmat_native_codegen::aot::NATIVE_OBJECT_SCHEMA_VERSION,
        runtime_fingerprint: environment.runtime_fingerprint,
        catalog_fingerprint: environment.catalog_fingerprint,
        archive_digest: Digest::sha256(archive),
        archive_bytes,
        payload_encoding: encoding,
        payload_digest: Digest::sha256(&payload),
        payload_bytes,
        native_link_tokens,
    };
    RuntimeArchive::new(manifest, payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn environment() -> ProgramEnvironment {
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256("runtime"),
            Digest::sha256("catalog"),
            "matlab",
        )
        .unwrap()
    }

    #[test]
    fn compressed_archive_is_exact_and_tamper_evident() {
        let archive = build_runtime_archive(
            b"!<arch>\nrepeated repeated repeated repeated",
            &environment(),
            vec!["-lm".into()],
            RuntimeArchiveEncoding::Zstd,
        )
        .unwrap();
        assert_eq!(
            archive.decode().unwrap(),
            b"!<arch>\nrepeated repeated repeated repeated"
        );
        let mut tampered = archive.clone();
        tampered.payload[0] ^= 1;
        assert!(RuntimeArchive::new(tampered.manifest, tampered.payload).is_err());
    }
}
