use runmat_execution::{Digest, ProgramEnvironment};

use crate::{AotError, AotResult};

use super::{
    RuntimeArchiveCapabilities, RuntimeArchiveEncoding, RuntimeArchiveManifest,
    MAX_RUNTIME_ARCHIVE_BYTES, MAX_RUNTIME_PAYLOAD_BYTES, RUNTIME_ARCHIVE_SCHEMA_VERSION,
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
        let probe = super::read_native_abi_probe(&archive)?;
        if &probe
            != self
                .manifest
                .native_target
                .native_parts()
                .map_err(|error| AotError::contract("aot.archive.probe", error.to_string()))?
                .1
        {
            return Err(AotError::contract(
                "aot.archive.probe",
                "runtime archive probe differs from its manifest target",
            ));
        }
        if super::read_native_accelerator_profile(&archive)?
            != self.manifest.capabilities.accelerators
        {
            return Err(AotError::contract(
                "aot.archive.accelerator_profile",
                "runtime accelerator inventory differs from target-built archive",
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
    capabilities: RuntimeArchiveCapabilities,
) -> AotResult<RuntimeArchive> {
    use runmat_execution::host::{NativeArchitecture, NativeCpuBaseline};

    let probe = super::read_native_abi_probe(archive)?;
    let baseline = match probe.architecture {
        NativeArchitecture::X86_64 => NativeCpuBaseline::X86_64V1,
        NativeArchitecture::Aarch64 => NativeCpuBaseline::ArmV8A,
    };
    let target = runmat_native_codegen::NativeTarget::from_probe(probe, baseline)
        .map_err(|error| AotError::contract("aot.archive.target", error.to_string()))?;
    build_archive(
        archive,
        environment,
        native_link_tokens,
        encoding,
        capabilities,
        target,
        runmat_execution::host::NativeProductMode::HostUnselected,
    )
}

pub(super) fn build_archive(
    archive: &[u8],
    environment: &ProgramEnvironment,
    native_link_tokens: Vec<String>,
    encoding: RuntimeArchiveEncoding,
    capabilities: RuntimeArchiveCapabilities,
    target: runmat_native_codegen::NativeTarget,
    product: runmat_execution::host::NativeProductMode,
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
    if capabilities.accelerators != super::read_native_accelerator_profile(archive)? {
        return Err(AotError::contract(
            "aot.archive.accelerator_profile",
            "declared accelerator inventory differs from target-built archive",
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
        target_triple: target
            .native_parts()
            .map_err(|error| AotError::contract("aot.archive.target", error.to_string()))?
            .0
            .triple()
            .map_err(|error| AotError::contract("aot.archive.target", error.to_string()))?,
        product,
        runmat_version: env!("CARGO_PKG_VERSION").to_string(),
        native_target: target,
        native_ir_schema_version: runmat_native_codegen::NATIVE_IR_SCHEMA_VERSION,
        native_object_schema_version: runmat_native_codegen::aot::NATIVE_OBJECT_SCHEMA_VERSION,
        runtime_fingerprint: environment.runtime_fingerprint,
        catalog_fingerprint: environment.catalog_fingerprint,
        archive_digest: Digest::sha256(archive),
        archive_bytes,
        payload_encoding: encoding,
        payload_digest: Digest::sha256(&payload),
        payload_bytes,
        capabilities,
        native_link_tokens,
    };
    // These bytes and their digests were constructed above. Packaging must
    // remain independent of the packer's architecture; loading the product
    // later uses RuntimeArchive::new and its current-host admission.
    manifest.validate_for_packaging()?;
    Ok(RuntimeArchive { manifest, payload })
}

#[cfg(test)]
mod tests {
    use super::*;

    mod cross_target;

    fn environment() -> ProgramEnvironment {
        ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
            Digest::sha256("runtime"),
            Digest::sha256("catalog"),
            "matlab",
        )
        .unwrap()
    }

    #[test]
    fn compressed_archive_is_exact_and_tamper_evident() {
        let bytes = super::super::test_archive::archive(
            runmat_execution::host::NativeAcceleratorProfile::NONE,
        );
        let archive = build_runtime_archive(
            &bytes,
            &environment(),
            vec!["-lm".into()],
            RuntimeArchiveEncoding::Zstd,
            RuntimeArchiveCapabilities::standalone_host(),
        )
        .unwrap();
        assert_eq!(archive.decode().unwrap(), bytes);
        let mut tampered = archive.clone();
        tampered.payload[0] ^= 1;
        assert!(RuntimeArchive::new(tampered.manifest, tampered.payload).is_err());

        let mut misstated = archive;
        misstated.manifest.capabilities.accelerators =
            runmat_execution::host::NativeAcceleratorProfile {
                wgpu: true,
                cuda: false,
            };
        assert!(misstated.decode().is_err());
    }

    #[test]
    fn archive_rejects_declared_none_for_physical_acceleration() {
        let bytes =
            super::super::test_archive::archive(runmat_execution::host::NativeAcceleratorProfile {
                wgpu: true,
                cuda: false,
            });
        let error = build_runtime_archive(
            &bytes,
            &environment(),
            Vec::new(),
            RuntimeArchiveEncoding::Raw,
            RuntimeArchiveCapabilities::standalone_host(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("aot.archive.accelerator_profile"));
    }

    #[test]
    fn frozen_runtime_archive_three_is_rejected_before_host_admission() {
        let error = RuntimeArchiveManifest::from_json(include_bytes!(
            "../../tests/fixtures/runtime-archive-manifest-3.json"
        ))
        .unwrap_err();
        assert!(error.to_string().contains("aot.archive.schema"));
        assert!(error.to_string().contains(&format!(
            "actual 3, expected {}",
            RUNTIME_ARCHIVE_SCHEMA_VERSION
        )));

        let changed_fields = br#"{"schema_version":3,"native_ir_schema_version":5,"native_object_schema_version":4,"native_link_tokens":"changed representation"}"#;
        let error = RuntimeArchiveManifest::from_json(changed_fields).unwrap_err();
        assert!(error.to_string().contains("aot.archive.schema"));
        assert!(error.to_string().contains(&format!(
            "actual 3, expected {}",
            RUNTIME_ARCHIVE_SCHEMA_VERSION
        )));
    }
}
