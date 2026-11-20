use super::*;
use crate::archive::{build_runtime_archive, RuntimeArchiveCapabilities, RuntimeArchiveEncoding};
use runmat_execution::host::{NativeAcceleratorProfile, NativeArchitecture};

fn runtime(encoding: RuntimeArchiveEncoding) -> RuntimeArchive {
    let bytes = crate::archive::test_archive::archive(NativeAcceleratorProfile::NONE);
    build_runtime_archive(
        &bytes,
        &runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        Vec::new(),
        encoding,
        RuntimeArchiveCapabilities::standalone_host(),
    )
    .unwrap()
}

fn pack(runtime: &RuntimeArchive) -> RuntimePack {
    build_runtime_pack(
        runtime,
        RuntimePackLicenses {
            license: b"First-party license fixture\n",
            free_use: b"Free-use grant fixture\n",
        },
    )
    .unwrap()
}

#[test]
fn stock_pack_round_trips_exact_payload_identity_and_notices_deterministically() {
    // This is the existing structural native archive fixture, not executable
    // product qualification. Real standalone linking remains a separate gate.
    let runtime = runtime(RuntimeArchiveEncoding::Raw);
    let pack = pack(&runtime);
    assert_eq!(self::pack(&runtime).bytes, pack.bytes);
    let descriptor =
        RuntimePackManifest::from_json(&serde_json::to_vec(&pack.manifest).unwrap()).unwrap();
    assert_eq!(descriptor, pack.manifest);
    let decoded = decode_runtime_pack(&descriptor, &pack.bytes).unwrap();
    assert_eq!(decoded.manifest, runtime.manifest);
    assert_eq!(decoded.payload(), runtime.payload());
    assert_eq!(decoded.decode().unwrap(), runtime.decode().unwrap());
}

#[test]
fn a_different_valid_pack_cannot_replace_the_pinned_pack() {
    let raw = pack(&runtime(RuntimeArchiveEncoding::Raw));
    let compressed = pack(&runtime(RuntimeArchiveEncoding::Zstd));
    assert_ne!(raw.manifest.pack_digest, compressed.manifest.pack_digest);
    assert!(decode_runtime_pack(&raw.manifest, &compressed.bytes).is_err());
    assert!(decode_runtime_pack(&compressed.manifest, &raw.bytes).is_err());
}

#[test]
fn foreign_pack_production_does_not_admit_it_for_the_build_host() {
    let mut probe = runmat_runtime::native::native_abi_probe();
    probe.architecture = match probe.architecture {
        NativeArchitecture::X86_64 => NativeArchitecture::Aarch64,
        NativeArchitecture::Aarch64 => NativeArchitecture::X86_64,
    };
    let bytes =
        crate::archive::test_archive::archive_for_probe(NativeAcceleratorProfile::NONE, &probe);
    let runtime = build_runtime_archive(
        &bytes,
        &runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        Vec::new(),
        RuntimeArchiveEncoding::Raw,
        RuntimeArchiveCapabilities::standalone_host(),
    )
    .unwrap();
    let pack = pack(&runtime);
    let descriptor =
        RuntimePackManifest::from_json(&serde_json::to_vec(&pack.manifest).unwrap()).unwrap();
    assert!(descriptor.validate().is_err());
    assert!(decode_runtime_pack(&descriptor, &pack.bytes).is_err());
}
