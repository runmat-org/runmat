use super::*;
use runmat_execution::host::{NativeAcceleratorProfile, NativeArchitecture};

#[test]
fn stock_packaging_uses_archive_target_without_granting_foreign_host_admission() {
    // Synthetic foreign metadata tests the producer/consumer boundary, not
    // actual execution or ABI layout on another architecture.
    let mut probe = runmat_runtime::native::native_abi_probe();
    probe.architecture = match probe.architecture {
        NativeArchitecture::X86_64 => NativeArchitecture::Aarch64,
        NativeArchitecture::Aarch64 => NativeArchitecture::X86_64,
    };
    let bytes = super::super::super::test_archive::archive_for_probe(
        NativeAcceleratorProfile::NONE,
        &probe,
    );
    let archive = build_runtime_archive(
        &bytes,
        &environment(),
        Vec::new(),
        RuntimeArchiveEncoding::Raw,
        RuntimeArchiveCapabilities::standalone_host(),
    )
    .unwrap();
    let (target, recorded_probe) = archive.manifest.native_target.native_parts().unwrap();
    assert_eq!(recorded_probe, &probe);
    assert_eq!(archive.manifest.target_triple, target.triple().unwrap());
    assert_ne!(
        archive.manifest.target_triple,
        target_lexicon::HOST.to_string()
    );
    archive.manifest.validate_for_packaging().unwrap();
    assert_eq!(archive.decode().unwrap(), bytes);
    assert!(archive.manifest.validate().is_err());
    assert!(RuntimeArchive::new(archive.manifest.clone(), archive.payload.clone()).is_err());

    let mut inconsistent = archive.clone();
    inconsistent.manifest.target_triple = target_lexicon::HOST.to_string();
    assert!(inconsistent.manifest.validate_for_packaging().is_err());

    // A self-consistent host label still cannot relabel foreign object bytes.
    inconsistent.manifest.native_target = runmat_native_codegen::NativeTarget::current();
    inconsistent.manifest.validate().unwrap();
    assert!(inconsistent.decode().is_err());
}
