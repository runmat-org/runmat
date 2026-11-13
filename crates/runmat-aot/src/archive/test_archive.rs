//! Structural target-built archive fixture for manifest and link unit tests.

use std::io::Cursor;

use ar_archive_writer::{
    write_archive_to_stream, ArchiveKind, NewArchiveMember, DEFAULT_OBJECT_READER,
};
use object::write::{Object as WritableObject, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
};
use runmat_execution::host::{
    NativeAcceleratorProfile, NativeArchitecture, NativeOperatingSystem, NATIVE_ABI_PROBE_SYMBOL,
    NATIVE_ACCELERATOR_PROFILE_SYMBOL,
};

pub(crate) fn archive(profile: NativeAcceleratorProfile) -> Vec<u8> {
    let probe = runmat_runtime::native::native_abi_probe();
    archive_for_probe(profile, &probe)
}

pub(crate) fn archive_for_probe(
    profile: NativeAcceleratorProfile,
    probe: &runmat_execution::host::NativeAbiProbe,
) -> Vec<u8> {
    let format = match probe.operating_system {
        NativeOperatingSystem::Linux => BinaryFormat::Elf,
        NativeOperatingSystem::Macos => BinaryFormat::MachO,
        NativeOperatingSystem::Windows => BinaryFormat::Coff,
    };
    let architecture = match probe.architecture {
        NativeArchitecture::X86_64 => Architecture::X86_64,
        NativeArchitecture::Aarch64 => Architecture::Aarch64,
    };
    let mut object = WritableObject::new(format, architecture, Endianness::Little);
    let section = object.add_section(
        Vec::new(),
        b".runmat_identity".to_vec(),
        SectionKind::ReadOnlyData,
    );
    for (name, bytes) in [
        (NATIVE_ABI_PROBE_SYMBOL, probe.canonical_bytes().to_vec()),
        (
            NATIVE_ACCELERATOR_PROFILE_SYMBOL,
            profile.canonical_bytes().to_vec(),
        ),
    ] {
        let offset = object.append_section_data(section, &bytes, 1);
        object.add_symbol(Symbol {
            name: name.as_bytes().to_vec(),
            value: offset,
            size: bytes.len() as u64,
            kind: SymbolKind::Data,
            scope: SymbolScope::Linkage,
            weak: false,
            section: SymbolSection::Section(section),
            flags: SymbolFlags::None,
        });
    }
    let bytes = object.write().expect("structural runtime object");
    let member =
        NewArchiveMember::new(bytes.as_slice(), &DEFAULT_OBJECT_READER, "runtime.o".into());
    let mut output = Cursor::new(Vec::new());
    write_archive_to_stream(&mut output, &[member], ArchiveKind::Gnu, false, None)
        .expect("structural runtime archive");
    output.into_inner()
}
