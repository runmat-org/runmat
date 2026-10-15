use super::*;
use ar_archive_writer::{
    write_archive_to_stream, ArchiveKind, NewArchiveMember, DEFAULT_OBJECT_READER,
};
use object::write::{Object as WritableObject, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
};
use runmat_execution::host::{NativeAbiField, NativeEnvironment};
use std::io::Cursor;

fn archive(objects: &[Vec<u8>]) -> Vec<u8> {
    let members = objects
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            NewArchiveMember::new(
                bytes.as_slice(),
                &DEFAULT_OBJECT_READER,
                format!("probe{i}.o"),
            )
        })
        .collect::<Vec<_>>();
    let mut output = Cursor::new(Vec::new());
    write_archive_to_stream(&mut output, &members, ArchiveKind::Gnu, false, None).unwrap();
    output.into_inner()
}
fn object(
    probe: &NativeAbiProbe,
    format: BinaryFormat,
    architecture: Architecture,
    bytes: &[u8],
) -> Vec<u8> {
    object_with_relocation(probe, format, architecture, bytes, false)
}
fn object_with_relocation(
    probe: &NativeAbiProbe,
    format: BinaryFormat,
    architecture: Architecture,
    bytes: &[u8],
    relocated: bool,
) -> Vec<u8> {
    let mut object = WritableObject::new(format, architecture, Endianness::Little);
    let section = object.add_section(
        Vec::new(),
        b".runmat_abi".to_vec(),
        SectionKind::ReadOnlyData,
    );
    let offset = object.append_section_data(section, bytes, 1);
    let name = if format == BinaryFormat::MachO {
        format!("_{NATIVE_ABI_PROBE_SYMBOL}")
    } else {
        NATIVE_ABI_PROBE_SYMBOL.to_string()
    };
    let symbol = object.add_symbol(Symbol {
        name: name.into_bytes(),
        value: offset,
        size: NATIVE_ABI_PROBE_BYTES as u64,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });
    if relocated {
        object
            .add_relocation(
                section,
                object::write::Relocation {
                    offset: offset + 450,
                    symbol,
                    addend: 0,
                    flags: object::RelocationFlags::Generic {
                        kind: object::RelocationKind::Absolute,
                        encoding: object::RelocationEncoding::Generic,
                        size: 64,
                    },
                },
            )
            .unwrap();
    }
    assert_eq!(probe.canonical_bytes().len(), NATIVE_ABI_PROBE_BYTES);
    object.write().unwrap()
}

#[test]
fn fixed_probe_is_read_without_foreign_execution_and_rejects_container_or_byte_substitution() {
    // Container fixtures are intentionally synthetic metadata, not evidence
    // that this host has executed or ABI-probed either foreign target.
    for (format, operating_system, environment) in [
        (
            BinaryFormat::Elf,
            NativeOperatingSystem::Linux,
            NativeEnvironment::Gnu,
        ),
        (
            BinaryFormat::MachO,
            NativeOperatingSystem::Macos,
            NativeEnvironment::Darwin,
        ),
        (
            BinaryFormat::Coff,
            NativeOperatingSystem::Windows,
            NativeEnvironment::Msvc,
        ),
    ] {
        for (architecture, physical_architecture) in [
            (Architecture::X86_64, NativeArchitecture::X86_64),
            (Architecture::Aarch64, NativeArchitecture::Aarch64),
        ] {
            let mut probe = runmat_runtime::native::native_abi_probe();
            probe.operating_system = operating_system;
            probe.environment = environment;
            probe.architecture = physical_architecture;
            let bytes = object(&probe, format, architecture, &probe.canonical_bytes());
            assert_eq!(
                read_native_abi_probe(&archive(std::slice::from_ref(&bytes))).unwrap(),
                probe
            );
            if format == BinaryFormat::Elf && architecture == Architecture::X86_64 {
                let relocated = object_with_relocation(
                    &probe,
                    format,
                    architecture,
                    &probe.canonical_bytes(),
                    true,
                );
                assert!(
                    read_native_abi_probe(&archive(&[relocated])).is_err(),
                    "canonical-looking bytes with unresolved target relocations must reject"
                );
            }
            assert!(read_native_abi_probe(&archive(&[bytes.clone(), bytes])).is_err());
            let other = if architecture == Architecture::X86_64 {
                Architecture::Aarch64
            } else {
                Architecture::X86_64
            };
            assert!(read_native_abi_probe(&archive(&[object(
                &probe,
                format,
                other,
                &probe.canonical_bytes()
            )]))
            .is_err());
            let mut corrupt = probe.canonical_bytes();
            corrupt[8] += 1;
            assert!(read_native_abi_probe(&archive(&[object(
                &probe,
                format,
                architecture,
                &corrupt
            )]))
            .is_err());
            let mut corrupt = probe.clone();
            corrupt.offsets[NativeAbiField::RequestFunction as usize] =
                corrupt.offset(NativeAbiField::RequestBlock);
            assert!(read_native_abi_probe(&archive(&[object(
                &probe,
                format,
                architecture,
                &corrupt.canonical_bytes()
            )]))
            .is_err());
            let mut trailing = probe.canonical_bytes().to_vec();
            trailing.push(0);
            assert!(NativeAbiProbe::from_canonical_bytes(&trailing).is_err());
        }
    }
    assert!(read_native_abi_probe(&archive(&[])).is_err());
}
