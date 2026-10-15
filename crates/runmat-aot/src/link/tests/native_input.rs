use super::*;
use ar_archive_writer::{
    write_archive_to_stream, ArchiveKind, NewArchiveMember, DEFAULT_OBJECT_READER,
};

#[test]
fn native_archive_checks_every_member_against_selected_target_not_build_host() {
    for (architecture, physical) in [
        (object::Architecture::X86_64, NativeArchitecture::X86_64),
        (object::Architecture::Aarch64, NativeArchitecture::Aarch64),
    ] {
        let object = object::write::Object::new(
            object::BinaryFormat::Elf,
            architecture,
            object::Endianness::Little,
        )
        .write()
        .unwrap();
        let target = NativeTargetIdentity::new(
            physical,
            runmat_execution::host::NativeOperatingSystem::Linux,
            64,
            runmat_execution::host::NativeAbi::new("fixture-only-abi").unwrap(),
            NativeObjectFormat::Elf,
        )
        .unwrap();
        validate(&object, TargetPackComponentKind::NativeObject, &target).unwrap();
        let members = [NewArchiveMember::new(
            object.as_slice(),
            &DEFAULT_OBJECT_READER,
            "a.o".into(),
        )];
        let mut archive = std::io::Cursor::new(Vec::new());
        write_archive_to_stream(&mut archive, &members, ArchiveKind::Gnu, false, None).unwrap();
        validate(
            archive.get_ref(),
            TargetPackComponentKind::StaticArchive,
            &target,
        )
        .unwrap();
        let mut changed = target;
        changed.architecture = if physical == NativeArchitecture::X86_64 {
            NativeArchitecture::Aarch64
        } else {
            NativeArchitecture::X86_64
        };
        assert!(validate(
            archive.get_ref(),
            TargetPackComponentKind::StaticArchive,
            &changed
        )
        .is_err());
    }
}
