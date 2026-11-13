//! Typed object-writer/header fixtures classify image purpose only. They are
//! never executed and do not qualify a real linker, runtime or target.
use super::*;

#[test]
fn elf_pie_and_executable_are_distinct_from_shared_dependency_images() {
    for (kind, flags, mapped_entry, executable) in [
        (object::elf::ET_EXEC, 0, true, true),
        (object::elf::ET_DYN, object::elf::DF_1_PIE, true, true),
        (object::elf::ET_DYN, 0, true, false),
        (object::elf::ET_DYN, object::elf::DF_1_PIE, false, false),
    ] {
        let bytes = elf(kind, flags, mapped_entry);
        let image = object::File::parse(bytes.as_slice()).unwrap();
        require_image_purpose(&image, NativeImagePurpose::Dependency).unwrap();
        assert_eq!(
            require_image_purpose(&image, NativeImagePurpose::FinalExecutable).is_ok(),
            executable
        );
    }
}

fn elf(kind: u16, flags: u32, mapped_entry: bool) -> Vec<u8> {
    use object::write::elf::{FileHeader, ProgramHeader, Writer};
    use object::{elf, Endianness};
    let mut bytes = Vec::new();
    let mut writer = Writer::new(Endianness::Little, true, &mut bytes);
    writer.reserve_file_header();
    writer.reserve_program_headers(2);
    let code = writer.reserve(1, 4096) as u64;
    let dynamic = writer.reserve(32, 8) as u64;
    writer
        .write_file_header(&FileHeader {
            os_abi: 0,
            abi_version: 0,
            e_type: kind,
            e_machine: elf::EM_X86_64,
            e_entry: code + u64::from(!mapped_entry),
            e_flags: 0,
        })
        .unwrap();
    writer.write_align_program_headers();
    writer.write_program_header(&ProgramHeader {
        p_type: elf::PT_LOAD,
        p_flags: elf::PF_R | elf::PF_X,
        p_offset: code,
        p_vaddr: code,
        p_paddr: code,
        p_filesz: 1,
        p_memsz: 1,
        p_align: 4096,
    });
    writer.write_program_header(&ProgramHeader {
        p_type: elf::PT_DYNAMIC,
        p_flags: elf::PF_R,
        p_offset: dynamic,
        p_vaddr: dynamic,
        p_paddr: dynamic,
        p_filesz: 32,
        p_memsz: 32,
        p_align: 8,
    });
    writer.pad_until(code as usize);
    writer.write(&[0]);
    writer.pad_until(dynamic as usize);
    for (tag, value) in [(elf::DT_FLAGS_1, u64::from(flags)), (elf::DT_NULL, 0)] {
        let entry = elf::Dyn64 {
            d_tag: object::endian::U64::new(Endianness::Little, u64::from(tag)),
            d_val: object::endian::U64::new(Endianness::Little, value),
        };
        writer.write(object::pod::bytes_of(&entry));
    }
    bytes
}

#[test]
fn macos_dylibs_and_pe_dlls_are_not_final_programs() {
    use object::{
        endian::{BigEndian, LittleEndian, U32},
        macho, pe,
    };
    for (kind, executable) in [(macho::MH_EXECUTE, true), (macho::MH_DYLIB, false)] {
        let header = macho::MachHeader64 {
            magic: U32::new(BigEndian, macho::MH_CIGAM_64),
            cputype: U32::new(LittleEndian, macho::CPU_TYPE_X86_64),
            cpusubtype: U32::new(LittleEndian, macho::CPU_SUBTYPE_X86_64_ALL),
            filetype: U32::new(LittleEndian, kind),
            ncmds: U32::default(),
            sizeofcmds: U32::default(),
            flags: U32::default(),
            reserved: U32::default(),
        };
        let image = object::File::parse(object::pod::bytes_of(&header)).unwrap();
        require_image_purpose(&image, NativeImagePurpose::Dependency).unwrap();
        assert_eq!(
            require_image_purpose(&image, NativeImagePurpose::FinalExecutable).is_ok(),
            executable
        );
    }
    for (flags, executable) in [
        (pe::IMAGE_FILE_EXECUTABLE_IMAGE, true),
        (pe::IMAGE_FILE_EXECUTABLE_IMAGE | pe::IMAGE_FILE_DLL, false),
        (0, false),
    ] {
        let bytes = pe_image(flags);
        let image = object::File::parse(bytes.as_slice()).unwrap();
        assert_eq!(
            require_image_purpose(&image, NativeImagePurpose::FinalExecutable).is_ok(),
            executable
        );
    }
}

fn pe_image(characteristics: u16) -> Vec<u8> {
    use object::{
        pe,
        write::pe::{NtHeaders, Writer},
    };
    let mut bytes = Vec::new();
    let mut writer = Writer::new(true, 4096, 512, &mut bytes);
    writer.reserve_dos_header();
    writer.reserve_nt_headers(0);
    writer.reserve_section_headers(0);
    writer.write_empty_dos_header().unwrap();
    writer.write_nt_headers(NtHeaders {
        machine: pe::IMAGE_FILE_MACHINE_AMD64,
        time_date_stamp: 0,
        characteristics,
        major_linker_version: 0,
        minor_linker_version: 0,
        address_of_entry_point: 0,
        image_base: 0x140000000,
        major_operating_system_version: 6,
        minor_operating_system_version: 0,
        major_image_version: 0,
        minor_image_version: 0,
        major_subsystem_version: 6,
        minor_subsystem_version: 0,
        subsystem: pe::IMAGE_SUBSYSTEM_WINDOWS_CUI,
        dll_characteristics: 0,
        size_of_stack_reserve: 0,
        size_of_stack_commit: 0,
        size_of_heap_reserve: 0,
        size_of_heap_commit: 0,
    });
    writer.write_section_headers();
    bytes
}

#[cfg(unix)]
#[test]
fn final_unix_program_requires_owner_execute_mode() {
    use std::os::unix::fs::PermissionsExt;
    let file = tempfile::NamedTempFile::new().unwrap();
    for (mode, executable) in [(0o644, false), (0o744, true)] {
        file.as_file()
            .set_permissions(std::fs::Permissions::from_mode(mode))
            .unwrap();
        let metadata = file.as_file().metadata().unwrap();
        for target in [
            runmat_execution::host::NativeOperatingSystem::Linux,
            runmat_execution::host::NativeOperatingSystem::Macos,
            runmat_execution::host::NativeOperatingSystem::Windows,
        ] {
            assert_eq!(
                require_unix_executable_mode(
                    &metadata,
                    target,
                    NativeImagePurpose::FinalExecutable
                )
                .is_ok(),
                executable || target == runmat_execution::host::NativeOperatingSystem::Windows
            );
            require_unix_executable_mode(&metadata, target, NativeImagePurpose::Dependency)
                .unwrap();
        }
    }
}
