//! Physical object/archive target validation, separate from host tool admission.
use crate::{AotError, AotResult};

#[derive(Clone, Copy)]
pub(crate) enum NativeImagePurpose {
    Dependency,
    FinalExecutable,
}
use object::Object;
use runmat_execution::host::{NativeArchitecture, NativeObjectFormat, NativeTargetIdentity};
use runmat_execution_artifact::target_pack::TargetPackComponentKind;
use runmat_package::NormalizedRelativePath;

use super::toolchain::NativeHostedDependency;

mod tapi;

pub(crate) fn validate(
    bytes: &[u8],
    kind: TargetPackComponentKind,
    target: &NativeTargetIdentity,
) -> AotResult<()> {
    match kind {
        TargetPackComponentKind::NativeObject => validate_object(bytes, target),
        TargetPackComponentKind::StaticArchive => {
            let archive = object::read::archive::ArchiveFile::parse(bytes).map_err(invalid)?;
            let mut objects = 0;
            for (ordinal, member) in archive.members().enumerate() {
                if ordinal >= 65_536 {
                    return Err(invalid("native archive exceeds its member bound"));
                }
                let member = member.map_err(invalid)?;
                let data = member.data(bytes).map_err(invalid)?;
                // Rust metadata is not linked as machine code. A parseable
                // native object still undergoes target checks regardless of name.
                if object::FileKind::parse(data).is_err() && member.name().ends_with(b".rmeta") {
                    continue;
                }
                validate_object(data, target)?;
                objects += 1;
            }
            if objects == 0 {
                return Err(invalid("native archive contains no target objects"));
            }
            Ok(())
        }
        _ => Err(invalid("non-native component cannot become a linker input")),
    }
}

fn validate_object(bytes: &[u8], target: &NativeTargetIdentity) -> AotResult<()> {
    let architecture = match target.architecture {
        NativeArchitecture::X86_64 => object::Architecture::X86_64,
        NativeArchitecture::Aarch64 => object::Architecture::Aarch64,
    };
    if target.object_format == NativeObjectFormat::Coff
        && object::FileKind::parse(bytes).map_err(invalid)? == object::FileKind::CoffImport
    {
        let import = object::read::coff::ImportFile::parse(bytes).map_err(invalid)?;
        if import.architecture() != architecture || import.sub_architecture().is_some() {
            return Err(invalid(
                "import object architecture differs from selected target",
            ));
        }
        return Ok(());
    }
    let object = object::File::parse(bytes).map_err(invalid)?;
    let format = match target.object_format {
        NativeObjectFormat::Elf => object::BinaryFormat::Elf,
        NativeObjectFormat::MachO => object::BinaryFormat::MachO,
        NativeObjectFormat::Coff => object::BinaryFormat::Coff,
    };
    if object.architecture() != architecture
        || object.sub_architecture().is_some()
        || object.format() != format
        // COFF's header width is independent of its checked machine type.
        || (format != object::BinaryFormat::Coff && !object.is_64())
        || !object.is_little_endian()
        || object.kind() != object::ObjectKind::Relocatable
    {
        return Err(invalid(
            "native link member differs from selected physical target",
        ));
    }
    Ok(())
}

pub(crate) fn validate_qualified_hosted_dependency(
    bytes: &[u8],
    target: &NativeTargetIdentity,
    dependency: &NativeHostedDependency,
    locator: &NormalizedRelativePath,
) -> AotResult<()> {
    if locator.as_str().ends_with(".tbd") || bytes.starts_with(b"--- !tapi-tbd") {
        return tapi::validate(bytes, target, dependency, locator);
    }
    validate_hosted_dependency(bytes, target)
}

pub(crate) fn validate_hosted_dependency(
    bytes: &[u8],
    target: &NativeTargetIdentity,
) -> AotResult<()> {
    if bytes.starts_with(b"!<arch>\n") {
        return validate(bytes, TargetPackComponentKind::StaticArchive, target);
    }
    let image = object::File::parse(bytes).map_err(invalid)?;
    if image.kind() == object::ObjectKind::Relocatable {
        return validate_object(bytes, target);
    }
    validate_image(bytes, target, NativeImagePurpose::Dependency)
}

pub(crate) fn validate_linked_program<'data, R: object::ReadRef<'data>>(
    bytes: R,
    target: &NativeTargetIdentity,
) -> AotResult<()> {
    validate_image(bytes, target, NativeImagePurpose::FinalExecutable)
}

fn validate_image<'data, R: object::ReadRef<'data>>(
    bytes: R,
    target: &NativeTargetIdentity,
    purpose: NativeImagePurpose,
) -> AotResult<()> {
    let image = object::File::parse(bytes).map_err(invalid)?;
    let architecture = match target.architecture {
        NativeArchitecture::X86_64 => object::Architecture::X86_64,
        NativeArchitecture::Aarch64 => object::Architecture::Aarch64,
    };
    let format = match target.object_format {
        NativeObjectFormat::Elf => object::BinaryFormat::Elf,
        NativeObjectFormat::MachO => object::BinaryFormat::MachO,
        NativeObjectFormat::Coff => object::BinaryFormat::Pe,
    };
    if image.architecture() != architecture
        || image.sub_architecture().is_some()
        || image.format() != format
        || !image.is_64()
        || !image.is_little_endian()
    {
        return Err(invalid("linked image differs from selected target"));
    }
    require_image_purpose(&image, purpose)
}

/// ObjectKind::Dynamic deliberately groups ELF PIE and shared libraries.
/// Purpose admission therefore uses the parsed format's executable flags,
/// not the broad cross-format kind alone. This is not an execution proof.
pub(super) fn require_image_purpose<'data, R: object::ReadRef<'data>>(
    image: &object::File<'data, R>,
    purpose: NativeImagePurpose,
) -> AotResult<()> {
    if matches!(purpose, NativeImagePurpose::Dependency) {
        return if matches!(
            image.kind(),
            object::ObjectKind::Executable | object::ObjectKind::Dynamic
        ) {
            Ok(())
        } else {
            Err(invalid("native dependency is not a loadable linked image"))
        };
    }
    let executable = match image {
        object::File::Elf64(file) => {
            use object::read::elf::{Dyn, FileHeader, ProgramHeader};
            let endian = file.endian();
            let entry = file.elf_header().e_entry(endian);
            let file_length = file
                .data()
                .len()
                .map_err(|()| invalid("ELF image length is unavailable"))?;
            let mut pie = false;
            let mut entry_is_code = false;
            for segment in file.elf_program_headers() {
                if segment.p_type(endian) == object::elf::PT_LOAD
                    && segment.p_flags(endian) & object::elf::PF_X != 0
                {
                    let start = segment.p_vaddr(endian);
                    entry_is_code |= entry >= start
                        && start
                            .checked_add(segment.p_filesz(endian))
                            .is_some_and(|end| entry < end)
                        && segment
                            .p_offset(endian)
                            .checked_add(segment.p_filesz(endian))
                            .is_some_and(|end| end <= file_length);
                }
                if let Some(dynamic) = segment.dynamic(endian, file.data()).map_err(invalid)? {
                    for item in dynamic
                        .iter()
                        .take_while(|item| item.d_tag(endian) != u64::from(object::elf::DT_NULL))
                    {
                        pie |= item.d_tag(endian) == u64::from(object::elf::DT_FLAGS_1)
                            && item.d_val(endian) & u64::from(object::elf::DF_1_PIE) != 0;
                    }
                }
            }
            entry_is_code
                && (image.kind() == object::ObjectKind::Executable
                    || (image.kind() == object::ObjectKind::Dynamic && pie))
        }
        object::File::MachO64(_) => image.kind() == object::ObjectKind::Executable,
        object::File::Pe64(_) => {
            image.kind() == object::ObjectKind::Executable
                && matches!(image.flags(),
            object::FileFlags::Coff { characteristics } if characteristics & object::pe::IMAGE_FILE_EXECUTABLE_IMAGE != 0
                && characteristics & (object::pe::IMAGE_FILE_DLL | object::pe::IMAGE_FILE_SYSTEM) == 0)
        }
        _ => false,
    };
    if !executable {
        return Err(invalid("final linked image is not an executable program"));
    }
    Ok(())
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.link.native_input", error.to_string())
}

#[cfg(unix)]
pub(super) fn require_unix_executable_mode(
    metadata: &std::fs::Metadata,
    target: runmat_execution::host::NativeOperatingSystem,
    purpose: NativeImagePurpose,
) -> AotResult<()> {
    use std::os::unix::fs::PermissionsExt;
    if target == runmat_execution::host::NativeOperatingSystem::Windows
        || !matches!(purpose, NativeImagePurpose::FinalExecutable)
    {
        return Ok(());
    }
    if metadata.permissions().mode() & 0o100 == 0 {
        return Err(invalid("final Unix program lacks owner-execute permission"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/native_input.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/image_purpose.rs"]
mod image_purpose_tests;
