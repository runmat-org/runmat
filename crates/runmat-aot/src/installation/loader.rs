//! Loader metadata read through the existing object decoder. This inventory
//! does not emulate target search rules or certify actual loader resolution.
use runmat_package::NormalizedRelativePath;

use super::invalid;
use crate::AotResult;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeLoaderReferenceKind {
    ElfInterpreter,
    ElfDynamic { tag: u64 },
    MachLoadCommand { command: u32 },
    PeImport,
    PeDelayImport,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLoaderReference {
    pub kind: NativeLoaderReferenceKind,
    /// Raw identifier bytes, without lossy UTF-8 conversion or path rewriting.
    pub value: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLoaderImage {
    pub path: NormalizedRelativePath,
    /// Original command order, including repeated references and search paths.
    pub references: Vec<NativeLoaderReference>,
}

pub(super) fn observe<'data, R: object::ReadRef<'data>>(
    bytes: R,
) -> AotResult<Vec<NativeLoaderReference>> {
    use object::read::elf::{Dyn, ProgramHeader};
    use object::Object;
    let image = object::File::parse(bytes).map_err(invalid)?;
    if !matches!(
        image.kind(),
        object::ObjectKind::Executable | object::ObjectKind::Dynamic
    ) {
        return Err(invalid("loader inventory requires a physical load image"));
    }
    let mut result = References {
        values: Vec::new(),
        bytes: 0,
    };
    match image {
        object::File::Elf64(file) => {
            let endian = file.endian();
            let headers = file.elf_program_headers();
            for header in headers {
                if let Some(value) = header.interpreter(endian, file.data()).map_err(invalid)? {
                    push(
                        &mut result,
                        NativeLoaderReferenceKind::ElfInterpreter,
                        value,
                    )?;
                }
                let Some(dynamic) = header.dynamic(endian, file.data()).map_err(invalid)? else {
                    continue;
                };
                let end = dynamic
                    .iter()
                    .take(65_537)
                    .position(|entry| entry.d_tag(endian) == u64::from(object::elf::DT_NULL))
                    .ok_or_else(|| {
                        invalid("ELF dynamic inventory exceeds its bound or lacks a terminator")
                    })?;
                let dynamic = &dynamic[..end];
                let address = dynamic
                    .iter()
                    .find(|entry| entry.d_tag(endian) == u64::from(object::elf::DT_STRTAB))
                    .map(|entry| entry.d_val(endian));
                let length = dynamic
                    .iter()
                    .find(|entry| entry.d_tag(endian) == u64::from(object::elf::DT_STRSZ))
                    .map(|entry| entry.d_val(endian));
                for entry in dynamic {
                    let tag = entry.d_tag(endian);
                    if ![
                        object::elf::DT_NEEDED,
                        object::elf::DT_SONAME,
                        object::elf::DT_RPATH,
                        object::elf::DT_RUNPATH,
                        object::elf::DT_AUXILIARY,
                        object::elf::DT_FILTER,
                        object::elf::DT_CONFIG,
                        object::elf::DT_AUDIT,
                        object::elf::DT_DEPAUDIT,
                    ]
                    .iter()
                    .any(|value| tag == u64::from(*value))
                    {
                        continue;
                    }
                    let (address, length) = address
                        .zip(length)
                        .ok_or_else(|| invalid("ELF loader reference has no string table"))?;
                    let mut strings = None;
                    for load in headers {
                        if load.p_type(endian) == object::elf::PT_LOAD {
                            if let Some(data) = load
                                .data_range(endian, file.data(), address, length)
                                .map_err(|()| invalid("invalid ELF loader string segment"))?
                            {
                                strings = Some(data);
                                break;
                            }
                        }
                    }
                    let strings =
                        strings.ok_or_else(|| invalid("ELF loader strings are not file backed"))?;
                    let table = object::read::StringTable::new(strings, 0, strings.len() as u64);
                    let offset = u32::try_from(entry.d_val(endian)).map_err(invalid)?;
                    push(
                        &mut result,
                        NativeLoaderReferenceKind::ElfDynamic { tag },
                        table.get(offset).map_err(|()| {
                            invalid("ELF loader reference has an invalid string offset")
                        })?,
                    )?;
                }
            }
        }
        object::File::MachO64(file) => {
            use object::read::macho::LoadCommandVariant;
            let endian = file.endian();
            let mut commands = file.macho_load_commands().map_err(invalid)?;
            while let Some(command) = commands.next().map_err(invalid)? {
                let reference = match command.variant().map_err(invalid)? {
                    LoadCommandVariant::Dylib(value) | LoadCommandVariant::IdDylib(value) => {
                        Some(value.dylib.name)
                    }
                    LoadCommandVariant::Rpath(value) => Some(value.path),
                    LoadCommandVariant::LoadDylinker(value)
                    | LoadCommandVariant::IdDylinker(value)
                    | LoadCommandVariant::DyldEnvironment(value) => Some(value.name),
                    // Obsolete resolution contracts are not silently ignored.
                    LoadCommandVariant::PreboundDylib(_)
                    | LoadCommandVariant::SubFramework(_)
                    | LoadCommandVariant::SubUmbrella(_)
                    | LoadCommandVariant::SubClient(_)
                    | LoadCommandVariant::SubLibrary(_) => {
                        return Err(invalid("unsupported legacy Mach loader contract"))
                    }
                    _ => None,
                };
                if let Some(reference) = reference {
                    push(
                        &mut result,
                        NativeLoaderReferenceKind::MachLoadCommand {
                            command: command.cmd(),
                        },
                        command.string(endian, reference).map_err(invalid)?,
                    )?;
                }
            }
        }
        object::File::Pe64(file) => {
            if let Some(table) = file.import_table().map_err(invalid)? {
                let mut descriptors = table.descriptors().map_err(invalid)?;
                while let Some(entry) = descriptors.next().map_err(invalid)? {
                    push(
                        &mut result,
                        NativeLoaderReferenceKind::PeImport,
                        table
                            .name(entry.name.get(object::LittleEndian))
                            .map_err(invalid)?,
                    )?;
                }
            }
            if let Some(table) = file
                .data_directories()
                .delay_load_import_table(file.data(), &file.section_table())
                .map_err(invalid)?
            {
                let mut descriptors = table.descriptors().map_err(invalid)?;
                while let Some(entry) = descriptors.next().map_err(invalid)? {
                    // The object decoder addresses this table by RVA. Do not
                    // reinterpret obsolete VA descriptors as RVA metadata.
                    const DELAY_IMPORT_RVA: u32 = 1;
                    if entry.attributes.get(object::LittleEndian) != DELAY_IMPORT_RVA {
                        return Err(invalid("unsupported PE delay-import addressing contract"));
                    }
                    push(
                        &mut result,
                        NativeLoaderReferenceKind::PeDelayImport,
                        table
                            .name(entry.dll_name_rva.get(object::LittleEndian))
                            .map_err(invalid)?,
                    )?;
                }
            }
        }
        _ => return Err(invalid("unsupported installed loader image format")),
    }
    Ok(result.values)
}

struct References {
    values: Vec<NativeLoaderReference>,
    bytes: usize,
}

fn push(result: &mut References, kind: NativeLoaderReferenceKind, value: &[u8]) -> AotResult<()> {
    if result.values.len() >= 65_536
        || value.is_empty()
        || value.len() > 65_536
        || result.bytes.saturating_add(value.len()) > 16 * 1024 * 1024
    {
        return Err(invalid("loader reference inventory exceeds its bound"));
    }
    result.bytes += value.len();
    result.values.push(NativeLoaderReference {
        kind,
        value: value.to_vec(),
    });
    Ok(())
}
