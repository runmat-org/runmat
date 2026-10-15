//! Read target-compiled probe bytes without loading or executing target code.
use crate::{AotError, AotResult};
use object::{Object, ObjectSection, ObjectSymbol};
use runmat_execution::host::{
    NativeAbiProbe, NativeArchitecture, NativeOperatingSystem, NATIVE_ABI_PROBE_BYTES,
    NATIVE_ABI_PROBE_SYMBOL,
};

pub fn read_native_abi_probe(archive: &[u8]) -> AotResult<NativeAbiProbe> {
    if archive.len() as u64 > super::MAX_RUNTIME_ARCHIVE_BYTES {
        return Err(invalid("archive exceeds its bound"));
    }
    let parsed = object::read::archive::ArchiveFile::parse(archive).map_err(invalid)?;
    let mut found = None;
    for (ordinal, member) in parsed.members().enumerate() {
        if ordinal >= 65536 {
            return Err(invalid("archive member count exceeds its bound"));
        }
        let member = member.map_err(invalid)?;
        let data = member.data(archive).map_err(invalid)?;
        // Rust metadata and archive bookkeeping are not executable objects.
        let Ok(file) = object::File::parse(data) else {
            continue;
        };
        for symbol in file.symbols() {
            if symbol.is_undefined() {
                continue;
            }
            let name = symbol.name().map_err(invalid)?;
            let expected_name = if file.format() == object::BinaryFormat::MachO {
                format!("_{NATIVE_ABI_PROBE_SYMBOL}")
            } else {
                NATIVE_ABI_PROBE_SYMBOL.to_string()
            };
            if name != expected_name {
                continue;
            }
            if file.kind() != object::ObjectKind::Relocatable {
                return Err(invalid("probe is not in a relocatable target object"));
            }
            if found.is_some() {
                return Err(invalid("duplicate target ABI probe definitions"));
            }
            if symbol.size() != 0 && symbol.size() != NATIVE_ABI_PROBE_BYTES as u64 {
                return Err(invalid(
                    "probe symbol size differs from its canonical extent",
                ));
            }
            let section = file
                .section_by_index(
                    symbol
                        .section_index()
                        .ok_or_else(|| invalid("probe has no section"))?,
                )
                .map_err(invalid)?;
            let offset = symbol
                .address()
                .checked_sub(section.address())
                .ok_or_else(|| invalid("probe address precedes its section"))?;
            let end = offset
                .checked_add(NATIVE_ABI_PROBE_BYTES as u64)
                .ok_or_else(|| invalid("probe extent overflows"))?;
            if section.relocations().any(|(position, relocation)| {
                if relocation.size() == 0 {
                    return true; // Unknown write extent cannot certify fixed bytes.
                }
                let width = u64::from(relocation.size()).div_ceil(8).max(1);
                position < end
                    && position
                        .checked_add(width)
                        .is_none_or(|relocation_end| relocation_end > offset)
            }) {
                return Err(invalid("probe bytes contain target relocations"));
            }
            let bytes = section
                .data_range(symbol.address(), NATIVE_ABI_PROBE_BYTES as u64)
                .map_err(invalid)?
                .ok_or_else(|| invalid("probe extends beyond initialized section bytes"))?;
            let probe = NativeAbiProbe::from_canonical_bytes(bytes).map_err(invalid)?;
            let architecture = match file.architecture() {
                object::Architecture::X86_64 => NativeArchitecture::X86_64,
                object::Architecture::Aarch64 => NativeArchitecture::Aarch64,
                _ => return Err(invalid("unsupported probe object architecture")),
            };
            let compatible_format = matches!(
                (file.format(), probe.operating_system),
                (object::BinaryFormat::Elf, NativeOperatingSystem::Linux)
                    | (object::BinaryFormat::MachO, NativeOperatingSystem::Macos)
                    | (object::BinaryFormat::Coff, NativeOperatingSystem::Windows)
            );
            if probe.architecture != architecture
                || !compatible_format
                || !file.is_64()
                || !file.is_little_endian()
            {
                return Err(invalid(
                    "probe target facts disagree with the containing object",
                ));
            }
            found = Some(probe);
        }
    }
    found.ok_or_else(|| invalid("target-built runtime archive has no ABI probe"))
}
fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.archive.abi_probe", error.to_string())
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod tests;
