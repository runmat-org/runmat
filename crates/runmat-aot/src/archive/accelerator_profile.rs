//! Authenticate target-built accelerator feature bytes in the runtime archive.

use object::{Object, ObjectSection, ObjectSymbol};
use runmat_execution::host::{
    NativeAcceleratorProfile, NATIVE_ACCELERATOR_PROFILE_BYTES, NATIVE_ACCELERATOR_PROFILE_SYMBOL,
};

use crate::{AotError, AotResult};

pub fn read_native_accelerator_profile(archive: &[u8]) -> AotResult<NativeAcceleratorProfile> {
    if archive.len() as u64 > super::MAX_RUNTIME_ARCHIVE_BYTES {
        return Err(invalid("archive exceeds its bound"));
    }
    let probe = super::read_native_abi_probe(archive)?;
    let parsed = object::read::archive::ArchiveFile::parse(archive).map_err(invalid)?;
    let mut found = None;
    for (ordinal, member) in parsed.members().enumerate() {
        if ordinal >= 65536 {
            return Err(invalid("archive member count exceeds its bound"));
        }
        let member = member.map_err(invalid)?;
        let data = member.data(archive).map_err(invalid)?;
        let Ok(file) = object::File::parse(data) else {
            continue;
        };
        for symbol in file.symbols() {
            if symbol.is_undefined() {
                continue;
            }
            let name = symbol.name().map_err(invalid)?;
            let matches = if file.format() == object::BinaryFormat::MachO {
                name.strip_prefix('_') == Some(NATIVE_ACCELERATOR_PROFILE_SYMBOL)
            } else {
                name == NATIVE_ACCELERATOR_PROFILE_SYMBOL
            };
            if !matches {
                continue;
            }
            if found.is_some() || !super::member_matches_probe_target(&file, &probe) {
                return Err(invalid(
                    "duplicate acceleration profile or incompatible target object",
                ));
            }
            if symbol.size() != 0 && symbol.size() != NATIVE_ACCELERATOR_PROFILE_BYTES as u64 {
                return Err(invalid("accelerator profile has an unexpected symbol size"));
            }
            let section = file
                .section_by_index(
                    symbol
                        .section_index()
                        .ok_or_else(|| invalid("accelerator profile has no section"))?,
                )
                .map_err(invalid)?;
            let offset = symbol
                .address()
                .checked_sub(section.address())
                .ok_or_else(|| invalid("accelerator profile precedes its section"))?;
            let end = offset
                .checked_add(NATIVE_ACCELERATOR_PROFILE_BYTES as u64)
                .ok_or_else(|| invalid("accelerator profile extent overflows"))?;
            if section.relocations().any(|(position, relocation)| {
                if relocation.size() == 0 {
                    return true;
                }
                let width = u64::from(relocation.size()).div_ceil(8).max(1);
                position < end
                    && position
                        .checked_add(width)
                        .is_none_or(|relocation_end| relocation_end > offset)
            }) {
                return Err(invalid("accelerator profile contains target relocations"));
            }
            let bytes = section
                .data_range(symbol.address(), NATIVE_ACCELERATOR_PROFILE_BYTES as u64)
                .map_err(invalid)?
                .ok_or_else(|| invalid("accelerator profile extends past its section"))?;
            found = Some(NativeAcceleratorProfile::from_canonical_bytes(bytes).map_err(invalid)?);
        }
    }
    Ok(found.unwrap_or(NativeAcceleratorProfile::NONE))
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.archive.accelerator_profile", error.to_string())
}
