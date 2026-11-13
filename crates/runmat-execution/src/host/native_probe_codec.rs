//! Fixed canonical byte encoding; no target process or allocation is needed.
use super::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl NativeAbiProbe {
    pub const fn canonical_bytes(&self) -> [u8; NATIVE_ABI_PROBE_BYTES] {
        let mut bytes = [0; NATIVE_ABI_PROBE_BYTES];
        let mut i = 0;
        while i < 8 {
            bytes[i] = MAGIC[i];
            i += 1;
        }
        put16(&mut bytes, 8, NATIVE_ABI_PROBE_SCHEMA_VERSION);
        put16(&mut bytes, 10, NATIVE_ABI_PROBE_BYTES as u16);
        bytes[12] = match self.architecture {
            NativeArchitecture::X86_64 => 1,
            NativeArchitecture::Aarch64 => 2,
        };
        bytes[13] = match self.operating_system {
            NativeOperatingSystem::Linux => 1,
            NativeOperatingSystem::Macos => 2,
            NativeOperatingSystem::Windows => 3,
        };
        bytes[14] = match self.environment {
            NativeEnvironment::Gnu => 1,
            NativeEnvironment::Musl => 2,
            NativeEnvironment::Darwin => 3,
            NativeEnvironment::Msvc => 4,
        };
        bytes[15] = match self.endianness {
            NativeEndianness::Little => 1,
            NativeEndianness::Big => 2,
        };
        put16(&mut bytes, 16, self.pointer_width);
        put16(&mut bytes, 18, self.abi_schema_version);
        put32(&mut bytes, 20, self.abi_encoded_version);
        put16(&mut bytes, 24, self.contract_length);
        i = 0;
        while i < self.contract.len() {
            bytes[26 + i] = self.contract[i];
            i += 1;
        }
        i = 0;
        let start = 26 + NATIVE_ABI_CONTRACT_CAPACITY;
        while i < self.layouts.len() {
            put32(&mut bytes, start + i * 8, self.layouts[i].size);
            put32(&mut bytes, start + i * 8 + 4, self.layouts[i].alignment);
            i += 1;
        }
        let start = start + self.layouts.len() * 8;
        i = 0;
        while i < self.offsets.len() {
            put32(&mut bytes, start + i * 4, self.offsets[i]);
            i += 1;
        }
        bytes
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ContractError> {
        let invalid =
            || ContractError::invalid("native ABI probe", "invalid canonical probe bytes");
        if bytes.len() != NATIVE_ABI_PROBE_BYTES
            || bytes[..8] != MAGIC
            || get16(bytes, 8) != NATIVE_ABI_PROBE_SCHEMA_VERSION
            || usize::from(get16(bytes, 10)) != bytes.len()
        {
            return Err(invalid());
        }
        let architecture = match bytes[12] {
            1 => NativeArchitecture::X86_64,
            2 => NativeArchitecture::Aarch64,
            _ => return Err(invalid()),
        };
        let operating_system = match bytes[13] {
            1 => NativeOperatingSystem::Linux,
            2 => NativeOperatingSystem::Macos,
            3 => NativeOperatingSystem::Windows,
            _ => return Err(invalid()),
        };
        let environment = match bytes[14] {
            1 => NativeEnvironment::Gnu,
            2 => NativeEnvironment::Musl,
            3 => NativeEnvironment::Darwin,
            4 => NativeEnvironment::Msvc,
            _ => return Err(invalid()),
        };
        let endianness = match bytes[15] {
            1 => NativeEndianness::Little,
            2 => NativeEndianness::Big,
            _ => return Err(invalid()),
        };
        let mut contract = [0; NATIVE_ABI_CONTRACT_CAPACITY];
        contract.copy_from_slice(&bytes[26..26 + NATIVE_ABI_CONTRACT_CAPACITY]);
        let start = 26 + NATIVE_ABI_CONTRACT_CAPACITY;
        let layouts = std::array::from_fn(|i| NativeProbeTypeLayout {
            size: get32(bytes, start + i * 8),
            alignment: get32(bytes, start + i * 8 + 4),
        });
        let start = start + NativeAbiType::ALL.len() * 8;
        let offsets = std::array::from_fn(|i| get32(bytes, start + i * 4));
        let probe = Self {
            architecture,
            operating_system,
            environment,
            endianness,
            pointer_width: get16(bytes, 16),
            abi_schema_version: get16(bytes, 18),
            abi_encoded_version: get32(bytes, 20),
            contract_length: get16(bytes, 24),
            contract,
            layouts,
            offsets,
        };
        probe.validate()?;
        if probe.canonical_bytes() != bytes {
            return Err(invalid());
        }
        Ok(probe)
    }
}
const fn put16(bytes: &mut [u8], offset: usize, value: u16) {
    let value = value.to_le_bytes();
    bytes[offset] = value[0];
    bytes[offset + 1] = value[1];
}
const fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    let value = value.to_le_bytes();
    let mut i = 0;
    while i < 4 {
        bytes[offset + i] = value[i];
        i += 1;
    }
}
fn get16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}
fn get32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

impl Serialize for NativeAbiProbe {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.canonical_bytes().as_slice().serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for NativeAbiProbe {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ProbeVisitor;
        impl<'de> serde::de::Visitor<'de> for ProbeVisitor {
            type Value = NativeAbiProbe;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    formatter,
                    "exactly {NATIVE_ABI_PROBE_BYTES} canonical ABI probe bytes"
                )
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut bytes = [0; NATIVE_ABI_PROBE_BYTES];
                for (i, byte) in bytes.iter_mut().enumerate() {
                    *byte = sequence
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                if sequence.next_element::<u8>()?.is_some() {
                    return Err(serde::de::Error::invalid_length(
                        NATIVE_ABI_PROBE_BYTES + 1,
                        &self,
                    ));
                }
                NativeAbiProbe::from_canonical_bytes(&bytes).map_err(serde::de::Error::custom)
            }
        }
        deserializer.deserialize_seq(ProbeVisitor)
    }
}
