use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{ContractError, Digest};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeArchitecture {
    X86_64,
    Aarch64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatingSystem {
    Macos,
    Linux,
    Windows,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeObjectFormat {
    MachO,
    Elf,
    Coff,
}

impl NativeObjectFormat {
    pub fn token(self) -> &'static str {
        match self {
            Self::MachO => "mach-o",
            Self::Elf => "elf",
            Self::Coff => "coff",
        }
    }

    pub fn from_token(value: &str) -> Result<Self, ContractError> {
        match value {
            "mach-o" => Ok(Self::MachO),
            "elf" => Ok(Self::Elf),
            "coff" => Ok(Self::Coff),
            _ => Err(ContractError::invalid(
                "native object format",
                "unsupported object format",
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct NativeAbi(String);

impl NativeAbi {
    pub fn new(value: impl Into<String>) -> Result<Self, ContractError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 256
            || !value.is_ascii()
            || value.chars().any(char::is_control)
        {
            return Err(ContractError::invalid(
                "native ABI",
                "must be 1..=256 printable ASCII bytes",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for NativeAbi {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for NativeAbi {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTargetIdentity {
    pub architecture: NativeArchitecture,
    pub operating_system: NativeOperatingSystem,
    pub pointer_width: u16,
    pub abi: NativeAbi,
    pub object_format: NativeObjectFormat,
}

impl NativeTargetIdentity {
    pub fn new(
        architecture: NativeArchitecture,
        operating_system: NativeOperatingSystem,
        pointer_width: u16,
        abi: NativeAbi,
        object_format: NativeObjectFormat,
    ) -> Result<Self, ContractError> {
        let target = Self {
            architecture,
            operating_system,
            pointer_width,
            abi,
            object_format,
        };
        target.validate()?;
        Ok(target)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if !matches!(self.pointer_width, 32 | 64) {
            return Err(ContractError::invalid(
                "native target pointer width",
                "must be 32 or 64",
            ));
        }
        let valid_pair = matches!(
            (self.operating_system, self.object_format),
            (NativeOperatingSystem::Macos, NativeObjectFormat::MachO)
                | (NativeOperatingSystem::Linux, NativeObjectFormat::Elf)
                | (NativeOperatingSystem::Windows, NativeObjectFormat::Coff)
        );
        if !valid_pair {
            return Err(ContractError::invalid(
                "native target",
                "operating system and object format are inconsistent",
            ));
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Digest {
        let mut bytes = b"runmat-native-target-v1\0".to_vec();
        bytes.extend_from_slice(&serde_json::to_vec(self).expect("native target is serializable"));
        Digest::sha256(bytes)
    }
}
