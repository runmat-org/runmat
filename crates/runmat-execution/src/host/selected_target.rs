//! Selected platform facts. ISA construction and actual ABI probing remain
//! backend responsibilities; this record never reads build-host facts.
use super::{NativeArchitecture, NativeOperatingSystem, NativeTargetIdentity};
use crate::{ContractError, Digest};
use serde::{Deserialize, Serialize};

pub const SELECTED_NATIVE_TARGET_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeCpuBaseline {
    X86_64V1,
    ArmV8A,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeEndianness {
    Little,
    Big,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeEnvironment {
    Gnu,
    Musl,
    Darwin,
    Msvc,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedNativeTarget {
    pub schema_version: u16,
    pub target: NativeTargetIdentity,
    pub cpu_baseline: NativeCpuBaseline,
    pub endianness: NativeEndianness,
    pub environment: NativeEnvironment,
}

impl SelectedNativeTarget {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.target.validate()?;
        if self.schema_version != SELECTED_NATIVE_TARGET_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: SELECTED_NATIVE_TARGET_SCHEMA_VERSION,
            });
        }
        let baseline = matches!(
            (self.target.architecture, self.cpu_baseline),
            (NativeArchitecture::X86_64, NativeCpuBaseline::X86_64V1)
                | (NativeArchitecture::Aarch64, NativeCpuBaseline::ArmV8A)
        );
        let platform = matches!(
            (self.target.operating_system, self.environment),
            (
                NativeOperatingSystem::Linux,
                NativeEnvironment::Gnu | NativeEnvironment::Musl
            ) | (NativeOperatingSystem::Macos, NativeEnvironment::Darwin)
                | (NativeOperatingSystem::Windows, NativeEnvironment::Msvc)
        );
        // These are admission limits for the first supported ISA baselines,
        // not a derivation of an ABI layout from the architecture.
        if !baseline
            || !platform
            || self.target.pointer_width != 64
            || self.endianness != NativeEndianness::Little
        {
            return Err(ContractError::invalid(
                "selected native target",
                "unsupported or inconsistent baseline/platform/explicit word layout",
            ));
        }
        Ok(())
    }

    pub fn triple(&self) -> Result<String, ContractError> {
        self.validate()?;
        let architecture = match self.target.architecture {
            NativeArchitecture::X86_64 => "x86_64",
            NativeArchitecture::Aarch64 => "aarch64",
        };
        let platform = match self.environment {
            NativeEnvironment::Gnu => "unknown-linux-gnu",
            NativeEnvironment::Musl => "unknown-linux-musl",
            NativeEnvironment::Darwin => "apple-darwin",
            NativeEnvironment::Msvc => "pc-windows-msvc",
        };
        Ok(format!("{architecture}-{platform}"))
    }

    pub fn fingerprint(&self) -> Result<Digest, ContractError> {
        self.validate()?;
        Ok(
            Digest::sha256_serialized(&("runmat-selected-native-target-v1", self))
                .expect("selected native target serializes"),
        )
    }
}
