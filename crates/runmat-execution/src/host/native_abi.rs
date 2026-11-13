//! Portable ABI evidence. Runtime/codegen produce and authenticate target
//! probes; structural decoding never substitutes the current host's layout.
use super::NativeAbi;
use crate::{ContractError, Digest};
use serde::{Deserialize, Serialize};

// Wire-compatible extraction of native-codegen's existing ABI record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeAbiBinding {
    pub schema_version: u16,
    pub encoded_version: u32,
    pub contract_fingerprint: Digest,
    pub layout_fingerprint: Digest,
}

impl NativeAbiBinding {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version == 0
            || self.encoded_version == 0
            || self
                .contract_fingerprint
                .bytes()
                .iter()
                .all(|byte| *byte == 0)
            || self
                .layout_fingerprint
                .bytes()
                .iter()
                .all(|byte| *byte == 0)
        {
            return Err(ContractError::invalid(
                "native ABI binding",
                "missing version or target-probed fingerprint",
            ));
        }
        Ok(())
    }

    pub fn require_exact(&self, expected: &Self) -> Result<(), ContractError> {
        self.validate()?;
        expected.validate()?;
        if self != expected {
            return Err(ContractError::invalid(
                "native ABI binding",
                "does not match the selected target probe",
            ));
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<NativeAbi, ContractError> {
        self.validate()?;
        NativeAbi::new(format!(
            "{}:{}:{}:{}",
            self.schema_version,
            self.encoded_version,
            self.contract_fingerprint,
            self.layout_fingerprint
        ))
    }
}
