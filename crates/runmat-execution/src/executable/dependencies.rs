//! Flat original compiler products. Members own local parts, never nested bundles.
use super::{ExecutableComponentPayload, ExecutableUnitManifest};
use crate::{ContractError, Digest};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::io::Write;

pub const EXECUTABLE_DEPENDENCY_MAX_PRODUCTS: usize = 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableDependencyDescriptor {
    pub identity: Digest,
    pub encoded_length: u64,
}

/// Local immutable parts of one original unit. The enclosing inventory closes
/// foreign references; a member is not an independently executable bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableDependencyProduct {
    pub manifest: ExecutableUnitManifest,
    pub payloads: Vec<ExecutableComponentPayload>,
}

impl ExecutableDependencyProduct {
    pub fn validate(&self) -> Result<(), ContractError> {
        if !self.manifest.dependencies.is_empty() {
            return Err(ContractError::invalid(
                "executable.dependencies",
                "nested inventories are forbidden",
            ));
        }
        self.manifest.validate()?;
        super::envelope::validate_payloads(&self.manifest, &self.payloads)
    }

    pub fn descriptor(&self) -> Result<ExecutableDependencyDescriptor, ContractError> {
        self.validate()?;
        let mut writer = DescriptorWriter {
            digest: Sha256::new(),
            bytes: 0,
        };
        serde_json::to_writer(&mut writer, self).map_err(|error| {
            ContractError::invalid("executable.dependencies", error.to_string())
        })?;
        Ok(ExecutableDependencyDescriptor {
            identity: Digest::from_bytes(writer.digest.finalize().into()),
            encoded_length: writer.bytes,
        })
    }
}

struct DescriptorWriter {
    digest: Sha256,
    bytes: u64,
}
impl Write for DescriptorWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(buffer.len() as u64)
            .filter(|count| *count <= super::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES as u64)
            .ok_or_else(|| std::io::Error::other("dependency product exceeds envelope budget"))?;
        self.digest.update(buffer);
        Ok(buffer.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn validate_descriptors(
    descriptors: &[ExecutableDependencyDescriptor],
) -> Result<(), ContractError> {
    if descriptors.len() > EXECUTABLE_DEPENDENCY_MAX_PRODUCTS {
        return Err(ContractError::Limit {
            field: "executable.dependencies",
            limit: EXECUTABLE_DEPENDENCY_MAX_PRODUCTS as u64,
        });
    }
    if descriptors
        .windows(2)
        .any(|pair| pair[0].identity >= pair[1].identity)
    {
        return Err(ContractError::invalid(
            "executable.dependencies",
            "members must be sorted and unique",
        ));
    }
    let total = descriptors.iter().try_fold(0_u64, |sum, descriptor| {
        if descriptor.encoded_length == 0 {
            return Err(ContractError::invalid(
                "executable.dependencies",
                "empty member",
            ));
        }
        sum.checked_add(descriptor.encoded_length)
            .ok_or(ContractError::Limit {
                field: "executable.dependencies",
                limit: super::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES as u64,
            })
    })?;
    if total > super::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES as u64 {
        return Err(ContractError::Limit {
            field: "executable.dependencies",
            limit: super::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES as u64,
        });
    }
    Ok(())
}

pub(super) fn validate_products(
    descriptors: &[ExecutableDependencyDescriptor],
    products: &[ExecutableDependencyProduct],
) -> Result<(), ContractError> {
    if descriptors.len() != products.len() {
        return Err(ContractError::invalid(
            "executable.dependencies",
            "inventory is incomplete",
        ));
    }
    for (descriptor, product) in descriptors.iter().zip(products) {
        if &product.descriptor()? != descriptor {
            return Err(ContractError::invalid(
                "executable.dependencies",
                "member differs from declared original identity",
            ));
        }
    }
    Ok(())
}
