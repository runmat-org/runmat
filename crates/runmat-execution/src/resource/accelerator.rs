use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::identity::AttemptId;
use crate::{ContractError, Digest};

macro_rules! token_type {
    ($name:ident, $field:literal, $max:expr) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, ContractError> {
                let value = value.into();
                validate_token($field, &value, $max)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl TryFrom<String> for $name {
            type Error = ContractError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::try_from(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

token_type!(AcceleratorClass, "accelerator class", 96);
token_type!(AcceleratorProviderId, "accelerator provider", 96);
token_type!(
    AcceleratorProviderVersion,
    "accelerator provider version",
    128
);
token_type!(AcceleratorDeviceId, "accelerator device identity", 256);
token_type!(
    AcceleratorAllocationDomainId,
    "accelerator allocation domain identity",
    256
);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorProvider {
    pub id: AcceleratorProviderId,
    pub version: AcceleratorProviderVersion,
    pub abi_fingerprint: Digest,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorFeature {
    Compute,
    UnifiedMemory,
    PeerToPeer,
    ConfidentialCompute,
    Custom(String),
}

impl AcceleratorFeature {
    fn validate(&self) -> Result<(), ContractError> {
        if let Self::Custom(value) = self {
            validate_token("custom accelerator feature", value, 128)?;
        }
        Ok(())
    }

    fn append_identity(&self, identity: &mut Vec<u8>) {
        match self {
            Self::Compute => identity.push(0),
            Self::UnifiedMemory => identity.push(1),
            Self::PeerToPeer => identity.push(2),
            Self::ConfidentialCompute => identity.push(3),
            Self::Custom(value) => {
                identity.push(4);
                append_bytes(identity, value.as_bytes());
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorDevice {
    /// Provider-view identity used for exact runtime authorization.
    pub id: AcceleratorDeviceId,
    /// Physical allocation domain. Distinct provider views of the same device
    /// share this identity and therefore cannot be leased concurrently.
    pub allocation_domain: AcceleratorAllocationDomainId,
    pub class: AcceleratorClass,
    pub provider: AcceleratorProvider,
    pub max_allocation_bytes: u64,
    pub features: std::collections::BTreeSet<AcceleratorFeature>,
    pub inventory_epoch: u64,
}

impl AcceleratorDevice {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.max_allocation_bytes == 0 || self.inventory_epoch == 0 {
            return Err(ContractError::invalid(
                "accelerator device",
                "memory and inventory epoch must be non-zero",
            ));
        }
        for feature in &self.features {
            feature.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorRequest {
    pub class: AcceleratorClass,
    pub count: u16,
    pub minimum_allocation_bytes: u64,
    #[serde(default)]
    pub provider: Option<AcceleratorProvider>,
    #[serde(default)]
    pub required_features: std::collections::BTreeSet<AcceleratorFeature>,
}

impl AcceleratorRequest {
    /// Minimum portable compute-device contract emitted for source that
    /// semantically requires accelerator residency without naming a provider.
    pub fn generic_compute(count: u16) -> Result<Self, ContractError> {
        let request = Self {
            class: AcceleratorClass::new("gpu")?,
            count,
            minimum_allocation_bytes: 0,
            provider: None,
            required_features: std::collections::BTreeSet::from([AcceleratorFeature::Compute]),
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.count == 0 {
            return Err(ContractError::invalid(
                "accelerator request",
                "count must be non-zero",
            ));
        }
        for feature in &self.required_features {
            feature.validate()?;
        }
        Ok(())
    }

    pub fn matches(&self, device: &AcceleratorDevice) -> bool {
        self.class == device.class
            && self.minimum_allocation_bytes <= device.max_allocation_bytes
            && self
                .provider
                .as_ref()
                .is_none_or(|provider| provider == &device.provider)
            && self.required_features.is_subset(&device.features)
    }

    fn same_constraint(&self, other: &Self) -> bool {
        self.class == other.class
            && self.minimum_allocation_bytes == other.minimum_allocation_bytes
            && self.provider == other.provider
            && self.required_features == other.required_features
    }

    /// Every device satisfying `self` also satisfies `other`.
    fn implies(&self, other: &Self) -> bool {
        self.class == other.class
            && self.minimum_allocation_bytes >= other.minimum_allocation_bytes
            && match (&self.provider, &other.provider) {
                (_, None) => true,
                (Some(left), Some(right)) => left == right,
                (None, Some(_)) => false,
            }
            && self.required_features.is_superset(&other.required_features)
    }
}

/// Canonically composes independently declared minimum requirements.
///
/// A stricter slot may satisfy one weaker slot of the same class. This keeps a
/// source-level generic GPU requirement from allocating a second device when
/// a packaged component already requires an exact CUDA provider. Incomparable
/// constraints remain separate: collapsing them would invent a requirement
/// that neither declaration made and could reject otherwise valid hosts.
pub fn merge_accelerator_requirements(
    requirements: impl IntoIterator<Item = AcceleratorRequest>,
) -> Result<Vec<AcceleratorRequest>, ContractError> {
    let mut merged: Vec<AcceleratorRequest> = Vec::new();
    for requirement in requirements {
        requirement.validate()?;
        if let Some(existing) = merged
            .iter_mut()
            .find(|existing| existing.same_constraint(&requirement))
        {
            existing.count = existing.count.max(requirement.count);
        } else {
            merged.push(requirement);
        }
    }
    let snapshot = merged.clone();
    for weaker_index in 0..merged.len() {
        let covered = snapshot
            .iter()
            .enumerate()
            .filter(|(stronger_index, stronger)| {
                *stronger_index != weaker_index
                    && !stronger.same_constraint(&snapshot[weaker_index])
                    && stronger.implies(&snapshot[weaker_index])
            })
            .map(|(_, stronger)| stronger.count)
            .fold(0_u16, u16::saturating_add);
        merged[weaker_index].count = merged[weaker_index].count.saturating_sub(covered);
    }
    merged.retain(|requirement| requirement.count > 0);
    merged.sort();
    let count = merged.iter().try_fold(0usize, |total, requirement| {
        total.checked_add(usize::from(requirement.count))
    });
    if count.is_none_or(|count| count > 16) {
        return Err(ContractError::Limit {
            field: "accelerators",
            limit: 16,
        });
    }
    Ok(merged)
}

/// Converts whole-program semantic capabilities into portable device leases.
/// Provider-specific requirements are merged separately from package and
/// foreign-artifact contracts.
pub fn accelerator_requirements_for_capabilities(
    capabilities: &runmat_types::CapabilitySet,
) -> Result<Vec<AcceleratorRequest>, ContractError> {
    if capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::Accelerator)
    {
        Ok(vec![AcceleratorRequest::generic_compute(1)?])
    } else {
        Ok(Vec::new())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorDeviceLease {
    pub device_id: AcceleratorDeviceId,
    pub allocation_domain: AcceleratorAllocationDomainId,
    pub class: AcceleratorClass,
    pub provider: AcceleratorProvider,
    pub max_allocation_bytes: u64,
    pub features: std::collections::BTreeSet<AcceleratorFeature>,
    pub inventory_epoch: u64,
    pub fencing_token: u64,
    pub lease_id: Digest,
}

impl AcceleratorDeviceLease {
    pub fn for_attempt(
        attempt_id: AttemptId,
        fencing_token: u64,
        device: &AcceleratorDevice,
    ) -> Result<Self, ContractError> {
        device.validate()?;
        let mut lease = Self {
            device_id: device.id.clone(),
            allocation_domain: device.allocation_domain.clone(),
            class: device.class.clone(),
            provider: device.provider.clone(),
            max_allocation_bytes: device.max_allocation_bytes,
            features: device.features.clone(),
            inventory_epoch: device.inventory_epoch,
            fencing_token,
            lease_id: Digest::sha256([]),
        };
        lease.lease_id = lease.expected_id(attempt_id);
        lease.validate_for_attempt(attempt_id)?;
        Ok(lease)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.max_allocation_bytes == 0 || self.inventory_epoch == 0 || self.fencing_token == 0 {
            return Err(ContractError::invalid(
                "accelerator device lease",
                "inventory epoch and fencing token must be non-zero",
            ));
        }
        for feature in &self.features {
            feature.validate()?;
        }
        Ok(())
    }

    pub fn validate_for_attempt(&self, attempt_id: AttemptId) -> Result<(), ContractError> {
        self.validate()?;
        if self.lease_id != self.expected_id(attempt_id) {
            return Err(ContractError::invalid(
                "accelerator device lease",
                "identity does not match its attempt, device, epoch, and fence",
            ));
        }
        Ok(())
    }

    /// Returns whether this lease names the exact cluster-facing device
    /// contract advertised by a process-local provider. Attempt and fence
    /// authority is validated separately by [`Self::validate_for_attempt`].
    pub fn matches_device(&self, device: &AcceleratorDevice) -> bool {
        self.device_id == device.id
            && self.allocation_domain == device.allocation_domain
            && self.class == device.class
            && self.provider == device.provider
            && self.max_allocation_bytes == device.max_allocation_bytes
            && self.features == device.features
            && self.inventory_epoch == device.inventory_epoch
    }

    fn expected_id(&self, attempt_id: AttemptId) -> Digest {
        let mut identity = b"runmat-accelerator-lease-v3\0".to_vec();
        identity.extend_from_slice(attempt_id.bytes().as_slice());
        append_bytes(&mut identity, self.device_id.as_str().as_bytes());
        append_bytes(&mut identity, self.allocation_domain.as_str().as_bytes());
        append_bytes(&mut identity, self.class.as_str().as_bytes());
        append_bytes(&mut identity, self.provider.id.as_str().as_bytes());
        append_bytes(&mut identity, self.provider.version.as_str().as_bytes());
        identity.extend_from_slice(self.provider.abi_fingerprint.bytes());
        identity.extend_from_slice(&self.max_allocation_bytes.to_be_bytes());
        identity.extend_from_slice(&(self.features.len() as u64).to_be_bytes());
        for feature in &self.features {
            feature.append_identity(&mut identity);
        }
        identity.extend_from_slice(&self.inventory_epoch.to_be_bytes());
        identity.extend_from_slice(&self.fencing_token.to_be_bytes());
        Digest::sha256(identity)
    }
}

fn append_bytes(identity: &mut Vec<u8>, value: &[u8]) {
    identity.extend_from_slice(&(value.len() as u64).to_be_bytes());
    identity.extend_from_slice(value);
}

fn validate_token(field: &'static str, value: &str, max_bytes: usize) -> Result<(), ContractError> {
    if value.is_empty()
        || value.len() > max_bytes
        || !value.is_ascii()
        || value.chars().any(char::is_control)
    {
        return Err(ContractError::invalid(
            field,
            format!("must be 1..={max_bytes} printable ASCII bytes"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_provider(id: &str) -> AcceleratorProvider {
        AcceleratorProvider {
            id: AcceleratorProviderId::new(id).unwrap(),
            version: AcceleratorProviderVersion::new("1").unwrap(),
            abi_fingerprint: Digest::sha256(id.as_bytes()),
        }
    }

    #[test]
    fn exact_provider_requirement_subsumes_one_generic_slot() {
        let generic = AcceleratorRequest::generic_compute(1).unwrap();
        let mut cuda = AcceleratorRequest::generic_compute(1).unwrap();
        cuda.provider = Some(exact_provider("runmat.cuda"));

        let merged = merge_accelerator_requirements([generic, cuda.clone()]).unwrap();
        assert_eq!(merged, vec![cuda]);
    }

    #[test]
    fn stronger_slots_cover_only_the_required_generic_count() {
        let generic = AcceleratorRequest::generic_compute(2).unwrap();
        let mut cuda = AcceleratorRequest::generic_compute(1).unwrap();
        cuda.provider = Some(exact_provider("runmat.cuda"));

        let merged = merge_accelerator_requirements([generic, cuda.clone()]).unwrap();
        assert_eq!(merged.len(), 2);
        assert!(merged.contains(&cuda));
        assert!(merged.iter().any(|request| {
            request.provider.is_none()
                && request.count == 1
                && request
                    .required_features
                    .contains(&AcceleratorFeature::Compute)
        }));
    }

    #[test]
    fn incomparable_provider_requirements_remain_distinct() {
        let mut cuda = AcceleratorRequest::generic_compute(1).unwrap();
        cuda.provider = Some(exact_provider("runmat.cuda"));
        let mut wgpu = AcceleratorRequest::generic_compute(1).unwrap();
        wgpu.provider = Some(exact_provider("runmat.wgpu"));

        let merged = merge_accelerator_requirements([cuda.clone(), wgpu.clone()]).unwrap();
        assert_eq!(merged.len(), 2);
        assert!(merged.contains(&cuda));
        assert!(merged.contains(&wgpu));
    }
}
