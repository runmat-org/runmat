use super::{
    NativeDeploymentProfile, NativeDeploymentServiceBehavior, NativeDeploymentServiceKind,
};
use crate::host::{
    NativeAbiBinding, NativeNumericalPolicy, NativeOptimization, SelectedNativeTarget,
};
use crate::{ContractError, Digest};
use runmat_types::{CapabilityRequirement, CapabilitySet, EffectKind, EffectSet};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION: u16 = 2;

/// Opaque portable carrier. The artifact codec alone computes the pack digest
/// from its canonical manifest. Constructing/decoding this ID authenticates no
/// bytes and grants no authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TargetPackId(Digest);
impl TargetPackId {
    pub fn from_digest(digest: Digest) -> Result<Self, ContractError> {
        nonzero(digest)?;
        Ok(Self(digest))
    }
    pub fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSelectedServiceBinding {
    pub kind: NativeDeploymentServiceKind,
    pub abi: Digest,
    /// Exact retained implementation bytes; labels never confer identity.
    pub implementation: Digest,
    pub behavior: NativeDeploymentServiceBehavior,
}

/// Exact selected build/runtime declaration, shared unchanged by recipe,
/// object, runtime archive, launcher and link plan. It is NOT a verified-byte
/// token or a qualification of the actual installed runtime provider.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSelectedProductBinding {
    pub schema_version: u16,
    pub pack: TargetPackId,
    pub selection: Digest,
    pub target: SelectedNativeTarget,
    pub abi: NativeAbiBinding,
    pub runtime: Digest,
    pub catalog: Digest,
    pub runtime_component: Digest,
    pub platform_service_abi: Digest,
    pub numerical: NativeNumericalPolicy,
    pub optimization: NativeOptimization,
    pub profile: NativeDeploymentProfile,
    /// Actual requirements of the selected authored program.
    pub reached_capabilities: CapabilitySet,
    /// Maximum runtime capability inventory provided by the selected pack.
    pub available_runtime_capabilities: CapabilitySet,
    /// Independent trusted caller envelope; inventory is never an implicit grant.
    pub authorized_runtime_capabilities: CapabilitySet,
    pub reached_effects: EffectSet,
    /// Non-effect runtime services demanded by canonical Native IR operations.
    pub required_services: BTreeSet<NativeDeploymentServiceKind>,
    pub services: Vec<NativeSelectedServiceBinding>,
}

impl NativeSelectedProductBinding {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION,
            });
        }
        self.target.validate()?;
        self.abi.validate()?;
        self.numerical.validate()?;
        for digest in [
            self.pack.digest(),
            self.selection,
            self.runtime,
            self.catalog,
            self.runtime_component,
            self.platform_service_abi,
        ] {
            nonzero(digest)?;
        }
        if self.target.target.abi != self.abi.identity()?
            || !self
                .reached_capabilities
                .0
                .is_subset(&self.authorized_runtime_capabilities.0)
            || !self
                .reached_capabilities
                .0
                .is_subset(&self.available_runtime_capabilities.0)
            || (self.profile == NativeDeploymentProfile::DeterministicDeployed
                && self.reached_effects.0.contains(&EffectKind::Unknown))
            || self
                .services
                .windows(2)
                .any(|pair| pair[0].kind >= pair[1].kind)
        {
            return Err(invalid(
                "inconsistent target, grant, profile or service inventory",
            ));
        }
        let mut needed = self
            .reached_effects
            .0
            .iter()
            .copied()
            .filter_map(NativeDeploymentServiceKind::for_effect)
            .collect::<BTreeSet<_>>();
        for kind in &self.required_services {
            kind.validate()?;
            if *kind == NativeDeploymentServiceKind::SourceEvaluation
                && ![
                    &self.reached_capabilities,
                    &self.available_runtime_capabilities,
                    &self.authorized_runtime_capabilities,
                ]
                .into_iter()
                .all(|capabilities| capabilities.0.contains(&CapabilityRequirement::HostRuntime))
            {
                return Err(invalid(
                    "source evaluation lacks an exact host-runtime grant",
                ));
            }
            needed.insert(*kind);
        }
        if !self.services.iter().map(|service| service.kind).eq(needed) {
            return Err(invalid(
                "service bindings do not match the exact reached effect closure",
            ));
        }
        for service in &self.services {
            service.kind.validate()?;
            nonzero(service.abi)?;
            nonzero(service.implementation)?;
            if self.profile == NativeDeploymentProfile::DeterministicDeployed
                && service.behavior != NativeDeploymentServiceBehavior::DeterministicDeclared
            {
                return Err(invalid(
                    "deterministic selection contains an ambient service declaration",
                ));
            }
        }
        Ok(())
    }
    pub fn fingerprint(&self) -> Result<Digest, ContractError> {
        self.validate()?;
        Ok(
            Digest::sha256_serialized(&("runmat-selected-native-product-v2", self))
                .expect("selected native product serializes"),
        )
    }
}

/// Mandatory tagged mode: absent fields must never silently downgrade a
/// selected product to a legacy host route or manufacture a runtime grant.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeProductMode {
    HostUnselected,
    Selected {
        binding: Box<NativeSelectedProductBinding>,
    },
}
impl NativeProductMode {
    pub fn validate(&self) -> Result<(), ContractError> {
        match self {
            Self::HostUnselected => Ok(()),
            Self::Selected { binding } => binding.validate(),
        }
    }
    pub fn selected(&self) -> Option<&NativeSelectedProductBinding> {
        match self {
            Self::HostUnselected => None,
            Self::Selected { binding } => Some(binding),
        }
    }
}
fn nonzero(digest: Digest) -> Result<(), ContractError> {
    if digest.bytes().iter().all(|byte| *byte == 0) {
        return Err(invalid("missing identity"));
    }
    Ok(())
}
fn invalid(message: &str) -> ContractError {
    ContractError::invalid("selected native product", message)
}
