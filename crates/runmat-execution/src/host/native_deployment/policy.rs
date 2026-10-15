use crate::ContractError;
use runmat_types::EffectKind;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeDeploymentProfile {
    HostedNative,
    DeterministicDeployed,
}

/// A declaration about implementation behavior, never qualification evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeDeploymentServiceBehavior {
    HostedAmbient,
    DeterministicDeclared,
}

/// Hosted dynamic dispatch is a distinct required service, not an effect
/// wildcard. Concrete service keys preserve the canonical language effect ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", content = "effect", rename_all = "snake_case")]
pub enum NativeDeploymentServiceKind {
    Effect(EffectKind),
    HostedDynamicDispatch,
}

impl NativeDeploymentServiceKind {
    pub fn for_effect(effect: EffectKind) -> Option<Self> {
        match effect {
            EffectKind::MayThrow => None,
            EffectKind::Unknown => Some(Self::HostedDynamicDispatch),
            effect => Some(Self::Effect(effect)),
        }
    }
    pub fn validate(self) -> Result<(), ContractError> {
        if matches!(
            self,
            Self::Effect(EffectKind::Unknown | EffectKind::MayThrow)
        ) {
            return Err(ContractError::invalid(
                "native deployment service",
                "unknown/control effects are not concrete service identities",
            ));
        }
        Ok(())
    }
}
