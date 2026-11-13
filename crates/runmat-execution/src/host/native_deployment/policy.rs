use crate::ContractError;
use runmat_types::EffectKind;
use serde::{Deserialize, Serialize};

/// Exact Rust-linked optional source compiler/service ABI. Target packs must
/// bind this ABI to the actual retained implementation archive; a label or
/// product manifest alone never certifies the archive bytes.
pub fn native_source_evaluation_service_abi() -> crate::Digest {
    crate::Digest::sha256(b"runmat-native-source-evaluation-factory-v2-runtime-ptr-owner-ptr")
}

/// One exact linked Rust factory constructs the selected invocation-local
/// state; individual EffectKind issuers still bind the distinct permissions.
pub fn native_local_service_abi() -> crate::Digest {
    crate::Digest::sha256(b"runmat-native-local-service-v1-product-invocation-issuers-inputs-owner")
}

/// Linked native HTTP/TCP owner, constructed from this invocation's reactor.
pub fn native_network_service_abi() -> crate::Digest {
    crate::Digest::sha256(b"runmat-native-network-service-v1-reactor-http-tcp-owner")
}

pub fn native_ui_service_abi() -> crate::Digest {
    crate::Digest::sha256(
        b"runmat-native-ui-v2-linked-console-optional-interaction-graphics-invocation",
    )
}

pub fn native_host_callback_service_abi() -> crate::Digest {
    crate::Digest::sha256(b"runmat-native-host-callback-v2-linked-call-router-invocation")
}

pub fn native_hosted_dynamic_dispatch_abi() -> crate::Digest {
    crate::Digest::sha256(b"runmat-native-hosted-dynamic-v1-scoped-builtin-revision-invocation")
}

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
    SourceEvaluation,
}

impl NativeDeploymentServiceKind {
    pub fn is_local_effect_service(self) -> bool {
        match self {
            Self::Effect(
                EffectKind::WorkspaceRead
                | EffectKind::WorkspaceWrite
                | EffectKind::EnvironmentRead
                | EffectKind::EnvironmentWrite
                | EffectKind::FilesystemRead
                | EffectKind::FilesystemWrite
                | EffectKind::Randomness,
            ) => true,
            Self::Effect(
                EffectKind::Network
                | EffectKind::UserInterface
                | EffectKind::Clock
                | EffectKind::HostCallback
                | EffectKind::MaySuspend
                | EffectKind::MayThrow
                | EffectKind::Unknown,
            )
            | Self::HostedDynamicDispatch
            | Self::SourceEvaluation => false,
        }
    }

    pub fn requires_local_input_payload(self) -> bool {
        match self {
            Self::Effect(
                EffectKind::EnvironmentRead
                | EffectKind::EnvironmentWrite
                | EffectKind::FilesystemRead
                | EffectKind::FilesystemWrite
                | EffectKind::Randomness,
            ) => true,
            Self::Effect(
                EffectKind::WorkspaceRead
                | EffectKind::WorkspaceWrite
                | EffectKind::Network
                | EffectKind::UserInterface
                | EffectKind::Clock
                | EffectKind::HostCallback
                | EffectKind::MaySuspend
                | EffectKind::MayThrow
                | EffectKind::Unknown,
            )
            | Self::HostedDynamicDispatch
            | Self::SourceEvaluation => false,
        }
    }

    pub fn for_effect(effect: EffectKind) -> Option<Self> {
        match effect {
            // Execution control is implemented by the runtime driver, not an
            // optional external service. Keep these effects in the program's
            // effect set without inventing a service grant for them.
            EffectKind::MayThrow | EffectKind::MaySuspend => None,
            EffectKind::Unknown => Some(Self::HostedDynamicDispatch),
            effect => Some(Self::Effect(effect)),
        }
    }
    pub fn validate(self) -> Result<(), ContractError> {
        if matches!(
            self,
            Self::Effect(EffectKind::Unknown | EffectKind::MayThrow | EffectKind::MaySuspend)
        ) {
            return Err(ContractError::invalid(
                "native deployment service",
                "unknown/control effects are not concrete service identities",
            ));
        }
        Ok(())
    }
}
