use super::{
    BuiltinContractDeclaration, BuiltinDescriptor, BuiltinDocumentation,
    BuiltinExtensionDescriptor, BuiltinIntegerAuditDescriptor, BuiltinIntegerCapabilityDescriptor,
    BuiltinLinkContract, BuiltinPlacementContract,
};
use runmat_types::EffectKind;
use serde::Serialize;

use crate::{BuiltinEffects, BuiltinSemantics};

pub const BUILTIN_CATALOG_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct BuiltinCatalogIdentity {
    pub name: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct BuiltinBindingIdentity {
    pub builtin: BuiltinCatalogIdentity,
    pub variant: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinBindingAvailability {
    Required,
    TargetConditional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinBindingDeclaration {
    pub identity: BuiltinBindingIdentity,
    pub availability: BuiltinBindingAvailability,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BuiltinCatalogEntry {
    pub identity: BuiltinCatalogIdentity,
    pub category: &'static str,
    pub documentation: BuiltinDocumentation,
    pub descriptor: &'static BuiltinDescriptor,
    pub contract: BuiltinContractDeclaration,
    pub placement: BuiltinPlacementContract,
    pub link: BuiltinLinkContract,
    pub bindings: &'static [BuiltinBindingDeclaration],
    pub extensions: &'static [BuiltinExtensionDescriptor],
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
    pub suppress_auto_output: bool,
}

impl BuiltinCatalogEntry {
    pub fn legacy_semantics(&self) -> BuiltinSemantics {
        let effects = self.contract.effect_set();
        BuiltinSemantics {
            compatibility: self.contract.compatibility,
            async_behavior: self.contract.async_behavior,
            effects: BuiltinEffects {
                workspace: effects.0.contains(&EffectKind::WorkspaceRead)
                    || effects.0.contains(&EffectKind::WorkspaceWrite),
                environment: effects.0.contains(&EffectKind::EnvironmentRead)
                    || effects.0.contains(&EffectKind::EnvironmentWrite),
                filesystem: effects.0.contains(&EffectKind::FilesystemRead)
                    || effects.0.contains(&EffectKind::FilesystemWrite),
                network: effects.0.contains(&EffectKind::Network),
                ui: effects.0.contains(&EffectKind::UserInterface),
                random: effects.0.contains(&EffectKind::Randomness),
                time: effects.0.contains(&EffectKind::Clock),
                host_callback: effects.0.contains(&EffectKind::HostCallback),
                unknown: effects.0.contains(&EffectKind::Unknown),
            },
            workspace_effect: self.contract.workspace_effect,
            environment_effect: self.contract.environment_effect,
            purity: self.contract.purity,
            semantic_kind: self.contract.semantic_kind,
        }
    }
}
