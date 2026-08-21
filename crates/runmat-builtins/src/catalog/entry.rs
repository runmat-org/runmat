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

const NATIVE_BINDING_SYMBOL_PREFIX: &str = "runmat_builtin_binding_v1_";

/// Stable native symbol exported by the runtime implementation for one
/// catalog binding. Encoding both identity components byte-for-byte avoids
/// punctuation folding and the collisions it can introduce.
pub fn native_binding_symbol(name: &str, variant: &str) -> String {
    fn append_hex(output: &mut String, value: &str) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in value.bytes() {
            output.push(HEX[usize::from(byte >> 4)] as char);
            output.push(HEX[usize::from(byte & 0x0f)] as char);
        }
    }

    let mut symbol = String::with_capacity(
        NATIVE_BINDING_SYMBOL_PREFIX.len() + (name.len() + variant.len()) * 2 + 1,
    );
    symbol.push_str(NATIVE_BINDING_SYMBOL_PREFIX);
    append_hex(&mut symbol, name);
    symbol.push('_');
    append_hex(&mut symbol, variant);
    symbol
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

#[cfg(test)]
mod tests {
    use super::native_binding_symbol;

    #[test]
    fn native_binding_symbols_preserve_the_complete_identity() {
        assert_eq!(
            native_binding_symbol("DataArray.read", "default"),
            "runmat_builtin_binding_v1_4461746141727261792e72656164_64656661756c74"
        );
        assert_ne!(
            native_binding_symbol("a.b", "c"),
            native_binding_symbol("a", "b.c")
        );
    }
}
