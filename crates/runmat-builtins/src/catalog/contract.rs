use runmat_types::{CapabilityRequirement, CapabilitySet, EffectKind, EffectSet};
use serde::Serialize;

use crate::{
    BuiltinAsyncBehavior, BuiltinCompatibility, BuiltinEnvironmentEffect, BuiltinPurity,
    BuiltinSemanticKind, BuiltinWorkspaceEffect,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinContractMaturity {
    Complete,
    DynamicByDesign,
    LegacyResolver,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinInferenceRuleId(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinContractDeclaration {
    pub maturity: BuiltinContractMaturity,
    pub inference_rule: BuiltinInferenceRuleId,
    pub compatibility: BuiltinCompatibility,
    pub async_behavior: BuiltinAsyncBehavior,
    pub purity: BuiltinPurity,
    pub semantic_kind: BuiltinSemanticKind,
    pub workspace_effect: Option<BuiltinWorkspaceEffect>,
    pub environment_effect: Option<BuiltinEnvironmentEffect>,
    pub effects: &'static [EffectKind],
    pub capabilities: &'static [CapabilityRequirement],
}

impl BuiltinContractDeclaration {
    pub fn effect_set(self) -> EffectSet {
        EffectSet(self.effects.iter().copied().collect())
    }

    pub fn capability_set(self) -> CapabilitySet {
        CapabilitySet(self.capabilities.iter().copied().collect())
    }
}
