use runmat_types::{CapabilityRequirement, CapabilitySet, EffectKind, EffectSet};
use serde::Serialize;

use crate::{
    BuiltinAsyncBehavior, BuiltinCompatibility, BuiltinEnvironmentEffect, BuiltinPurity,
    BuiltinSemanticKind, BuiltinWorkspaceEffect,
};

mod acceleration;
mod aggregate;
mod array;
mod cells;
mod introspection;
mod io;
mod logical;
mod math;
mod parallel;
mod stats;
mod structs;

pub use acceleration::*;
pub use aggregate::*;
pub use array::*;
pub use cells::*;
pub use introspection::*;
pub use io::*;
pub use logical::*;
pub use math::*;
pub use parallel::*;
pub use stats::*;
pub use structs::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinContractMaturity {
    Complete,
    DynamicByDesign,
    LegacyResolver,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinInferenceRule {
    Acceleration(AccelerationInferenceRule),
    Aggregate(AggregateInferenceRule),
    Array(ArrayInferenceRule),
    Cells(CellInferenceRule),
    Introspection(IntrospectionInferenceRule),
    Io(IoInferenceRule),
    Logical(LogicalInferenceRule),
    Math(MathInferenceRule),
    Parallel(ParallelInferenceRule),
    Stats(StatsInferenceRule),
    Structs(StructInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinContractDeclaration {
    pub maturity: BuiltinContractMaturity,
    pub inference_rule: BuiltinInferenceRule,
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
