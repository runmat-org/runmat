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
pub enum BuiltinInferenceRule {
    Acceleration(AccelerationInferenceRule),
    Aggregate(AggregateInferenceRule),
    Array(ArrayInferenceRule),
    Introspection(IntrospectionInferenceRule),
    Math(MathInferenceRule),
    Parallel(ParallelInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AccelerationInferenceRule {
    Gather,
    GpuArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AggregateInferenceRule {
    Struct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ArrayInferenceRule {
    Full,
    Zeros,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum IntrospectionInferenceRule {
    Feval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MathInferenceRule {
    Abs,
    PhaseAngle,
    Exp,
    Expm1,
    Log1p,
    Logarithm(LogarithmBase),
    NumericConversion(runmat_types::NumericClass),
    NumericConversionWithLike(runmat_types::NumericClass),
    NumericComponent(NumericComponentRule),
    Signum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogarithmBase {
    Natural,
    Common,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NumericComponentRule {
    Conjugate,
    ImaginaryPart,
    RealPart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ParallelInferenceRule {
    Barrier,
    Broadcast,
    Cat,
    Codistributed,
    CodistributedBuild,
    Codistributor,
    Codistributor1d,
    Codistributor2dbc,
    CodistributorIsComplete,
    Distributed,
    FetchNext,
    FetchOutputs,
    FunctionalReduce,
    Gcp,
    GetCodistributor,
    GetCurrentJob,
    GetCurrentTask,
    GetCurrentWorker,
    GlobalIndices,
    Gplus,
    Iscodistributed,
    LocalPart,
    Parfeval,
    ParfevalOnAll,
    Parpool,
    Probe,
    Receive,
    Redistribute,
    Send,
    SendReceive,
    SpmdIndex,
    SpmdSize,
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
