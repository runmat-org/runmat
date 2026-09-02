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
    Stats(StatsInferenceRule),
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
pub enum StatsInferenceRule {
    Random(StatsRandomInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StatsRandomInferenceRule {
    Binomial,
    Gamma,
    Weibull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MathInferenceRule {
    Abs,
    Atan2,
    Hypot,
    Gamma,
    GammaLn,
    PhaseAngle,
    Exp,
    Expm1,
    Log1p,
    Log2,
    Logarithm(LogarithmBase),
    Root(RootKind),
    NumericLimit(NumericLimitRule),
    NumericConversion(runmat_types::NumericClass),
    NumericConversionWithLike(runmat_types::NumericClass),
    NumericComponent(NumericComponentRule),
    Rounding(RoundingFunction),
    Remainder(RemainderFunction),
    Round,
    Signum,
    Trigonometric(TrigonometricFunction),
    Hyperbolic(HyperbolicFunction),
    PiScaledTrigonometric(PiScaledTrigonometricFunction),
    DegreeTrigonometric(DegreeTrigonometricFunction),
    InverseTrigonometric(InverseTrigonometricFunction),
    InverseHyperbolic(InverseHyperbolicFunction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RoundingFunction {
    Ceil,
    Fix,
    Floor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RemainderFunction {
    Modulus,
    Remainder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TrigonometricFunction {
    Sin,
    Cos,
    Tan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HyperbolicFunction {
    Sine,
    Cosine,
    Tangent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PiScaledTrigonometricFunction {
    Sin,
    Cos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DegreeTrigonometricFunction {
    Sin,
    Cos,
    Tan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InverseTrigonometricFunction {
    Sine,
    Cosine,
    Tangent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InverseHyperbolicFunction {
    Cosine,
    Sine,
    Tangent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogarithmBase {
    Natural,
    Common,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RootKind {
    Principal,
    RealOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NumericLimitRule {
    Integer(IntegerLimitKind),
    Floating(FloatingLimitKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum IntegerLimitKind {
    Minimum,
    Maximum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FloatingLimitKind {
    SmallestNormal,
    LargestFinite,
    LargestConsecutiveInteger,
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
