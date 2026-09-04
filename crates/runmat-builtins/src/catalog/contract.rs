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
    Logical(LogicalInferenceRule),
    Math(MathInferenceRule),
    Parallel(ParallelInferenceRule),
    Stats(StatsInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalInferenceRule {
    Elementwise(LogicalElementwiseRule),
    MetadataPredicate(MetadataPredicate),
    NumericClassification(NumericClassificationPredicate),
    Relational(RelationalOperator),
    ScalarReduction(ScalarLogicalReduction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalElementwiseRule {
    Binary(LogicalBinaryOperator),
    Unary(LogicalUnaryOperator),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalBinaryOperator {
    And,
    Or,
    Xor,
}

impl LogicalBinaryOperator {
    pub const fn name(self) -> &'static str {
        match self {
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalUnaryOperator {
    Not,
}

impl LogicalUnaryOperator {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Not => "not",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RelationalOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl RelationalOperator {
    pub const fn is_equality(self) -> bool {
        matches!(self, Self::Equal | Self::NotEqual)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Equal => "eq",
            Self::NotEqual => "ne",
            Self::LessThan => "lt",
            Self::LessThanOrEqual => "le",
            Self::GreaterThan => "gt",
            Self::GreaterThanOrEqual => "ge",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MetadataPredicate {
    Cell,
    CellString,
    GpuArray,
    Logical,
    Numeric,
    Real,
    Sparse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NumericClassificationPredicate {
    Finite,
    Infinite,
    Nan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarLogicalReduction {
    AllFinite,
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
    Accumulation(AccumulationInferenceRule),
    Binning(BinningInferenceRule),
    Combinatorics(CombinatoricsInferenceRule),
    Creation(ArrayCreationInferenceRule),
    Grouping(GroupingInferenceRule),
    Introspection(ArrayIntrospectionInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AccumulationInferenceRule {
    Indexed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinningInferenceRule {
    Discretize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CombinatoricsInferenceRule {
    CartesianProduct,
    Permutations,
    SelectionCombinations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GroupingInferenceRule {
    Counts,
    GroupedApply,
    IndexLabels,
    SortedGroups,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ArrayCreationInferenceRule {
    Full,
    Zeros,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ArrayIntrospectionInferenceRule {
    ShapePredicate(ShapePredicate),
    ShapeQuery(ShapeQuery),
    ShapeScalarQuery(ShapeScalarQuery),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShapeQuery {
    Size,
    ElementCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShapePredicate {
    Empty,
    Scalar,
    Vector,
    Matrix,
    Row,
    Column,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShapeScalarQuery {
    Length,
    Rank,
    Height,
    Width,
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
    AngleConversion(AngleConversionInferenceRule),
    Atan2,
    Bitwise(BitwiseInferenceRule),
    Discrete(DiscreteInferenceRule),
    ErrorFunction(ErrorFunctionInferenceRule),
    GammaFunction(GammaFunctionInferenceRule),
    IntegerDivide,
    Hypot,
    PhaseAngle,
    Exp,
    Expm1,
    Log1p,
    Log2,
    Logarithm(LogarithmBase),
    LogicalReduction(LogicalReductionKind),
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
pub enum AngleConversionInferenceRule {
    DegreesToRadians,
    RadiansToDegrees,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DiscreteInferenceRule {
    Binary(BinaryNumberTheoryRule),
    Factor,
    Factorial,
    IsPrime,
    Primes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinaryNumberTheoryRule {
    Gcd,
    Lcm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ErrorFunctionInferenceRule {
    Erf,
    InverseComplementary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GammaFunctionInferenceRule {
    Gamma,
    LogGamma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BitwiseInferenceRule {
    Binary(BinaryBitwiseOperator),
    Complement,
    Get,
    Set,
    Shift,
    SwapBytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinaryBitwiseOperator {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalReductionKind {
    All,
    Any,
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
