use crate::{NumericClass, ReductionNaN};

/// Closed operation semantics for an owner-admitted fused reduction. This is
/// shared by the runtime owner, physical recipe, and executing providers; it
/// does not by itself grant a provider capability or execution permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FusionReductionOperation {
    Sum,
    Mean,
    Minimum,
    Maximum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FusionReductionCount {
    None,
    SliceLength,
    NonNan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FusionReductionEmpty {
    PositiveZero,
    NaN,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FusionReductionTies {
    NotApplicable,
    FirstWithNegativeZero,
    FirstWithPositiveZero,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FusionReductionContract {
    pub operation: FusionReductionOperation,
    pub input_class: NumericClass,
    /// Sum/mean arithmetic uses Double. Extrema select represented input
    /// values, so their accumulator preserves the input class.
    pub accumulator_class: NumericClass,
    pub output_class: NumericClass,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    /// Zero-based owner-resolved axes in source order.
    pub axes: Vec<usize>,
    /// The owner's scalar/no-op path forwards rather than adding +0.
    pub identity: bool,
    pub nan_mode: ReductionNaN,
    pub count: FusionReductionCount,
    pub empty: FusionReductionEmpty,
    pub ties: FusionReductionTies,
}
