//! Operation-owned host semantics for a prospective fused reduction.
//!
//! This receipt is not provider capability or selected-binding evidence. It
//! deliberately does not bless current resident-path class inconsistencies.
use super::ReductionNaN;
use runmat_types::{NumericClass, NumericDomain, ValueFact};
use runmat_value::Value;
use std::{future::Future, pin::Pin};

pub struct FusionReductionContext<'a> {
    pub input: &'a ValueFact,
    pub controls: &'a [Value],
    pub requested_outputs: usize,
}
impl FusionReductionContext<'_> {
    /// This first contract covers real floating data and host-only controls.
    /// Device controls must retain ordinary dispatch, not perform hidden probes
    /// while constructing a supposedly payload-free semantic receipt.
    pub fn real_floating_input(&self) -> Option<(NumericClass, Vec<usize>)> {
        if self.requested_outputs != 1
            || !self.controls.iter().all(|value| {
                matches!(
                    value,
                    Value::Num(_)
                        | Value::Int(_)
                        | Value::String(_)
                        | Value::CharArray(_)
                        | Value::StringArray(_)
                        | Value::Tensor(_)
                )
            })
        {
            return None;
        }
        let numeric = self.input.numeric()?;
        if numeric.domain != NumericDomain::Real
            || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        {
            return None;
        }
        let shape = self
            .input
            .shape
            .known_dims()?
            .into_iter()
            .collect::<Option<Vec<_>>>()?;
        checked_reduction_elements(&shape)?;
        Some((numeric.class, shape))
    }
}

pub type FusionReductionFuture<'a> =
    Pin<Box<dyn Future<Output = crate::BuiltinResult<Option<FusionReductionContract>>> + 'a>>;
pub type FusionReductionResolver =
    for<'a> fn(FusionReductionContext<'a>) -> FusionReductionFuture<'a>;

pub struct FusionReductionDeclaration {
    pub revision: u32,
    pub resolve: FusionReductionResolver,
}

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
    /// Sum/mean arithmetic uses Double. Extrema only select represented input
    /// values, so their accumulator preserves the input class without rounding.
    pub accumulator_class: NumericClass,
    pub output_class: NumericClass,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    /// Zero-based owner-resolved axes, not guessed from an arbitrary operand.
    pub axes: Vec<usize>,
    /// Owner's scalar/no-op path forwards the value instead of adding +0.
    pub identity: bool,
    pub nan_mode: ReductionNaN,
    pub count: FusionReductionCount,
    pub empty: FusionReductionEmpty,
    pub ties: FusionReductionTies,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_fact::value_fact;

    #[test]
    fn zero_cardinality_precedes_overflow_and_other_overflow_is_unknown() {
        assert_eq!(checked_reduction_elements(&[usize::MAX, 2, 0]), Some(0));
        assert_eq!(checked_reduction_elements(&[usize::MAX, 2]), None);
        assert_eq!(checked_reduction_elements(&[]), Some(1));
    }

    #[test]
    fn contract_input_is_real_floating_single_output_and_does_not_touch_payloads() {
        let fact = value_fact(&Value::Num(3.0));
        let request = FusionReductionContext {
            input: &fact,
            controls: &[],
            requested_outputs: 1,
        };
        assert_eq!(
            request.real_floating_input(),
            Some((NumericClass::Double, vec![1, 1]))
        );
        let extra = FusionReductionContext {
            requested_outputs: 2,
            ..request
        };
        assert!(extra.real_floating_input().is_none());
        for value in [
            Value::Complex(1.0, 2.0),
            Value::Bool(true),
            Value::String("unsupported".into()),
        ] {
            let fact = value_fact(&value);
            assert!(FusionReductionContext {
                input: &fact,
                controls: &[],
                requested_outputs: 1
            }
            .real_floating_input()
            .is_none());
        }
    }
}

pub fn checked_reduction_elements(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |size, &extent| size.checked_mul(extent))
}
