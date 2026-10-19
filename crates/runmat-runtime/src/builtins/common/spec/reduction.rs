//! Operation-owned host semantics for a prospective fused reduction.
//!
//! This receipt is not provider capability or selected-binding evidence. It
//! deliberately does not bless current resident-path class inconsistencies.
pub use runmat_types::{
    FusionReductionContract, FusionReductionCount, FusionReductionEmpty, FusionReductionOperation,
    FusionReductionTies,
};
use runmat_types::{NumericClass, NumericDomain, StorageFact, ValueFact};
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
pub type FusionReductionStagesFuture<'a> =
    Pin<Box<dyn Future<Output = crate::BuiltinResult<Option<Vec<FusionReductionContract>>>> + 'a>>;
pub struct FusionReductionDeclaration {
    pub revision: u32,
    /// Operation fixed by this selected builtin. The resolved contract may
    /// refine axes/classes, but a legacy kernel may never guess its operation
    /// from a name or treat an unsupported extrema as Sum.
    pub operation: FusionReductionOperation,
    pub prepare: FusionReductionPreparer,
}

/// Parsed by the operation owner before producer admission can touch a device.
/// Resolving the final, domain-refined fact is pure and does not parse again.
pub trait PreparedFusionReduction {
    fn resolve(&self, input: &ValueFact) -> crate::BuiltinResult<Option<FusionReductionContract>>;

    /// Carrier certification is independent of any input sign theorem.
    /// Only the prepared operation's ordinary public output owner may opt in.
    fn publication_storage(
        &self,
        _input: &ValueFact,
        _output: &ValueFact,
        _stages: &[FusionReductionContract],
    ) -> Option<StorageFact> {
        None
    }

    /// Ordered arithmetic stages for one public reduction call. The default
    /// covers reductions whose public operation is one physical reduction.
    /// A multi-axis owner may preserve intermediate class and publication
    /// boundaries by returning several contracts from its parsed controls.
    fn resolve_stages(
        &self,
        input: &ValueFact,
    ) -> crate::BuiltinResult<Option<Vec<FusionReductionContract>>> {
        self.resolve(input)
            .map(|contract| contract.map(|stage| vec![stage]))
    }

    /// Optional represented publication and sign theorem. This is deliberately
    /// absent for every owner unless its actual final publication certifies it.
    fn nonnegative_output_storage(
        &self,
        _input: &ValueFact,
        _output: &ValueFact,
        _stages: &[FusionReductionContract],
    ) -> Option<StorageFact> {
        None
    }
}

pub struct FusionReductionControls<'a> {
    pub values: &'a [Value],
    pub requested_outputs: usize,
}
impl FusionReductionControls<'_> {
    pub fn supported(&self) -> bool {
        self.requested_outputs == 1
            && self.values.iter().all(|value| {
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
    }
}
pub type FusionReductionPrepareFuture<'a> = Pin<
    Box<dyn Future<Output = crate::BuiltinResult<Option<Box<dyn PreparedFusionReduction>>>> + 'a>,
>;
pub type FusionReductionPreparer =
    for<'a> fn(FusionReductionControls<'a>) -> FusionReductionPrepareFuture<'a>;

impl FusionReductionDeclaration {
    pub async fn certify_publication_storage(
        &self,
        context: FusionReductionContext<'_>,
        output: &ValueFact,
    ) -> crate::BuiltinResult<Option<StorageFact>> {
        if context.real_floating_input().is_none()
            || !matches!(
                context.input.storage,
                StorageFact::Scalar | StorageFact::Dense
            )
            || matches!(output.storage, StorageFact::Sparse | StorageFact::Opaque)
        {
            return Ok(None);
        }
        let Some(prepared) = (self.prepare)(FusionReductionControls {
            values: context.controls,
            requested_outputs: context.requested_outputs,
        })
        .await?
        else {
            return Ok(None);
        };
        let Some(stages) = prepared.resolve_stages(context.input)? else {
            return Ok(None);
        };
        Ok(prepared
            .publication_storage(context.input, output, &stages)
            .filter(|storage| matches!(storage, StorageFact::Scalar | StorageFact::Dense)))
    }

    pub fn resolve<'a>(&'a self, context: FusionReductionContext<'a>) -> FusionReductionFuture<'a> {
        Box::pin(async move {
            // Preserve the standalone resolver's existing unsupported-input gate.
            if context.real_floating_input().is_none() {
                return Ok(None);
            }
            let Some(prepared) = (self.prepare)(FusionReductionControls {
                values: context.controls,
                requested_outputs: context.requested_outputs,
            })
            .await?
            else {
                return Ok(None);
            };
            prepared.resolve(context.input)
        })
    }

    pub fn resolve_stages<'a>(
        &'a self,
        context: FusionReductionContext<'a>,
    ) -> FusionReductionStagesFuture<'a> {
        Box::pin(async move {
            if context.real_floating_input().is_none() {
                return Ok(None);
            }
            let Some(prepared) = (self.prepare)(FusionReductionControls {
                values: context.controls,
                requested_outputs: context.requested_outputs,
            })
            .await?
            else {
                return Ok(None);
            };
            prepared.resolve_stages(context.input)
        })
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
