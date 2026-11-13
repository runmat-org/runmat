//! Exact request-local domain proof for the ordinary two-operand maximum.
//! Reduction, comparison options, and physical provider equivalence are not
//! authorized by this declaration.

use crate::builtins::common::spec::{
    FusionAdmission, FusionAdmissionContext, FusionAdmissionEvidence, FusionAdmissionFuture,
    FusionAdmissionWork, FusionProofStage, FusionSemanticDeclaration,
};
use runmat_accelerate_api::{NumericalDomainEvidence, ProviderPrecision};
use runmat_builtins::{BuiltinBindingIdentity, MAX_CATALOG_ENTRY};
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
use runmat_value::{NumericScalar, Value};

pub(super) const SEMANTICS: FusionSemanticDeclaration = FusionSemanticDeclaration {
    binding: BuiltinBindingIdentity {
        builtin: MAX_CATALOG_ENTRY.identity,
        variant: "default",
    },
    revision: 2,
    admit: Some(FusionAdmission::unbounded(admit)),
    admit_static: None,
    required_real_unary: None,
    required_arithmetic: None,
    helpers: &[],
    scalar_operation: None,
};

fn positive_zero(value: &Value, class: NumericClass) -> bool {
    match (value, class) {
        (Value::Num(value), NumericClass::Double) => value.to_bits() == 0.0_f64.to_bits(),
        (Value::Tensor(tensor), NumericClass::Single) if tensor.len() == 1 => {
            matches!(tensor.numeric_value_at(0), Some(NumericScalar::F32(value)) if value.to_bits() == 0.0_f32.to_bits())
        }
        (Value::Tensor(tensor), NumericClass::Double) if tensor.len() == 1 => {
            matches!(tensor.numeric_value_at(0), Some(NumericScalar::F64(value)) if value.to_bits() == 0.0_f64.to_bits())
        }
        _ => false,
    }
}

fn admit(context: FusionAdmissionContext<'_>) -> FusionAdmissionFuture<'_> {
    Box::pin(async move {
        let ([left, right], [left_fact, right_fact]) = (context.operands, context.facts) else {
            return Ok(None);
        };
        let (Some(left_numeric), Some(right_numeric)) = (left_fact.numeric(), right_fact.numeric())
        else {
            return Ok(None);
        };
        if left_numeric != right_numeric
            || left_numeric.domain != NumericDomain::Real
            || !matches!(
                left_numeric.class,
                NumericClass::Single | NumericClass::Double
            )
            || left_fact.shape.element_count() == Some(0)
            || right_fact.shape.element_count() == Some(0)
        {
            return Ok(None);
        }
        let has_exact_zero = left_fact.shape.element_count() == Some(1)
            && left.is_some_and(|value| positive_zero(value, left_numeric.class))
            || right_fact.shape.element_count() == Some(1)
                && right.is_some_and(|value| positive_zero(value, right_numeric.class));
        if !has_exact_zero {
            return Ok(None);
        }
        *context.work = FusionAdmissionWork::DomainProof {
            binding: SEMANTICS.binding,
            provider_precision: if left_numeric.class == NumericClass::Single {
                ProviderPrecision::F32
            } else {
                ProviderPrecision::F64
            },
            stages: vec![FusionProofStage::RangePropagation { operands: 2 }],
        };
        Ok(Some(FusionAdmissionEvidence {
            output_storage: None,
            output_kind: ValueKindFact::Numeric(NumericFact {
                class: left_numeric.class,
                domain: NumericDomain::Real,
            }),
            numerical_domain: NumericalDomainEvidence::NonnegativeOrNaN,
            arithmetic: None,
            sign: Some(
                crate::builtins::common::spec::FusionSignEvidence::NonnegativeOrNaN {
                    class: left_numeric.class,
                },
            ),
        }))
    })
}

#[cfg(test)]
#[path = "fusion_pairwise/tests.rs"]
mod tests;
