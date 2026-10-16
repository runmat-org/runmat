pub(super) mod acceleration;
pub(super) mod array;
mod distributed;
pub(super) mod introspection;
pub(super) mod io;
mod local;
pub(super) mod math;
pub(super) mod parallel;
pub(super) mod stats;

use crate::{BuiltinCatalogEntry, BuiltinInferenceRule, MathInferenceRule};
use runmat_types::{CallInference, CallRequest, InferenceDiagnostic, ValueFact};

pub fn infer(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    distributed::infer(entry, request)
}

/// Selected real-numeric fusion calls consume one fact, not effect or
/// capability sets. Return `None` for rule families without a certified
/// facts-only route; callers with finite authority must reject those before
/// allocating or probing rather than substituting a different inference.
pub fn infer_fusion_numeric_fact(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
) -> Option<Result<ValueFact, Vec<InferenceDiagnostic>>> {
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(rule)) => Some(
            crate::catalog::entries::math::elementwise::binary_arithmetic::infer_binary_arithmetic_fusion_fact(
                rule, request,
            ),
        ),
        BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(function)) => Some(
            super::math_trigonometric::infer_fusion_numeric_fact(request, function),
        ),
        _ => None,
    }
}

pub(super) fn infer_local(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    local::infer(entry, request)
}

#[cfg(test)]
mod fusion_fact_tests {
    use super::*;
    use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};

    #[test]
    fn pure_numeric_fact_route_matches_public_inference() {
        let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        for (entry, arguments) in [
            (
                &crate::PLUS_CATALOG_ENTRY,
                vec![input.clone(), input.clone()],
            ),
            (&crate::SIN_CATALOG_ENTRY, vec![input.clone()]),
        ] {
            let request = CallRequest {
                arguments,
                literals: Default::default(),
                outputs: Default::default(),
            };
            let public = infer(entry, &request);
            assert!(public.diagnostics.is_empty());
            assert_eq!(public.outputs.len(), 1);
            assert_eq!(
                infer_fusion_numeric_fact(entry, &request).unwrap().unwrap(),
                public.outputs[0]
            );
        }
    }
}
