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
        BuiltinInferenceRule::Identity(rule) if rule.has_fusion_fact() => Some(
            rule.fusion_fact(request).ok_or_else(Vec::new),
        ),
        BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(rule)) => Some(
            crate::catalog::entries::math::elementwise::binary_arithmetic::infer_binary_arithmetic_fusion_fact(
                rule, request,
            ),
        ),
        BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(function)) => Some(
            super::math_trigonometric::infer_fusion_numeric_fact(request, function),
        ),
        BuiltinInferenceRule::Math(MathInferenceRule::NumericConversionWithLike(target))
            if request.arguments.len() == 1 =>
        {
            Some(
                super::math::numeric_conversion::infer_single_fusion_fact(request, target)
                    .ok_or_else(Vec::new),
            )
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Root(crate::RootKind::Principal))
            if request.arguments.len() == 1 =>
        {
            Some(
                super::math::roots::infer_principal_fusion_fact(request)
                    .ok_or_else(Vec::new),
            )
        }
        _ => None,
    }
}

/// Structural admission for the same facts-only rule families above. Image
/// source geometry calls this before retaining an invocation; it never
/// substitutes a second numerical inference implementation.
pub fn finite_fusion_fact_route(entry: &BuiltinCatalogEntry, arity: usize) -> bool {
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Identity(rule) => rule.has_fusion_fact(),
        BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(_)) => arity == 2,
        BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(_)) => arity == 1,
        BuiltinInferenceRule::Math(MathInferenceRule::NumericConversionWithLike(_))
        | BuiltinInferenceRule::Math(MathInferenceRule::Root(crate::RootKind::Principal)) => {
            arity == 1
        }
        _ => false,
    }
}

pub(super) fn infer_local(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    local::infer(entry, request)
}

#[cfg(test)]
mod fusion_fact_tests;
