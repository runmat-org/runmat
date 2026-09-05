use crate::{BuiltinCatalogEntry, MathInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: MathInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        MathInferenceRule::AngleConversion(rule) => {
            super::super::math::angle_conversion::infer(rule, request, entry)
        }
        MathInferenceRule::Atan2 => super::super::math_binary::infer_atan2(request, entry),
        MathInferenceRule::Bitwise(rule) => {
            super::super::math::bitwise::infer(rule, request, entry)
        }
        MathInferenceRule::Discrete(rule) => {
            super::super::math::discrete::infer(rule, request, entry)
        }
        MathInferenceRule::ErrorFunction(rule) => {
            super::super::math::error_functions::infer(rule, request, entry)
        }
        MathInferenceRule::GammaFunction(rule) => {
            super::super::math::gamma_functions::infer(rule, request, entry)
        }
        MathInferenceRule::IntegerDivide => {
            super::super::math::integer_division::infer(request, entry)
        }
        MathInferenceRule::Hypot => super::super::math_binary::infer_hypot(request, entry),
        MathInferenceRule::MagnitudePhaseSign(kind) => {
            super::super::math::magnitude_phase_sign::infer(kind, request, entry)
        }
        MathInferenceRule::Exponential(kind) => {
            super::super::math::exponentials::infer(kind, request, entry)
        }
        MathInferenceRule::Logarithm(kind) => {
            super::super::math::logarithms::infer(kind, request, entry)
        }
        MathInferenceRule::LogicalReduction(kind) => {
            super::super::math_reduction::infer_logical(request, entry, kind)
        }
        MathInferenceRule::Root(kind) => super::super::math::roots::infer(kind, request, entry),
        MathInferenceRule::NumericLimit(rule) => {
            super::super::numeric_limit::infer_numeric_limit(request, entry, rule)
        }
        MathInferenceRule::NumericConversion(target) => {
            super::super::math::numeric_conversion::infer_numeric_conversion_call(
                request, entry, target,
            )
        }
        MathInferenceRule::NumericConversionWithLike(target) => {
            super::super::math::numeric_conversion::infer_numeric_conversion_with_like_call(
                request, entry, target,
            )
        }
        MathInferenceRule::NumericComponent(rule) => {
            super::super::math::complex_components::infer(request, entry, rule)
        }
        MathInferenceRule::Rounding(function) => {
            super::super::math_rounding::infer_rounding(request, entry, function)
        }
        MathInferenceRule::Round => super::super::math_rounding::infer_round(request, entry),
        MathInferenceRule::Remainder(function) => {
            super::super::math_binary::infer_remainder(request, entry, function)
        }
        MathInferenceRule::Trigonometric(function) => {
            super::super::math_trigonometric::infer_trigonometric(request, entry, function)
        }
        MathInferenceRule::Hyperbolic(function) => {
            super::super::math_hyperbolic::infer_hyperbolic(request, entry, function)
        }
        MathInferenceRule::PiScaledTrigonometric(function) => {
            super::super::math_trigonometric::infer_pi_scaled_trigonometric(
                request, entry, function,
            )
        }
        MathInferenceRule::PowerOfTwo(rule) => {
            super::super::math::powers_of_two::infer(rule, request, entry)
        }
        MathInferenceRule::DegreeTrigonometric(function) => {
            super::super::math_degree_trigonometric::infer_degree_trigonometric(
                request, entry, function,
            )
        }
        MathInferenceRule::InverseTrigonometric(function) => {
            super::super::math_inverse::infer_inverse_trigonometric(request, entry, function)
        }
        MathInferenceRule::InverseHyperbolic(function) => {
            super::super::math_inverse::infer_inverse_hyperbolic(request, entry, function)
        }
    }
}
