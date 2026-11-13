//! Conditional chart-handle facts and host-visible form diagnostics.
use super::options::{BinscatterOptionKind, BINSCATTER_MAX_NUM_BINS};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, InferenceDiagnostic, LiteralContext,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};

fn chart_handle() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn known_bad_vector(fact: &ValueFact) -> bool {
    match fact.shape.known_dims() {
        Some(dims) => {
            dims.iter()
                .filter(|extent| extent.is_some_and(|n| n > 1))
                .count()
                > 1
        }
        None => false,
    }
}

fn known_bad_control(
    kind: BinscatterOptionKind,
    fact: &ValueFact,
    literal: Option<&LiteralValue>,
) -> bool {
    let count = fact.shape.element_count();
    let numeric_real = matches!(&fact.kind, ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real);
    let numeric_complex = matches!(&fact.kind, ValueKindFact::Numeric(number) if number.domain == NumericDomain::Complex);
    let invalid_for_option = match kind {
        BinscatterOptionKind::NumBins => {
            if numeric_complex || known_bad_vector(fact) || count.is_some_and(|n| n != 1 && n != 2)
            {
                return true;
            }
            let numeric_literal = literal.and_then(LiteralContext::numeric_from_literal);
            numeric_literal.is_some_and(|value| {
                !value.is_finite()
                    || value < 1.0
                    || value.fract() != 0.0
                    || value > BINSCATTER_MAX_NUM_BINS as f64
            }) || matches!(literal, Some(LiteralValue::Bool(false)))
        }
        BinscatterOptionKind::XLimits | BinscatterOptionKind::YLimits => {
            numeric_complex || known_bad_vector(fact) || count.is_some_and(|n| n != 2)
        }
        BinscatterOptionKind::FaceAlpha => {
            numeric_complex
                || count.is_some_and(|n| n != 1)
                || literal
                    .and_then(LiteralContext::numeric_from_literal)
                    .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        }
        BinscatterOptionKind::ShowEmptyBins => {
            count.is_some_and(|n| n != 1)
                || numeric_complex
                || literal
                    .and_then(LiteralContext::numeric_from_literal)
                    .is_some_and(|value| value != 0.0 && value != 1.0)
        }
        BinscatterOptionKind::DisplayName | BinscatterOptionKind::DisplayStyle => {
            !matches!(
                fact.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            ) || (matches!(fact.kind, ValueKindFact::String) && count.is_some_and(|n| n != 1))
                || (kind == BinscatterOptionKind::DisplayStyle
                    && literal.is_some_and(|literal| match literal {
                        LiteralValue::String(value) | LiteralValue::Character(value) => {
                            !value.trim().eq_ignore_ascii_case("tile")
                        }
                        _ => false,
                    }))
        }
        BinscatterOptionKind::UnsupportedMatlabOption => true,
    };
    invalid_for_option
        || (!numeric_real
            && !matches!(fact.kind, ValueKindFact::Logical)
            && matches!(
                kind,
                BinscatterOptionKind::NumBins
                    | BinscatterOptionKind::XLimits
                    | BinscatterOptionKind::YLimits
            )
            && !matches!(fact.kind, ValueKindFact::Unknown))
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    if args.len() < 2 {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "binscatter requires x and y observations or table-variable selectors",
        ));
    }
    // A numeric scalar might be an axes handle. A known non-scalar first
    // operand is unambiguously direct X data, so validate its XY geometry.
    let direct_xy = args.first().is_some_and(|first| {
        matches!(
            first.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) && first.shape.element_count().is_some_and(|count| count != 1)
    });
    if direct_xy {
        for (index, fact) in args.iter().take(2).enumerate() {
            if !matches!(
                fact.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            ) || fact
                .numeric()
                .is_some_and(|number| number.domain != NumericDomain::Real)
                || known_bad_vector(fact)
            {
                diagnostics.push(InferenceDiagnostic::error(
                    "DATA",
                    if index == 0 {
                        "binscatter x must be a real numeric vector"
                    } else {
                        "binscatter y must be a real numeric vector"
                    },
                ));
            }
        }
        if args.len() >= 2
            && args[0].shape.element_count().is_some()
            && args[1].shape.element_count().is_some()
            && args[0].shape.element_count() != args[1].shape.element_count()
        {
            diagnostics.push(InferenceDiagnostic::error(
                "LENGTH",
                "binscatter x and y must contain the same number of elements",
            ));
        }
        if let Some(third) = args.get(2) {
            let option_start = if matches!(
                third.kind,
                ValueKindFact::String | ValueKindFact::Character
            ) {
                2
            } else if matches!(
                third.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) {
                if known_bad_control(
                    BinscatterOptionKind::NumBins,
                    third,
                    request.literals.literal_args.get(2),
                ) {
                    diagnostics.push(InferenceDiagnostic::error("OPTION", "binscatter NumBins must be a positive scalar or two-element vector no greater than 250"));
                }
                3
            } else {
                args.len()
            };
            if !(args.len() - option_start).is_multiple_of(2) {
                diagnostics.push(InferenceDiagnostic::error(
                    "OPTION",
                    "binscatter name-value controls require paired values",
                ));
            }
            for index in (option_start..args.len()).step_by(2) {
                let Some(name) = request.literals.literal_string_at(index) else {
                    continue;
                };
                let Some(kind) = BinscatterOptionKind::parse(&name) else {
                    diagnostics.push(InferenceDiagnostic::error(
                        "OPTION",
                        "binscatter option is unknown or unsupported",
                    ));
                    continue;
                };
                if kind == BinscatterOptionKind::UnsupportedMatlabOption {
                    diagnostics.push(InferenceDiagnostic::error(
                        "OPTION",
                        "binscatter option is unknown or unsupported",
                    ));
                    continue;
                }
                if let Some(value) = args.get(index + 1) {
                    if known_bad_control(kind, value, request.literals.literal_args.get(index + 1))
                    {
                        diagnostics.push(InferenceDiagnostic::error(
                            "OPTION",
                            "binscatter option value is invalid",
                        ));
                    }
                }
            }
        }
    }
    // With exactly two inputs, a scalar numeric first operand can be X or
    // an axes handle. If the second input is a non-scalar observation vector,
    // direct X/Y has incompatible lengths and axes/X lacks its Y operand.
    if !direct_xy
        && args.len() == 2
        && args[0].shape.element_count() == Some(1)
        && args[1].shape.element_count().is_some_and(|n| n != 1)
        && matches!(
            args[0].kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    {
        diagnostics.push(InferenceDiagnostic::error(
            "LENGTH",
            "binscatter x and y must contain the same number of elements",
        ));
    }
    // The third argument may be positional NumBins for direct X/Y, or Y for
    // axes/X/Y. When X is scalar and both other known lengths disagree, neither
    // interpretation can produce a chart.
    if !direct_xy
        && args.len() == 3
        && args[0].shape.element_count() == Some(1)
        && matches!(
            args[0].kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
        && matches!(
            args[1].kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
        && matches!(
            args[2].kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
        && args[1].shape.element_count().is_some_and(|second| {
            second != 1
                && args[2]
                    .shape
                    .element_count()
                    .is_some_and(|third| third != second)
        })
    {
        diagnostics.push(InferenceDiagnostic::error(
            "LENGTH",
            "binscatter x and y must contain the same number of elements",
        ));
    }
    let mut contract = CallContract::fixed(vec![if diagnostics.is_empty() {
        chart_handle()
    } else {
        ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue)
    }]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
