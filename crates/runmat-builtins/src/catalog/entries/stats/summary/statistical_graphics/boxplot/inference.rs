//! Conditional handle facts and source-known boxplot argument diagnostics.
use super::options::BoxplotOptionKind;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn handles() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

fn bad_sample(fact: &ValueFact) -> bool {
    if fact.storage == StorageFact::Sparse {
        return true;
    }
    if !matches!(
        fact.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) {
        return true;
    }
    fact.shape.element_count() == Some(0) || fact.shape.rank().is_some_and(|rank| rank > 2)
}

fn bad_group(fact: &ValueFact, sample: &ValueFact) -> bool {
    if matches!(
        fact.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        })
    ) {
        return true;
    }
    if !matches!(
        fact.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Cell(_)
            | ValueKindFact::Object(_)
            | ValueKindFact::Unknown
    ) {
        return true;
    }
    if let ValueKindFact::Object(object) = &fact.kind {
        if object
            .runtime_class
            .as_ref()
            .is_some_and(|class| !class.is(runmat_types::standard::CATEGORICAL))
        {
            return true;
        }
    }
    if matches!(fact.kind, ValueKindFact::Cell(_) | ValueKindFact::Object(_)) {
        return false;
    }
    let group_count = if matches!(fact.kind, ValueKindFact::Character) {
        first_known_extent(&fact.shape)
    } else {
        fact.shape.element_count()
    };
    let (Some(group_count), Some(sample_count)) = (group_count, sample.shape.element_count())
    else {
        return false;
    };
    // Matrix inputs also admit row-wise groups, which are repeated per column.
    let rows = first_known_extent(&sample.shape);
    let matrix_rows = match &sample.shape {
        ShapeFact::Shaped { dims } => matches!(
            dims.as_slice(),
            [DimensionFact::Known(rows), DimensionFact::Known(cols)] if *rows > 1 && *cols > 1
        ),
        _ => false,
    };
    group_count != sample_count && !(matrix_rows && rows == Some(group_count))
}

fn first_known_extent(shape: &ShapeFact) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } => match dims.first() {
            Some(DimensionFact::Known(extent)) => Some(*extent),
            _ => None,
        },
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => None,
    }
}

fn known_bad_option(
    kind: BoxplotOptionKind,
    value: &ValueFact,
    literal: Option<&LiteralValue>,
    factor_count: Option<usize>,
) -> bool {
    use BoxplotOptionKind::*;
    let numeric = literal.and_then(LiteralContext::numeric_from_literal);
    match kind {
        Whisker | Jitter | OutlierSize => {
            !matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            ) || value.shape.element_count().is_some_and(|count| count != 1)
                || numeric.is_some_and(|n| {
                    !n.is_finite()
                        || (kind == OutlierSize
                            && (n <= 0.0 || !((n as f32).is_finite() && (n as f32) > 0.0)))
                        || (kind != OutlierSize && n < 0.0)
                })
        }
        Orientation | PlotStyle | BoxStyle | MedianStyle | ExtremeMode | LabelOrientation
        | LabelVerbosity | Symbol | Notch | FullFactors => known_bad_scalar_text(value),
        FactorDirection => {
            !matches!(
                value.kind,
                ValueKindFact::Character
                    | ValueKindFact::String
                    | ValueKindFact::Cell(_)
                    | ValueKindFact::Unknown
            ) || factor_count.is_some_and(|count| {
                let length = if matches!(value.kind, ValueKindFact::Character) {
                    first_known_extent(&value.shape)
                } else {
                    value.shape.element_count()
                };
                length.is_some_and(|length| length != 1 && length != count)
            })
        }
        FactorGap | FactorSeparator
            if matches!(value.kind, ValueKindFact::String | ValueKindFact::Character) =>
        {
            known_bad_scalar_text(value)
                || literal.is_some_and(|literal| match literal {
                    LiteralValue::String(text)
                    | LiteralValue::Character(text)
                    | LiteralValue::Keyword(text) => !text.eq_ignore_ascii_case("auto"),
                    _ => false,
                })
        }
        FactorGap | FactorSeparator | Positions | Widths | DataLim => {
            if known_bad_numeric_vector(value) {
                return true;
            }
            match kind {
                FactorGap => {
                    value
                        .shape
                        .element_count()
                        .zip(factor_count)
                        .is_some_and(|(len, count)| len > count)
                        || literal.is_some_and(|literal| {
                            any_numeric_literal(literal, |n| !n.is_finite() || n < 0.0)
                        })
                }
                FactorSeparator => {
                    value
                        .shape
                        .element_count()
                        .zip(factor_count)
                        .is_some_and(|(len, count)| len > count)
                        || literal.is_some_and(|literal| {
                            any_separator_literal_invalid(literal, factor_count)
                        })
                }
                Positions => {
                    literal.is_some_and(|literal| any_numeric_literal(literal, |n| !n.is_finite()))
                }
                Widths => {
                    value.shape.element_count() == Some(0)
                        || literal.is_some_and(|literal| {
                            any_numeric_literal(literal, |n| !n.is_finite() || n <= 0.0)
                        })
                }
                DataLim => {
                    value.shape.element_count().is_some_and(|len| len != 2)
                        || literal.is_some_and(|literal| any_numeric_literal(literal, f64::is_nan))
                }
                _ => false,
            }
        }
        Labels | GroupOrder => {
            matches!(value.kind, ValueKindFact::Logical)
                || matches!(
                    value.kind,
                    ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Complex,
                        ..
                    })
                )
        }
        _ => false,
    }
}

fn known_bad_numeric_vector(value: &ValueFact) -> bool {
    value.storage == StorageFact::Sparse
        || !matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        )
        || value.shape.rank().is_some_and(|rank| rank > 2)
        || matches!(&value.shape, ShapeFact::Shaped { dims } if
            matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows > 1)
                && matches!(dims.get(1), Some(DimensionFact::Known(cols)) if *cols > 1))
}

fn any_numeric_literal(literal: &LiteralValue, invalid: impl Fn(f64) -> bool) -> bool {
    match literal {
        LiteralValue::Vector(values) => values
            .iter()
            .any(|value| numeric_option_literal(value).is_some_and(&invalid)),
        value => numeric_option_literal(value).is_some_and(invalid),
    }
}

fn numeric_option_literal(literal: &LiteralValue) -> Option<f64> {
    match literal {
        LiteralValue::Bool(value) => Some(f64::from(u8::from(*value))),
        other => LiteralContext::numeric_from_literal(other),
    }
}

fn any_separator_literal_invalid(literal: &LiteralValue, factor_count: Option<usize>) -> bool {
    match literal {
        LiteralValue::Vector(values) => values
            .iter()
            .any(|value| separator_literal_invalid(value, factor_count)),
        value => separator_literal_invalid(value, factor_count),
    }
}

fn separator_literal_invalid(literal: &LiteralValue, factor_count: Option<usize>) -> bool {
    let selector = match literal {
        LiteralValue::Integer { text, .. } => text
            .parse::<u128>()
            .ok()
            .and_then(|value| usize::try_from(value).ok()),
        other => numeric_option_literal(other)
            .and_then(|value| runmat_types::decode_shape_dimension(value).ok()),
    };
    match selector {
        Some(0) => true,
        Some(selector) => factor_count.is_some_and(|count| selector > count),
        None => matches!(
            literal,
            LiteralValue::Integer { .. }
                | LiteralValue::Number(_)
                | LiteralValue::Real { .. }
                | LiteralValue::Complex { .. }
                | LiteralValue::Bool(_)
        ),
    }
}

fn known_bad_scalar_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
        ValueKindFact::Character => first_known_extent(&value.shape).is_some_and(|rows| rows != 1),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    let mut input_roles_unresolved = false;
    if args.is_empty() {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "boxplot requires numeric sample data",
        ));
    }
    if let Some(first) = args.first() {
        if bad_sample(first) {
            diagnostics.push(InferenceDiagnostic::error(
                "DATA",
                "boxplot X must be a nonempty real numeric vector or matrix",
            ));
        }
        // A scalar numeric first argument can also denote axes. Only a known
        // non-scalar first argument fixes the direct-data argument positions.
        let direct = first.shape.element_count().is_some_and(|count| count != 1);
        input_roles_unresolved = !direct && args.len() > 1;
        if direct {
            let second_is_group =
                args.get(1)
                    .is_some_and(|second| match request.literals.literal_string_at(1) {
                        Some(name) => BoxplotOptionKind::parse(&name).is_none(),
                        None => match second.kind {
                            ValueKindFact::Numeric(_)
                            | ValueKindFact::Logical
                            | ValueKindFact::Cell(_)
                            | ValueKindFact::Object(_) => true,
                            ValueKindFact::String => {
                                second.shape.element_count().is_some_and(|count| count != 1)
                            }
                            ValueKindFact::Character => {
                                first_known_extent(&second.shape).is_some_and(|rows| rows != 1)
                            }
                            _ => false,
                        },
                    });
            let option_start = if second_is_group { 2 } else { 1 };
            let factor_count = if second_is_group && matches!(args[1].kind, ValueKindFact::Cell(_))
            {
                None
            } else {
                Some(1)
            };
            if second_is_group && bad_group(&args[1], first) {
                diagnostics.push(InferenceDiagnostic::error(
                    "GROUP",
                    "boxplot grouping data are incompatible with X",
                ));
            }
            if !(args.len() - option_start).is_multiple_of(2) {
                diagnostics.push(InferenceDiagnostic::error(
                    "OPTION",
                    "boxplot name-value options must be paired",
                ));
            }
            for index in (option_start..args.len()).step_by(2) {
                let Some(name) = request.literals.literal_string_at(index) else {
                    continue;
                };
                let Some(kind) = BoxplotOptionKind::parse(&name) else {
                    diagnostics.push(InferenceDiagnostic::error(
                        "OPTION",
                        "boxplot option name is unsupported",
                    ));
                    continue;
                };
                if let Some(value) = args.get(index + 1) {
                    if known_bad_option(
                        kind,
                        value,
                        request.literals.literal_args.get(index + 1),
                        factor_count,
                    ) {
                        diagnostics.push(InferenceDiagnostic::error(
                            "OPTION",
                            "boxplot option value is invalid",
                        ));
                    }
                }
            }
        }
    }
    let mut contract =
        CallContract::fixed(vec![if diagnostics.is_empty() && !input_roles_unresolved {
            handles()
        } else {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }]);
    // The established runtime pads additional requested outputs with empty
    // values. Their contents are not a second graphics handle contract.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
