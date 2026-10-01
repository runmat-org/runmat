//! Conditional handle facts and source-known boxplot argument diagnostics.
use super::options::BoxplotOptionKind;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
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
    if matches!(
        fact.kind,
        ValueKindFact::Cell(_) | ValueKindFact::Object(_) | ValueKindFact::Character
    ) {
        return false;
    }
    let (Some(group_count), Some(sample_count), Some(dims)) = (
        fact.shape.element_count(),
        sample.shape.element_count(),
        sample.shape.known_dims(),
    ) else {
        return false;
    };
    // Matrix inputs also admit row-wise groups, which are repeated per column.
    let rows = dims.first().copied().flatten();
    group_count != sample_count && rows != Some(group_count)
}

fn known_bad_option(kind: BoxplotOptionKind, value: &ValueFact, numeric: Option<f64>) -> bool {
    use BoxplotOptionKind::*;
    match kind {
        Whisker | Jitter | OutlierSize => {
            !matches!(
                value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            ) || value.shape.element_count().is_some_and(|count| count != 1)
                || numeric.is_some_and(|n| {
                    !n.is_finite()
                        || (kind == OutlierSize
                            && (n <= 0.0 || !((n as f32).is_finite() && (n as f32) > 0.0)))
                        || (kind != OutlierSize && n < 0.0)
                })
        }
        Orientation | PlotStyle | BoxStyle | MedianStyle | ExtremeMode | LabelOrientation
        | LabelVerbosity | Symbol | Notch | FullFactors | FactorDirection => !matches!(
            value.kind,
            ValueKindFact::Character
                | ValueKindFact::String
                | ValueKindFact::Cell(_)
                | ValueKindFact::Unknown
        ),
        _ => false,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
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
                            ValueKindFact::Character => second
                                .shape
                                .known_dims()
                                .and_then(|dims| dims.first().copied().flatten())
                                .is_some_and(|rows| rows != 1),
                            _ => false,
                        },
                    });
            let option_start = if second_is_group { 2 } else { 1 };
            if second_is_group && bad_group(&args[1], first) {
                diagnostics.push(InferenceDiagnostic::error(
                    "GROUP",
                    "boxplot grouping data are incompatible with X",
                ));
            }
            if (args.len() - option_start) % 2 != 0 {
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
                        request
                            .literals
                            .literal_args
                            .get(index + 1)
                            .and_then(LiteralContext::numeric_from_literal),
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
    let mut contract = CallContract::fixed(vec![if diagnostics.is_empty() {
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
