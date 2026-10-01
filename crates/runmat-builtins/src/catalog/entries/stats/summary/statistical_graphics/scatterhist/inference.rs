//! Conditional chart-handle facts and known-invalid form diagnostics.
use super::{ScatterhistOptionKind, SCATTERHIST_MAX_BINS};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn handles() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    )
}

fn known_nonvector(shape: &ShapeFact) -> bool {
    let ShapeFact::Shaped { dims } = shape else {
        return false;
    };
    dims.iter()
        .filter(|dim| matches!(dim, runmat_types::DimensionFact::Known(n) if *n > 1))
        .count()
        > 1
}

fn bad_observation(fact: &ValueFact) -> bool {
    matches!(fact.storage, StorageFact::Sparse | StorageFact::Opaque)
        || known_nonvector(&fact.shape)
        || fact.shape.element_count() == Some(0)
        || !matches!(
            &fact.kind,
            ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real
        ) && !matches!(fact.kind, ValueKindFact::Logical | ValueKindFact::Unknown)
}

fn known_observation(fact: &ValueFact) -> bool {
    matches!(fact.storage, StorageFact::Scalar | StorageFact::Dense)
        && fact.shape.element_count().is_some()
        && match &fact.kind {
            ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
            ValueKindFact::Logical => true,
            _ => false,
        }
}

fn numeric_control(fact: &ValueFact) -> bool {
    matches!(&fact.kind, ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real)
        || matches!(fact.kind, ValueKindFact::Logical | ValueKindFact::Unknown)
}

fn literal_numeric(literal: Option<&LiteralValue>) -> Option<f64> {
    literal.and_then(LiteralContext::numeric_from_literal)
}

fn bad_option(
    kind: ScatterhistOptionKind,
    fact: &ValueFact,
    literal: Option<&LiteralValue>,
) -> bool {
    use ScatterhistOptionKind as K;
    let count = fact.shape.element_count();
    match kind {
        K::NBins => {
            !numeric_control(fact)
                || count.is_some_and(|n| n != 1 && n != 2)
                || literal_numeric(literal).is_some_and(|n| {
                    !n.is_finite() || n.fract() != 0.0 || n < 2.0 || n > SCATTERHIST_MAX_BINS as f64
                })
        }
        K::Group => match &fact.kind {
            ValueKindFact::Numeric(number) => number.domain != NumericDomain::Real,
            ValueKindFact::Logical
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Cell(_)
            | ValueKindFact::Unknown => false,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_some_and(|class| !class.is(runmat_types::standard::CATEGORICAL)),
            _ => true,
        },
        K::Kernel | K::Legend | K::PlotGroup => {
            count.is_some_and(|n| n != 1)
                && !matches!(fact.kind, ValueKindFact::String | ValueKindFact::Character)
        }
        K::MarkerSize | K::LineWidth | K::Bandwidth => {
            !numeric_control(fact)
                || count == Some(0)
                || literal_numeric(literal).is_some_and(|n| {
                    !n.is_finite() || n < 0.0 || (kind == K::Bandwidth && n == 0.0)
                })
        }
        K::Color => {
            matches!(&fact.kind, ValueKindFact::Numeric(number) if number.domain == NumericDomain::Complex)
        }
        K::Location | K::Direction | K::Marker | K::LineStyle | K::Style => !matches!(
            fact.kind,
            ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Unknown
        ),
        K::Parent => false,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    let mut unknown_control = false;
    if args.len() < 2 {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "scatterhist requires x and y observations",
        ));
    }
    for (index, fact) in args.iter().take(2).enumerate() {
        if bad_observation(fact) {
            diagnostics.push(InferenceDiagnostic::error(
                "DATA",
                if index == 0 {
                    "scatterhist x must be a nonempty real numeric vector"
                } else {
                    "scatterhist y must be a nonempty real numeric vector"
                },
            ));
        }
    }
    if args.len() >= 2
        && args[0]
            .shape
            .element_count()
            .zip(args[1].shape.element_count())
            .is_some_and(|(x, y)| x != y)
    {
        diagnostics.push(InferenceDiagnostic::error(
            "LENGTH",
            "scatterhist x and y must contain the same number of elements",
        ));
    }
    if args.len() > 2 {
        if !(args.len() - 2).is_multiple_of(2) {
            diagnostics.push(InferenceDiagnostic::error(
                "OPTION",
                "scatterhist name-value options must be paired",
            ));
        }
        for index in (2..args.len()).step_by(2) {
            let name = &args[index];
            if !matches!(
                name.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            ) || name.shape.element_count().is_some_and(|n| n != 1)
            {
                diagnostics.push(InferenceDiagnostic::error(
                    "OPTION",
                    "scatterhist option names must be scalar text",
                ));
                continue;
            }
            let Some(text) = request.literals.literal_string_at(index) else {
                unknown_control = true;
                continue;
            };
            let Some(kind) = ScatterhistOptionKind::parse(&text) else {
                diagnostics.push(InferenceDiagnostic::error(
                    "OPTION",
                    "scatterhist option is unknown or unsupported",
                ));
                continue;
            };
            if let Some(value) = args.get(index + 1) {
                if bad_option(kind, value, request.literals.literal_args.get(index + 1)) {
                    diagnostics.push(InferenceDiagnostic::error(
                        "OPTION",
                        "scatterhist option value is invalid",
                    ));
                }
            }
        }
    }
    let valid = diagnostics.is_empty();
    let known_data = args.iter().take(2).all(known_observation);
    let contract = CallContract::fixed(vec![if valid && known_data && !unknown_control {
        handles()
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    }]);
    let mut contract = contract;
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
