use super::super::super::super::argument_error;
use runmat_types::{
    CallRequest, InferenceDiagnostic, LiteralContext, LiteralValue, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn check(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    for (index, input) in request.arguments.iter().take(2).enumerate() {
        if matches!(input.residency, ResidencyFact::Device { .. })
            || matches!(input.storage, StorageFact::Sparse | StorageFact::Opaque)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-NCHOOSEK-REPRESENTATION",
                "nchoosek requires supported host values",
                index,
            ));
        }
    }
    if let Some(selection) = request.arguments.get(1) {
        let invalid_kind = match &selection.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Complex,
            _ => true,
        };
        if invalid_kind
            || selection
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-NCHOOSEK-SELECTION",
                "nchoosek selection must be a real numeric scalar",
                1,
            ));
        }
    }
    let selection = literal_count(request.literals.literal_args.get(1));
    if matches!(selection, Some(None)) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NCHOOSEK-SELECTION-VALUE",
            "nchoosek selection must be a nonnegative representable integer",
            1,
        ));
    }
    if request.arguments.first().is_some_and(|input| {
        input.is_scalar()
            && input
                .numeric()
                .is_some_and(|numeric| numeric.domain == NumericDomain::Real)
    }) && matches!(
        literal_count(request.literals.literal_args.first()),
        Some(None)
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NCHOOSEK-POPULATION-VALUE",
            "nchoosek scalar population must be a nonnegative representable integer",
            0,
        ));
    }
    selection.flatten()
}

fn literal_count(literal: Option<&LiteralValue>) -> Option<Option<usize>> {
    let literal = literal?;
    if let LiteralValue::Integer { text, .. } = literal {
        return Some(text.parse::<usize>().ok());
    }
    let value = LiteralContext::numeric_from_literal(literal)?;
    Some(
        (value.is_finite()
            && value >= 0.0
            && value.fract() == 0.0
            && value <= usize::MAX as f64
            && (usize::BITS != 64 || value != usize::MAX as f64))
            .then_some(value as usize),
    )
}

pub(super) fn check_vector(input: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let invalid = input
        .shape
        .known_dims()
        .is_some_and(|dims| match dims.as_slice() {
            [] | [_] => false,
            [Some(rows), Some(columns)] => *rows > 1 && *columns > 1,
            [_, _] => false,
            _ => true,
        });
    if invalid {
        diagnostics.push(argument_error(
            "RM-CATALOG-NCHOOSEK-VECTOR",
            "nchoosek first input must be a scalar or vector",
            0,
        ));
    }
}

pub(super) fn vector_output_shape(input: &ValueFact, selection: Option<usize>) -> ShapeFact {
    let Some(k) = selection else {
        return ShapeFact::Ranked { rank: 2 };
    };
    let rows = if k == 0 {
        Some(1)
    } else {
        input.shape.element_count().and_then(|n| rows(n, k))
    };
    ShapeFact::from(vec![rows, Some(k)])
}

fn rows(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let count = k.min(n - k);
    let mut rows = 1_u128;
    for index in 1..=count {
        rows = rows.checked_mul((n - count + index) as u128)? / index as u128;
        // Runtime rejects above this materialized element bound; inference
        // must also avoid work proportional to enormous literal selections.
        if rows.checked_mul(k as u128)? > 50_000_000 {
            return None;
        }
    }
    usize::try_from(rows).ok()
}
