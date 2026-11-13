use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let single = request.arguments.len() == 1;
    for (index, input) in request.arguments.iter().enumerate() {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || input
            .numeric()
            .is_some_and(|n| n.domain == NumericDomain::Complex)
            || matches!(input.residency, ResidencyFact::Device { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GOBJECTS-SIZE",
                "sizes must be real host numeric data, not logical or resident controls",
                index,
            ));
        }
        let invalid_shape = if single && !input.is_scalar() {
            input.shape.rank().is_some_and(|rank| rank != 2)
                || input
                    .shape
                    .known_dims()
                    .is_some_and(|dims| dims.first().copied().flatten().is_some_and(|n| n != 1))
        } else {
            input.shape.element_count().is_some_and(|n| n != 1)
        };
        if invalid_shape {
            diagnostics.push(argument_error(
                "RM-CATALOG-GOBJECTS-SIZE-SHAPE",
                "use scalar dimensions or one numeric row vector",
                index,
            ));
        }
    }
    let dimensions = if request.arguments.is_empty() {
        Some(vec![Some(1), Some(1)])
    } else if single {
        match request.literals.literal_args.first() {
            Some(LiteralValue::Empty) => Some(vec![Some(0), Some(0)]),
            Some(LiteralValue::Vector(values)) => {
                let dims: Vec<_> = values
                    .iter()
                    .map(|value| dimension(value, &mut diagnostics, 0))
                    .collect();
                Some(match dims.len() {
                    0 => vec![Some(0), Some(0)],
                    1 => vec![dims[0], dims[0]],
                    _ => dims,
                })
            }
            Some(value) if !matches!(value, LiteralValue::Unknown) => {
                let dim = dimension(value, &mut diagnostics, 0);
                Some(vec![dim, dim])
            }
            _ if request.arguments[0].is_scalar() => Some(vec![None, None]),
            _ => None,
        }
    } else {
        Some(
            (0..request.arguments.len())
                .map(|index| {
                    request
                        .literals
                        .literal_args
                        .get(index)
                        .and_then(|value| dimension(value, &mut diagnostics, index))
                })
                .collect(),
        )
    };
    let shape = dimensions.map_or(ShapeFact::Unknown, ShapeFact::from);
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn dimension(
    value: &LiteralValue,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    index: usize,
) -> Option<usize> {
    if matches!(value, LiteralValue::Unknown) {
        return None;
    }
    let exact = match value {
        LiteralValue::Integer { .. } => value
            .integer_value_i128()
            .and_then(|n| usize::try_from(n.max(0)).ok()),
        _ => LiteralContext::numeric_from_literal(value).and_then(|n| {
            (n.is_finite() && n.fract() == 0.0 && n < 2.0_f64.powi(usize::BITS as i32))
                .then_some(n.max(0.0) as usize)
        }),
    };
    if exact.is_none() {
        diagnostics.push(argument_error(
            "RM-CATALOG-GOBJECTS-DIMENSION",
            "size must be a finite integer representable on the host",
            index,
        ));
    }
    exact
}
