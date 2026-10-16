use crate::catalog::inference::{argument_error, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy)]
pub(super) enum SequenceKind {
    Linear,
    Logarithmic,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    sequence: SequenceKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-SEQUENCE-ARITY",
            "a sequence requires two endpoints and an optional count",
            0,
        ));
    }
    let endpoints = request.arguments.iter().take(2).collect::<Vec<_>>();
    for (index, endpoint) in endpoints.iter().enumerate() {
        if endpoint
            .shape
            .element_count()
            .is_some_and(|count| count != 1)
            || !matches!(
                endpoint.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double | NumericClass::Single,
                        ..
                    })
            )
        {
            diagnostics.push(argument_error(
                "RM-SEQUENCE-ENDPOINT",
                "endpoints must be single or double numeric scalars",
                index,
            ));
        }
    }
    let kind = match endpoints.as_slice() {
        [first, second] => match (first.numeric(), second.numeric()) {
            (Some(first), Some(second))
                if matches!(first.class, NumericClass::Double | NumericClass::Single)
                    && matches!(second.class, NumericClass::Double | NumericClass::Single) =>
            {
                numeric_kind(
                    if first.class == NumericClass::Single || second.class == NumericClass::Single {
                        NumericClass::Single
                    } else {
                        NumericClass::Double
                    },
                    if first.domain == NumericDomain::Complex
                        || second.domain == NumericDomain::Complex
                    {
                        NumericDomain::Complex
                    } else {
                        NumericDomain::Real
                    },
                )
            }
            _ => ValueKindFact::Unknown,
        },
        _ => ValueKindFact::Unknown,
    };
    let count = request
        .literals
        .literal_args
        .get(2)
        .and_then(|literal| count(literal, sequence))
        .or_else(|| {
            (request.arguments.len() == 2).then_some(match sequence {
                SequenceKind::Linear => 100,
                SequenceKind::Logarithmic => 50,
            })
        });
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = kind;
    output.shape = ShapeFact::from(vec![Some(1), count]);
    output.storage = StorageFact::Dense;
    finish_fixed(entry, request, output, diagnostics)
}

fn count(literal: &LiteralValue, sequence: SequenceKind) -> Option<usize> {
    if let LiteralValue::Integer { text, .. } = literal {
        if text.starts_with('-') {
            return Some(0);
        }
        return text.parse::<usize>().ok();
    }
    let numeric = runmat_types::LiteralContext::numeric_from_literal(literal)?;
    if numeric.is_nan() && matches!(sequence, SequenceKind::Linear) {
        return Some(1);
    }
    if !numeric.is_finite() || numeric >= usize::MAX as f64 {
        return None;
    }
    Some(if numeric <= 0.0 {
        0
    } else {
        numeric.floor() as usize
    })
}
