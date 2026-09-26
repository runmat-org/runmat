mod options;
use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-CORR-ARITY",
            "corr expects observations",
            0,
        ));
    }
    let left = request
        .arguments
        .first()
        .map(|v| matrix(v, 0, &mut diagnostics))
        .unwrap_or((DimensionFact::Unknown, DimensionFact::Unknown));
    let second = request.arguments.get(1);
    let second_text = second
        .is_some_and(|v| matches!(v.kind, ValueKindFact::String | ValueKindFact::Character))
        || request
            .literals
            .literal_args
            .get(1)
            .and_then(literal_text)
            .is_some();
    let second_unknown =
        second.is_some_and(|v| matches!(v.kind, ValueKindFact::Unknown)) && !second_text;
    let has_right = second.is_some() && !second_text && !second_unknown;
    let right = if has_right {
        matrix(second.unwrap(), 1, &mut diagnostics)
    } else if second_unknown {
        (DimensionFact::Unknown, DimensionFact::Unknown)
    } else {
        left.clone()
    };
    if let (DimensionFact::Known(l), DimensionFact::Known(r)) = (&left.0, &right.0) {
        if l != r {
            diagnostics.push(argument_error(
                "RM-CATALOG-CORR-ROWS",
                "X and Y must have equal observation counts",
                1,
            ));
        }
    }
    if !second_unknown {
        options::validate(request, if has_right { 2 } else { 1 }, &mut diagnostics);
    }
    let shape = ShapeFact::Shaped {
        dims: vec![left.1, right.1],
    };
    let count = request.outputs.requested.known_count().unwrap_or(1);
    let outputs = (0..count)
        .map(|slot| {
            output(if slot < 2 {
                shape.clone()
            } else {
                ShapeFact::Scalar
            })
        })
        .collect();
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}
fn matrix(
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (DimensionFact, DimensionFact) {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CORR-INPUT",
            "corr observations must be real numeric arrays",
            index,
        ));
    }
    match &value.shape {
        ShapeFact::Scalar => (DimensionFact::Known(1), DimensionFact::Known(1)),
        ShapeFact::Shaped { dims } => {
            if dims
                .iter()
                .skip(2)
                .any(|d| matches!(d,DimensionFact::Known(n)if *n!=1))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CORR-RANK",
                    "corr observations must be two-dimensional",
                    index,
                ));
            }
            (
                dims.first().cloned().unwrap_or(DimensionFact::Known(1)),
                dims.get(1).cloned().unwrap_or(DimensionFact::Known(1)),
            )
        }
        _ => (DimensionFact::Unknown, DimensionFact::Unknown),
    }
}
fn output(shape: ShapeFact) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    )
}
