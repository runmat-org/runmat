use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let input = request.arguments.first();
    let kind = match input.map(|value| &value.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })) => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        Some(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })) => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => ValueKindFact::Unknown,
    };
    let dims = input.and_then(|value| value.shape.known_dims());
    let rows = dims
        .as_ref()
        .and_then(|dims| dims.first().copied().flatten());
    let columns = dims
        .as_ref()
        .and_then(|dims| dims.get(1).copied().flatten());
    let matrix = |rows, columns| {
        ValueFact::proven(
            kind.clone(),
            ShapeFact::from(vec![rows, columns]),
            StorageFact::Dense,
        )
    };
    // Complete-row removal and option values can change retained component
    // counts; row reinsertion and variable-axis extents remain invariant.
    let mut contract = CallContract::fixed(vec![
        matrix(columns, None),
        matrix(rows, None),
        matrix(None, Some(1)),
        matrix(rows, Some(1)),
        matrix(None, Some(1)),
        matrix(Some(1), columns),
    ]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.is_empty() || !(request.arguments.len() - 1).is_multiple_of(2) {
        result.diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "pca requires X and complete name/value pairs",
        ));
    }
    if input.is_some_and(|input| {
        !matches!(
            input.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single | NumericClass::Double,
                    domain: NumericDomain::Real
                })
        )
    }) {
        result.diagnostics.push(InferenceDiagnostic::error(
            "INPUT",
            "pca requires real single or double observations",
        ));
    }
    result
}
